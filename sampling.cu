// Production CUDA sampler: kernels, host dispatch, and Torch bindings.
// This translation unit has no dependencies on the experimental repository files.
#include <torch/csrc/utils/pybind.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>

using RAND = curandStatePhilox4_32_10_t;
constexpr int RAND_SETUP_THREADS = 256;
struct HistogramLayout {
    int passes, max_bins;
    int widths[5], shifts[5];
    int shards = 2;
};
constexpr int HIST_SHARDS = 2;

__device__ __forceinline__ float safe_float(float x) {
    return isnan(x) ? 0.f : (isinf(x) ? copysignf(FLT_MAX, x) : x);
}

// Exact floor(p*2^exponent) for finite nonnegative probabilities. The caller
// keeps the result below INT64_MAX, including for subnormal p.
__device__ __forceinline__ unsigned long long quantize_power2(float p, int exponent) {
    unsigned bits = __float_as_uint(p);
    int e = (bits >> 23) & 255;
    unsigned m = (bits & 0x7fffffu) | (e ? 0x800000u : 0u);
    if (!m)
        return 0;
    int shift = exponent + (e ? e - 150 : -149);
    if (shift >= 0)
        return static_cast<unsigned long long>(m) << shift;
    return shift <= -24 ? 0 : static_cast<unsigned long long>(m) >> (-shift);
}

template <typename T, bool Maximum> __device__ __forceinline__ T warp_reduce(T value) {
#if __CUDA_ARCH__ >= 800
    if constexpr (Maximum && std::is_same<T, float>::value) {
        unsigned bits = __float_as_uint(value);
        unsigned ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        ordered = __reduce_max_sync(0xffffffff, ordered);
        return __uint_as_float((ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered);
    } else if constexpr (!Maximum && std::is_same<T, unsigned>::value) {
        return __reduce_add_sync(0xffffffff, value);
    } else
#endif
    {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            T other = __shfl_down_sync(0xffffffff, value, offset);
            if constexpr (Maximum)
                value = value > other ? value : other;
            else
                value += other;
        }
        return value;
    }
}

template <typename T, bool Maximum = false> __device__ T block_reduce(T value, T *scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    value = warp_reduce<T, Maximum>(value);
    if (lane == 0)
        scratch[warp] = value;
    __syncthreads();
    if (warp == 0) {
        T total = lane < blockDim.x / 32 ? scratch[lane] : T(0);
        if constexpr (Maximum) {
            if (lane >= blockDim.x / 32)
                total = T(-FLT_MAX);
        }
        total = warp_reduce<T, Maximum>(total);
        if (lane == 0)
            scratch[0] = total;
    }
    __syncthreads();
    T total = scratch[0];
    __syncthreads(); // Prevent a subsequent reduction from overwriting readers' data.
    return total;
}

template <typename T> __device__ T block_scan(T value, T *scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        T other = __shfl_up_sync(0xffffffff, value, offset);
        if (lane >= offset)
            value += other;
    }
    if (lane == 31)
        scratch[warp] = value;
    __syncthreads();
    if (warp == 0) {
        T total = lane < blockDim.x / 32 ? scratch[lane] : T(0);
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            T other = __shfl_up_sync(0xffffffff, total, offset);
            if (lane >= offset)
                total += other;
        }
        if (lane < blockDim.x / 32)
            scratch[lane] = total;
    }
    __syncthreads();
    if (warp)
        value += scratch[warp - 1];
    __syncthreads();
    return value;
}

__global__ void create_rng_kernel(RAND *states, unsigned long long seed, int64_t count) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        curand_init(seed, i, 0, states + i);
}

__device__ unsigned long long random_below(RAND *state, unsigned long long total) {
    // Rejection removes modulo bias, including for int64 totals beyond 2^53.
    const unsigned long long cutoff = (0ULL - total) % total;
    unsigned long long r;
    do {
        unsigned hi = curand(state), lo = curand(state);
        r = (static_cast<unsigned long long>(hi) << 32) | lo;
    } while (r < cutoff);
    return r % total;
}

__device__ unsigned random_below32(RAND *state, unsigned total) {
    unsigned value = curand(state);
    unsigned high = __umulhi(value, total), low = value * total;
    if (low < total) {
        const unsigned cutoff = (0u - total) % total;
        while (low < cutoff) {
            value = curand(state);
            high = __umulhi(value, total);
            low = value * total;
        }
    }
    return high;
}

// Compile-time constraints belong to host launch code.
template <typename Weight, bool Penalties, bool Cooperative = false>
__host__ constexpr void validate_sampling_configuration() {
    static_assert(std::is_same<Weight, unsigned>::value || std::is_same<Weight, int64_t>::value,
                  "weights must be uint32 or int64");
    static_assert(!Cooperative || (std::is_same<Weight, unsigned>::value && !Penalties),
                  "cooperative sampling requires uint32 weights without penalties");
}

template <typename Weight, bool Penalties, bool Cooperative = false, int CachedCandidates = 0>
__device__ __forceinline__ void
sampling_body(int T, int V, const float *logits, float *penalties, int *outputs, RAND *states,
              float *probs, Weight *weights, float inv_temperature, int top_k, float top_p,
              float presence, float repetition, float decay, HistogramLayout layout,
              const unsigned *first_histogram = nullptr) {
    const int t = threadIdx.x, b = blockIdx.x;
    logits += (int64_t(b) * T + T - 1) * V;
    probs += int64_t(b) * V;
    weights += int64_t(b) * V;
    if constexpr (Penalties)
        penalties += int64_t(b) * V;
    __shared__ float fscratch[1024 / 32];
    __shared__ Weight iscratch[1024 / 32];
    const int histogram_stride = layout.max_bins + 1;
    using Mass = typename std::conditional<sizeof(Weight) == 4, unsigned, unsigned long long>::type;
    constexpr float mass_scale = sizeof(Weight) == 4 ? 0x1p31f : 0x1p62f;
    extern __shared__ __align__(8) unsigned char histogram_storage[];
    Mass *mass = reinterpret_cast<Mass *>(histogram_storage);
    unsigned *counts = reinterpret_cast<unsigned *>(mass + layout.shards * histogram_stride);
    float *candidate_cache = reinterpret_cast<float *>(counts + layout.shards * histogram_stride);
    unsigned candidate_count = 0, candidate_overflow = 0;
    __shared__ unsigned prefix, above_count;
    __shared__ Mass histogram_scan[32];
    __shared__ unsigned count_scan[32];
    __shared__ Mass above_mass;
    __shared__ float compensation;
    __shared__ Weight total_weight;
    __shared__ Weight random_weight;
    __shared__ int selected, owner;
    __shared__ Weight residual;

    const bool vectorized = (V & 3) == 0;
    if constexpr (!Cooperative) {
        float maximum = -FLT_MAX;
        if (vectorized) {
            for (int i = t; i < V / 4; i += blockDim.x) {
                float4 x = reinterpret_cast<const float4 *>(logits)[i];
                float4 penalty;
                if constexpr (Penalties)
                    penalty = reinterpret_cast<const float4 *>(penalties)[i];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float v = safe_float(reinterpret_cast<float *>(&x)[j]);
                    if constexpr (Penalties)
                        v -= reinterpret_cast<float *>(&penalty)[j];
                    v = safe_float(v * inv_temperature);
                    reinterpret_cast<float *>(&x)[j] = v;
                    maximum = fmaxf(maximum, v);
                }
                reinterpret_cast<float4 *>(probs)[i] = x;
            }
        } else {
            for (int i = t; i < V; i += blockDim.x) {
                float x = safe_float(logits[i]);
                if constexpr (Penalties)
                    x -= penalties[i];
                x = safe_float(x * inv_temperature);
                probs[i] = x;
                maximum = fmaxf(maximum, x);
            }
        }
        maximum = block_reduce<float, true>(maximum, fscratch);
        float denominator = 0.f;
        if (vectorized) {
            for (int i = t; i < V / 4; i += blockDim.x) {
                float4 x = reinterpret_cast<const float4 *>(probs)[i];
                float sum = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float p = exp2f(reinterpret_cast<float *>(&x)[j] - maximum);
                    reinterpret_cast<float *>(&x)[j] = p;
                    sum += p;
                }
                reinterpret_cast<float4 *>(probs)[i] = x;
                denominator += sum;
            }
        } else {
            for (int i = t; i < V; i += blockDim.x) {
                float p = exp2f(probs[i] - maximum);
                probs[i] = p;
                denominator += p;
            }
        }
        denominator = block_reduce(denominator, fscratch);
        const float inv_denominator = 1.f / denominator;
        if (vectorized) {
            for (int i = t; i < V / 4; i += blockDim.x) {
                float4 x = reinterpret_cast<const float4 *>(probs)[i];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    reinterpret_cast<float *>(&x)[j] *= inv_denominator;
                reinterpret_cast<float4 *>(probs)[i] = x;
            }
        } else {
            for (int i = t; i < V; i += blockDim.x)
                probs[i] *= inv_denominator;
        }
    }
    if (t == 0) {
        prefix = 0;
        above_mass = 0;
        above_count = 0;
        compensation = 1.f;
    }
    __syncthreads();

    // Nonnegative IEEE float bits are ordered like their values. Refine only
    // the boundary bucket with the requested bit partition.
    if (top_k < V || top_p < 1.) {
        for (int level = 0; level < layout.passes; ++level) {
            const int shift = layout.shifts[level], bins = 1 << layout.widths[level];
            unsigned overflow = 1;
            if constexpr (CachedCandidates > 0) {
                if (level >= 2)
                    overflow = block_reduce(candidate_overflow, iscratch);
            }
            for (int bin = t; bin < bins; bin += blockDim.x) {
                for (int shard = 0; shard < layout.shards; ++shard) {
                    const int index = shard * histogram_stride + bin;
                    mass[index] = 0;
                    counts[index] = 0;
                }
            }
            __syncthreads();
            const unsigned boundary = prefix;
            const int shard = (t >> 5) % layout.shards;
            if (Cooperative && level == 0) {
                const int tiles = (V + 2047) / 2048;
                const int stored_bins = bins / 2;
                const unsigned *h = first_histogram + int64_t(b) * tiles * stored_bins * 2;
                for (int bin = t; bin < bins; bin += blockDim.x) {
                    Mass sum = 0;
                    unsigned count = 0;
                    for (int tile = 0; bin < stored_bins && tile < tiles; ++tile) {
                        uint2 value = reinterpret_cast<const uint2 *>(h)[tile * stored_bins + bin];
                        sum += value.x;
                        count += value.y;
                    }
                    mass[bin] = sum;
                    counts[bin] = count;
                }
            } else {
                auto insert = [&](float p) {
                    unsigned bits = __float_as_uint(p);
                    if (level == 0 || (bits >> (shift + layout.widths[level])) ==
                                          (boundary >> (shift + layout.widths[level]))) {
                        if constexpr (CachedCandidates > 0) {
                            if (level == 1) {
                                if (candidate_count < CachedCandidates)
                                    candidate_cache[candidate_count * blockDim.x + t] = p;
                                else
                                    candidate_overflow = 1;
                                ++candidate_count;
                            }
                        }
                        int bin = shard * histogram_stride + ((bits >> shift) & (bins - 1));
                        Mass q;
                        if constexpr (sizeof(Weight) == 4)
                            q = __float2uint_rn(p * mass_scale);
                        else
                            q = __float2ull_rz(p * mass_scale);
                        atomicAdd(mass + bin, q);
                        atomicAdd(counts + bin, 1u);
                    }
                };
                if (CachedCandidates > 0 && level >= 2 && !overflow) {
                    for (int i = 0; i < int(candidate_count); ++i)
                        insert(candidate_cache[i * blockDim.x + t]);
                } else if (vectorized) {
                    for (int i = t; i < V / 4; i += blockDim.x) {
                        float4 x = reinterpret_cast<const float4 *>(probs)[i];
                        insert(x.x);
                        insert(x.y);
                        insert(x.z);
                        insert(x.w);
                    }
                } else {
                    for (int i = t; i < V; i += blockDim.x)
                        insert(probs[i]);
                }
            }
            __syncthreads();
            // Scan contiguous descending bin chunks in parallel. Only the
            // winning chunk (at most ceil(bins/blockDim.x) bins) is searched
            // serially; no thread walks the entire histogram.
            const Mass previous_mass = above_mass;
            const unsigned previous_count = above_count;
            const int bin_chunk = (bins + int(blockDim.x) - 1) / int(blockDim.x);
            const int begin_bin = t * bin_chunk;
            const int end_bin = min(begin_bin + bin_chunk, bins);
            Mass chunk_mass = 0;
            unsigned chunk_count = 0;
            for (int position = begin_bin; position < end_bin; ++position) {
                const int bin = bins - 1 - position;
                for (int shard = 0; shard < layout.shards; ++shard) {
                    const Mass value = mass[shard * histogram_stride + bin];
                    chunk_mass += value;
                    chunk_count += counts[shard * histogram_stride + bin];
                }
            }
            const Mass cumulative_mass = block_scan(chunk_mass, histogram_scan);
            const unsigned cumulative_count = block_scan(chunk_count, count_scan);
            // top_p=1 disables the mass bound, even when rounded softmax
            // probabilities sum slightly above one. Preserve rare top-k weights.
            const Mass raw_target = top_p >= 1.f ? ~Mass(0) : static_cast<Mass>(top_p * mass_scale);
            const Mass target = raw_target > 0 ? raw_target : Mass(1);
            Mass before_mass = previous_mass + cumulative_mass - chunk_mass;
            unsigned before_count = previous_count + cumulative_count - chunk_count;
            if (chunk_count && before_count < unsigned(top_k) && before_mass < target &&
                (previous_count + cumulative_count >= unsigned(top_k) ||
                 previous_mass + cumulative_mass >= target)) {
                for (int position = begin_bin; position < end_bin; ++position) {
                    const int bin = bins - 1 - position;
                    Mass m = 0;
                    unsigned c = 0;
                    for (int shard = 0; shard < layout.shards; ++shard) {
                        const Mass value = mass[shard * histogram_stride + bin];
                        m += value;
                        c += counts[shard * histogram_stride + bin];
                    }
                    if (c && (before_count + c >= unsigned(top_k) || before_mass + m >= target)) {
                        prefix |= unsigned(bin) << shift;
                        above_mass = before_mass;
                        above_count = before_count;
                        if (level == layout.passes - 1) {
                            const float by_k = float(top_k - int(before_count)) / float(c);
                            const Mass remaining =
                                raw_target > before_mass ? raw_target - before_mass : 0;
                            const float by_p = m > 0 ? float(remaining) / float(m) : 1.f;
                            compensation = fmaxf(0.f, fminf(1.f, fminf(by_k, by_p)));
                            if (before_mass == 0 && compensation == 0.f && top_p > 0.f)
                                compensation = FLT_MIN;
                        }
                        break;
                    }
                    before_mass += m;
                    before_count += c;
                }
            }
            __syncthreads();
        }
    }
    const float threshold = __uint_as_float(prefix);
    if constexpr (Cooperative) {
        if (t == 0) {
            unsigned *control = const_cast<unsigned *>(first_histogram);
            control[0] = __float_as_uint(threshold);
            control[1] = __float_as_uint(compensation);
        }
        return;
    }
    float retained = 0.f;
    if (vectorized) {
        for (int i = t; i < V / 4; i += blockDim.x) {
            float4 x = reinterpret_cast<const float4 *>(probs)[i];
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = reinterpret_cast<float *>(&x)[j];
                sum += p > threshold ? p : (p == threshold ? p * compensation : 0.f);
            }
            retained += sum;
        }
    } else {
        for (int i = t; i < V; i += blockDim.x) {
            float p = probs[i];
            retained += p > threshold ? p : (p == threshold ? p * compensation : 0.f);
        }
    }
    retained = block_reduce(retained, fscratch);
    // Uint32 normalizes near its full range with overflow headroom. Int64 uses an exact
    // power-of-two scale so conversion preserves the input float's bits and
    // the sum stays safely below 2^63, without per-token FP64 multiplication/division.
    const float factor = retained > 0.f ? 4.26e9f / retained : 0.f;
    const int exponent = retained > 0.f ? 61 - ilogbf(retained) : 0;
    Weight local = 0;
    if (vectorized) {
        for (int i = t; i < V / 4; i += blockDim.x) {
            float4 x = reinterpret_cast<const float4 *>(probs)[i];
            Weight values[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = reinterpret_cast<float *>(&x)[j];
                const float filtered =
                    p > threshold ? p : (p == threshold ? p * compensation : 0.f);
                Weight weight;
                if constexpr (sizeof(Weight) == 4) {
                    weight = Weight(isfinite(factor) ? filtered * factor
                                                     : (filtered / retained) * 4.26e9f);
                } else {
                    weight = Weight(quantize_power2(filtered, exponent));
                }
                values[j] = weight;
                local += weight;
            }
            if constexpr (sizeof(Weight) == 4) {
                reinterpret_cast<uint4 *>(weights)[i] =
                    make_uint4(values[0], values[1], values[2], values[3]);
            } else {
                reinterpret_cast<longlong2 *>(weights)[2 * i] =
                    make_longlong2(values[0], values[1]);
                reinterpret_cast<longlong2 *>(weights)[2 * i + 1] =
                    make_longlong2(values[2], values[3]);
            }
        }
    } else {
        for (int i = t; i < V; i += blockDim.x) {
            float p = probs[i];
            float filtered = p > threshold ? p : (p == threshold ? p * compensation : 0.f);
            Weight weight;
            if constexpr (sizeof(Weight) == 4) {
                weight =
                    Weight(isfinite(factor) ? filtered * factor : (filtered / retained) * 4.26e9f);
            } else {
                weight = Weight(quantize_power2(filtered, exponent));
            }
            weights[i] = weight;
            local += weight;
        }
    }
    Weight cumulative = block_scan(local, iscratch);
    if (t == blockDim.x - 1)
        total_weight = cumulative;
    __syncthreads();
    if (t == 0) {
        selected = 0;
        owner = 0;
        residual = 0;
        if constexpr (sizeof(Weight) == 4) {
            random_weight = total_weight > 0 ? random_below32(states + b, total_weight) : 0;
        } else {
            random_weight = total_weight > 0 ? random_below(states + b, total_weight) : 0;
        }
    }
    __syncthreads();
    // First select the thread's strided token group using an exact integer interval.
    if (total_weight > 0 && random_weight >= cumulative - local && random_weight < cumulative) {
        owner = t;
        residual = random_weight - (cumulative - local);
    }
    __syncthreads();
    // Distribute the winning group's tokens over the block instead of making
    // one thread serially read V/blockDim.x weights. Integer chunk scans preserve
    // the same categorical interval and skip zero-weight tokens.
    const int group_size = vectorized
                               ? 4 * ((V / 4 - owner + int(blockDim.x) - 1) / int(blockDim.x))
                               : (V - owner + int(blockDim.x) - 1) / int(blockDim.x);
    const int chunk_size = (group_size + int(blockDim.x) - 1) / int(blockDim.x);
    const int begin = t * chunk_size, end = min(begin + chunk_size, group_size);
    Weight chunk_weight = 0;
    for (int j = begin; j < end; ++j)
        chunk_weight += weights[vectorized ? 4 * (owner + (j / 4) * blockDim.x) + j % 4
                                           : owner + j * blockDim.x];
    Weight chunk_prefix = block_scan(chunk_weight, iscratch);
    if (total_weight > 0 && residual >= chunk_prefix - chunk_weight && residual < chunk_prefix) {
        Weight r = residual - (chunk_prefix - chunk_weight);
        for (int j = begin; j < end; ++j) {
            const int i =
                vectorized ? 4 * (owner + (j / 4) * blockDim.x) + j % 4 : owner + j * blockDim.x;
            if (r < weights[i]) {
                selected = i;
                break;
            }
            r -= weights[i];
        }
    }
    __syncthreads();
    if (t == 0)
        outputs[b] = selected;
    if constexpr (Penalties) {
        for (int i = t; i < V; i += blockDim.x) {
            float p = penalties[i];
            penalties[i] = fmaf(p, decay, i == selected ? (p == 0.f ? presence : repetition) : 0.f);
        }
    }
}

template <typename Weight, bool Penalties>
__global__ void sampling_kernel(int T, int V, const float *logits, float *penalties, int *outputs,
                                RAND *states, float *probs, Weight *weights, float inv_temperature,
                                int top_k, float top_p, float presence, float repetition,
                                float decay, HistogramLayout layout) {
    sampling_body<Weight, Penalties>(T, V, logits, penalties, outputs, states, probs, weights,
                                     inv_temperature, top_k, top_p, presence, repetition, decay,
                                     layout);
}

#include <cooperative_groups.h>

// B=1,V=65536 cooperative fast path. Every exponential uses one global
// maximum, so equal logits retain equal probabilities across tiles.
__device__ __forceinline__ void cooperative_max(int T, int V, const float *logits, float *probs,
                                                float *partials, float inv_temperature) {
    const int t = threadIdx.x, tile = blockIdx.x, begin = tile * 2048, end = min(begin + 2048, V);
    logits += int64_t(T - 1) * V;
    __shared__ float scratch[32];
    float maximum = -FLT_MAX;
    for (int i = begin + 4 * t; i < end; i += 4 * blockDim.x) {
        float4 x = reinterpret_cast<const float4 *>(logits)[i / 4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v = safe_float(safe_float(reinterpret_cast<float *>(&x)[j]) * inv_temperature);
            maximum = fmaxf(maximum, v);
            reinterpret_cast<float *>(&x)[j] = v;
        }
        reinterpret_cast<float4 *>(probs)[i / 4] = x;
    }
    maximum = block_reduce<float, true>(maximum, scratch);
    if (t == 0)
        partials[2 * tile] = maximum;
}
__device__ __forceinline__ void global_exp_body(int V, float *probs, float *partials) {
    const int t = threadIdx.x, tile = blockIdx.x, b = blockIdx.y, tiles = (V + 2047) / 2048;
    __shared__ float scratch[32];
    float maximum = t < tiles ? partials[(b * tiles + t) * 2] : -FLT_MAX;
    maximum = block_reduce<float, true>(maximum, scratch);
    probs += int64_t(b) * V;
    const int begin = tile * 2048, end = min(begin + 2048, V);
    float sum = 0.f;
    for (int i = begin + 4 * t; i < end; i += 4 * blockDim.x) {
        float4 x = reinterpret_cast<float4 *>(probs)[i / 4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p = exp2f(reinterpret_cast<float *>(&x)[j] - maximum);
            sum += p;
            reinterpret_cast<float *>(&x)[j] = p;
        }
        reinterpret_cast<float4 *>(probs)[i / 4] = x;
    }
    sum = block_reduce(sum, scratch);
    if (t == 0)
        partials[(b * tiles + tile) * 2 + 1] = sum;
}

__device__ __forceinline__ void cooperative_normalize_histogram(int V, float *probs,
                                                                const float *partials,
                                                                unsigned *histograms, int width,
                                                                int shift, bool needed) {
    const int t = threadIdx.x, tile = blockIdx.x, tiles = (V + 2047) / 2048,
              bins = 1 << (width - 1);
    __shared__ float scratch[32];
    extern __shared__ unsigned storage[];
    unsigned *mass = storage;
    unsigned *counts = storage + bins;
    const float denominator = block_reduce(t < tiles ? partials[2 * t + 1] : 0.f, scratch);
    const float scale = 1.f / denominator;
    if (needed)
        for (int bin = t; bin < bins; bin += blockDim.x) {
            mass[bin] = 0;
            counts[bin] = 0;
        }
    __syncthreads();
    const int begin = tile * 2048, end = min(begin + 2048, V);
    for (int i = begin + 4 * t; i < end; i += 4 * blockDim.x) {
        float4 x = reinterpret_cast<float4 *>(probs)[i / 4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p = reinterpret_cast<float *>(&x)[j] * scale;
            reinterpret_cast<float *>(&x)[j] = p;
            if (needed) {
                int bin = __float_as_uint(p) >> shift;
                atomicAdd(mass + bin, __float2uint_rn(p * 0x1p31f));
                atomicAdd(counts + bin, 1u);
            }
        }
        reinterpret_cast<float4 *>(probs)[i / 4] = x;
    }
    __syncthreads();
    if (needed) {
        unsigned *h = histograms + int64_t(tile) * bins * 2;
        for (int bin = t; bin < bins; bin += blockDim.x) {
            h[2 * bin] = mass[bin];
            h[2 * bin + 1] = counts[bin];
        }
    }
}
__global__ void cooperative_sampling_kernel(int T, int V, const float *logits, int *outputs,
                                            RAND *states, float *probs, unsigned *weights,
                                            float *partials, unsigned *histograms,
                                            float inv_temperature, int k, float p,
                                            HistogramLayout layout) {
    // All tiles share a maximum and denominator, preserving ties across tiles.
    cooperative_max(T, V, logits, probs, partials, inv_temperature);
    cooperative_groups::this_grid().sync();
    global_exp_body(V, probs, partials);
    cooperative_groups::this_grid().sync();
    cooperative_normalize_histogram(V, probs, partials, histograms, layout.widths[0],
                                    layout.shifts[0], k < V || p < 1.f);
    cooperative_groups::this_grid().sync();
    // Refine the histogram in block zero and publish the cutoff and tie fraction.
    if (blockIdx.x == 0) {
        if (blockDim.x == 512)
            sampling_body<unsigned, false, true, 16>(T, V, logits, nullptr, outputs, states, probs,
                                                     weights, inv_temperature, k, p, 0.f, 0.f, 0.f,
                                                     layout, histograms);
        else
            sampling_body<unsigned, false, true, 8>(T, V, logits, nullptr, outputs, states, probs,
                                                    weights, inv_temperature, k, p, 0.f, 0.f, 0.f,
                                                    layout, histograms);
    }
    cooperative_groups::this_grid().sync();
    // Apply the cutoff, then quantize using the mass retained by the whole grid.
    const int t = threadIdx.x, tile = blockIdx.x, tiles = 32;
    const float threshold = __uint_as_float(histograms[0]),
                compensation = __uint_as_float(histograms[1]);
    const int i = tile * 512 + t;
    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
    if (t < 512) {
        x = reinterpret_cast<float4 *>(probs)[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p = reinterpret_cast<float *>(&x)[j];
            reinterpret_cast<float *>(&x)[j] =
                p > threshold ? p : (p == threshold ? p * compensation : 0.f);
        }
    }
    __shared__ float fscratch[32];
    __shared__ unsigned iscratch[32];
    float retained = block_reduce(x.x + x.y + x.z + x.w, fscratch);
    if (t == 0)
        partials[tile] = retained;
    cooperative_groups::this_grid().sync();
    retained = block_reduce(t < tiles ? partials[t] : 0.f, fscratch);
    const float factor = retained > 0.f ? 4.26e9f / retained : 0.f;
    uint4 w = make_uint4(0, 0, 0, 0);
    if (t < 512) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float value = reinterpret_cast<float *>(&x)[j];
            reinterpret_cast<unsigned *>(&w)[j] =
                unsigned(isfinite(factor) ? value * factor : (value / retained) * 4.26e9f);
        }
        reinterpret_cast<uint4 *>(weights)[i] = w;
    }
    unsigned tile_weight = block_reduce(w.x + w.y + w.z + w.w, iscratch);
    if (t == 0)
        histograms[2 + tile] = tile_weight;
    cooperative_groups::this_grid().sync();
    // Select a tile by integer mass, then select a token within that tile.
    if (tile == 0) {
        __shared__ unsigned total, random, residual, chosen;
        unsigned local = t < tiles ? histograms[2 + t] : 0;
        unsigned cumulative = block_scan(local, iscratch);
        if (t == blockDim.x - 1)
            total = cumulative;
        __syncthreads();
        if (!total) {
            if (t == 0)
                outputs[0] = 0;
            return;
        }
        if (t == 0)
            random = random_below32(states, total);
        __syncthreads();
        if (local && random >= cumulative - local && random < cumulative) {
            chosen = t;
            residual = random - (cumulative - local);
        }
        __syncthreads();
        w = t < 512 ? reinterpret_cast<uint4 *>(weights)[chosen * 512 + t] : make_uint4(0, 0, 0, 0);
        local = w.x + w.y + w.z + w.w;
        cumulative = block_scan(local, iscratch);
        if (local && residual >= cumulative - local && residual < cumulative) {
            unsigned r = residual - (cumulative - local);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned q = reinterpret_cast<unsigned *>(&w)[j];
                if (r < q) {
                    outputs[0] = chosen * 2048 + 4 * t + j;
                    break;
                }
                r -= q;
            }
        }
    }
}

static bool try_cooperative_sampling(int T, int V, int block_size, at::Tensor &logits,
                                     at::Tensor &states, at::Tensor &probs, at::Tensor &weights,
                                     at::Tensor &out, float inv_temperature, int k, float p,
                                     HistogramLayout layout, cudaStream_t stream) {
    validate_sampling_configuration<unsigned, false, true>();
    static thread_local int configured = -1;
    static thread_local bool available = false;
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    if (configured != device) {
        const auto *properties = at::cuda::getDeviceProperties(device);
        available = properties->cooperativeLaunch && properties->multiProcessorCount >= 32 &&
                    properties->sharedMemPerBlockOptin >= 66688;
        if (available) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                cooperative_sampling_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65664));
            int active;
            C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, cooperative_sampling_kernel, 1024, 65552));
            available = active * properties->multiProcessorCount >= 32;
        }
        configured = device;
    }
    if (!available)
        return false;
    const int tiles = 32, bins = 1 << (layout.widths[0] - 1);
    auto partials = at::empty({tiles * 2}, logits.options());
    auto histogram = at::empty({tiles * bins * 2}, logits.options().dtype(at::kInt));
    const float *l = logits.data_ptr<float>();
    int *o = out.data_ptr<int>();
    RAND *s = reinterpret_cast<RAND *>(states.data_ptr());
    float *prob = probs.data_ptr<float>();
    unsigned *w = reinterpret_cast<unsigned *>(weights.data_ptr());
    float *part = partials.data_ptr<float>();
    unsigned *h = reinterpret_cast<unsigned *>(histogram.data_ptr());
    void *args[] = {&T, &V, &l, &o, &s, &prob, &w, &part, &h, &inv_temperature, &k, &p, &layout};
    const size_t bytes = layout.shards * (layout.max_bins + 1) * 8 + 8 * 1024 * sizeof(float);
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)cooperative_sampling_kernel, dim3(tiles),
                                               dim3(block_size), args, bytes, stream));
    return true;
}

at::Tensor create_rng(int64_t seed, int64_t B) {
    TORCH_CHECK(B > 0 && B <= INT32_MAX, "B must be positive and fit in int32");
    auto state = at::empty({B * int64_t(sizeof(RAND))},
                           at::TensorOptions().dtype(at::kChar).device(at::kCUDA));
    create_rng_kernel<<<(B + RAND_SETUP_THREADS - 1) / RAND_SETUP_THREADS, RAND_SETUP_THREADS, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<RAND *>(state.data_ptr()), seed, B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return state;
}

// Opt into enough shared memory for 2048 bins per shard. Cache this
// per specialization and device so normal calls do not pay attribute overhead.
template <typename Weight, bool Penalties, typename... Args>
static void launch_sampling(int B, cudaStream_t stream, int block_size, size_t bytes,
                            Args... args) {
    validate_sampling_configuration<Weight, Penalties>();
    static thread_local int configured_device = -1;
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    if (configured_device != device) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            sampling_kernel<Weight, Penalties>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            std::min(65664,
                     int(at::cuda::getDeviceProperties(device)->sharedMemPerBlockOptin) - 1024)));
        configured_device = device;
    }
    sampling_kernel<Weight, Penalties><<<B, block_size, bytes, stream>>>(args...);
}

static at::Tensor sample_impl(at::Tensor &logits, at::Tensor *penalties, at::Tensor &states,
                              double presence, double repetition, double decay, double temperature,
                              int64_t top_k, double top_p, int64_t quantization_bits,
                              const std::vector<int64_t> &histogram_bits, int64_t block_size) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.is_contiguous(),
                "logits must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(logits.dim() >= 1 && logits.dim() <= 3,
                "logits must have shape (V), (B,V), or (B,T,V)");
    const int64_t V = logits.size(-1), B = logits.dim() >= 2 ? logits.size(0) : 1;
    const int64_t T = logits.dim() == 3 ? logits.size(1) : 1;
    TORCH_CHECK(V > 0 && V <= (1 << 20), "V must be in [1, 1048576]");
    TORCH_CHECK(B > 0 && B <= INT32_MAX && T > 0 && T <= INT32_MAX, "invalid B or T");
    TORCH_CHECK(std::isfinite(temperature) && temperature >= .001 && temperature <= 1000.,
                "temperature must be in [0.001, 1000]");
    TORCH_CHECK(block_size == 256 || block_size == 512 || block_size == 1024,
                "block_size must be 256, 512, or 1024");
    TORCH_CHECK(quantization_bits == 32 || quantization_bits == 64,
                "quantization_bits must be 32 or 64");
    TORCH_CHECK(states.device() == logits.device() && states.scalar_type() == at::kChar &&
                    states.is_contiguous() && states.numel() >= B * int64_t(sizeof(RAND)),
                "states must be a sufficiently large contiguous CUDA int8 tensor from create_rng");
    if (penalties) {
        TORCH_CHECK(
            penalties->device() == logits.device() && penalties->scalar_type() == at::kFloat &&
                penalties->is_contiguous() &&
                ((penalties->dim() == 1 && B == 1 && penalties->size(0) == V) ||
                 (penalties->dim() == 2 && penalties->size(0) == B && penalties->size(1) == V)),
            "penalties must be contiguous float32 with shape (B,V), or (V) for B=1, on the logits "
            "device");
    }
    if (top_k <= 0 || top_k > V)
        top_k = V;
    if (!std::isfinite(top_p) || top_p < 0. || top_p > 1.)
        top_p = 1.;
    if (top_p == 0.) {
        top_k = 1;
        top_p = 1.;
    }
    HistogramLayout layout{};
    const std::vector<int64_t> bits =
        histogram_bits.empty() ? std::vector<int64_t>{10, 10, 11} : histogram_bits;
    TORCH_CHECK(bits.size() >= 3 && bits.size() <= 5,
                "histogram_bits must specify 3, 4, or 5 passes");
    int remaining = 31;
    layout.passes = bits.size();
    for (int i = 0; i < layout.passes; ++i) {
        TORCH_CHECK(bits[i] >= 1 && bits[i] <= 11, "each histogram width must be in [1,11]");
        remaining -= bits[i];
        layout.widths[i] = bits[i];
        layout.shifts[i] = remaining;
        layout.max_bins = std::max(layout.max_bins, 1 << bits[i]);
    }
    TORCH_CHECK(remaining == 0, "histogram_bits must sum to 31");
    const c10::cuda::CUDAGuard guard(logits.device());
    auto probs = at::empty({B, V}, logits.options());
    auto weights =
        at::empty({B, V}, logits.options().dtype(quantization_bits == 32 ? at::kInt : at::kLong));
    auto out = at::empty({B}, logits.options().dtype(at::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    const float inv_temperature = float(1.4426950408889634 / temperature);
    float device_top_p = float(top_p);
    if (top_p > 0. && device_top_p == 0.f)
        device_top_p = FLT_MIN;
    const size_t bytes_per_bin =
        (quantization_bits == 32 ? sizeof(unsigned) : sizeof(unsigned long long)) +
        sizeof(unsigned);
    layout.shards = 2;
    const size_t shared_budget =
        at::cuda::getDeviceProperties(logits.get_device())->sharedMemPerBlockOptin - 1024;
    while (layout.shards > 1 &&
           layout.shards * (layout.max_bins + 1) * bytes_per_bin > shared_budget)
        layout.shards /= 2;
    const size_t histogram_bytes = layout.shards * (layout.max_bins + 1) * bytes_per_bin;
    if (B == 1 && V == 65536 && quantization_bits == 32 && !penalties &&
        (block_size == 512 || block_size == 1024) &&
        try_cooperative_sampling(T, V, block_size, logits, states, probs, weights, out,
                                 inv_temperature, top_k, device_top_p, layout, stream)) {
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
#define LAUNCH(W, P)                                                                               \
    launch_sampling<W, P>(B, stream, block_size, histogram_bytes, T, V, logits.data_ptr<float>(),  \
                          penalties ? penalties->data_ptr<float>() : nullptr, out.data_ptr<int>(), \
                          reinterpret_cast<RAND *>(states.data_ptr()), probs.data_ptr<float>(),    \
                          reinterpret_cast<W *>(weights.data_ptr()), inv_temperature, top_k,       \
                          device_top_p, presence, repetition, decay, layout)
    if (quantization_bits == 32) {
        if (penalties) {
            LAUNCH(unsigned, true);
        } else {
            LAUNCH(unsigned, false);
        }
    } else {
        if (penalties) {
            LAUNCH(int64_t, true);
        } else {
            LAUNCH(int64_t, false);
        }
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor sample(at::Tensor &logits, at::Tensor &states, double temperature, int64_t top_k,
                  double top_p, int64_t quantization_bits,
                  const std::vector<int64_t> &histogram_bits, int64_t block_size) {
    return sample_impl(logits, nullptr, states, 0., 0., 0., temperature, top_k, top_p,
                       quantization_bits, histogram_bits, block_size);
}

at::Tensor sample_with_penalties(at::Tensor &logits, at::Tensor &penalties, at::Tensor &states,
                                 double presence, double repetition, double decay,
                                 double temperature, int64_t top_k, double top_p,
                                 int64_t quantization_bits,
                                 const std::vector<int64_t> &histogram_bits, int64_t block_size) {
    return sample_impl(logits, &penalties, states, presence, repetition, decay, temperature, top_k,
                       top_p, quantization_bits, histogram_bits, block_size);
}

#ifndef RAPID_SAMPLING_NO_BINDINGS
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("create_rng", create_rng, py::arg("seed"), py::arg("B"));
    m.def("sample_with_penalties", sample_with_penalties, py::arg("logits"), py::arg("penalties"),
          py::arg("states"), py::arg("presence_penalty"), py::arg("repetition_penalty"),
          py::arg("penalty_decay"), py::arg("temperature"), py::arg("top_k"), py::arg("top_p"),
          py::arg("quantization_bits") = 32, py::arg("histogram_bits") = std::vector<int64_t>{},
          py::arg("block_size") = 1024);
    m.def("sample", sample, py::arg("logits"), py::arg("states"), py::arg("temperature"),
          py::arg("top_k"), py::arg("top_p"), py::arg("quantization_bits") = 32,
          py::arg("histogram_bits") = std::vector<int64_t>{}, py::arg("block_size") = 1024);
}

#endif
