# Rapid sampling CUDA

A CUDA sampler with temperature, fractional top-k/top-p filtering, histogram
reduction, and uint32 or int64 weights. It supports local editable installation.
All device code, host dispatch, and Torch bindings live in **sampling.cu**. There
is no C++ binding source, external project header, or separately launched prepass.
`rapid_sampling.py` imports Torch and exposes the native extension functions.

## Install

Install a CUDA-enabled PyTorch first. Building requires a compatible CUDA toolkit
with `nvcc`, a C++ compiler, setuptools, and wheel.

```bash
cd production
pip install -e .
```

Set `CUDA_HOME` if your toolkit is not found automatically, and
`TORCH_CUDA_ARCH_LIST` to target specific GPUs (e.g. `12.0`). To build using your
already installed PyTorch instead of an isolated build environment, use
`pip install -e . --no-build-isolation`. Install SciPy separately to run the G-tests.
Import the extension as `rapid_sampling`. Rebuild the editable installation after
changing `sampling.cu`; Python changes take effect directly.

## Example

```python
import torch
import rapid_sampling as sampler

logits = torch.randn(1, 1, 65536, device="cuda", dtype=torch.float32)
states = sampler.create_rng(seed=123, B=1)
token = sampler.sample(
    logits, states, temperature=1., top_k=-1, top_p=.9,
    quantization_bits=32, histogram_bits=[10, 10, 11], block_size=1024,
)
print(token.item())
```

Reuse `states` to advance the random sequence. Recreating states with the same
seed reproduces the sequence for the same input, parameters, and launch path.
`python production/example.py` also demonstrates int64 and repetition penalties.

## API and semantics

`create_rng(seed, B)` returns opaque RNG state on the **current CUDA device**.
Create it with that device selected when using multiple GPUs. Sampling runs on
PyTorch's current stream for the logits device. State mutates in place; serialize
calls sharing state, or give independent streams their own state tensors.

`sample(logits, states, temperature, top_k, top_p,
quantization_bits=32, histogram_bits=[], block_size=1024)` returns CUDA int32
indices with shape `(B,)`.

- Logits must be contiguous CUDA float32 with shape `(V,)`, `(B,V)`, or `(B,T,V)`.
  For `(B,T,V)`, **only the final timestep** is sampled. Logit offsets are
  `(b*T + T-1)*V`; probability and weight scratch buffers each have shape `(B,V)`.
- `1 <= V <= 1048576`; temperature must be finite and in `[0.001,1000]`.
- Nonpositive `top_k`, or `top_k > V`, means all tokens. `top_p=1` disables the
  mass cutoff, preserving rare top-k weights even if softmax rounds its sum.
  Invalid/nonfinite top-p becomes 1. `top_p=0` means top-k=1 with no mass cutoff.
- Filtering shares a **fractional boundary weight among equal probabilities**.
  It does not retain an entire boundary token as conventional nucleus sampling
  often does. Equal-logit ties are treated symmetrically, including top-k=1.
- `quantization_bits` is 32 or 64. Uint32 is the default and fastest mode for
  `(1,1,65536)`. Int64 offers finer weight quantization, but softmax and cutoff
  probabilities remain float32. Device computation uses float32 and integers,
  with no FP64. Python numeric arguments use ordinary host conversion.
- Default histogram widths are `[10,10,11]`. Supply 3, 4, or 5 widths in `[1,11]`
  totaling 31, for example `[10,8,9,4]` or `[7,6,6,6,6]`.
- `block_size` can be 256, 512, or 1024; default 1024. Random sequences can differ
  across block sizes or backends while preserving the intended distribution.
- `safe_float` maps NaN to zero and infinities to signed `FLT_MAX`.

`sample_with_penalties(logits, penalties, states,
presence_penalty, repetition_penalty, penalty_decay, temperature, top_k, top_p,
quantization_bits=32, histogram_bits=[], block_size=1024)` additionally subtracts
penalties before temperature scaling. Penalties must be contiguous CUDA float32
of shape `(B,V)` (also `(V,)` when B=1). After sampling, each entry becomes:

```text
old * penalty_decay + (if selected: presence_penalty when old==0,
                                   otherwise repetition_penalty;
                      otherwise: 0)
```

Each call allocates `(B,V)` float32 probabilities, `(B,V)` integer weights, and
`(B,)` output. Cooperative calls additionally allocate small tile summaries and
first-pass histograms. RNG setup is a separate kernel; each sample call launches
one sampling kernel. CUDA graph capture is supported after warming up the same
configuration, including RNG advancement on replay.

## Algorithm and pseudocode

Nonnegative float32 probabilities have the same order as their unsigned bit
patterns. Each histogram pass partitions the remaining bits of the cutoff bucket
and accumulates **integer mass and token count**. This avoids sorting the vocabulary.
Counts enforce top-k, and mass enforces top-p. Only the boundary bucket needs
refinement in the next pass; the default 3 passes examine all 31 non-sign bits.

```text
for each batch row b:
    x = safe_float(logits[b, last_timestep])
    x = safe_float((x - optional_penalties[b]) * log2(e) / temperature)
    p = exp2(x - max(x)) / sum(exp2(x - max(x)))

    boundary_bits = 0; mass_above = 0; count_above = 0
    if top_k < V or top_p < 1:
        for width in histogram_widths:
            zero sharded mass and count histograms
            insert probabilities matching the previously chosen boundary prefix
            scan buckets from high to low to find the first cutoff crossing
            add fully retained bucket mass/count; extend boundary_bits
        fraction = clamp(min(remaining_k / boundary_count,
                             remaining_mass / boundary_mass), 0, 1)
    else:
        boundary_bits = 0; fraction = 1

    filtered[i] = p[i]                 if bits(p[i]) > boundary_bits
                  p[i] * fraction     if bits(p[i]) == boundary_bits
                  0                   otherwise
    retained = sum(filtered)
    if uint32:
        weights[i] = floor(filtered[i] * (4.26e9 / retained))
    else:
        exponent = 61 - ilogb(retained)
        weights[i] = exact_floor(filtered[i] * 2**exponent)
    r = unbiased_random_integer_below(sum(weights))
    return first i with inclusive_integer_cumsum(weights)[i] > r
```

For top-p=1, `remaining_mass` imposes no cutoff. Histogram mass uses rounded
`p*2^31` in uint32 mode and truncated `p*2^62` in int64 mode. Boundary compensation
uses this integer mass, so the implemented cutoff approximates ideal real-valued
filtering. The uint32 target leaves overflow headroom; int64 uses a power-of-two
scale, with totals near `2^61` to `2^62` and safely below `2^63`. The int64 conversion
decodes float bits to preserve exact power-of-two scaling, including subnormals.
Integer CDF scans and rejection-based RNG avoid floating CDF rounding and modulo
bias. Quantization can remove probabilities smaller than one integer unit.

There are two production launch paths:

- **Cooperative grid:** B=1, V=65536, uint32, no penalties, 512/1024 threads, on
  GPUs supporting cooperative launch with sufficient shared memory and residency.
  Thirty-two tiles use grid barriers between global maximum, global exponential
  sum, normalization/first histogram, block-zero histogram refinement, parallel
  retained-mass reduction/quantization, and integer tile/token selection. A common
  maximum and denominator preserve equal probabilities across tiles. A bounded
  candidate cache falls back to a full scan on overflow.
- **One block per row:** all other supported configurations, including int64 and
  penalties. The block performs softmax, histogram refinement, quantization, and
  integer sampling itself. Shared histograms use padded shards, and block scans
  combine warp summaries.

The standalone experimental prepass kernels and alternate atomic strategies are
outside this package. Template validation is host-only; device code has no
`static_assert` or builtin assumptions.

## Smoke and statistical tests

After installation, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s production -p test_smoke.py -v
CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s production -p test_gtest.py -v
```

From inside a standalone production directory:

```bash
python -m unittest test_smoke test_gtest -v
```

Tests generate their own inputs and need no saved logits. Smoke checks cover last-T offsets, both precisions, penalties, RNG,
3/4/5-pass layouts, all block sizes, cooperative ties/cache overflow, validation,
and CUDA graph replay.

G-tests compare observed categorical counts to an independent CPU reference for
fractional top-k/top-p and ties. They exercise both precisions and all block
sizes, plus cross-tile ties and spatial groups after candidate-cache overflow.
The likelihood-ratio statistic is `G=2*sum(observed*log(observed/expected))`;
zero observations contribute zero. Zero-probability categories must never occur.
Sparse bins are pooled to expected count >=10. Chi-square survival probabilities
use `df=number_of_bins-1`, with no fitted parameters. The threshold `1e-5/64`
controls family false rejection across fewer than 64 comparisons. Reference and
statistical calculations use CPU float64; this does not introduce device FP64.
Statistical tests can reject by chance, and passing is not proof of exactness.

## Speed tests

```bash
# Focused synthetic B=1,T=1,V=65536, both block dimensions and p=.9/1.
CUDA_VISIBLE_DEVICES=0 python production/benchmark.py --output /tmp/speed.json
# Real logits are optional; load onto CPU before selecting rows and moving to CUDA.
CUDA_VISIBLE_DEVICES=0 python production/benchmark.py --logits /home/zrc/rapid-sampling/sampling.pt \
    --rows 0 56 112 167 --blocks 512 1024 --bits 32 64 \
    --layouts 10,10,11 10,8,9,4 7,6,6,6,6 --output /tmp/real-speed.json
```

The benchmark reports CUDA-event and host wall times for eager calls and graph
replay, with warmup and repeated trials. Eager measurements include launch gaps
and allocation effects; graph replay reuses captured allocations. These are
Python API speed tests, not the raw CUDA microbenchmarks in the experimental
repository. JSON records configuration, GPU identity, software versions, trial
timings, and before/after GPU/process telemetry. Other compute processes mark a
result **contended**. Endpoint telemetry cannot rule out interference during a
trial; use an idle GPU for performance comparisons. There is no universal speed
threshold, and contended timings should not guide kernel selection.

To reproduce the full B={1,8,64,512}, V={32768,50304,65536,100000,129280,151936,204800,524288}
matrix with real logits from sampling.pt (cropped/cyclically repeated per row
to V, with saved rows cycled to B), temperature=1, top-k=-1, top-p=.9/1,
uint32, and 1024 threads:

```bash
CUDA_VISIBLE_DEVICES=2 python production/benchmark_shapes.py --output production/timings.json
```

The script saves a Markdown table alongside JSON trial data and contention
telemetry. The current measurements are in [timings.md](timings.md).
