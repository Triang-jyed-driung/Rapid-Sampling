
# Rapid-Sampling

Fast LLM sampling kernels (3-7x faster than FlashInfer!) implemented in CUDA.

# Warning: Prototype Quality Code, DO NOT Use in Production!

## Overview

Here I write optimized CUDA kernels for fast token sampling in Large Language Models (LLMs). The implementation includes support for various sampling strategies with repetition and presence penalties, achieving significant performance improvements over existing solutions like FlashInfer.

## Features

- 1.5-25x faster than FlashInfer
- Support for batch, temperature, top-k, top-p, presence and repetition penalty

## Performance
Run benchmarking: `python bench.py`

Example performance result:
```json
{
  "batch_size": 1,
  "vocab_size": 262144,
  "top_p": 0.3,
  "top_k": -1,
  "logit_type": 4,
  "median_flashinfer_time": 0.0007960796356201172,
  "median_my_kernel_time": 0.00022029876708984375,
  "speedup_ratio": 3.6136363636363638
}
```

## Requirements

- PyTorch with CUDA support
- C++ compiler supporting C++17 or later
- NVIDIA GPU with compute capability 7.0+ (should support: Maximum number of 32-bit registers per thread block = 64K)


```python
import torch
from torch.utils.cpp_extension import load

sample = load(
    name="sample",
    sources = ["sampling.cpp","sampling.cu"],
    extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"], # "--use_fast_math" if you want
    verbose=True,
)
```

## Usage

### Basic Example

Run `example.py`

```python
import torch
from torch.utils.cpp_extension import load

sample = load(
    name="sample",
    sources = ["sampling.cpp","sampling.cu"],
    extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
    verbose=True,
)

batch_size = 4
vocab_size = 131072
temperature = 1.0
top_p = 0.5
top_k = -1

# Setup random states
states = sample.setup_rand(0, batch_size)

# Generate random logits (in practice, these would come from your model)
logits = torch.rand(batch_size, vocab_size).to(0)

# Perform sampling
samples = sample.batch_sampling_temperature_topk_topp(
    logits, 
    states, 
    temperature, 
    top_k, 
    top_p
)
print(samples)
```

### With Repetition Penalties

```python
# Initialize penalties tensor
penalties = torch.zeros(batch_size, vocab_size).to(0)

# Perform sampling with repetition control
samples = sample.batch_sampling_repetition_temperature_topk_topp(
    logits,
    penalties,
    states,
    presence_penalty=0.1,
    repetition_penalty=0.1,
    penalty_decay=0.95,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

## API Reference

### Functions

#### `setup_rand(seed, B)`
Initialize random states for sampling.

**Parameters:**
- `seed` (int): Random seed
- `B` (int): Batch size

**Returns:** Tensor containing random states

#### `batch_sampling_temperature_topk_topp(logits, states, temperature, top_k, top_p)`
Perform batched sampling with temperature, top-k, and top-p constraints.

**Parameters:**
- `logits` (Tensor): Input logits tensor of shape (V,), (B, V) or (B, T, V). V must be a multiple of 4 and does not exceed 1048576.
- `states` (Tensor): Random states from `setup_rand`
- `temperature` (float): Sampling temperature
- `top_k` (int): Top-k parameter (-1 for no limit)
- `top_p` (float): Top-p parameter (0.0-1.0)

**Returns:** Sampled token indices of shape (B,)

#### `batch_sampling_repetition_temperature_topk_topp(logits, penalties, states, presence_penalty, repetition_penalty, penalty_decay, temperature, top_k, top_p)`
Perform batched sampling with repetition and presence penalties.

**Parameters:**
- `logits` (Tensor): Input logits tensor of shape (V,), (B, V) or (B, T, V). V must be a multiple of 4 and does not exceed 1048576.
- `penalties` (Tensor): Penalty tensor of shape (B, V)
- `states` (Tensor): Random states from `setup_rand`
- `presence_penalty` (float): Presence penalty coefficient
- `repetition_penalty` (float): Repetition penalty coefficient  
- `penalty_decay` (float): Decay factor for penalties
- `temperature` (float): Sampling temperature
- `top_k` (int): Top-k parameter (-1 for unused)
- `top_p` (float): Top-p parameter

**Returns:** Sampled token indices of shape (B,)

## Technical Details
### Key Optimizations

1. **Quaternary Search**: Fast threshold finding for top-k/top-p sampling, offering superior performance compared to binary search. The algorithm requires only 4-14 iterations to find the optimal threshold. While [FlashInfer uses ternary search](https://github.com/flashinfer-ai/flashinfer/blob/f6a9899d157522a2651c886cbfb68c6210e11918/include/flashinfer/sampling.cuh#L1711), our quaternary search approach further reduces global memory pressure and computational overhead.

2. **128-bit Vectorization**: The implementation leverages `float4` operations for vectorized memory loads and computations, achieving optimal memory bandwidth utilization. As a result, the vocabulary size `V` must be a multiple of 4 for proper alignment.

3. **Monotonic Inclusive Scans**: Implements numerically stable cumulative probability calculations through monotonic inclusive scans. While the [FlashInfer blog on LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html) states that parallel prefix-sum cannot guarantee monotonic outputs due to floating-point arithmetic properties, I disagree with the statement:
   - Floating-point addition and multiplication **are** commutative, as guaranteed by IEEE 754 standards.
   - Although floating-point addition is not associative, this challenge can be addressed by avoiding generic libraries like CUB and implementing controlled accumulation patterns yourself.

Within a single warp, the sequence `__shfl_up_sync 16 8 4 2 1` maintains monotonicity, whereas `1 2 4 8 16` does not. For blocks containing 1024 threads, we implement two approaches to guarantee monotonicity while maintaining efficiency:
   - Method 1: Shared memory reduction with schedule `512 256 ... 8 4 2 1`
   - Method 2: Warp-level accumulation followed by sequential processing of 32 values by the first thread
   
Both methods maintain monotonicity based on the principle that "the whole is greater than the part" and the monotonicity property of floating-point addition: if `a ≤ a'`, then `a + b ≤ a' + b`.

4. **L2 Cache Optimization**: Implements persistent L2 cache policies for improved memory access patterns:
   ```cpp
   cudaStreamAttrValue stream_attribute;
   stream_attribute.accessPolicyWindow.base_ptr  = probs.data_ptr();
   stream_attribute.accessPolicyWindow.num_bytes = B*V*4;
   stream_attribute.accessPolicyWindow.hitRatio  = 1;
   stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
   stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
   cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
   ```

5. **Two-Phase Selection Strategy**: Employs a two-stage selection process for efficient token identification:
   - Phase 1: Identify the responsible thread (`idxt`) among the 1024 threads in a block
   - Phase 2: Select the specific token from the subset managed by thread `idxt` using the formula `int idn = idxt*4 + (t/4)*4*d + (t%4)`
   
   Therefore, this design supports maximum vocabulary sizes up to 1048576 tokens.

6. **Single-Precision Optimization**: Avoids double-precision operations entirely (having only 1% of float32 throughput). For quaternary search, we utilize floating-point values reinterpreted as unsigned integers for precise calculations. Similarly, monotonic inclusive scans maintain numerical accuracy without requiring double-precision arithmetic.
