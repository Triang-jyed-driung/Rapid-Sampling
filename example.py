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
states = sample.setup_rand(0, batch_size)
logits = torch.rand(batch_size, vocab_size).to(0)
samples = sample.batch_sampling_temperature_topk_topp(logits, states, temperature, top_k, top_p)
print(samples)