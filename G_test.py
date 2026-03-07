import torch
from torch.utils.cpp_extension import load
torch.random.manual_seed(42)
import tqdm
sample = load(
    name="sample",
    sources = ["sampling.cpp","sampling.cu"],
    # extra_cuda_cflags=["-g"],
    extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
    verbose=True,
)

batch_size = 4096
vocab_size = 65536
temperature = 1.0
top_p = 1.0
top_k = -1
states = sample.setup_rand(1, batch_size)

import tqdm
from scipy.stats import power_divergence
s = torch.load("./sampling.pt")
si = s.to(torch.float32)
threshold = 1e-4
num_failure = 0
for l in tqdm.tqdm(range(len(s))):
    sj = si[l].unsqueeze(0).repeat(batch_size, 1)
    cnts = torch.zeros(65536, dtype=torch.int32, device="cuda:0")
    for z in range(64):
        results = sample.batch_sampling_temperature_topk_topp(sj, states, temperature, top_k, top_p)
        # breakpoint()
        assert all(results >= 0)
        assert all(results < 65536)
        ONE = torch.ones((batch_size,), dtype=torch.int32, device="cuda:0")
        cnts.index_add_(0, results, ONE)
    actual_freq = cnts.to(torch.double)
    expected_freq = torch.softmax(sj[1].to(torch.double), dim=-1) * cnts.sum()
    statistic, p_value = power_divergence(actual_freq.cpu(), f_exp=expected_freq.cpu(), lambda_='log-likelihood')
    print(f"G-statistic: {statistic}")
    print(f"P-value: {p_value}")
    num_failure += (p_value < threshold)

print(f"Total number of failures: {num_failure}")
if num_failure == 0:
    print("ALL tests passed!")
elif num_failure <= 1:
    print(f"WARNING: {num_failure} tests failed.")
else:
    print(f"ERROR: {num_failure} tests failed!!!")