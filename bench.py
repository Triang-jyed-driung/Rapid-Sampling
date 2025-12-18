import torch
from torch.utils.cpp_extension import load
import flashinfer
torch.manual_seed(42)

sample = load(
    name="sample",
    sources = ["sampling.cpp","sampling.cu"],
    extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
    verbose=True,
)

import torch
import flashinfer
import numpy as np
# from scipy.stats import entropy
import os
import json
from datetime import datetime
import time
from tqdm import tqdm

def benchmark_kernels():
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512][::-1]
    vocab_sizes = [256, 32768, 50304, 65536, 100256, 131072, 151936, 262144][::-1]
    top_ps = [0.1, 0.3, 0.7, 0.9, 1]
    top_ks = [-1, 1, 10, 100, 1000, 10000]  # -1 means unused
    
    # batch_sizes = [1, 32, 128][::-1]
    # vocab_sizes = [256, 4096, 32768][::-1]
    # top_ps = [0.2, 0.3, 0.5, 0.8, 0.9999, 1]
    # top_ks = [-1]
    results = []
    
    total_combinations = len(batch_sizes) * len(vocab_sizes) * len(top_ps) * len(top_ks) * 5
    print(f"Total combinations to benchmark: {total_combinations}")
    
    combination_idx = 0
    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            for top_p in top_ps:
                for top_k in top_ks:
                    for logit_type in range(5):
                        combination_idx += 1
                        print(f"\n[{combination_idx}/{total_combinations}] "
                            f"Batch: {batch_size}, Vocab: {vocab_size}, Top-P: {top_p}, Top-K: {top_k}, logit type: {logit_type}")
                        # Skip if both top_p and top_k are disabled (both default values)
                        if top_p == 0.0 and top_k <= 0:
                            continue
                        # Initialize states for your kernel
                        states = sample.setup_rand(0, batch_size)  # Adjust according to your implementation
                        temperature = 1.0
                        
                        # Benchmark FlashInfer
                        flashinfer_times = []
                        my_kernel_times = []
                        for _ in range(5):
                            logits = (lambda a: 
                                torch.zeros(batch_size, vocab_size, dtype=torch.float32, device='cuda')   if a==0 else
                                torch.rand (batch_size, vocab_size, dtype=torch.float32, device='cuda')   if a==1 else
                                torch.randn(batch_size, vocab_size, dtype=torch.float32, device='cuda')/2 if a==2 else
                                torch.randn(batch_size, vocab_size, dtype=torch.float32, device='cuda')*3 if a==3 else
                                torch.rand (batch_size, vocab_size, dtype=torch.float32, device='cuda')*4 if a==4 else ValueError
                            )(logit_type)
                            torch.cuda.synchronize()
                            start_time = time.time()
                            
                            samples_fi = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                                logits, top_k, top_p
                            )
                            
                            torch.cuda.synchronize()
                            end_time = time.time()
                            flashinfer_times.append(end_time - start_time)
                            torch.cuda.synchronize()
                            start_time = time.time()
                            
                            samples_my = sample.batch_sampling_temperature_topk_topp(
                                logits, states, temperature, top_k, top_p
                            )
                            
                            torch.cuda.synchronize()
                            end_time = time.time()
                            my_kernel_times.append(end_time - start_time)
                        
                        # Calculate medians
                        median_flashinfer = np.median(flashinfer_times)
                        median_my_kernel = np.median(my_kernel_times)
                        
                        result = {
                            'batch_size': batch_size,
                            'vocab_size': vocab_size,
                            'top_p': top_p,
                            'top_k': top_k,
                            'logit_type': logit_type,
                            'median_flashinfer_time': median_flashinfer,
                            'median_my_kernel_time': median_my_kernel,
                            'speedup_ratio': median_flashinfer / median_my_kernel,
                            # 'timestamp': datetime.now().isoformat()
                        }
                        
                        results.append(result)
                        
                        print(f"  FlashInfer: {median_flashinfer*1000:.3f}ms")
                        print(f"  Your Kernel: {median_my_kernel*1000:.3f}ms")
                        print(f"  Speedup: {result['speedup_ratio']:.2f}x")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sampling_benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'benchmark_config': {
                'batch_sizes': batch_sizes,
                'vocab_sizes': vocab_sizes,
                'top_ps': top_ps,
                'top_ks': top_ks,
                'runs_per_setting': 5,
                'temperature': 1.0
            },
            'results': results,
            'summary': {
                'total_combinations': len(results),
                'fastest_kernel_count': {
                    'flashinfer': sum(1 for r in results if r['speedup_ratio'] < 1.0),
                    'your_kernel': sum(1 for r in results if r['speedup_ratio'] >= 1.0)
                }
            }
        }, f, indent=2)
    
    print(f"\nBenchmark complete! Results saved to {filename}")
    
    # Print summary
    flashinfer_faster = sum(1 for r in results if r['speedup_ratio'] < 1.0)
    my_kernel_faster = sum(1 for r in results if r['speedup_ratio'] >= 1.0)
    
    print(f"\nSummary:")
    print(f"FlashInfer faster: {flashinfer_faster} configurations")
    print(f"Your kernel faster: {my_kernel_faster} configurations")
    
    return results

def detailed_analysis(results_file):
    """Analyze the benchmark results"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Group by batch size
    batch_groups = {}
    for result in results:
        bs = result['batch_size']
        if bs not in batch_groups:
            batch_groups[bs] = []
        batch_groups[bs].append(result)
    
    print("\nPerformance by Batch Size:")
    for batch_size, group in sorted(batch_groups.items()):
        avg_speedup = np.mean([r['speedup_ratio'] for r in group])
        faster_count = sum(1 for r in group if r['speedup_ratio'] >= 1.0)
        print(f"Batch {batch_size}: Avg speedup {avg_speedup:.2f}x, "
              f"Faster configs: {faster_count}/{len(group)}")
    
    # Find best/worst cases for your kernel
    sorted_by_speedup = sorted(results, key=lambda x: x['speedup_ratio'], reverse=True)
    
    print(f"\nTop 5 best performing cases for your kernel:")
    for i, result in enumerate(sorted_by_speedup[:5]):
        print(f"{i+1}. BS:{result['batch_size']}, VS:{result['vocab_size']}, "
              f"TP:{result['top_p']}, TK:{result['top_k']} -> {result['speedup_ratio']:.2f}x")
    
    print(f"\nTop 5 worst performing cases for your kernel:")
    for i, result in enumerate(sorted(results, key=lambda x: x['speedup_ratio'])[:5]):
        print(f"{i+1}. BS:{result['batch_size']}, VS:{result['vocab_size']}, "
              f"TP:{result['top_p']}, TK:{result['top_k']} -> {result['speedup_ratio']:.2f}x")

if __name__ == "__main__":
    # Run the main benchmark
    results = benchmark_kernels()
    
    # If you want to analyze existing results:
    # detailed_analysis("./sampling_benchmark_results_20251218_144133.json")