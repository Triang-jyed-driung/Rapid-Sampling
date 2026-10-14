"""Benchmark the requested (B,1,V) matrix using saved real logits, cropped/repeated per row."""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import statistics
import time

import torch
import rapid_sampling as sampler
from benchmark import telemetry, foreign_processes

BATCHES = [1, 8, 64, 512]
VOCABS = [32768, 50304, 65536, 100000, 129280, 151936, 204800, 524288]


def measure(call, capture, trials, target_ms):
    deadline = time.perf_counter() + .25
    while time.perf_counter() < deadline:
        for _ in range(5):
            call()
        torch.cuda.synchronize()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        launch = graph.replay
    else:
        launch = call
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        launch()
    end.record()
    end.synchronize()
    pilot_ms = start.elapsed_time(end)/5
    iterations = max(5, min(500, round(target_ms/max(pilot_ms, .001))))
    device, host = [], []
    for _ in range(trials):
        torch.cuda.synchronize()
        before = time.perf_counter()
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        host.append((time.perf_counter()-before)*1e6/iterations)
        device.append(start.elapsed_time(end)*1000/iterations)
    return {"iterations": iterations, "device_trials_us": device,
            "device_median_us": statistics.median(device),
            "device_min_us": min(device), "host_median_us": statistics.median(host)}


def write_report(report, output):
    output.write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# CUDA shape timings", "", f"Measured {report['date']} on {report['gpu']}.", "",
             "T=1; real logits from sampling.pt, cropped/cyclically repeated per row to V, "
             "saved rows cycled to B; contiguous float32; "
             "temperature=1, top-k=-1, uint32 weights, histogram widths (10,10,11), 1024 threads. "
             f"Values are microseconds per full batch call, median of {report['trials']} CUDA-event trials after 250 ms warmup per mode. "
             "Graph replay reuses captured scratch allocations; eager includes Python launch gaps and allocation effects. "
             "RNG initialization and input generation are excluded.", ""]
    for mode, title in (("graph", "CUDA graph replay"), ("eager", "Eager Python API")):
        for p in (.9, 1.):
            lines += [f"## {title}, top-p={p}", "", "| V | B=1 | B=8 | B=64 | B=512 |",
                      "|---:|---:|---:|---:|---:|"]
            for vocab in VOCABS:
                values = []
                for batch in BATCHES:
                    row = next((r for r in report['results'] if r['batch']==batch and
                                r['vocab']==vocab and r['top_p']==p), None)
                    values.append(f"{row[mode]['device_median_us']:.2f}" if row and row['status']=='clear' else "—")
                lines.append(f"| {vocab} | "+" | ".join(values)+" |")
            lines.append("")
    lines += ["Only cases with successful telemetry and no foreign compute process observed at either endpoint "
              "are displayed. Endpoint checks cannot exclude transient interference. Full trial timings, "
              "host measurements, and GPU/process snapshots are in the accompanying JSON.", ""]
    output.with_suffix('.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logits', type=Path, default=Path('/home/zrc/rapid-sampling/sampling.pt'))
    parser.add_argument('--output', type=Path, default=Path('timings.json'))
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--target-ms', type=float, default=50.)
    args = parser.parse_args()
    if args.trials < 1 or args.target_ms <= 0:
        parser.error('trials and target-ms must be positive')
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid)
    saved = torch.load(args.logits, map_location='cpu', weights_only=True)
    if not isinstance(saved, torch.Tensor) or saved.ndim != 2 or not torch.isfinite(saved).all():
        parser.error('saved logits must be a finite (rows,V) tensor')
    saved = saved.float().contiguous().cuda()
    report = {"date": datetime.now().astimezone().isoformat(timespec='seconds'),
              "gpu": properties.name, "uuid": uuid, "torch": torch.__version__,
              "cuda": torch.version.cuda, "pid": os.getpid(), "trials": args.trials,
              "target_ms": args.target_ms, "distribution": "real logits; row-wise crop/cyclic repeat, batch rows cycled",
              "logits_path": str(args.logits), "saved_shape": list(saved.shape), "warmup_ms": 250,
              "temperature": 1., "top_k": -1, "quantization_bits": 32,
              "histogram_bits": [10,10,11], "block_size": 1024, "results": []}
    for batch in BATCHES:
        for vocab in VOCABS:
            rows = torch.arange(batch, device='cuda') % saved.shape[0]
            columns = torch.arange(vocab, device='cuda') % saved.shape[1]
            logits = saved[rows[:, None], columns[None, :]].contiguous().unsqueeze(1)
            states = sampler.create_rng(123, batch)
            for p in (.9, 1.):
                call = lambda: sampler.sample(
                    logits, states, 1., -1, p, 32, [10,10,11], 1024)
                before = telemetry()
                if not before['ok'] or foreign_processes(before, uuid):
                    raise RuntimeError('GPU unavailable or telemetry failed; rerun on an idle GPU')
                row = {"batch": batch, "timesteps": 1, "vocab": vocab, "top_p": p, "before": before}
                # Alternate ordering to reduce systematic warmup bias.
                for mode in ('graph','eager') if p < 1 else ('eager','graph'):
                    row[mode] = measure(call, mode=='graph', args.trials, args.target_ms)
                after = telemetry()
                row['after'] = after
                row['status'] = 'clear' if after['ok'] and not foreign_processes(after, uuid) else 'contended'
                report['results'].append(row)
                write_report(report, args.output)
                print(f"B={batch} V={vocab} p={p}: graph={row['graph']['device_median_us']:.2f} us, "
                      f"eager={row['eager']['device_median_us']:.2f} us [{row['status']}]", flush=True)
                if row['status'] != 'clear':
                    raise RuntimeError('Foreign GPU workload appeared; partial results saved')
            del logits, states
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
