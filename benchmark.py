"""CUDA-event and host speed tests; save configuration and GPU occupancy alongside timings."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import rapid_sampling as sampler


def telemetry():
    commands = {
        "gpus": ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,power.draw,memory.free",
                 "--format=csv,noheader,nounits"],
        "processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                      "--format=csv,noheader,nounits"],
    }
    result = {"ok": True}
    for key, command in commands.items():
        try:
            result[key] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            result[key] = str(error)
            result["ok"] = False
    return result


def foreign_processes(snapshot, uuid):
    found = []
    for line in snapshot["processes"].splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0].removeprefix("GPU-").lower() == uuid.removeprefix("GPU-").lower() and fields[1] != str(os.getpid()):
            found.append(line)
    return found


def measure(call, iterations, trials, capture):
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
    times, host = [], []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        host.append((time.perf_counter()-before)*1e6/iterations)
        times.append(start.elapsed_time(end)*1000/iterations)
    return {"device_median_us": statistics.median(times), "device_min_us": min(times),
            "host_median_us": statistics.median(host), "device_trials_us": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, help="Optional saved CPU/CUDA tensor, loaded onto CPU first")
    parser.add_argument("--rows", type=int, nargs="+", default=[0])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--blocks", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--bits", type=int, nargs="+", default=[32])
    parser.add_argument("--top-p", type=float, nargs="+", default=[.9, 1.])
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--layouts", nargs="+", default=["10,10,11"],
                        help="Space-separated bit partitions, e.g. 10,10,11 10,8,9,4 7,6,6,6,6")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.timesteps, args.vocab, args.iterations, args.trials) <= 0:
        parser.error("dimensions, iterations and trials must be positive")
    layouts = [list(map(int, text.split(","))) for text in args.layouts]
    torch.manual_seed(42)
    saved = torch.load(args.logits, map_location="cpu", weights_only=True) if args.logits else None
    if saved is not None and (not isinstance(saved, torch.Tensor) or saved.ndim != 2):
        parser.error("saved logits must be a tensor with shape (rows,V)")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    uuid = str(properties.uuid)
    report = {"gpu": properties.name, "uuid": uuid, "torch": torch.__version__,
              "cuda": torch.version.cuda, "pid": os.getpid(), "arguments": vars(args).copy(), "results": []}
    report["arguments"] = {key: str(value) if isinstance(value, Path) else value
                           for key, value in report["arguments"].items()}
    for row in args.rows if saved is not None else [None]:
        if saved is not None:
            logits = saved[row].float().expand(args.batch, args.timesteps, -1).contiguous().cuda()
        else:
            logits = torch.randn(args.batch, args.timesteps, args.vocab, device="cuda")*3
        states = sampler.create_rng(123, args.batch)
        for bits in args.bits:
            for layout in layouts:
                for p in args.top_p:
                    # Rotate block order to reduce systematic thermal/order bias.
                    blocks = args.blocks if p < 1. else list(reversed(args.blocks))
                    for block in blocks:
                        call = lambda: sampler.sample(
                            logits, states, args.temperature, args.top_k, p, bits, layout, block,
                        )
                        before = telemetry()
                        result = {"row": row, "shape": list(logits.shape), "bits": bits,
                                  "layout": layout, "block": block, "top_p": p,
                                  "before": before,
                                  "eager": measure(call, args.iterations, args.trials, False),
                                  "graph": measure(call, args.iterations, args.trials, True)}
                        after = telemetry()
                        result["after"] = after
                        foreign = foreign_processes(before, uuid)+foreign_processes(after, uuid)
                        result["foreign_compute_processes"] = sorted(set(foreign))
                        result["timing_status"] = ("contended" if foreign else
                            "telemetry unavailable" if not before["ok"] or not after["ok"] else
                            "no foreign process observed at endpoints")
                        report["results"].append(result)
                        print(f"row={row} bits={bits} layout={layout} block={block} p={p}: "
                              f"eager={result['eager']['device_median_us']:.2f} us, "
                              f"graph={result['graph']['device_median_us']:.2f} us; "
                              f"{result['timing_status']}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
