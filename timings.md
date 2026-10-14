# CUDA shape timings

Measured 2026-10-14T12:21:24+08:00 on NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

T=1; real logits from sampling.pt, cropped/cyclically repeated per row to V, saved rows cycled to B; contiguous float32; temperature=1, top-k=-1, uint32 weights, histogram widths (10,10,11), 1024 threads. Values are microseconds per full batch call, median of 7 CUDA-event trials after 250 ms warmup per mode. Graph replay reuses captured scratch allocations; eager includes Python launch gaps and allocation effects. RNG initialization and input generation are excluded.

## CUDA graph replay, top-p=0.9

| V | B=1 | B=8 | B=64 | B=512 |
|---:|---:|---:|---:|---:|
| 32768 | 26.62 | 30.74 | 32.77 | 206.55 |
| 50304 | 34.81 | 43.00 | 43.06 | 321.15 |
| 65536 | 24.58 | 51.22 | 53.27 | 416.72 |
| 100000 | 57.31 | 73.71 | 77.90 | 693.27 |
| 129280 | 69.67 | 90.12 | 102.48 | 968.62 |
| 151936 | 79.92 | 104.44 | 122.70 | 1347.11 |
| 204800 | 102.56 | 135.78 | 194.25 | 2520.06 |
| 524288 | 243.98 | 328.95 | 779.02 | 8350.61 |

## CUDA graph replay, top-p=1.0

| V | B=1 | B=8 | B=64 | B=512 |
|---:|---:|---:|---:|---:|
| 32768 | 14.35 | 20.48 | 20.48 | 152.35 |
| 50304 | 20.48 | 28.67 | 28.69 | 242.51 |
| 65536 | 12.29 | 34.81 | 36.84 | 321.22 |
| 100000 | 34.48 | 51.20 | 55.28 | 561.62 |
| 129280 | 41.97 | 63.47 | 75.23 | 807.38 |
| 151936 | 49.12 | 73.72 | 90.82 | 1185.69 |
| 204800 | 62.42 | 96.27 | 151.72 | 2163.20 |
| 524288 | 150.65 | 236.89 | 667.11 | 6508.35 |

## Eager Python API, top-p=0.9

| V | B=1 | B=8 | B=64 | B=512 |
|---:|---:|---:|---:|---:|
| 32768 | 26.64 | 30.76 | 32.79 | 206.60 |
| 50304 | 34.83 | 43.02 | 43.65 | 319.67 |
| 65536 | 24.60 | 51.24 | 53.41 | 415.93 |
| 100000 | 57.29 | 73.73 | 78.17 | 693.69 |
| 129280 | 69.68 | 90.14 | 102.13 | 975.14 |
| 151936 | 79.93 | 104.47 | 122.32 | 1351.81 |
| 204800 | 102.56 | 135.73 | 193.41 | 2523.61 |
| 524288 | 244.03 | 329.03 | 719.69 | 8344.76 |

## Eager Python API, top-p=1.0

| V | B=1 | B=8 | B=64 | B=512 |
|---:|---:|---:|---:|---:|
| 32768 | 14.37 | 20.51 | 20.50 | 154.54 |
| 50304 | 20.50 | 28.69 | 28.71 | 243.15 |
| 65536 | 14.68 | 34.83 | 34.86 | 321.34 |
| 100000 | 34.34 | 51.22 | 53.25 | 560.73 |
| 129280 | 41.66 | 63.49 | 73.85 | 811.88 |
| 151936 | 49.15 | 73.75 | 89.58 | 1181.62 |
| 204800 | 62.25 | 96.31 | 150.22 | 2163.85 |
| 524288 | 150.61 | 237.10 | 625.08 | 6507.38 |

Only cases with successful telemetry and no foreign compute process observed at either endpoint are displayed. Endpoint checks cannot exclude transient interference. Full trial timings, host measurements, and GPU/process snapshots are in the accompanying JSON.
