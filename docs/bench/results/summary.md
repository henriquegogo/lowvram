# Benchmark summary

Runs: 20261008-160122  
Machine: NVIDIA GeForce RTX 3060 Laptop GPU (driver 595.91.07), 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, 31 GiB RAM, Linux 7.0.0-38-generic, PyTorch 2.14.1+cu130 (CUDA 13.0)

## Overhead when the working set fits

| Workload | Unmodified | lowvram | Change |
|---|---:|---:|---:|
| GEMM FP16 (TFLOP/s) | 26.29 | 26.29 | -0.0% |
| GEMM FP32 (TFLOP/s) | 8.82 | 8.79 | -0.3% |
| Sweep 1 GiB (GB/s) | 309 | 308 | -0.2% |
| Sweep 4 GiB (GB/s) | 301 | 303 | +0.7% |
| Layers 2 GiB, batch 16 (ms) | 7.6 | 7.4 | -2.3% |
| Layers 4 GiB, batch 16 (ms) | 15.6 | 14.9 | -4.7% |
| Layers 4 GiB, batch 2048 (ms) | 351.6 | 339.0 | -3.6% |

## Layer stack beyond VRAM (seconds per forward pass)

| Mode | b16 6 GiB | b16 8 GiB | b16 12 GiB | b2048 6 GiB | b2048 8 GiB | b2048 12 GiB |
|---|---:|---:|---:|---:|---:|---:|
| Unmodified | OOM | OOM | OOM | OOM | OOM | OOM |
| lowvram | 1.83 | 2.49 | 3.76 | 2.20 | 3.01 | 4.72 |
| lowvram + ReadMostly | 1.26 | 1.61 | 2.70 | 1.63 | 2.32 | 3.30 |
| lowvram + prefetch | 2.98 | 3.75 | 5.92 | 3.34 | 4.50 | 6.64 |
| lowvram + remote map | 0.33 | 0.51 | 1.12 | 4.57 | 9.12 | 16.98 |
| manual streaming | 1.06 | 1.46 | 2.25 | 1.57 | 2.08 | 3.10 |
| manual streaming, overlapped | 1.05 | 1.38 | 2.24 | 1.05 | 1.41 | 2.07 |
| CPU (FP32) | -- | 0.96 | -- | -- | 27.85 | -- |

## YuE2-3B (7.26 GB of BF16 weights)

| Mode | AR prompt tok/s | AR decode tok/s | NAR s/step |
|---|---:|---:|---:|
| Unmodified | OOM | OOM | OOM |
| Accelerate offload | 607 | 1.9 | 1.32 |
| CPU (BF16) | 46 | 6.8 | 52.18 |
| lowvram | 611 | 43.1 | 1.90 |
| AR half only (oracle) | 1694 | 45.3 | -- |

## Dense LLM (Qwen3-4B)

| Mode | prompt tok/s | decode tok/s |
|---|---:|---:|
| Unmodified | OOM | OOM |
| Accelerate offload | 837 | 1.17 |
| CPU (BF16) | 25 | 3.63 |
| lowvram | 499 | 0.58 |

## Named numbers

- `layersHiCpu`: 27.85
- `layersHiLv`: 3.01
- `layersHiOff`: 2.08
- `layersHiOvl`: 1.41
- `layersHiPf`: 4.50
- `layersHiRem`: 9.12
- `layersHiRm`: 2.32
- `layersLoCpu`: 0.96
- `layersLoLv`: 2.49
- `layersLoOff`: 1.46
- `layersLoOvl`: 1.38
- `layersLoPf`: 3.75
- `layersLoRem`: 0.51
- `layersLoRm`: 1.61
- `llmAccDecode`: 1.17
- `llmAccPrefill`: 837
- `llmCpuDecode`: 3.63
- `llmCpuPrefill`: 25
- `llmLvDecode`: 0.58
- `llmLvPrefill`: 499
- `llmWeightsGiB`: 7.5
- `pcieDtoH`: 6.3
- `pcieHtoD`: 6.1
- `sweepFitGBps`: 304
- `sweepOversubGBps`: 8.5
- `usableVram`: 4.89
- `yueAccDecode`: 1.9
- `yueAccNar`: 1.32
- `yueAccPrefill`: 607
- `yueCpuDecode`: 6.8
- `yueCpuNar`: 52.2
- `yueLvDecode`: 43.1
- `yueLvDecodeRange`: 41.0--46.0
- `yueLvNar`: 1.90
- `yueLvOfOracle`: 95
- `yueLvOverAcc`: 23
- `yueLvPrefill`: 611
- `yueNarSlowdown`: 1.4
- `yueOracleDecode`: 45.3
- `yueOraclePrefill`: 1694
- `yueRepeats`: 3
