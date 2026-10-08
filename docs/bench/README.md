# Benchmarks

Measures lowvram against unmodified PyTorch, manual and framework offloading, CPU
execution and driver hints. Every configuration runs in a fresh process, once plainly
(baseline) and once under lowvram where that applies.

## Requirements

- Linux, an NVIDIA GPU (Pascal or newer) and driver, [uv](https://docs.astral.sh/uv/)
- About 30 GB of RAM for the 12 GiB configurations and the CPU runs
- lowvram built at the repository root (`run_all.sh` runs `make` itself)

`uv sync` installs PyTorch (CUDA 13.0 wheels), Transformers and Accelerate into `.venv/`.

## Models

The real-model benchmarks need weights that are not in git:

```sh
# YuE2-3B (7.3 GB), loaded from the Hugging Face cache
uv run hf download m-a-p/YuE2-3B

# Qwen3-4B (8 GB), loaded from models/ (git-ignored)
uv run hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
```

`llm.py` works with any Hugging Face causal LM: `LLM=models/<name> ./run_all.sh llm`.

## Running

```sh
./run_all.sh          # everything, ~1.5 h on an RTX 3060 Laptop GPU
./run_all.sh micro    # synthetic benchmarks only (~40 min)
./run_all.sh yue2     # YuE2-3B only
./run_all.sh llm      # dense LLM only
REPEAT=5 ./run_all.sh yue2                # more repetitions of the real models
BENCH_RUN_ID=<id> ./run_all.sh llm        # append to an existing run
uv run python plot.py                     # figures, tables, numbers, summary.md
```

Single configurations can be run directly, e.g.
`../../lowvram uv run python micro.py layers --gib 8 --batch 16 --mode readmostly`.

Close GPU-heavy applications first: if the GPU also drives the display, the desktop
holds part of its memory and adds noise. Do not edit `run_all.sh` while it runs; bash
reads scripts incrementally.

## Benchmarks

| Script | Configuration | Question |
|---|---|---|
| `micro.py pcie` | 1 GiB pinned copies | Link bandwidth: the ceiling for any offloading |
| `micro.py gemm` | 8192² FP16/FP32 | Compute overhead when everything fits |
| `micro.py sweep` | 1–12 GiB of 256 MiB tensors, updated in a fixed order | Bandwidth when it fits; the cost of cyclic access beyond VRAM |
| `micro.py layers` | chain of 8192² FP16 matmuls, 2–12 GiB, batch 16 or 2048 | Paging vs streaming vs CPU, memory- vs compute-bound |
| `yue2.py --phase ar` | YuE2-3B token generation, 512-token prompt, 96 tokens | Real model whose hot set (3.4 GiB) fits but whose total (6.8 GiB) does not |
| `yue2.py --phase nar` | YuE2-3B flow-matching step, 1,538-token sequence | Same model, every weight used every step |
| `llm.py` | Qwen3-4B, 1,024-token prompt, 48 tokens | Dense model: every weight used for every token |

Modes of `micro.py layers`:

| Mode | Runs under | Meaning |
|---|---|---|
| `gpu` | baseline or lowvram | weights allocated on the GPU |
| `offload` | baseline | weights in pinned RAM, each layer copied before use |
| `overlap` | baseline | as `offload`, next layer's copy overlapped with compute |
| `cpu` | baseline | FP32 on the CPU |
| `readmostly` | lowvram | `gpu` + `cuMemAdvise(READ_MOSTLY)` on the weights |
| `prefetch` | lowvram | `gpu` + `cuMemPrefetchAsync` of the next two layers |
| `remote` | lowvram | layers beyond a 3.5 GiB budget kept in RAM and read directly by the GPU |

The hint modes measure what a program that knows its access pattern could recover;
lowvram itself issues no hints. Modes of the real models: `gpu` (plain loading; out of
memory without lowvram), `accelerate` (`device_map="auto"`, 4 GiB weight budget), `cpu`,
and for YuE2 `ar-only`, an oracle with the unused NAR half deleted so the rest fits.

Real-model inputs are random token ids and latents of realistic shape: timing depends on
shapes, not values.

## Result format

`results/<run id>.jsonl` holds one JSON object per configuration:

```json
{"bench": "layers", "run_id": "20261008-160122", "timestamp": "...",
 "config": {"gib": 8.0, "batch": 16, "mode": "readmostly", "reps": 5},
 "status": "ok",                       // or "oom"
 "metrics": {"median_s": 1.61, "times_s": [...], "start_free_bytes": ..., ...},
 "env": {"lowvram": true, "lowvram_commit": "e6be53d", "gpu": "...", "driver": "...",
         "torch": "...", "pcie_max": "4, 16", ...}}
```

`plot.py` takes the median over repeated configurations. `results/summary.md` is its
human-readable output.
