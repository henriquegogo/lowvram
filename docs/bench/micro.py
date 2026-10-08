"""Synthetic benchmarks: PCIe bandwidth, GEMM, working-set sweep and a layer stack.

Each invocation measures one configuration in a fresh process. Run it plainly for the
baseline, or under lowvram (`../../lowvram uv run python micro.py ...`) for managed memory.

  micro.py pcie
  micro.py gemm --dtype fp16
  micro.py sweep --gib 8
  micro.py layers --gib 8 --batch 16 --mode gpu|offload|overlap|cpu|readmostly|prefetch|remote
"""
import argparse
import time

import torch

from common import GiB, MiB, Driver, emit, pcie_now, run_guarded, summary, sync, timeit

DIM = 8192                       # layer stack: DIM x DIM fp16 matrices, 128 MiB each
LAYER_BYTES = DIM * DIM * 2


def bench_pcie(args):
    """Pinned host <-> device copy bandwidth, the ceiling for any offloading scheme."""
    n = 1 * GiB
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    h2d = summary(timeit(lambda: dev.copy_(host, non_blocking=True), warmup=2, reps=args.reps))
    link = pcie_now()
    d2h = summary(timeit(lambda: host.copy_(dev, non_blocking=True), warmup=2, reps=args.reps))
    return {"h2d_GBps": n / h2d["median_s"] / 1e9, "d2h_GBps": n / d2h["median_s"] / 1e9,
            "h2d": h2d, "d2h": d2h, "pcie_under_load": link}


def bench_gemm(args):
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    a = torch.randn(DIM, DIM, device="cuda", dtype=dtype)
    b = torch.randn(DIM, DIM, device="cuda", dtype=dtype)
    c = torch.empty(DIM, DIM, device="cuda", dtype=dtype)
    s = summary(timeit(lambda: torch.matmul(a, b, out=c), warmup=5, reps=args.reps))
    return {"tflops": 2 * DIM**3 / s["median_s"] / 1e12, **s}


def bench_sweep(args):
    """Update a working set of 256 MiB tensors once per pass (cyclic access, LRU worst case)."""
    chunk = 256 * MiB // 4
    tensors = [torch.zeros(chunk, device="cuda", dtype=torch.float32) for _ in range(int(args.gib * 4))]

    def one_pass():
        for t in tensors:
            t.add_(1.0)

    s = summary(timeit(one_pass, warmup=1, reps=args.reps))
    moved = 2 * len(tensors) * 256 * MiB          # read + write
    return {"effective_GBps": moved / s["median_s"] / 1e9, "pcie_under_load": pcie_now(), **s}


def make_weights(n_layers, device, dtype=torch.float16, pin=False):
    scale = DIM ** -0.5
    ws = []
    for _ in range(n_layers):
        if device == "cpu":
            w = torch.randn(DIM, DIM, dtype=dtype) * scale
            ws.append(w.pin_memory() if pin else w)
        else:
            ws.append(torch.randn(DIM, DIM, device=device, dtype=dtype).mul_(scale))
    return ws


def bench_layers(args):
    """A chain of DIM x DIM fp16 matmuls: a stand-in for transformer inference.

    gpu       weights allocated on the GPU (under lowvram: managed, paged on demand)
    offload   weights in pinned RAM, each layer copied to the GPU right before use
    overlap   like offload, but the next layer's copy overlaps the current matmul
    cpu       fp32 weights computed on the CPU (no fast fp16 GEMM on this CPU)
    readmostly / prefetch / remote: gpu mode under lowvram plus one driver hint
    """
    n = int(round(args.gib * GiB / LAYER_BYTES))
    mode = args.mode
    if mode in ("readmostly", "prefetch", "remote") and not args.allow_unmanaged:
        from common import lowvram_active
        assert lowvram_active(), f"mode {mode} needs managed memory: run under lowvram"

    if mode == "cpu":
        torch.set_num_threads(args.threads or torch.get_num_threads())
        ws = make_weights(n, "cpu", torch.float32)
        x = torch.randn(args.batch, DIM)

        def forward():
            y = x
            for w in ws:
                y = y @ w
            return y
        s = summary(timeit(forward, warmup=1, reps=args.reps))
        return {"layers": n, "threads": torch.get_num_threads(), **s}

    x = torch.randn(args.batch, DIM, device="cuda", dtype=torch.float16)

    if mode in ("offload", "overlap"):
        ws = make_weights(n, "cpu", pin=True)
        bufs = [torch.empty(DIM, DIM, device="cuda", dtype=torch.float16) for _ in range(2)]
        if mode == "offload":
            def forward():
                y = x
                for w in ws:
                    bufs[0].copy_(w)
                    y = y @ bufs[0]
                return y
        else:
            copy_stream = torch.cuda.Stream()
            ready = [torch.cuda.Event() for _ in range(2)]
            free = [torch.cuda.Event() for _ in range(2)]

            def forward():
                compute = torch.cuda.current_stream()
                with torch.cuda.stream(copy_stream):
                    bufs[0].copy_(ws[0], non_blocking=True)
                    ready[0].record()
                y = x
                for i in range(n):
                    b = i % 2
                    if i + 1 < n:
                        nb = (i + 1) % 2
                        with torch.cuda.stream(copy_stream):
                            if i >= 1:
                                copy_stream.wait_event(free[nb])
                            bufs[nb].copy_(ws[i + 1], non_blocking=True)
                            ready[nb].record()
                    compute.wait_event(ready[b])
                    y = y @ bufs[b]
                    free[b].record()
                return y
        s = summary(timeit(forward, warmup=1, reps=args.reps))
        return {"layers": n, "weights_GBps": n * LAYER_BYTES / s["median_s"] / 1e9, **s}

    # gpu and hint modes
    ws = make_weights(n, "cuda")
    sync()
    drv = Driver() if mode in ("readmostly", "prefetch", "remote") else None
    if mode == "readmostly":
        for w in ws:
            drv.read_mostly(w)
    resident = n
    if mode == "remote":
        # Keep the first layers that fit a VRAM budget as normal managed memory;
        # map the rest from host RAM for direct access, so they never migrate.
        resident = min(n, int(args.budget_gib * GiB // LAYER_BYTES))
        for w in ws[resident:]:
            drv.remote(w)
            drv.prefetch_to_host(w, torch.cuda.current_stream())
        sync()
    prefetch_stream = torch.cuda.Stream() if mode == "prefetch" else None

    def forward():
        y = x
        for i, w in enumerate(ws):
            if prefetch_stream is not None:
                for j in range(i + 1, min(n, i + 1 + args.prefetch_depth)):
                    if i == 0 or j == i + args.prefetch_depth:
                        drv.prefetch_to_gpu(ws[j], prefetch_stream)
            y = y @ w
        return y

    s = summary(timeit(forward, warmup=1, reps=args.reps))
    return {"layers": n, "resident_layers": resident, "weights_GBps": n * LAYER_BYTES / s["median_s"] / 1e9,
            "pcie_under_load": pcie_now(), **s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bench", choices=["pcie", "gemm", "sweep", "layers"])
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--gib", type=float, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--mode", default="gpu",
                   choices=["gpu", "offload", "overlap", "cpu", "readmostly", "prefetch", "remote"])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--budget-gib", type=float, default=3.5, help="remote mode: GiB of layers kept migratable")
    p.add_argument("--prefetch-depth", type=int, default=2)
    p.add_argument("--allow-unmanaged", action="store_true")
    args = p.parse_args()

    config = {k: v for k, v in vars(args).items() if k not in ("reps", "allow_unmanaged")}
    if args.bench != "layers":
        for k in ("batch", "mode", "budget_gib", "prefetch_depth", "threads"):
            config.pop(k)
    if args.bench in ("pcie", "gemm"):
        config.pop("gib")
    if args.bench != "gemm":
        config.pop("dtype")
    if args.bench == "layers" and args.mode != "remote":
        config.pop("budget_gib")
    if args.bench == "layers" and args.mode != "prefetch":
        config.pop("prefetch_depth")
    if args.bench == "layers" and args.mode != "cpu":
        config.pop("threads")
    config["reps"] = args.reps
    t0 = time.time()
    run_guarded(args.bench, config, {"pcie": lambda: bench_pcie(args), "gemm": lambda: bench_gemm(args),
                                     "sweep": lambda: bench_sweep(args), "layers": lambda: bench_layers(args)}[args.bench])
    print(f"  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
