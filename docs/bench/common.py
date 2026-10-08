"""Shared helpers for the lowvram benchmarks: timing, metadata, results, driver hints."""
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GiB = 1 << 30
MiB = 1 << 20


def lowvram_active() -> bool:
    """True when lowvram is preloaded into this process."""
    with open("/proc/self/maps") as f:
        return any(line.rstrip().endswith("/lowvram") for line in f)


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def environment() -> dict:
    """Machine and software state recorded with every result."""
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "kernel": platform.release(),
        "glibc": " ".join(platform.libc_ver()),
        "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"),
        "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "lowvram": lowvram_active(),
        "lowvram_commit": sh(f"git -C {REPO} rev-parse --short HEAD"),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        env.update({
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "reported_free_bytes": free,
            "reported_total_bytes": total,
            "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
            "pcie_max": sh("nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv,noheader"),
        })
    return env


def pcie_now() -> str:
    """Current PCIe generation and width; links downshift when idle, so call this under load."""
    return sh("nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader")


def sync():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def timeit(fn, warmup: int = 1, reps: int = 5) -> list[float]:
    """Wall-clock seconds of fn() per repetition, GPU-synchronized."""
    for _ in range(warmup):
        fn()
        sync()
    times = []
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t)
    return times


def summary(times: list[float]) -> dict:
    return {
        "times_s": times,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def emit(bench: str, config: dict, status: str, **metrics):
    """Append one result as a JSON line to $BENCH_OUT (or stdout) and print a short line."""
    record = {
        "bench": bench,
        "run_id": os.environ.get("BENCH_RUN_ID", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
        "status": status,
        "metrics": metrics,
        "env": environment(),
    }
    line = json.dumps(record)
    out = os.environ.get("BENCH_OUT")
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")
    short = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
    tag = "lowvram" if record["env"]["lowvram"] else "baseline"
    print(f"[{bench}] {tag} {config} -> {status} {short}", file=sys.stderr, flush=True)


def run_guarded(bench: str, config: dict, fn):
    """Run fn() -> metrics dict; record OOM as a result instead of crashing."""
    start_free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    try:
        metrics = fn()
        metrics["start_free_bytes"] = start_free
    except torch.OutOfMemoryError as e:
        emit(bench, config, "oom", error=str(e).splitlines()[0])
        return
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            emit(bench, config, "oom", error=str(e).splitlines()[0])
            return
        raise
    emit(bench, config, "ok", **metrics)


# --- CUDA driver hints (cuMemAdvise / cuMemPrefetchAsync) on managed memory ----------------
# Used only by the "hint" experiments, which run under lowvram so that tensors are managed.

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


LOC_DEVICE, LOC_HOST = 1, 2
ADVISE_SET_READ_MOSTLY = 1
ADVISE_SET_PREFERRED_LOCATION = 3
ADVISE_SET_ACCESSED_BY = 5


class Driver:
    def __init__(self):
        lib = ctypes.CDLL("libcuda.so.1")
        self.advise = lib.cuMemAdvise_v2
        self.advise.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_int, CUmemLocation]
        self.prefetch = lib.cuMemPrefetchAsync_v2
        self.prefetch.argtypes = [ctypes.c_ulonglong, ctypes.c_size_t, CUmemLocation, ctypes.c_uint, ctypes.c_void_p]

    @staticmethod
    def _check(result, what):
        if result != 0:
            raise RuntimeError(f"{what} failed with CUresult {result}")

    def read_mostly(self, t: torch.Tensor):
        loc = CUmemLocation(LOC_DEVICE, 0)
        self._check(self.advise(t.data_ptr(), t.nbytes, ADVISE_SET_READ_MOSTLY, loc), "cuMemAdvise(READ_MOSTLY)")

    def remote(self, t: torch.Tensor, device: int = 0):
        """Keep t in host RAM and map it for direct access from the GPU (no migration)."""
        self._check(self.advise(t.data_ptr(), t.nbytes, ADVISE_SET_PREFERRED_LOCATION, CUmemLocation(LOC_HOST, 0)),
                    "cuMemAdvise(PREFERRED_LOCATION=host)")
        self._check(self.advise(t.data_ptr(), t.nbytes, ADVISE_SET_ACCESSED_BY, CUmemLocation(LOC_DEVICE, device)),
                    "cuMemAdvise(ACCESSED_BY=gpu)")

    def prefetch_to_gpu(self, t: torch.Tensor, stream: torch.cuda.Stream, device: int = 0):
        self._check(self.prefetch(t.data_ptr(), t.nbytes, CUmemLocation(LOC_DEVICE, device), 0,
                                  ctypes.c_void_p(stream.cuda_stream)), "cuMemPrefetchAsync")

    def prefetch_to_host(self, t: torch.Tensor, stream: torch.cuda.Stream):
        self._check(self.prefetch(t.data_ptr(), t.nbytes, CUmemLocation(LOC_HOST, 0), 0,
                                  ctypes.c_void_p(stream.cuda_stream)), "cuMemPrefetchAsync(host)")
