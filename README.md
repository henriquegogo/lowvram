# lowvram

Run CUDA programs (PyTorch and others) that need more memory than your GPU has,
without changing their code.

```sh
lowvram python script.py
```

GPU allocations become CUDA managed memory: the NVIDIA driver keeps the pages
in use on the GPU and moves idle ones to system RAM. Instead of
`CUDA out of memory`, the program keeps running, slower when its working set
does not fit in VRAM.

## Build

Requirements: Linux (x86_64 or aarch64), glibc, a C compiler and the NVIDIA
driver. The CUDA toolkit is not needed.

```sh
make
```

This produces a single file, `lowvram`. To use it from anywhere, copy or
symlink it into your `PATH`:

```sh
ln -s "$PWD/lowvram" ~/.local/bin/lowvram
```

## Usage

```sh
lowvram <command> [args...]
```

Examples:

```sh
lowvram uv run script.py
lowvram python train.py --batch-size 64
```

The preload is inherited by child processes, so wrappers such as `uv run`, and
subprocesses started by the program, are covered too. Equivalent manual form:

```sh
LD_PRELOAD=/path/to/lowvram uv run script.py
```

## How it works

- `cuMemAlloc` is replaced by `cuMemAllocManaged`. The program, its kernels and
  libraries (cuBLAS, cuDNN, ...) still see ordinary device pointers. When VRAM
  is full the driver evicts the least recently used pages to RAM and pages them
  back in on access.
- `cuMemGetInfo` and `cuDeviceTotalMem` report VRAM plus half of system RAM, so
  programs that size themselves from free memory (e.g. `device_map="auto"`)
  use the extra space instead of offloading on their own.
- The CUDA runtime looks driver functions up through `dlsym` and
  `cuGetProcAddress`, so both are hooked to return the replacements.
- `lowvram` is a shared library with an entry point. When executed, it sets
  `LD_PRELOAD` to its own path and runs the command. When preloaded, only the
  hooks are active.

## Limitations

- Performance: once the working set exceeds VRAM, pages move over PCIe and
  kernels slow down. It trades speed for being able to run at all.
- Not intercepted, still limited to real VRAM: stream-ordered allocations
  (`cudaMallocAsync`) and virtual memory management (`cuMemCreate`/`cuMemMap`).
  In PyTorch, do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Reported free memory is real free VRAM plus a fixed amount; it does not
  shrink as managed memory is allocated. Programs that allocate until "free"
  reaches zero can exhaust system RAM.
- Linux only.

## License

[MIT](LICENSE)
