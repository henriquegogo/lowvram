/*
 * lowvram - let CUDA programs (e.g. PyTorch) use more memory than the GPU has.
 *
 * Every cuMemAlloc becomes cuMemAllocManaged: the NVIDIA driver keeps pages in
 * use on the GPU and moves idle ones to system RAM. Reported VRAM grows by half
 * the system RAM, so programs that size themselves from free memory use it too.
 *
 * The CUDA runtime looks driver functions up with dlsym/cuGetProcAddress, so
 * those are hooked to hand out the replacements.
 *
 * Not covered: cuMemAllocAsync pools and cuMemCreate/cuMemMap (e.g. PyTorch
 * expandable_segments:True) bypass this and stay limited to real VRAM.
 *
 * One file is both the launcher and the library: run as a program, it puts
 * itself in LD_PRELOAD and execs the command; preloaded, only the hooks run.
 *
 * Use: lowvram uv run script.py
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

static void *(*real_dlsym)(void *, const char *);
static CUresult (*real_alloc_managed)(CUdeviceptr *, size_t, unsigned);
static CUresult (*real_mem_info)(size_t *, size_t *);
static CUresult (*real_total_mem)(size_t *, int);
static CUresult (*real_proc)(const char *, void **, int, uint64_t);
static CUresult (*real_proc_v2)(const char *, void **, int, uint64_t, int *);
static size_t extra;

static void init_once(void) {
    real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!real_dlsym) real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda) return;  /* no NVIDIA driver: hook nothing */
    real_alloc_managed = real_dlsym(cuda, "cuMemAllocManaged");
    real_mem_info = real_dlsym(cuda, "cuMemGetInfo_v2");
    real_total_mem = real_dlsym(cuda, "cuDeviceTotalMem_v2");
    real_proc = real_dlsym(cuda, "cuGetProcAddress");
    real_proc_v2 = real_dlsym(cuda, "cuGetProcAddress_v2");
    extra = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGESIZE) / 2;
}

static void init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, init_once);
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t size) {
    init();
    return real_alloc_managed(dptr, size, 1 /* CU_MEM_ATTACH_GLOBAL */);
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    init();
    CUresult result = real_mem_info(free, total);
    if (result == 0) *free += extra, *total += extra;
    return result;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, int device) {
    init();
    CUresult result = real_total_mem(bytes, device);
    if (result == 0) *bytes += extra;
    return result;
}

static void *hook_for(const char *name);

CUresult cuGetProcAddress(const char *symbol, void **pfn, int version, uint64_t flags) {
    init();
    CUresult result = real_proc(symbol, pfn, version, flags);
    void *hook = hook_for(symbol);
    if (result == 0 && hook) *pfn = hook;
    return result;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int version, uint64_t flags, int *status) {
    init();
    CUresult result = real_proc_v2(symbol, pfn, version, flags, status);
    void *hook = hook_for(symbol);
    if (result == 0 && hook) *pfn = hook;
    return result;
}

/* cuGetProcAddress takes base names ("cuMemAlloc"); dlsym takes exported ones. */
static void *hook_for(const char *name) {
    if (!strcmp(name, "cuMemAlloc") || !strcmp(name, "cuMemAlloc_v2")) return (void *)cuMemAlloc_v2;
    if (!strcmp(name, "cuMemGetInfo") || !strcmp(name, "cuMemGetInfo_v2")) return (void *)cuMemGetInfo_v2;
    if (!strcmp(name, "cuDeviceTotalMem") || !strcmp(name, "cuDeviceTotalMem_v2")) return (void *)cuDeviceTotalMem_v2;
    if (!strcmp(name, "cuGetProcAddress")) return (void *)cuGetProcAddress;
    if (!strcmp(name, "cuGetProcAddress_v2")) return (void *)cuGetProcAddress_v2;
    return NULL;
}

void *dlsym(void *handle, const char *name) {
    init();
    void *hook = real_alloc_managed ? hook_for(name) : NULL;
    return hook ? hook : real_dlsym(handle, name);
}

/* Launcher: the entry point when this file is executed instead of preloaded. */
#if defined(__x86_64__)
const char interp[] __attribute__((section(".interp"))) = "/lib64/ld-linux-x86-64.so.2";
#elif defined(__aarch64__)
const char interp[] __attribute__((section(".interp"))) = "/lib/ld-linux-aarch64.so.1";
#endif

__attribute__((force_align_arg_pointer, noreturn)) void lowvram_main(void) {
    /* No C startup code runs here, so argv comes from /proc. */
    static char cmdline[1 << 20], *argv[1 << 14];
    FILE *f = fopen("/proc/self/cmdline", "r");
    size_t len = f ? fread(cmdline, 1, sizeof cmdline - 1, f) : 0;
    int argc = 0;
    for (size_t i = 0; i < len && argc < (1 << 14) - 1; i += strlen(cmdline + i) + 1) argv[argc++] = cmdline + i;
    argv[argc] = NULL;
    if (argc < 2) {
        fprintf(stderr, "usage: lowvram <command> [args...]\n");
        _exit(2);
    }
    char self[4096], preload[8192];
    ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n < 0) _exit(127);
    self[n] = 0;
    const char *old = getenv("LD_PRELOAD");
    snprintf(preload, sizeof preload, old && *old ? "%s:%s" : "%s", self, old);
    setenv("LD_PRELOAD", preload, 1);
    execvp(argv[1], argv + 1);
    perror(argv[1]);
    _exit(127);
}
