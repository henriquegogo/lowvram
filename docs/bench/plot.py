"""Turn benchmark results into figures, LaTeX tables/macros and a Markdown summary.

  uv run python plot.py [results/<run>.jsonl ...]     (default: the newest run)

Writes:
  ../paper/figures/*.pdf, *.png     figures for the paper, blog and slides
  ../paper/generated/numbers.tex    \\newcommand macros with every number quoted in the text
  ../paper/generated/*.tex          tables
  results/summary.md                the same numbers in Markdown
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "paper"
FIG = PAPER / "figures"
GEN = PAPER / "generated"
GiB = 1 << 30

# Categorical palette (fixed order, validated for CVD on adjacent pairs) and text inks.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(r):
    return (r["bench"], json.dumps({k: v for k, v in r["config"].items() if k != "reps"}, sort_keys=True),
            r["env"]["lowvram"])


def index(rows):
    """(bench, config, lowvram) -> list of records (repeats of the same configuration)."""
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


class Results:
    def __init__(self, rows):
        self.rows = rows
        self.idx = index(rows)

    def get(self, bench, lowvram, **config):
        """All records of one configuration (exact match on the given config fields)."""
        hits = []
        for (b, cfg, lv), recs in self.idx.items():
            if b != bench or lv != lowvram:
                continue
            c = json.loads(cfg)
            if all(c.get(k) == v for k, v in config.items()):
                hits.extend(recs)
        return hits

    def metric(self, bench, lowvram, name, **config):
        """Median over repeats of a metric; None when absent; 'oom' when every repeat ran out of memory."""
        recs = self.get(bench, lowvram, **config)
        if not recs:
            return None
        ok = [r["metrics"][name] for r in recs if r["status"] == "ok" and name in r["metrics"]]
        if ok:
            return statistics.median(ok)
        return "oom" if all(r["status"] == "oom" for r in recs) else None

    def spread(self, bench, lowvram, name, **config):
        vals = [r["metrics"][name] for r in self.get(bench, lowvram, **config)
                if r["status"] == "ok" and name in r["metrics"]]
        return (min(vals), max(vals), len(vals)) if vals else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- figures

def usable_vram(res):
    """GiB of VRAM a benchmark process could use: from the start-of-run measurement when
    recorded, else from PyTorch's OOM messages (memory in use by the process + still free)."""
    import re
    vals = [r["metrics"]["start_free_bytes"] / GiB for r in res.rows
            if not r["env"]["lowvram"] and r["metrics"].get("start_free_bytes")]
    if vals:
        return statistics.median(vals)
    units = {"GiB": 1, "MiB": 1 / 1024, "KiB": 1 / 2**20}
    for r in res.rows:
        m = re.search(r"of which ([\d.]+) (\w+) is free.*this process has ([\d.]+) (\w+) memory in use",
                      r["metrics"].get("error", ""))
        if m:
            vals.append(float(m[1]) * units[m[2]] + float(m[3]) * units[m[4]])
    return statistics.median(vals) if vals else None


def fig_sweep(res, numbers):
    sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, (lv, label) in enumerate([(False, "Unmodified PyTorch"), (True, "lowvram")]):
        xs, ys, oom = [], [], []
        for g in sizes:
            v = res.metric("sweep", lv, "effective_GBps", gib=float(g))
            if isinstance(v, float):
                xs.append(g)
                ys.append(v)
            elif v == "oom":
                oom.append(g)
        if xs:
            ax.plot(xs, ys, color=C[i], marker=MARKERS[i], label=label, zorder=3 - i)
    pcie = res.metric("pcie", False, "h2d_GBps")
    if isinstance(pcie, float):
        ax.axhline(pcie, color=INK2, lw=1, ls="--")
        ax.text(12, pcie * 1.15, f"pinned copy over PCIe ({pcie:.1f} GB/s)", ha="right", color=INK2, fontsize=8)
    vram = usable_vram(res)
    if vram:
        ax.axvline(vram, color=INK2, lw=1, ls=":")
        ax.text(vram + 0.15, 30, f"free VRAM\n{vram:.1f} GiB", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("working set (GiB), touched once per pass in a fixed order")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("Cyclic access: full speed while it fits, PCIe-bound after", loc="left")
    ax.legend(loc="center right", fontsize=8)
    save(fig, "sweep")
    big = [res.metric("sweep", True, "effective_GBps", gib=float(g)) for g in (6, 8, 10, 12)]
    big = [v for v in big if isinstance(v, float)]
    if big:
        numbers["sweepOversubGBps"] = f"{statistics.median(big):.1f}"
    small = [res.metric("sweep", True, "effective_GBps", gib=float(g)) for g in (1, 2, 3, 4)]
    small = [v for v in small if isinstance(v, float)]
    if small:
        numbers["sweepFitGBps"] = f"{statistics.median(small):.0f}"


LAYER_SERIES = [
    ("lowvram", True, "gpu"),
    ("lowvram + ReadMostly", True, "readmostly"),
    ("lowvram + prefetch", True, "prefetch"),
    ("lowvram + remote map", True, "remote"),
    ("manual streaming", False, "offload"),
    ("manual streaming, overlapped", False, "overlap"),
    ("CPU (FP32)", False, "cpu"),
]


def fig_layers(res, numbers):
    sizes = [2, 4, 6, 8, 12]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True)
    for ax, batch in zip(axes, (16, 2048)):
        for i, (label, lv, mode) in enumerate(LAYER_SERIES):
            xs, ys = [], []
            for g in sizes:
                v = res.metric("layers", lv, "median_s", gib=float(g), batch=batch, mode=mode)
                if isinstance(v, float):
                    xs.append(g)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, color=C[i], marker=MARKERS[i], label=label)
        xs = [g for g in sizes if isinstance(res.metric("layers", False, "median_s", gib=float(g), batch=batch, mode="gpu"), float)]
        ax.plot(xs, [res.metric("layers", False, "median_s", gib=float(g), batch=batch, mode="gpu") for g in xs],
                color=INK2, ls="--", lw=1.2, marker="o", mfc="white", label="unmodified PyTorch", zorder=5)
        base = [g for g in sizes if res.metric("layers", False, "median_s", gib=float(g), batch=batch, mode="gpu") == "oom"]
        if base:
            ax.axvspan(min(base) - 0.5, max(sizes) + 0.5, color=GRID, alpha=0.5, lw=0)
        ax.set_yscale("log")
        ax.set_title(f"batch {batch} ({'memory-bound' if batch == 16 else 'compute-bound'})", loc="left")
        ax.set_xlabel("weights (GiB)")
        ax.set_xticks(sizes)
    axes[0].set_ylabel("seconds per forward pass")
    fig.text(0.5, -0.02, "shaded: unmodified PyTorch runs out of memory", ha="center", color=INK2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12), fontsize=8)
    fig.tight_layout()
    save(fig, "layers")


def fig_yue2(res, numbers):
    ar_modes = [("Unmodified", False, "gpu"), ("Accelerate offload", False, "accelerate"),
                ("CPU (BF16)", False, "cpu"), ("lowvram", True, "gpu"), ("AR half only (oracle)", False, "ar-only")]
    nar_modes = [("Unmodified", False, "gpu"), ("Accelerate offload", False, "accelerate"),
                 ("CPU (BF16)", False, "cpu"), ("lowvram", True, "gpu")]
    color = {"Accelerate offload": C[1], "CPU (BF16)": C[2], "lowvram": C[0], "AR half only (oracle)": C[6]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, phase, modes, metric, unit, fmt in (
            (axes[0], "ar", ar_modes, "decode_tok_s", "tokens per second (higher is better)", "{:.1f}"),
            (axes[1], "nar", nar_modes, "median_s", "seconds per ODE step (lower is better)", "{:.2f}")):
        labels, vals = [], []
        for label, lv, mode in modes:
            v = res.metric("yue2", lv, metric, phase=phase, mode=mode)
            if v is None:
                continue
            labels.append(label)
            vals.append(v)
        ys = range(len(labels))
        top = max([v for v in vals if isinstance(v, float)] or [1])
        for y, label, v in zip(ys, labels, vals):
            if isinstance(v, float):
                ax.barh(y, v, color=color.get(label, C[0]), height=0.6)
                ax.text(v + top * 0.02, y, fmt.format(v), va="center", color=INK, fontsize=8)
            else:
                ax.text(top * 0.02, y, "out of memory", va="center", color=INK2, fontsize=8, style="italic")
        ax.set_yticks(list(ys), labels)
        ax.set_ylim(len(labels) - 0.5, -0.9)
        ax.set_xlim(0, top * 1.22)
        ax.set_xlabel(unit)
        ax.grid(axis="y", visible=False)
    axes[0].set_title("Token generation (AR, 3.4 GiB hot)", loc="left")
    axes[1].set_title("Flow-matching step (NAR, all 6.8 GiB)", loc="left")
    fig.tight_layout()
    save(fig, "yue2")


def fig_yue2_warmup(res, numbers):
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for i, (label, lv, mode, c) in enumerate([("lowvram", True, "gpu", C[0]),
                                               ("AR half only (oracle)", False, "ar-only", C[6])]):
        recs = [r for r in res.get("yue2", lv, phase="ar", mode=mode) if r["status"] == "ok"]
        if not recs:
            continue
        lat = recs[0]["metrics"]["token_latencies_s"]
        ax.plot(range(1, len(lat) + 1), [t * 1e3 for t in lat], color=c, label=label, lw=1.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([20, 30, 50, 100, 200, 300])
    ax.set_xlabel("generated token")
    ax.set_ylabel("latency (ms)")
    ax.set_title("lowvram reaches the oracle's latency after a few tokens", loc="left")
    ax.legend(fontsize=8)
    save(fig, "yue2_warmup")


LLM_MODES = [("Unmodified", False, "gpu"), ("Accelerate offload", False, "accelerate"),
             ("CPU (BF16)", False, "cpu"), ("lowvram", True, "gpu")]


def fig_llm(res, numbers):
    """Dense LLM: prompt processing (compute-bound) and token generation (memory-bound)."""
    if not res.get("llm", True) and not res.get("llm", False):
        return
    color = {"Accelerate offload": C[1], "CPU (BF16)": C[2], "lowvram": C[0]}
    model = next(r["config"]["model"] for r in res.rows if r["bench"] == "llm")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ax, metric, unit, title in ((axes[0], "prefill_tok_s", "tokens per second", "Prompt processing"),
                                    (axes[1], "decode_tok_s", "tokens per second", "Token generation")):
        labels, vals = [], []
        for label, lv, mode in LLM_MODES:
            v = res.metric("llm", lv, metric, mode=mode)
            if v is not None:
                labels.append(label)
                vals.append(v)
        top = max([v for v in vals if isinstance(v, float)] or [1])
        for y, label, v in zip(range(len(labels)), labels, vals):
            if isinstance(v, float):
                ax.barh(y, v, color=color.get(label, C[0]), height=0.6)
                ax.text(v + top * 0.02, y, f"{v:.1f}" if v < 100 else f"{v:.0f}", va="center", color=INK, fontsize=8)
            else:
                ax.text(top * 0.02, y, "out of memory", va="center", color=INK2, fontsize=8, style="italic")
        ax.set_yticks(list(range(len(labels))), labels)
        ax.set_ylim(len(labels) - 0.5, -0.9)
        ax.set_xlim(0, top * 1.22)
        ax.set_xlabel(unit + " (higher is better)")
        ax.set_title(f"{title} ({model})", loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "llm")


# --------------------------------------------------------------------------- tables and numbers

def fmt(v, spec):
    if v == "oom":
        return "OOM"
    if v is None:
        return "--"
    return spec.format(v)


def tables(res, numbers):
    GEN.mkdir(parents=True, exist_ok=True)
    md = ["# Benchmark summary", ""]
    run_ids = sorted({r["run_id"] for r in res.rows})
    env = next((r["env"] for r in res.rows if not r["env"]["lowvram"] and "gpu" in r["env"]), {})
    md.append(f"Runs: {', '.join(run_ids)}  ")
    md.append(f"Machine: {env.get('gpu')} (driver {env.get('driver')}), {env.get('cpu')}, "
              f"{env.get('ram_bytes', 0) / GiB:.0f} GiB RAM, Linux {env.get('kernel')}, "
              f"PyTorch {env.get('torch')} (CUDA {env.get('cuda_runtime')})")
    md.append("")

    # Table 1: overhead when the working set fits.
    rows = []
    for name, bench, metric, cfg, spec in [
        ("GEMM FP16 (TFLOP/s)", "gemm", "tflops", {"dtype": "fp16"}, "{:.2f}"),
        ("GEMM FP32 (TFLOP/s)", "gemm", "tflops", {"dtype": "fp32"}, "{:.2f}"),
        ("Sweep 1 GiB (GB/s)", "sweep", "effective_GBps", {"gib": 1.0}, "{:.0f}"),
        ("Sweep 4 GiB (GB/s)", "sweep", "effective_GBps", {"gib": 4.0}, "{:.0f}"),
        ("Layers 2 GiB, batch 16 (ms)", "layers", "median_s", {"gib": 2.0, "batch": 16, "mode": "gpu"}, "ms"),
        ("Layers 4 GiB, batch 16 (ms)", "layers", "median_s", {"gib": 4.0, "batch": 16, "mode": "gpu"}, "ms"),
        ("Layers 4 GiB, batch 2048 (ms)", "layers", "median_s", {"gib": 4.0, "batch": 2048, "mode": "gpu"}, "ms"),
    ]:
        b = res.metric(bench, False, metric, **cfg)
        l = res.metric(bench, True, metric, **cfg)
        if spec == "ms":
            b = b * 1e3 if isinstance(b, float) else b
            l = l * 1e3 if isinstance(l, float) else l
            spec = "{:.1f}"
        change = f"{(l / b - 1) * 100:+.1f}\\%" if isinstance(b, float) and isinstance(l, float) else "--"
        rows.append((name, fmt(b, spec), fmt(l, spec), change))
    tex = ["\\begin{tabular}{lrrr}", "\\toprule", "Workload & Unmodified & lowvram & Change \\\\", "\\midrule"]
    tex += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_overhead.tex").write_text("\n".join(tex) + "\n")
    md += ["## Overhead when the working set fits", "", "| Workload | Unmodified | lowvram | Change |",
           "|---|---:|---:|---:|"]
    md += [f"| {a} | {b} | {c} | {d.replace(chr(92), '')} |" for a, b, c, d in rows]
    md.append("")

    # Table 2: layer stack beyond VRAM.
    sizes = [6, 8, 12]
    tex = ["\\begin{tabular}{l" + "r" * (2 * len(sizes)) + "}", "\\toprule",
           " & \\multicolumn{3}{c}{batch 16} & \\multicolumn{3}{c}{batch 2048} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "Mode & " + " & ".join(f"{g}\\,GiB" for g in sizes * 2) + " \\\\", "\\midrule"]
    md += ["## Layer stack beyond VRAM (seconds per forward pass)", "",
           "| Mode | " + " | ".join(f"b16 {g} GiB" for g in sizes) + " | " +
           " | ".join(f"b2048 {g} GiB" for g in sizes) + " |", "|---" + "|---:" * 6 + "|"]
    for label, lv, mode in [("Unmodified", False, "gpu")] + LAYER_SERIES:
        cells = []
        for batch in (16, 2048):
            for g in sizes:
                cells.append(fmt(res.metric("layers", lv, "median_s", gib=float(g), batch=batch, mode=mode), "{:.2f}"))
        tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_layers.tex").write_text("\n".join(tex) + "\n")
    md.append("")

    # Table 3: YuE2.
    tex = ["\\begin{tabular}{lrrrr}", "\\toprule",
           " & \\multicolumn{2}{c}{Token generation (AR)} & \\multicolumn{2}{c}{Flow-matching step (NAR)} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
           "Mode & prompt tok/s & decode tok/s & s/step & GPU weights \\\\", "\\midrule"]
    md += ["## YuE2-3B (7.26 GB of BF16 weights)", "",
           "| Mode | AR prompt tok/s | AR decode tok/s | NAR s/step |", "|---|---:|---:|---:|"]
    for label, lv, mode in [("Unmodified", False, "gpu"), ("Accelerate offload", False, "accelerate"),
                            ("CPU (BF16)", False, "cpu"), ("lowvram", True, "gpu"),
                            ("AR half only (oracle)", False, "ar-only")]:
        pre = res.metric("yue2", lv, "prefill_tok_s", phase="ar", mode=mode)
        dec = res.metric("yue2", lv, "decode_tok_s", phase="ar", mode=mode)
        nar = res.metric("yue2", lv, "median_s", phase="nar", mode=mode)
        where = {"gpu": "all (paged)" if lv else "all", "accelerate": "part", "cpu": "none",
                 "ar-only": "AR half"}[mode]
        tex.append(f"{label} & {fmt(pre, '{:.0f}')} & {fmt(dec, '{:.1f}')} & {fmt(nar, '{:.2f}')} & {where} \\\\")
        md.append(f"| {label} | {fmt(pre, '{:.0f}')} | {fmt(dec, '{:.1f}')} | {fmt(nar, '{:.2f}')} |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_yue2.tex").write_text("\n".join(tex) + "\n")
    md.append("")

    # Table 4: dense LLM.
    llm_rows = []
    for label, lv, mode in LLM_MODES:
        pre = res.metric("llm", lv, "prefill_tok_s", mode=mode)
        dec = res.metric("llm", lv, "decode_tok_s", mode=mode)
        if pre is not None or dec is not None:
            llm_rows.append((label, fmt(pre, "{:.0f}"), fmt(dec, "{:.2f}")))
    if llm_rows:
        tex = ["\\begin{tabular}{lrr}", "\\toprule", "Mode & prompt tok/s & decode tok/s \\\\", "\\midrule"]
        tex += [f"{a} & {b} & {c} \\\\" for a, b, c in llm_rows]
        tex += ["\\bottomrule", "\\end{tabular}"]
        (GEN / "table_llm.tex").write_text("\n".join(tex) + "\n")
        model = next(r["config"]["model"] for r in res.rows if r["bench"] == "llm")
        md += [f"## Dense LLM ({model})", "", "| Mode | prompt tok/s | decode tok/s |", "|---|---:|---:|"]
        md += [f"| {a} | {b} | {c} |" for a, b, c in llm_rows]
        md.append("")
        w = next((r["metrics"]["weight_bytes"] for r in res.rows if r["bench"] == "llm" and r["status"] == "ok"), None)
        if w:
            numbers["llmWeightsGiB"] = f"{w / GiB:.1f}"
        for tag, lv, mode in (("Lv", True, "gpu"), ("Acc", False, "accelerate"), ("Cpu", False, "cpu")):
            v = res.metric("llm", lv, "prefill_tok_s", mode=mode)
            if isinstance(v, float):
                numbers[f"llm{tag}Prefill"] = f"{v:.0f}"
            v = res.metric("llm", lv, "decode_tok_s", mode=mode)
            if isinstance(v, float):
                numbers[f"llm{tag}Decode"] = f"{v:.2f}"

    # Named numbers for the text.
    def put(name, v, spec):
        if isinstance(v, float):
            numbers[name] = spec.format(v)

    put("pcieHtoD", res.metric("pcie", False, "h2d_GBps"), "{:.1f}")
    put("pcieDtoH", res.metric("pcie", False, "d2h_GBps"), "{:.1f}")
    lv_dec = res.metric("yue2", True, "decode_tok_s", phase="ar", mode="gpu")
    or_dec = res.metric("yue2", False, "decode_tok_s", phase="ar", mode="ar-only")
    ac_dec = res.metric("yue2", False, "decode_tok_s", phase="ar", mode="accelerate")
    cpu_dec = res.metric("yue2", False, "decode_tok_s", phase="ar", mode="cpu")
    put("yueLvDecode", lv_dec, "{:.1f}")
    put("yueOracleDecode", or_dec, "{:.1f}")
    put("yueAccDecode", ac_dec, "{:.1f}")
    put("yueCpuDecode", cpu_dec, "{:.1f}")
    if all(isinstance(v, float) for v in (lv_dec, or_dec, ac_dec)):
        numbers["yueLvOfOracle"] = f"{lv_dec / or_dec * 100:.0f}"
        numbers["yueLvOverAcc"] = f"{lv_dec / ac_dec:.0f}"
    put("yueLvPrefill", res.metric("yue2", True, "prefill_tok_s", phase="ar", mode="gpu"), "{:.0f}")
    put("yueOraclePrefill", res.metric("yue2", False, "prefill_tok_s", phase="ar", mode="ar-only"), "{:.0f}")
    put("yueAccPrefill", res.metric("yue2", False, "prefill_tok_s", phase="ar", mode="accelerate"), "{:.0f}")
    lv_nar = res.metric("yue2", True, "median_s", phase="nar", mode="gpu")
    ac_nar = res.metric("yue2", False, "median_s", phase="nar", mode="accelerate")
    put("yueLvNar", lv_nar, "{:.2f}")
    put("yueAccNar", ac_nar, "{:.2f}")
    put("yueCpuNar", res.metric("yue2", False, "median_s", phase="nar", mode="cpu"), "{:.1f}")
    if isinstance(lv_nar, float) and isinstance(ac_nar, float):
        numbers["yueNarSlowdown"] = f"{lv_nar / ac_nar:.1f}"
    sp = res.spread("yue2", True, "decode_tok_s", phase="ar", mode="gpu")
    if sp:
        numbers["yueLvDecodeRange"] = f"{sp[0]:.1f}--{sp[1]:.1f}"
        numbers["yueRepeats"] = str(sp[2])
    if env:
        numbers["usableVram"] = f"{usable_vram(res):.2f}"

    for g in (8,):
        for batch in (16, 2048):
            tag = f"{'Lo' if batch == 16 else 'Hi'}"
            for label, lv, mode in [("Lv", True, "gpu"), ("Rm", True, "readmostly"), ("Pf", True, "prefetch"),
                                    ("Rem", True, "remote"), ("Off", False, "offload"),
                                    ("Ovl", False, "overlap"), ("Cpu", False, "cpu")]:
                put(f"layers{tag}{label}", res.metric("layers", lv, "median_s", gib=float(g), batch=batch, mode=mode),
                    "{:.2f}")

    bad = [k for k in numbers if not k.isalpha()]
    assert not bad, f"LaTeX macro names must be letters only: {bad}"
    lines = ["% Generated by docs/bench/plot.py from the raw results. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(numbers.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    md += ["## Named numbers", ""] + [f"- `{k}`: {v}" for k, v in sorted(numbers.items())]
    (HERE / "results" / "summary.md").write_text("\n".join(md) + "\n")


def main():
    paths = sys.argv[1:]
    if not paths:
        runs = sorted((HERE / "results").glob("*.jsonl"))
        if not runs:
            sys.exit("no results/*.jsonl")
        paths = [runs[-1]]
    res = Results(load(paths))
    numbers = {}
    fig_sweep(res, numbers)
    fig_layers(res, numbers)
    fig_yue2(res, numbers)
    fig_yue2_warmup(res, numbers)
    fig_llm(res, numbers)
    tables(res, numbers)
    print(f"{len(res.rows)} results -> {FIG}, {GEN}, results/summary.md")


if __name__ == "__main__":
    main()
