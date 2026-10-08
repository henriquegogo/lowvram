"""Real workload: the YuE2-3B music model (7.26 GB of bf16 weights) on a 6 GiB GPU.

YuE2 is an AR-NAR "mixture of transformers": every layer holds two attention blocks
and two MLPs. Token generation (AR) uses only the AR half; the flow-matching step
(NAR) runs both halves over the whole sequence. The two phases therefore differ in
working set (about 3.4 vs 6.8 GiB) and in arithmetic intensity (decode is bound by
memory traffic, the NAR step by compute).

Inputs are synthetic token ids and latents of realistic shape: timing depends on the
shapes, not on the values.

  yue2.py --phase ar  --mode gpu|accelerate|cpu|ar-only
  yue2.py --phase nar --mode gpu|accelerate|cpu
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from common import emit, pcie_now, run_guarded, summary, sync

MODEL = "m-a-p/YuE2-3B"
TEXT_VOCAB = 151643          # token ids below this are ordinary text tokens


def load(mode: str, budget_gib: float):
    t = time.perf_counter()
    kwargs = dict(trust_remote_code=True, dtype=torch.bfloat16)
    if mode == "gpu":
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda", **kwargs)
    elif mode == "accelerate":
        # The usual way to run a model larger than VRAM: Accelerate places whole
        # decoder layers on the GPU up to a budget and offloads the rest to RAM,
        # copying them to the GPU in every forward pass.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="auto", max_memory={0: f"{budget_gib}GiB", "cpu": "24GiB"}, **kwargs)
    elif mode == "ar-only":
        # Oracle for token generation: drop the NAR half (unused by AR decoding) so the
        # AR weights fit in VRAM as ordinary device memory. Needs knowledge of the model.
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cpu", **kwargs)
        for layer in model.model.layers:
            for name in ("nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp"):
                delattr(layer, name)
        for name in ("llm2vae", "vae2llm", "time_embedder", "latent_pos_embed"):
            delattr(model, name)
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cpu", **kwargs)
    model.eval()
    sync()
    placement = {}
    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            placement[str(dev)] = placement.get(str(dev), 0) + 1
    return model, time.perf_counter() - t, placement



@torch.no_grad()
def bench_ar(model, args):
    dev = "cpu" if args.mode == "cpu" else "cuda"
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, TEXT_VOCAB, (1, args.prompt), generator=g).to(dev)
    cache = DynamicCache()
    sync()
    t = time.perf_counter()
    out = model(input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    sync()
    prefill_s = time.perf_counter() - t
    token = out.logits[:, -1:].argmax(-1)
    lat = []
    for _ in range(args.tokens):
        t = time.perf_counter()
        out = model(input_ids=token, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
        token = out.logits[:, -1:].argmax(-1)
        sync()
        lat.append(time.perf_counter() - t)
    steady = lat[args.skip:] if len(lat) > args.skip else lat
    s = summary(steady)
    return {"prefill_s": prefill_s, "prefill_tok_s": args.prompt / prefill_s,
            "decode_tok_s": 1 / s["median_s"], "token_latencies_s": lat,
            "pcie_under_load": pcie_now() if dev == "cuda" else "", **s}


@torch.no_grad()
def bench_nar(model, args):
    """One flow-matching velocity evaluation per ODE step over [AR context | latent frames]."""
    dev = "cpu" if args.mode == "cpu" else "cuda"
    g = torch.Generator().manual_seed(0)
    a, frames = args.prompt, args.frames
    s_len = a + frames + 2                                   # AR tokens, LATENT_START, frames, LATENT_END
    tokens = torch.randint(0, TEXT_VOCAB, (1, s_len), generator=g).to(dev)
    ar_mask = torch.zeros(1, s_len, dtype=torch.bool, device=dev)
    ar_mask[0, :a] = True
    nar_mask = ~ar_mask
    content = torch.zeros_like(ar_mask)
    content[0, a + 1:a + 1 + frames] = True
    x_t = torch.randn(frames, model.config.latent_dim, generator=g).to(dev)
    steps = []
    for i in range(args.steps):
        sync()
        t = time.perf_counter()
        v = model.nar_velocity(tokens, ar_mask, nar_mask, content, x_t, t_value=float(i) / args.steps)
        sync()
        steps.append(time.perf_counter() - t)
        x_t = x_t + 0.01 * v.float().to(x_t.device)
    steady = steps[args.skip_steps:] if len(steps) > args.skip_steps else steps
    s = summary(steady)
    return {"sequence": s_len, "step_latencies_s": steps, "pcie_under_load": pcie_now() if dev == "cuda" else "", **s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["ar", "nar"], required=True)
    p.add_argument("--mode", choices=["gpu", "accelerate", "cpu", "ar-only"], required=True)
    p.add_argument("--budget-gib", type=float, default=4.0, help="accelerate: VRAM budget for weights")
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--tokens", type=int, default=96)
    p.add_argument("--skip", type=int, default=16, help="AR: warm-up tokens excluded from the median")
    p.add_argument("--frames", type=int, default=1024)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--skip-steps", type=int, default=1)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    config = {"phase": args.phase, "mode": args.mode, "prompt": args.prompt}
    if args.mode == "accelerate":
        config["budget_gib"] = args.budget_gib
    if args.phase == "ar":
        config.update(tokens=args.tokens, skip=args.skip)
    else:
        config.update(frames=args.frames, steps=args.steps, skip_steps=args.skip_steps)

    def run():
        model, load_s, placement = load(args.mode, args.budget_gib)
        metrics = bench_ar(model, args) if args.phase == "ar" else bench_nar(model, args)
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        return {"load_s": load_s, "placement": placement, "torch_peak_allocated_bytes": peak, **metrics}

    t0 = time.time()
    run_guarded("yue2", config, run)
    print(f"  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
