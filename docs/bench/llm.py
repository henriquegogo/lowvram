"""Generic real workload: any Hugging Face causal language model larger than VRAM.

Unlike YuE2, a dense LLM reads all of its weights for every token, so its hot set is
the whole model. Prompt processing is compute-bound (many tokens per weight read);
token generation is bound by memory traffic (one token per weight read).

  llm.py --model models/Qwen3-4B --mode gpu|accelerate|cpu [--prompt 2048 --tokens 64]

Inputs are random token ids: timing depends on shapes, not values.
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from common import emit, pcie_now, run_guarded, summary, sync

HERE = Path(__file__).resolve().parent


def load(path: str, mode: str, budget_gib: float):
    t = time.perf_counter()
    kwargs = dict(dtype=torch.bfloat16)
    if mode == "gpu":
        model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", **kwargs)
    elif mode == "accelerate":
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", max_memory={0: f"{budget_gib}GiB", "cpu": "24GiB"}, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu", **kwargs)
    model.eval()
    sync()
    return model, time.perf_counter() - t


@torch.no_grad()
def bench(model, args, vocab):
    dev = "cpu" if args.mode == "cpu" else "cuda"
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, vocab, (1, args.prompt), generator=g).to(dev)
    # Warm-up prefill on a short prompt, so the timed prefill does not include one-off setup.
    model(input_ids=prompt[:, :16], past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
    sync()
    cache = DynamicCache()
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
    return {"prefill_s": prefill_s, "prefill_tok_s": args.prompt / prefill_s, "decode_tok_s": 1 / s["median_s"],
            "token_latencies_s": lat, "pcie_under_load": pcie_now() if dev == "cuda" else "", **s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(HERE / "models" / "Qwen3-4B"))
    p.add_argument("--mode", choices=["gpu", "accelerate", "cpu"], required=True)
    p.add_argument("--budget-gib", type=float, default=4.0)
    p.add_argument("--prompt", type=int, default=1024)
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--skip", type=int, default=8)
    args = p.parse_args()

    name = Path(args.model).name
    config = {"model": name, "mode": args.mode, "prompt": args.prompt, "tokens": args.tokens, "skip": args.skip}
    if args.mode == "accelerate":
        config["budget_gib"] = args.budget_gib

    def run():
        vocab = AutoConfig.from_pretrained(args.model).vocab_size
        model, load_s = load(args.model, args.mode, args.budget_gib)
        weights = sum(t.numel() * t.element_size() for t in model.parameters())
        metrics = bench(model, args, min(vocab, 150000))
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        return {"load_s": load_s, "weight_bytes": weights, "torch_peak_allocated_bytes": peak, **metrics}

    t0 = time.time()
    run_guarded("llm", config, run)
    print(f"  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
