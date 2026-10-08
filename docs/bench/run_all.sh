#!/usr/bin/env bash
# Run the full benchmark matrix. Every configuration is a fresh process.
# Results: results/<run id>.jsonl (one JSON object per configuration).
#
#   ./run_all.sh            everything (~1 hour on an RTX 3060 Laptop GPU)
#   ./run_all.sh micro      synthetic benchmarks only
#   ./run_all.sh yue2       YuE2-3B only (needs m-a-p/YuE2-3B in the Hugging Face cache)
#   ./run_all.sh llm        dense LLM only (needs models/Qwen3-4B, see README)
# BENCH_RUN_ID=<existing id> appends to an existing run.
set -u
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
LOWVRAM="$REPO/lowvram"
make -C "$REPO" -s lowvram

export BENCH_RUN_ID=${BENCH_RUN_ID:-$(date +%Y%m%d-%H%M%S)}
mkdir -p results
export BENCH_OUT="$PWD/results/$BENCH_RUN_ID.jsonl"
export HF_HUB_OFFLINE=1
WHAT=${1:-all}
REPEAT=${REPEAT:-3}

uv sync -q
PY="uv run --no-sync python"
base() { $PY "$@"; }
lv() { "$LOWVRAM" $PY "$@"; }
both() { base "$@"; lv "$@"; }

echo "run $BENCH_RUN_ID -> $BENCH_OUT"
nvidia-smi --query-gpu=name,memory.used,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv

if [[ $WHAT == all || $WHAT == micro ]]; then
    # Hardware ceilings.
    base micro.py pcie

    # 1. Overhead when the working set fits.
    both micro.py gemm --dtype fp16 --reps 20
    both micro.py gemm --dtype fp32 --reps 20
    for g in 1 2 3 4; do both micro.py sweep --gib $g; done
    for b in 16 2048; do
        for g in 2 4; do
            both micro.py layers --gib $g --batch $b --mode gpu
            base micro.py layers --gib $g --batch $b --mode offload
            base micro.py layers --gib $g --batch $b --mode overlap
        done
    done

    # 2. Working set beyond VRAM: the cyclic sweep (worst case for LRU eviction).
    for g in 5 6 8 10 12; do both micro.py sweep --gib $g; done

    # 3. Layer stack beyond VRAM: transparent paging vs manual streaming vs hints.
    for b in 16 2048; do
        for g in 6 8 12; do
            both micro.py layers --gib $g --batch $b --mode gpu            # baseline: OOM
            base micro.py layers --gib $g --batch $b --mode offload
            base micro.py layers --gib $g --batch $b --mode overlap
            lv micro.py layers --gib $g --batch $b --mode readmostly
            lv micro.py layers --gib $g --batch $b --mode prefetch
            lv micro.py layers --gib $g --batch $b --mode remote
        done
    done

    # 4. CPU execution of the same stack (fp32; the CPU has no fast fp16 GEMM).
    for g in 2 4 8; do base micro.py layers --gib $g --batch 16 --mode cpu --reps 3; done
    base micro.py layers --gib 8 --batch 2048 --mode cpu --reps 2
fi

if [[ $WHAT == all || $WHAT == yue2 ]]; then
    for i in $(seq "$REPEAT"); do
        both yue2.py --phase ar --mode gpu                 # baseline: OOM; lowvram: paged
        base yue2.py --phase ar --mode ar-only              # oracle: NAR half removed, fits in VRAM
        base yue2.py --phase ar --mode accelerate --tokens 32 --skip 8
        both yue2.py --phase nar --mode gpu
        base yue2.py --phase nar --mode accelerate
    done
    base yue2.py --phase ar --mode cpu --tokens 24 --skip 8
    base yue2.py --phase nar --mode cpu --steps 2 --skip-steps 1
fi

if [[ $WHAT == all || $WHAT == llm ]]; then
    LLM=${LLM:-models/Qwen3-4B}
    for i in $(seq "$REPEAT"); do
        both llm.py --model "$LLM" --mode gpu                    # baseline: OOM; lowvram: paged
        base llm.py --model "$LLM" --mode accelerate --tokens 24
    done
    base llm.py --model "$LLM" --mode cpu --prompt 256 --tokens 24
fi

echo "done: $(wc -l < "$BENCH_OUT") results in $BENCH_OUT"
