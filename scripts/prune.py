#!/usr/bin/env python
"""One-shot pruning + evaluation CLI.

Example:
    python scripts/prune.py --model tinyllama-1.1b --method cws --sparsity 0.5 \\
        --device cuda --benchmark
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cws.data import get_c4_calibration, get_wikitext2_test
from cws.eval_ppl import eval_ppl
from cws.models import load_model_and_tokenizer
from cws.models.adapters import capture_block0_inputs, find_blocks
from cws.sequential import METHODS, prune_model_sequential


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Registry key or HF hub id/path")
    p.add_argument("--method", choices=METHODS + ("dense",), default="cws")
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--blocksize", type=int, default=128, help="0 => global (None) CWS variant")
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--benchmark", action="store_true", help="also run lm-eval-harness zero-shot tasks")
    p.add_argument("--full-tasks", action="store_true", help="include LAMBADA alongside the default suite")
    p.add_argument("--save-model", default=None, help="directory to save the pruned checkpoint to")
    p.add_argument("--out", default=None, help="path to write a results JSON to")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype, device=args.device)

    result = {
        "model": args.model,
        "method": args.method,
        "sparsity": args.sparsity,
        "blocksize": args.blocksize,
    }

    if args.method != "dense":
        print(f"Collecting {args.nsamples} C4 calibration samples ...")
        calib_samples = get_c4_calibration(
            tokenizer, nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed
        )

        blocks = find_blocks(model)
        print(f"Found {len(blocks)} sequential blocks. Capturing block-0 inputs ...")
        calib_inputs = capture_block0_inputs(model, blocks, calib_samples, args.device)

        t0 = time.time()
        stats = prune_model_sequential(
            model,
            calib_inputs,
            method=args.method,
            sparsity=args.sparsity,
            blocksize=(args.blocksize or None),
            percdamp=args.percdamp,
            device=args.device,
        )
        result["prune_seconds"] = time.time() - t0
        result["achieved_sparsity"] = sum(
            v for block in stats.values() for v in block.values()
        ) / sum(len(block) for block in stats.values())

    print("Evaluating WikiText-2 perplexity ...")
    eval_chunks = get_wikitext2_test(tokenizer, seqlen=2048)
    result["wikitext2_ppl"] = eval_ppl(model, eval_chunks, device=args.device)
    print(f"WikiText-2 PPL: {result['wikitext2_ppl']:.3f}")

    if args.benchmark:
        from cws.eval_tasks import DEFAULT_TASKS, FULL_TASKS, run_lm_eval

        tasks = FULL_TASKS if args.full_tasks else DEFAULT_TASKS
        print(f"Running zero-shot benchmark: {tasks} ...")
        accs, _ = run_lm_eval(model, tokenizer, tasks=tasks, device=args.device)
        result["lm_eval"] = accs
        print(json.dumps(accs, indent=2))

    if args.save_model:
        print(f"Saving pruned model to {args.save_model}")
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"Wrote results to {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
