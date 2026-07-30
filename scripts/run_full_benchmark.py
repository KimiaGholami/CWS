#!/usr/bin/env python
"""Full benchmark suite (PPL + zero-shot lm-eval tasks) at 50% and 80%
sparsity, matching the CWS paper's Tables 1/2/4/5/8/9.

Example:
    python scripts/run_full_benchmark.py --models tinyllama-1.1b hgrn-1.3b llama-7b \\
        --methods cws sparsegpt wanda --device cuda --out results/benchmark.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cws.data import get_c4_calibration, get_wikitext2_test
from cws.eval_ppl import eval_ppl
from cws.eval_tasks import FULL_TASKS, run_lm_eval
from cws.models import load_model_and_tokenizer
from cws.models.adapters import capture_block0_inputs, find_blocks
from cws.sequential import prune_model_sequential

BENCHMARK_SPARSITIES = (0.5, 0.8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=["cws"])
    p.add_argument(
        "--sparsities", nargs="+", type=float, default=list(BENCHMARK_SPARSITIES)
    )
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--include-dense", action="store_true")
    p.add_argument("--out", default="results/benchmark.json")
    return p.parse_args()


def evaluate(model, tokenizer, device):
    eval_chunks = get_wikitext2_test(tokenizer, seqlen=2048)
    ppl = eval_ppl(model, eval_chunks, device=device)
    accs, _ = run_lm_eval(model, tokenizer, tasks=FULL_TASKS, device=device)
    return {"wikitext2_ppl": ppl, **accs}


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    all_results = []

    for model_name in args.models:
        if args.include_dense:
            print(f"=== {model_name} | dense baseline ===")
            model, tokenizer = load_model_and_tokenizer(model_name, dtype=dtype, device=args.device)
            metrics = evaluate(model, tokenizer, args.device)
            all_results.append({"model": model_name, "method": "dense", "sparsity": 0.0, **metrics})
            del model

        for method in args.methods:
            for sparsity in args.sparsities:
                print(f"=== {model_name} | {method} | sparsity={sparsity} ===")
                model, tokenizer = load_model_and_tokenizer(model_name, dtype=dtype, device=args.device)

                calib_samples = get_c4_calibration(tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)
                blocks = find_blocks(model)
                calib_inputs = capture_block0_inputs(model, blocks, calib_samples, args.device)
                prune_model_sequential(
                    model,
                    calib_inputs,
                    method=method,
                    sparsity=sparsity,
                    blocksize=args.blocksize,
                    device=args.device,
                )

                metrics = evaluate(model, tokenizer, args.device)
                print(json.dumps(metrics, indent=2))
                all_results.append(
                    {"model": model_name, "method": method, "sparsity": sparsity, **metrics}
                )
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {len(all_results)} results to {args.out}")


if __name__ == "__main__":
    main()
