#!/usr/bin/env python
"""Sparsity sweep (30%-80%) across models/methods, reporting WikiText-2 PPL.

Mirrors the CWS paper's Tables 3/6/10 sweep. Each (model, method, sparsity)
combination reloads the dense checkpoint fresh, since pruning + OBS
correction mutates weights in place.

Example:
    python scripts/run_sparsity_sweep.py --models tinyllama-1.1b hgrn-1.3b \\
        --methods cws sparsegpt wanda --device cuda --out results/sweep.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cws.data import get_c4_calibration, get_wikitext2_test
from cws.eval_ppl import eval_ppl
from cws.models import load_model_and_tokenizer
from cws.models.adapters import capture_block0_inputs, find_blocks
from cws.sequential import prune_model_sequential

DEFAULT_SPARSITIES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=["cws"])
    p.add_argument(
        "--sparsities", nargs="+", type=float, default=list(DEFAULT_SPARSITIES)
    )
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--out", default="results/sweep.csv")
    p.add_argument(
        "--plot", action="store_true", help="also render PPL-vs-sparsity plots per model"
    )
    p.add_argument("--plot-dir", default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    rows = []

    for model_name in args.models:
        eval_tokenizer = None
        for method in args.methods:
            for sparsity in args.sparsities:
                print(f"=== {model_name} | {method} | sparsity={sparsity} ===")
                model, tokenizer = load_model_and_tokenizer(
                    model_name, dtype=dtype, device=args.device
                )
                eval_tokenizer = tokenizer

                calib_samples = get_c4_calibration(
                    tokenizer, nsamples=args.nsamples, seqlen=args.seqlen
                )
                blocks = find_blocks(model)
                calib_inputs = capture_block0_inputs(
                    model, blocks, calib_samples, args.device
                )
                prune_model_sequential(
                    model,
                    calib_inputs,
                    method=method,
                    sparsity=sparsity,
                    blocksize=args.blocksize,
                    device=args.device,
                )

                eval_chunks = get_wikitext2_test(tokenizer, seqlen=2048)
                ppl = eval_ppl(model, eval_chunks, device=args.device)
                print(f"PPL: {ppl:.3f}")
                rows.append(
                    {
                        "model": model_name,
                        "method": method,
                        "sparsity": sparsity,
                        "wikitext2_ppl": ppl,
                    }
                )
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()

        # dense baseline, once per model
        model, tokenizer = load_model_and_tokenizer(
            model_name, dtype=dtype, device=args.device
        )
        eval_chunks = get_wikitext2_test(tokenizer, seqlen=2048)
        ppl = eval_ppl(model, eval_chunks, device=args.device)
        rows.append(
            {"model": model_name, "method": "dense", "sparsity": 0.0, "wikitext2_ppl": ppl}
        )
        del model

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "method", "sparsity", "wikitext2_ppl"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")

    if args.plot:
        from cws.plotting import plot_sweep_csv

        out_paths = plot_sweep_csv(args.out, args.plot_dir)
        print(f"Wrote {len(out_paths)} plot(s) to {args.plot_dir}")


if __name__ == "__main__":
    main()
