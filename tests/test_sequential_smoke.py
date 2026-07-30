"""End-to-end smoke test on a tiny randomly-initialized LlamaForCausalLM
(no downloads): exercises block discovery, input capture, Hessian
accumulation, and every prune method through the sequential driver."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cws.eval_ppl import eval_ppl
from cws.models.adapters import capture_block0_inputs, find_blocks, find_prunable_linears
from cws.sequential import METHODS, prune_model_sequential


def make_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    model.config.use_cache = False
    return model


def run():
    torch.manual_seed(0)
    device = "cpu"
    model = make_tiny_llama()

    blocks = find_blocks(model)
    assert len(blocks) == 3, f"expected 3 blocks, found {len(blocks)}"

    n_linears = len(find_prunable_linears(blocks[0]))
    assert n_linears == 7, f"expected 7 Linear layers per Llama block, found {n_linears}"

    calib_samples = [torch.randint(0, 256, (1, 16)) for _ in range(8)]
    calib_inputs = capture_block0_inputs(model, blocks, calib_samples, device)
    assert len(calib_inputs) == 8

    eval_chunks = [torch.randint(0, 256, (1, 16)) for _ in range(4)]
    dense_ppl = eval_ppl(model, eval_chunks, device=device)
    print(f"dense PPL (random weights, meaningless value): {dense_ppl:.3f}")

    for method in METHODS:
        model = make_tiny_llama()
        torch.manual_seed(0)
        blocks = find_blocks(model)
        calib_inputs = capture_block0_inputs(model, blocks, calib_samples, device)

        stats = prune_model_sequential(
            model,
            calib_inputs,
            method=method,
            sparsity=0.5,
            blocksize=8,
            device=device,
            verbose=False,
        )
        for block_idx, block_stats in stats.items():
            for name, achieved in block_stats.items():
                assert abs(achieved - 0.5) < 1e-6, (
                    f"{method} block {block_idx} layer {name}: "
                    f"expected ~0.5 sparsity, got {achieved}"
                )

        ppl = eval_ppl(model, eval_chunks, device=device)
        assert torch.isfinite(torch.tensor(ppl)), f"{method}: non-finite PPL {ppl}"
        print(f"{method}: achieved sparsity ~0.5 across all layers, PPL={ppl:.3f} OK")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    run()
