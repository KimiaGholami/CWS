"""Zero-shot downstream benchmark wrapper around EleutherAI's lm-evaluation-harness,
matching the CWS paper's task suite: ARC-Easy, ARC-Challenge, HellaSwag, PIQA,
WinoGrande, and (for the recovery-training tables) LAMBADA."""

DEFAULT_TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
)

FULL_TASKS = DEFAULT_TASKS + ("lambada_openai",)


def run_lm_eval(model, tokenizer, tasks=DEFAULT_TASKS, batch_size=8, device="cpu"):
    """Run zero-shot lm-eval-harness tasks against an already-pruned model
    in memory (no re-serialization needed)."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    results = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=0)
    accs = {}
    for task in tasks:
        task_results = results["results"].get(task, {})
        acc = task_results.get("acc,none", task_results.get("acc_norm,none"))
        accs[task] = acc
    accs["avg"] = sum(a for a in accs.values() if a is not None) / len(
        [a for a in accs.values() if a is not None]
    )
    return accs, results
