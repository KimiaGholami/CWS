"""PPL-vs-sparsity plots, matching the style of CWS.pdf's Figures 1-3:
log-scale perplexity on the y-axis, sparsity (30-80%) on the x-axis, one
line per method, and the dense baseline as a horizontal dashed reference."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHOD_STYLE = {
    "cws": dict(color="#1f77b4", marker="o", linewidth=2.2, zorder=5),
    "sparsegpt": dict(color="#d62728", marker="s"),
    "wanda": dict(color="#2ca02c", marker="^"),
    "ria": dict(color="#9467bd", marker="v"),
    "awp": dict(color="#8c564b", marker="D"),
    "magnitude": dict(color="#7f7f7f", marker="x"),
}
METHOD_LABEL = {
    "cws": "CWS",
    "sparsegpt": "SparseGPT",
    "wanda": "Wanda",
    "ria": "RIA",
    "awp": "AWP",
    "magnitude": "Magnitude",
}


def plot_ppl_vs_sparsity(
    df: pd.DataFrame,
    model_name: str,
    out_path: str,
    dense_ppl: float | None = None,
    title: str | None = None,
) -> None:
    """`df` must have columns [method, sparsity, wikitext2_ppl] for one model.
    `sparsity` is a fraction (0.3-0.8); `dense_ppl` (if given, or if a
    method == "dense" row is present) is drawn as a dashed horizontal line."""
    df = df.copy()
    if dense_ppl is None:
        dense_rows = df[df["method"] == "dense"]
        if len(dense_rows) > 0:
            dense_ppl = dense_rows["wikitext2_ppl"].iloc[0]
    df = df[df["method"] != "dense"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for method, group in df.groupby("method"):
        group = group.sort_values("sparsity")
        style = METHOD_STYLE.get(method, {})
        ax.plot(
            group["sparsity"] * 100,
            group["wikitext2_ppl"],
            label=METHOD_LABEL.get(method, method),
            **style,
        )

    if dense_ppl is not None:
        ax.axhline(dense_ppl, color="black", linestyle="--", linewidth=1, label="Dense")

    ax.set_yscale("log")
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Perplexity on WikiText-2")
    ax.set_title(title or f"Sparsity-vs-perplexity: {model_name}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sweep_csv(csv_path: str, out_dir: str) -> list[str]:
    """Render one PPL-vs-sparsity figure per model found in a sweep CSV
    (as written by `scripts/run_sparsity_sweep.py`)."""
    df = pd.read_csv(csv_path)
    out_paths = []
    for model_name, group in df.groupby("model"):
        out_path = str(Path(out_dir) / f"{model_name}_sparsity_sweep.png")
        plot_ppl_vs_sparsity(group, model_name, out_path)
        out_paths.append(out_path)
    return out_paths
