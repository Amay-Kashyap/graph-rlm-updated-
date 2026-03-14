from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_pareto(metrics_path: Path, output_path: Path) -> Path:
    metrics = pd.read_csv(metrics_path)
    overall = metrics[metrics["category"] == "ALL"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in overall.itertuples(index=False):
        ax.scatter(row.avg_tokens_per_query, row.accuracy_pct, s=120, label=row.system)
        ax.annotate(row.system, (row.avg_tokens_per_query, row.accuracy_pct), textcoords="offset points", xytext=(6, 6))

    ax.set_xscale("log")
    ax.set_xlabel("Average tokens per query (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Graph-RLM Pareto Frontier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
