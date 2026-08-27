# -*- coding: utf-8 -*-
"""Validation figures (matching balance heatmaps)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def plot_matching_balance_heatmap(
    balance_df: pd.DataFrame,
    output_path: Path,
    *,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Write a covariate × method heatmap of mean absolute pair SMD."""
    log = logger or LOGGER
    if balance_df.empty or "method" not in balance_df.columns:
        log.info("Matching balance heatmap skipped: empty balance table.")
        return None

    pivot = balance_df.pivot_table(
        index="covariate",
        columns="method",
        values="mean_abs_smd",
        aggfunc="mean",
    )
    if pivot.empty:
        log.info("Matching balance heatmap skipped: nothing to pivot.")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height = max(4.0, 0.35 * len(pivot.index) + 1.5)
    width = max(6.0, 1.8 * len(pivot.columns) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean |SMD| across pairs"},
    )
    ax.set_title("Post-matching balance (mean absolute SMD per covariate)")
    ax.set_xlabel("Matching method")
    ax.set_ylabel("Covariate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Matching balance heatmap: %s", output_path)
    return output_path
