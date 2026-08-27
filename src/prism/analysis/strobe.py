# -*- coding: utf-8 -*-
"""STROBE-style participant selection flow diagram for the Prism analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import seaborn as sns


@dataclass
class StrobeStage:
    """One vertical box in the STROBE flow (remaining population after a step).

    Attributes:
        label: Display text for the main (kept) box.
        n: Number of patients remaining after this stage.
        excluded_label: Optional reason shown in the lateral exclusion box
            leading into this stage. ``None`` means no exclusion side-box
            (typically the first / source stage).
        n_excluded: Number of patients dropped between the previous stage and
            this one. Side-box is omitted when ``None`` or ``0``.
    """

    label: str
    n: int
    excluded_label: str | None = None
    n_excluded: int | None = None


def stages_to_dataframe(stages: Sequence[StrobeStage]) -> pd.DataFrame:
    """Tabular export of STROBE stage counts (one row per stage)."""
    rows = [
        {
            "label": stage.label,
            "n": int(stage.n),
            "n_excluded": (
                int(stage.n_excluded) if stage.n_excluded is not None else None
            ),
            "excluded_label": stage.excluded_label,
        }
        for stage in stages
    ]
    return pd.DataFrame(rows, columns=["label", "n", "n_excluded", "excluded_label"])


def _format_n(n: int) -> str:
    """Thousands separator compatible with default matplotlib fonts."""
    return f"{n:,}".replace(",", " ")


def _box_text(label: str, n: int) -> str:
    return f"{label}\n(n = {_format_n(n)})"


def _exclusion_text(label: str, n: int) -> str:
    return f"Exclus :\n{label}\n(n = {_format_n(n)})"


def render_strobe_diagram(
    stages: Sequence[StrobeStage],
    out_path: Path | str,
    title: str | None = None,
    *,
    show_zero_exclusions: bool = False,
) -> Path:
    """Draw a STROBE-style participant selection flow diagram and save it as PNG.

    Main population boxes are stacked top→bottom and linked by vertical arrows.
    For each stage with ``n_excluded > 0`` (or ``show_zero_exclusions=True``),
    a lateral exclusion box is drawn to the right, linked by a horizontal arrow.

    Args:
        stages: Ordered list of flow stages (first = source cohort).
        out_path: Destination PNG path (parent directories are created).
        title: Optional figure title.
        show_zero_exclusions: When ``True``, draw lateral boxes even if
            ``n_excluded == 0``. Default ``False`` omits empty boxes.

    Returns:
        Resolved path of the written PNG.
    """
    if not stages:
        raise ValueError("render_strobe_diagram requires at least one StrobeStage.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")

    n_stages = len(stages)
    fig_height = max(4.0, 2.2 * n_stages + (1.0 if title else 0.0))
    fig_width = 10.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, float(n_stages) + 0.6)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, pad=12)

    # Layout constants (figure coordinates in the 0–10 / 0–n_stages space).
    main_x = 2.6
    main_w = 3.6
    main_h = 0.70
    excl_x = 7.0
    excl_w = 2.6
    excl_h = 0.70
    box_face = "#E8F0FE"
    box_edge = "#4C72B0"
    excl_face = "#FCE8E6"
    excl_edge = "#C44E52"

    centers_y: list[float] = []
    for i, stage in enumerate(stages):
        # Top stage near y = n_stages − 0.2, then descending.
        cy = float(n_stages - i) - 0.15
        centers_y.append(cy)

        # Main (kept) box.
        main_box = FancyBboxPatch(
            (main_x - main_w / 2, cy - main_h / 2),
            main_w,
            main_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=box_edge,
            facecolor=box_face,
            mutation_aspect=0.5,
            zorder=2,
        )
        ax.add_patch(main_box)
        ax.text(
            main_x,
            cy,
            _box_text(stage.label, stage.n),
            ha="center",
            va="center",
            fontsize=9,
            zorder=3,
            wrap=True,
        )

        # Lateral exclusion box (if applicable).
        n_excl = stage.n_excluded
        show_excl = (
            stage.excluded_label is not None
            and n_excl is not None
            and (show_zero_exclusions or n_excl > 0)
        )
        if show_excl:
            excl_box = FancyBboxPatch(
                (excl_x - excl_w / 2, cy - excl_h / 2),
                excl_w,
                excl_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.2,
                edgecolor=excl_edge,
                facecolor=excl_face,
                mutation_aspect=0.5,
                zorder=2,
            )
            ax.add_patch(excl_box)
            ax.text(
                excl_x,
                cy,
                _exclusion_text(str(stage.excluded_label), int(n_excl)),
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )
            arrow = FancyArrowPatch(
                (main_x + main_w / 2, cy),
                (excl_x - excl_w / 2, cy),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color=excl_edge,
                zorder=1,
            )
            ax.add_patch(arrow)

    # Vertical arrows between consecutive main boxes.
    for i in range(len(centers_y) - 1):
        y_from = centers_y[i] - main_h / 2
        y_to = centers_y[i + 1] + main_h / 2
        arrow = FancyArrowPatch(
            (main_x, y_from),
            (main_x, y_to),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color=box_edge,
            zorder=1,
        )
        ax.add_patch(arrow)

    # Tiny legend for colour coding.
    legend_handles = [
        mpatches.Patch(facecolor=box_face, edgecolor=box_edge, label="Conservés"),
        mpatches.Patch(facecolor=excl_face, edgecolor=excl_edge, label="Exclus"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
