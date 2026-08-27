# -*- coding: utf-8 -*-
"""Plotting helpers for Prism analysis visualisations."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PHYSICIAN_COLOR_PALETTE: str = "husl"


def _physician_marker_shape(_physician_id: object) -> str:
    """Marker shape for figure points; always a circle for real-world P* labels."""
    return "o"


def _natural_physician_hue_order(values) -> list:
    """Stable sort: P1, P2, ..., P9, P10, ... then other strings."""
    uniq = list(dict.fromkeys(values))

    def _key(pid: object) -> tuple:
        s = str(pid).strip().upper()
        if len(s) >= 2 and s[0] == "P" and s[1:].isdigit():
            return (0, int(s[1:]))
        return (1, s)

    return sorted(uniq, key=_key)


def physician_color_map(
    values,
    *,
    palette: str = PHYSICIAN_COLOR_PALETTE,
) -> dict[str, tuple]:
    """Map each physician id to a color, stable for a fixed roster.

    Ids are coerced to ``str`` and ordered by :func:`_natural_physician_hue_order`
    (``P1``, ``P2``, … ``P10``, then other strings). ``husl`` then assigns one
    equally spaced hue per id. A physician keeps the same color on every figure
    that uses this map, even if that figure omits some colleagues.
    """
    hue_order = _natural_physician_hue_order(str(v) for v in values)
    n_hue = len(hue_order)
    if n_hue == 0:
        return {}
    colors = sns.color_palette(palette, n_colors=n_hue)
    return {h: colors[i] for i, h in enumerate(hue_order)}


def _normalize_physician_color_map(color_map: Mapping) -> dict[str, object]:
    return {str(k): v for k, v in color_map.items()}


def _resolve_physician_color_map(
    values,
    *,
    color_map: Optional[Mapping] = None,
    hue_order: Optional[list] = None,
    palette: str = PHYSICIAN_COLOR_PALETTE,
) -> dict[str, object]:
    """Prefer an explicit map; otherwise colour the given roster (or ``hue_order``)."""
    if color_map:
        return _normalize_physician_color_map(color_map)
    source = hue_order if hue_order is not None else values
    return physician_color_map(source, palette=palette)


def physician_face_colors(
    physicians,
    color_map: Mapping,
    *,
    palette: str = PHYSICIAN_COLOR_PALETTE,
) -> list:
    """Per-row face colors from an identity map (unknown ids get a fallback hue)."""
    resolved = _normalize_physician_color_map(color_map)
    fallback = sns.color_palette(palette, n_colors=1)[0]
    return [resolved.get(str(phy), fallback) for phy in physicians]


def _color_for_physician(
    physician_id: object,
    color_map: Mapping,
    *,
    palette: str = PHYSICIAN_COLOR_PALETTE,
):
    resolved = _normalize_physician_color_map(color_map)
    if str(physician_id) in resolved:
        return resolved[str(physician_id)]
    return sns.color_palette(palette, n_colors=1)[0]


def _truncate_physician_id_display(physician_id: object, max_chars: int = 7) -> str:
    """Libellé court pour les annot. graphiques (lisibilité) ; les données restent inchangées."""
    return (str(physician_id).strip())[:max_chars]


def format_physician_display_label(physician_id: object, max_len: int = 7) -> str:
    """Short display label for physician ids on crowded multi-physician figures."""
    return _truncate_physician_id_display(physician_id, max_chars=max_len)


def _scatter_hue_physician_continuous(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    *,
    hue_order: Optional[list] = None,
    color_map: Optional[Mapping] = None,
    palette: str = PHYSICIAN_COLOR_PALETTE,
    size: float = 8.0,
    linewidth: float = 1.0,
) -> None:
    """Continuous-axis scatter, hue-colored by physician (circle markers).

    Drops rows with NaN in any of the three columns. Marker shape is always a
    circle (see :func:`_physician_marker_shape`). When ``color_map`` is given
    it is used as-is (stable identity colors); otherwise hues are assigned
    from ``hue_order`` or a natural sort of the physicians in ``data``.
    """
    if data.empty or hue_col not in data.columns:
        return
    plot_df = data.dropna(subset=[x_col, y_col, hue_col]).copy()
    if plot_df.empty:
        return
    resolved = _resolve_physician_color_map(
        plot_df[hue_col].tolist(),
        color_map=color_map,
        hue_order=hue_order,
        palette=palette,
    )
    point_s = float(size) ** 2 * 1.15
    for _, row in plot_df.iterrows():
        phy = row[hue_col]
        col = _color_for_physician(phy, resolved, palette=palette)
        mk = _physician_marker_shape(phy)
        ax.scatter(
            float(row[x_col]),
            float(row[y_col]),
            s=point_s,
            marker=mk,
            facecolors=col,
            edgecolors="black",
            linewidths=linewidth,
            zorder=5,
        )


def _scatter_hue_physician_categorical(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    *,
    order: list[str],
    hue_order: Optional[list] = None,
    color_map: Optional[Mapping] = None,
    palette: str = PHYSICIAN_COLOR_PALETTE,
    size: float = 8.0,
    linewidth: float = 1.0,
    jitter: float = 0.0,
    rng_seed: int = 42,
) -> None:
    """
    Categorical-axis scatter, hue-colored by physician, circle markers.
    Default jitter=0 keeps points on the category center (vertical alignment); set jitter>0 for horizontal spread.
    When ``color_map`` is given it is used as-is; otherwise hues follow
    ``hue_order`` or a natural sort of the physicians in ``data``.
    """
    if data.empty or hue_col not in data.columns:
        return
    plot_df = data.dropna(subset=[x_col, y_col, hue_col]).copy()
    if plot_df.empty:
        return
    resolved = _resolve_physician_color_map(
        plot_df[hue_col].tolist(),
        color_map=color_map,
        hue_order=hue_order,
        palette=palette,
    )
    x_index = {cat: i for i, cat in enumerate(order)}
    rng = np.random.default_rng(rng_seed) if jitter > 0 else None
    point_s = float(size) ** 2 * 1.15
    for _, row in plot_df.iterrows():
        cat = row[x_col]
        if cat in x_index:
            xi = x_index[cat]
        else:
            xi = None
            for i, c in enumerate(order):
                if pd.isna(cat) and pd.isna(c):
                    xi = i
                    break
                if cat == c:
                    xi = i
                    break
                try:
                    if pd.notna(cat) and pd.notna(c) and float(cat) == float(c):
                        xi = i
                        break
                except (TypeError, ValueError):
                    continue
        if xi is None:
            continue
        phy = row[hue_col]
        col = _color_for_physician(phy, resolved, palette=palette)
        if jitter <= 0 or rng is None:
            x0 = float(xi)
        else:
            x0 = float(xi) + float(rng.uniform(-jitter, jitter))
        mk = _physician_marker_shape(phy)
        ax.scatter(
            x0,
            row[y_col],
            s=point_s,
            marker=mk,
            facecolors=col,
            edgecolors="black",
            linewidths=linewidth,
            zorder=5,
        )


USAGE_BOTH = "matching + GLMM"
USAGE_MATCHING = "matching seul"
USAGE_GLMM = "GLMM seul"

_USAGE_COLORS = {
    USAGE_BOTH: "#4C72B0",
    USAGE_MATCHING: "#55A868",
    USAGE_GLMM: "#C44E52",
}


def questionnaire_display_label(column_name: str) -> str:
    """Human-readable label for a ``qa__*`` column."""
    if column_name.startswith("qa__"):
        return column_name[4:]
    return column_name


def plot_questionnaire_completion(
    df: pd.DataFrame,
    out_path,
    *,
    title: str = (
        "Taux de réponse avant imputation — questions utilisées (matching / GLMM)"
    ),
) -> None:
    """Horizontal bar chart of pre-impute response rates for used questionnaire columns.

    Expected columns: ``display_label``, ``response_rate_pct``, ``usage``.
    Sorted ascending by rate. Y-axis 0–100 %.
    """
    out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        return

    plot_df = df.copy()
    plot_df = plot_df.sort_values("response_rate_pct", ascending=True)
    n = len(plot_df)
    fig_h = max(3.0, 0.45 * n + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = [
        _USAGE_COLORS.get(str(u), "#999999") for u in plot_df["usage"].tolist()
    ]
    ax.barh(
        plot_df["display_label"].astype(str),
        plot_df["response_rate_pct"].astype(float),
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Taux de réponse (%)")
    ax.set_title(title)
    ax.axvline(20, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_USAGE_COLORS[k], label=k)
        for k in (USAGE_BOTH, USAGE_MATCHING, USAGE_GLMM)
        if k in set(plot_df["usage"].astype(str))
    ]
    handles.append(
        plt.Line2D([0], [0], color="#888888", linestyle="--", linewidth=0.8, label="seuil 20 %")
    )
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
