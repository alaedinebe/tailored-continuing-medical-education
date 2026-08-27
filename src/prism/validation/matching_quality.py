# -*- coding: utf-8 -*-
"""Post-matching balance metrics (pair-level SMD)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pair_smd_stats(
    df_match: pd.DataFrame,
    feature_cols: list[str],
    ref_indices: np.ndarray,
    match_indices: np.ndarray,
) -> dict[str, float | int]:
    """Compute absolute standardized mean differences on matched pairs."""
    empty = {
        "n_pairs": 0,
        "mean_abs_smd": float("nan"),
        "p90_abs_smd": float("nan"),
        "worst_pair_mean_smd": float("nan"),
        "worst_pair_p90_smd": float("nan"),
        "worst_pair_max_smd": float("nan"),
        "frac_pairs_worst_smd_gt_2": float("nan"),
    }
    if len(ref_indices) == 0 or len(match_indices) == 0 or not feature_cols:
        return empty

    sub = df_match[feature_cols].astype(float)
    std = sub.std(ddof=0).replace(0, np.nan)
    all_smds: list[float] = []
    pair_worst: list[float] = []

    for ref_i, match_i in zip(ref_indices, match_indices, strict=False):
        row_a = sub.iloc[int(ref_i)]
        row_b = sub.iloc[int(match_i)]
        diffs = (row_a - row_b).abs() / std
        diffs = diffs.replace([np.inf, -np.inf], np.nan).dropna()
        if diffs.empty:
            continue
        vals = diffs.to_numpy(dtype=float)
        all_smds.extend(vals.tolist())
        pair_worst.append(float(np.max(vals)))

    if not all_smds:
        return {**empty, "n_pairs": int(len(ref_indices))}

    worst_arr = np.asarray(pair_worst, dtype=float)
    all_arr = np.asarray(all_smds, dtype=float)
    return {
        "n_pairs": int(len(pair_worst)),
        "mean_abs_smd": float(np.mean(all_arr)),
        "p90_abs_smd": float(np.quantile(all_arr, 0.9)),
        "worst_pair_mean_smd": float(np.mean(worst_arr)),
        "worst_pair_p90_smd": float(np.quantile(worst_arr, 0.9)),
        "worst_pair_max_smd": float(np.max(worst_arr)),
        "frac_pairs_worst_smd_gt_2": float(np.mean(worst_arr > 2.0)),
    }


def balance_table_by_covariate(
    df_match: pd.DataFrame,
    feature_cols: list[str],
    ref_indices: np.ndarray,
    match_indices: np.ndarray,
    method: str,
) -> pd.DataFrame:
    """Per-covariate mean absolute SMD across pairs."""
    rows: list[dict[str, object]] = []
    if len(ref_indices) == 0 or not feature_cols:
        return pd.DataFrame(columns=["method", "covariate", "mean_abs_smd", "max_abs_smd"])

    sub = df_match[feature_cols].astype(float)
    std = sub.std(ddof=0).replace(0, np.nan)
    for col in feature_cols:
        diffs: list[float] = []
        for ref_i, match_i in zip(ref_indices, match_indices, strict=False):
            sd = std[col]
            if pd.isna(sd) or sd < 1e-12:
                continue
            val = abs(float(sub.iloc[int(ref_i)][col]) - float(sub.iloc[int(match_i)][col])) / float(sd)
            if np.isfinite(val):
                diffs.append(val)
        if not diffs:
            continue
        arr = np.asarray(diffs, dtype=float)
        rows.append(
            {
                "method": method,
                "covariate": col,
                "mean_abs_smd": float(np.mean(arr)),
                "max_abs_smd": float(np.max(arr)),
            }
        )
    return pd.DataFrame(rows)
