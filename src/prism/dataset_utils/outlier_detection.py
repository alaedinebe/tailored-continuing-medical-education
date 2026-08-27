# -*- coding: utf-8 -*-
"""
Cell-level outlier detection for wide patient-level cohorts.

Real-world extracts occasionally contain isolated data-entry errors (e.g. a
weight of 747 instead of 74.7) that silently corrupt matching / SMD metrics,
because those computations use a non-robust (population) standard deviation:
one extreme value inflates both the numerator (pairwise distance) and the
denominator (pooled std) at once.

This module flags such cells with a robust univariate rule — the median +
MAD (median absolute deviation) "modified z-score" of Iglewicz & Hoaglin
(1993) — and optionally repairs them in place with the column median.

Scope / known limitations:
  * Univariate only: a value that is plausible per-column but jointly
    implausible across columns is not caught.
  * Columns with a degenerate MAD (near-constant / binary indicators) are
    exempted — an isolated bad value there stays invisible to this detector.
  * Does not address the separate zero-imputation issue (missing biomarkers
    filled with 0.0 downstream by :func:`prism.dataset_utils.imputation.zero_impute`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_MODIFIED_ZSCORE_THRESHOLD = 3.5
DEFAULT_MIN_UNIQUE_VALUES = 3
MAD_SCALE_CONSTANT = 0.6745  # Iglewicz & Hoaglin (1993)

REPORT_COLUMNS: tuple[str, ...] = (
    "member_id",
    "physician_id",
    "column",
    "raw_value",
    "median",
    "mad",
    "modified_zscore",
    "method",
)


def compute_modified_zscores(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Median/MAD "modified z-score", NaN-safe.

    Returns ``(zscores, median, mad)``. ``zscores`` is all-NaN (same index as
    ``series``) when the column has too few valid values or ``mad == 0``
    (degenerate / near-constant column) — callers should treat that as "no
    outliers detectable here" rather than crash.
    """
    values = pd.to_numeric(series, errors="coerce")
    median = float(values.median(skipna=True)) if values.notna().any() else float("nan")
    mad = float((values - median).abs().median(skipna=True)) if values.notna().any() else float("nan")
    if not np.isfinite(median) or not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64"), median, mad
    zscores = MAD_SCALE_CONSTANT * (values - median) / mad
    return zscores, median, mad


def detect_outlier_cells(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD,
    min_unique_values: int = DEFAULT_MIN_UNIQUE_VALUES,
) -> pd.DataFrame:
    """Boolean mask (same index, ``columns``) marking implausible cells.

    A column is skipped (never flagged) when it has fewer than
    ``min_unique_values`` distinct non-null values or a degenerate MAD (both
    are proxies for "near-constant / binary indicator column", where a
    modified z-score is not a meaningful statistic).
    """
    mask = pd.DataFrame(False, index=df.index, columns=list(columns))
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if series.dropna().nunique() < min_unique_values:
            continue
        zscores, _median, mad = compute_modified_zscores(series)
        if mad == 0 or not np.isfinite(mad):
            continue
        mask[col] = zscores.abs() > threshold
        mask[col] = mask[col].fillna(False)
    return mask


def null_and_repair_outlier_cells(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD,
    min_unique_values: int = DEFAULT_MIN_UNIQUE_VALUES,
    member_id_col: str = "member_pseudo_id",
    physician_col: str | None = None,
    repair: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect implausible cells and optionally repair them with the column median.

    Args:
        df: Input DataFrame.
        columns: Candidate numeric covariate columns to check (callers should
            pass only genuinely continuous covariates — never id/outcome/
            reserved columns, never label-encoded questionnaire columns).
        threshold: Modified z-score cutoff (absolute value).
        min_unique_values: Minimum distinct non-null values required for a
            column to be eligible for detection.
        member_id_col: Column used as the row identifier in the audit report,
            when present in ``df``.
        physician_col: Optional column used as the physician identifier in
            the audit report, so a reviewer can spot if one physician
            concentrates an outsized share of flags for a given column.
        repair: ``"median"`` nulls the flagged cell and immediately refills
            it with the column median (so a downstream generic imputer sees
            no remaining NaN there); ``"none"`` leaves ``df`` completely
            untouched — only the audit report is produced (report-only /
            detect-only mode, zero analytical impact).

    Returns:
        ``(df_out, report_df)``. ``report_df`` has one row per flagged cell
        (columns: ``member_id``, ``physician_id``, ``column``, ``raw_value``,
        ``median``, ``mad``, ``modified_zscore``); it is empty-but-columned
        when nothing was flagged.
    """
    if repair not in {"median", "none"}:
        raise ValueError(f"Unsupported repair mode: {repair!r} (expected 'median' or 'none').")

    out = df.copy()
    mask = detect_outlier_cells(
        out, columns, threshold=threshold, min_unique_values=min_unique_values
    )

    rows: list[dict[str, object]] = []
    for col in columns:
        if col not in mask.columns:
            continue
        flagged_idx = out.index[mask[col]]
        if len(flagged_idx) == 0:
            continue
        _zscores, median, mad = compute_modified_zscores(out[col])
        for idx in flagged_idx:
            raw_value = out.at[idx, col]
            rows.append(
                {
                    "member_id": out.at[idx, member_id_col] if member_id_col in out.columns else None,
                    "physician_id": out.at[idx, physician_col] if physician_col and physician_col in out.columns else None,
                    "column": col,
                    "raw_value": float(raw_value) if pd.notna(raw_value) else None,
                    "median": median,
                    "mad": mad,
                    "modified_zscore": MAD_SCALE_CONSTANT * (float(raw_value) - median) / mad
                    if pd.notna(raw_value)
                    else None,
                    "method": "mad_zscore",
                }
            )
        if repair == "median":
            out.loc[flagged_idx, col] = median
        # repair == "none": df is left untouched, report-only.

    report_df = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    return out, report_df


def apply_hard_bounds(
    df: pd.DataFrame,
    bounds: dict[str, dict[str, float]],
    *,
    member_id_col: str = "member_pseudo_id",
    physician_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Null and median-repair values outside explicit clinical plausibility bounds.

    Unlike :func:`null_and_repair_outlier_cells` (a statistical heuristic prone
    to false positives on legitimately skewed biomarkers), these bounds encode
    unambiguous domain knowledge (e.g. a diastolic blood pressure below 30 mmHg
    is incompatible with life) — so this is always applied when configured,
    independent of any ``auto_repair`` toggle.

    Args:
        df: Input DataFrame.
        bounds: ``{column: {"min": ..., "max": ...}}``; either bound may be
            omitted. Columns absent from ``df`` are skipped silently.
        member_id_col: Row identifier column used in the audit report.
        physician_col: Optional physician identifier column used in the
            audit report.

    Returns:
        ``(df_out, report_df)`` — same schema as
        :func:`null_and_repair_outlier_cells`, with ``method="hard_bound"``
        and ``mad``/``modified_zscore`` left ``None`` (not applicable).
    """
    out = df.copy()
    rows: list[dict[str, object]] = []
    for col, bound in bounds.items():
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        lo = bound.get("min")
        hi = bound.get("max")
        mask = pd.Series(False, index=out.index)
        if lo is not None:
            mask = mask | (series < lo)
        if hi is not None:
            mask = mask | (series > hi)
        mask = mask.fillna(False)
        flagged_idx = out.index[mask]
        if len(flagged_idx) == 0:
            continue
        valid = series[~mask].dropna()
        median = float(valid.median()) if not valid.empty else float("nan")
        for idx in flagged_idx:
            raw_value = out.at[idx, col]
            rows.append(
                {
                    "member_id": out.at[idx, member_id_col] if member_id_col in out.columns else None,
                    "physician_id": out.at[idx, physician_col] if physician_col and physician_col in out.columns else None,
                    "column": col,
                    "raw_value": float(raw_value) if pd.notna(raw_value) else None,
                    "median": median,
                    "mad": None,
                    "modified_zscore": None,
                    "method": "hard_bound",
                }
            )
        out.loc[flagged_idx, col] = median

    report_df = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    return out, report_df
