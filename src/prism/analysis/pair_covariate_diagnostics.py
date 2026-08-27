# -*- coding: utf-8 -*-
"""Per-method pair diagnostics on age, LDL stage, and smoking discordance."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

AGE_COL = "age"
LDL_COL = "biomarker_ldl_cholesterol_calculated_serum"
SMOKER_COL = "is_smoker"

LDL_STAGE_BINS = [-np.inf, 40.0, 55.0, 70.0, 110.0, np.inf]
LDL_STAGE_LABELS = ("<40", "40-55", "55-70", "70-110", ">110")

PAIR_CSV_METHODS = frozenset(
    {"euclidean", "mahalanobis", "rf_matching", "learning", "mutual_info"}
)

OUTPUT_FILENAME = "pair_covariate_diagnostics_by_method.csv"


def summarize_abs_diff(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Absolute pairwise differences: mean, min, p25, p75, max (NaN if empty)."""
    empty = {
        "mean": float("nan"),
        "min": float("nan"),
        "p25": float("nan"),
        "p75": float("nan"),
        "max": float("nan"),
    }
    diffs = (pd.to_numeric(a, errors="coerce") - pd.to_numeric(b, errors="coerce")).abs()
    diffs = diffs.replace([np.inf, -np.inf], np.nan).dropna()
    if diffs.empty:
        return empty
    arr = diffs.to_numpy(dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def ldl_stage(series: pd.Series) -> pd.Series:
    """Bin LDL (mg/dL) into clinical stages; right=False so 40 ∈ 40-55."""
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        values,
        bins=LDL_STAGE_BINS,
        labels=LDL_STAGE_LABELS,
        right=False,
        include_lowest=True,
    )


def _pair_cols(df: pd.DataFrame, stem: str) -> tuple[str, str] | None:
    a, b = f"{stem}_A", f"{stem}_B"
    if a in df.columns and b in df.columns:
        return a, b
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    """Coerce smoker-like columns to nullable boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    text = series.astype("string").str.strip().str.casefold()
    mapped = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
        }
    )
    return mapped.astype("boolean")


def diagnostics_from_pairs_frame(df: pd.DataFrame, method: str) -> dict[str, Any]:
    """Compute age / LDL / smoker diagnostics for one method's pairs table."""
    row: dict[str, Any] = {
        "method": method,
        "n_pairs": int(len(df)),
    }

    age_cols = _pair_cols(df, AGE_COL)
    if age_cols is None:
        for key in ("mean", "min", "p25", "p75", "max"):
            row[f"age_abs_diff_{key}"] = float("nan")
    else:
        age_stats = summarize_abs_diff(df[age_cols[0]], df[age_cols[1]])
        for key, val in age_stats.items():
            row[f"age_abs_diff_{key}"] = val

    ldl_cols = _pair_cols(df, LDL_COL)
    if ldl_cols is None:
        for key in ("mean", "min", "p25", "p75", "max"):
            row[f"ldl_abs_diff_{key}"] = float("nan")
        row["ldl_same_stage_n"] = 0
        row["ldl_same_stage_frac"] = float("nan")
        for label in LDL_STAGE_LABELS:
            row[f"ldl_same_stage_n_{label}"] = 0
    else:
        ldl_stats = summarize_abs_diff(df[ldl_cols[0]], df[ldl_cols[1]])
        for key, val in ldl_stats.items():
            row[f"ldl_abs_diff_{key}"] = val
        st_a = ldl_stage(df[ldl_cols[0]])
        st_b = ldl_stage(df[ldl_cols[1]])
        both = st_a.notna() & st_b.notna()
        same = both & (st_a.astype("string") == st_b.astype("string"))
        n_both = int(both.sum())
        n_same = int(same.sum())
        row["ldl_same_stage_n"] = n_same
        row["ldl_same_stage_frac"] = float(n_same / n_both) if n_both else float("nan")
        same_labels = st_a[same].astype("string")
        for label in LDL_STAGE_LABELS:
            row[f"ldl_same_stage_n_{label}"] = int((same_labels == label).sum())

    smoker_cols = _pair_cols(df, SMOKER_COL)
    if smoker_cols is None:
        row["n_smoker_discordant"] = 0
    else:
        a = _bool_series(df[smoker_cols[0]])
        b = _bool_series(df[smoker_cols[1]])
        valid = a.notna() & b.notna()
        row["n_smoker_discordant"] = int((valid & (a != b)).sum())

    return row


def _method_from_pairs_path(path: Path) -> str | None:
    stem = path.stem
    if not stem.endswith("_pairs"):
        return None
    method = stem[: -len("_pairs")]
    if method not in PAIR_CSV_METHODS:
        return None
    return method


def write_pair_covariate_diagnostics(
    results_dir: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Scan clinical_reviews/*_pairs.csv and write per-method diagnostics CSV."""
    log = logger or LOGGER
    results_dir = Path(results_dir)
    clinical_dir = results_dir / "clinical_reviews"
    if not clinical_dir.is_dir():
        log.info("Pair covariate diagnostics skipped: no clinical_reviews/ directory.")
        return None

    rows: list[dict[str, Any]] = []
    for path in sorted(clinical_dir.glob("*_pairs.csv")):
        method = _method_from_pairs_path(path)
        if method is None:
            continue
        try:
            pairs = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            log.warning("Could not read pairs CSV %s: %s", path, exc)
            continue
        row = diagnostics_from_pairs_frame(pairs, method)
        rows.append(row)
        log.info(
            "Pair diagnostics [%s]: n_pairs=%d age_abs_diff_mean=%s "
            "ldl_same_stage_n=%s n_smoker_discordant=%s",
            method,
            row["n_pairs"],
            row.get("age_abs_diff_mean"),
            row.get("ldl_same_stage_n"),
            row.get("n_smoker_discordant"),
        )

    if not rows:
        log.info("Pair covariate diagnostics skipped: no matching *_pairs.csv files.")
        return None

    out = pd.DataFrame(rows)
    out_path = results_dir / OUTPUT_FILENAME
    out.to_csv(out_path, index=False)
    log.info("Saved pair covariate diagnostics to %s (%d methods)", out_path, len(out))
    return out_path
