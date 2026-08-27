# -*- coding: utf-8 -*-
"""
Auto-selection of analysis covariates and imputation helpers.

The external (RDS) dataset returns a wide table with many heterogeneous columns
(biomarkers, questionnaire codes, profile fields). For the GLMM / matching
pipeline we want a numerical, mostly-complete subset.

Procedures:
  1) :func:`select_low_missing_covariates` picks numeric columns with NaN ratio
     at most ``max_missing`` (default 20%).
  2) :func:`preprocess_wide_cohort_for_analysis` drops excluded / sparse columns
     and optionally zero-fills empty cells in the retained set.
  3) :func:`zero_impute` fills remaining NaNs with 0 (preprocess path when
     ``analysis.imputation.method`` is ``zero``).
  4) :func:`mice_impute` runs IterativeImputer on a column subset (used by
     ``Analysis`` after feature selection when method is ``mice``).
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (side-effect import)
from sklearn.impute import IterativeImputer

from src.prism.dataset_utils.outlier_detection import (
    DEFAULT_MIN_UNIQUE_VALUES,
    DEFAULT_MODIFIED_ZSCORE_THRESHOLD,
    REPORT_COLUMNS as OUTLIER_REPORT_COLUMNS,
    null_and_repair_outlier_cells,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_MISSING = 0.20
DEFAULT_QUESTIONNAIRE_PREFIX = "qa__"
DEFAULT_MIN_QUESTIONNAIRE_RESPONSE_RATE = 0.20

DEFAULT_RESERVED_COLUMNS: tuple[str, ...] = (
    "member_pseudo_id",
    "professional_id",
    "recommendation",
    "n_recos",
    "approximate_birth_date",
    "checkup_id",
)

COLUMNS_TO_DROP: tuple[str, ...] = (
    "pdt_111",
    "pdt_14",
    "pdt_13",
    "pgn_2010",
    "pgn_997",
    "pgn_999",
    "slp_10007",
)


def _coerce_boolean_to_int(series: pd.Series) -> pd.Series:
    """Convert booleans/Boolean nullable to {0, 1, NaN} float."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int8").astype("float")
    if isinstance(series.dtype, pd.BooleanDtype):
        return series.astype("Int8").astype("float")
    return series


def _try_to_numeric(series: pd.Series) -> pd.Series | None:
    """Try to coerce a column to numeric. Return None if too lossy.

    A column is considered numeric if at least 95% of its non-null values can
    be parsed numerically. That cutoff lets us recover columns like
    ``"systolic_blood_pressure"`` that are strings in the warehouse but really
    numeric.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    coerced = pd.to_numeric(series, errors="coerce")
    n_nonnull = series.notna().sum()
    if n_nonnull == 0:
        return None
    parseable_ratio = coerced.notna().sum() / n_nonnull
    if parseable_ratio >= 0.95:
        return coerced
    return None


def select_low_missing_covariates(
    df: pd.DataFrame,
    max_missing: float = DEFAULT_MAX_MISSING,
    reserved_columns: Iterable[str] = DEFAULT_RESERVED_COLUMNS,
    extra_exclude: Sequence[str] = (),
) -> tuple[list[str], pd.DataFrame]:
    """Pick numeric columns with <= ``max_missing`` NaN ratio.

    Args:
        df: Input wide patient-level DataFrame.
        max_missing: Maximum acceptable fraction of NaN per column (0..1).
        reserved_columns: Identifier / outcome columns excluded from the search.
        extra_exclude: Additional columns to skip (e.g. raw birth date).

    Returns:
        Tuple ``(covariates, df_with_numeric)`` where ``covariates`` is the list
        of selected column names and ``df_with_numeric`` is a copy of ``df`` in
        which the selected columns were coerced to ``float`` dtype.
    """
    excluded = set(reserved_columns) | set(extra_exclude)
    n_rows = len(df)
    if n_rows == 0:
        return [], df.copy()

    out = df.copy()
    selected: list[str] = []
    for col in out.columns:
        if col in excluded:
            continue
        series = _coerce_boolean_to_int(out[col])
        coerced = _try_to_numeric(series)
        if coerced is None:
            continue
        missing_ratio = coerced.isna().mean()
        if missing_ratio <= max_missing:
            out[col] = coerced.astype(float)
            selected.append(col)

    LOGGER.info(
        "Auto-selected %d covariates with NaN ratio <= %.0f%% (out of %d candidate columns).",
        len(selected),
        max_missing * 100,
        df.shape[1] - len(excluded),
    )
    if selected:
        miss = out[selected].isna().mean().sort_values(ascending=False)
        LOGGER.info("Top missing rates among selected: %s",
                    ", ".join(f"{c}={miss[c] * 100:.1f}%" for c in miss.head(5).index))
    return selected, out


def _questionnaire_answered_mask(series: pd.Series) -> pd.Series:
    """Boolean mask: non-null, non-blank answers."""
    s = series.astype("string")
    return s.notna() & (s.str.strip() != "")


def _questionnaire_response_rate(series: pd.Series) -> float:
    """Share of rows with a non-empty answer (0..1)."""
    answered = _questionnaire_answered_mask(series)
    if len(answered) == 0:
        return 0.0
    return float(answered.mean())


def questionnaire_response_rates(
    df: pd.DataFrame,
    *,
    column_prefix: str = DEFAULT_QUESTIONNAIRE_PREFIX,
) -> dict[str, float]:
    """Per-column response rate (0..1) for questionnaire columns, before imputation.

    Blank / NA answers count as unanswered. Non-questionnaire columns are ignored.
    """
    rates: dict[str, float] = {}
    for col in df.columns:
        if not str(col).startswith(column_prefix):
            continue
        rates[str(col)] = _questionnaire_response_rate(df[col])
    return rates


def questionnaire_response_detail(
    df: pd.DataFrame,
    *,
    column_prefix: str = DEFAULT_QUESTIONNAIRE_PREFIX,
) -> pd.DataFrame:
    """One row per questionnaire column: rate, answered count, cohort size."""
    n_members = int(len(df))
    rows: list[dict[str, object]] = []
    for col in df.columns:
        if not str(col).startswith(column_prefix):
            continue
        mask = _questionnaire_answered_mask(df[col])
        n_answered = int(mask.sum())
        rate = float(n_answered / n_members) if n_members else 0.0
        rows.append(
            {
                "column_name": str(col),
                "response_rate": rate,
                "n_answered": n_answered,
                "n_members": n_members,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["column_name", "response_rate", "n_answered", "n_members"],
    )


def encode_questionnaire_column(series: pd.Series) -> pd.Series:
    """Ordinal-encode answer codes; missing / blank answers stay NaN."""
    s = series.astype("string")
    valid = s.notna() & (s.str.strip() != "")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if not valid.any():
        return out
    codes, _ = pd.factorize(s[valid], sort=True)
    out.loc[valid] = codes.astype(float)
    return out


def select_questionnaire_covariates(
    df: pd.DataFrame,
    *,
    column_prefix: str = DEFAULT_QUESTIONNAIRE_PREFIX,
    min_response_rate: float = DEFAULT_MIN_QUESTIONNAIRE_RESPONSE_RATE,
    reserved_columns: Iterable[str] = DEFAULT_RESERVED_COLUMNS,
) -> tuple[list[str], pd.DataFrame]:
    """Keep questionnaire columns answered by at least ``min_response_rate`` of members.

    Each retained column of answer codes is label-encoded to float for MICE /
    matching. Columns that are already numeric are kept as-is: they carry an
    encoding built upstream (0/1 option indicators of a multiple-choice
    question).
    """
    excluded = set(reserved_columns)
    out = df.copy()
    selected: list[str] = []
    n_candidates = 0
    for col in out.columns:
        if col in excluded or not str(col).startswith(column_prefix):
            continue
        n_candidates += 1
        if _questionnaire_response_rate(out[col]) < min_response_rate:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            # Already encoded upstream (0/1 indicators of a multiple-choice
            # question). Factorizing again would go through ``astype("string")``
            # and could silently rewrite the levels — an indicator constant at
            # 1.0 would come back as 0.0 everywhere.
            out[col] = out[col].astype(float)
        else:
            out[col] = encode_questionnaire_column(out[col])
        selected.append(col)

    LOGGER.info(
        "Questionnaire: %d / %d questions kept (>= %.0f%% of members answered).",
        len(selected),
        n_candidates,
        min_response_rate * 100,
    )
    if selected:
        miss = out[selected].isna().mean().sort_values(ascending=False)
        LOGGER.info(
            "Top missing rates among kept questionnaire columns: %s",
            ", ".join(f"{c}={miss[c] * 100:.1f}%" for c in miss.head(5).index),
        )
    return selected, out


def preprocess_wide_cohort_for_analysis(
    df: pd.DataFrame,
    *,
    max_missing: float = DEFAULT_MAX_MISSING,
    reserved_columns: Iterable[str] = DEFAULT_RESERVED_COLUMNS,
    extra_exclude: Sequence[str] = (),
    candidate_columns: Sequence[str] | None = None,
    questionnaire_prefix: str = DEFAULT_QUESTIONNAIRE_PREFIX,
    min_questionnaire_response_rate: float = DEFAULT_MIN_QUESTIONNAIRE_RESPONSE_RATE,
    include_questionnaire: bool = True,
    impute: bool = True,
    max_iter: int = 10,
    random_state: int = 0,
    detect_outliers: bool = True,
    outlier_auto_repair: bool = False,
    outlier_threshold: float = DEFAULT_MODIFIED_ZSCORE_THRESHOLD,
    outlier_min_unique_values: int = DEFAULT_MIN_UNIQUE_VALUES,
    outlier_columns: Sequence[str] | None = None,
    member_id_col: str = "member_pseudo_id",
    physician_col: str | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Drop sparse columns, encode questionnaire, repair outliers, zero-impute.

    Note: explicit clinical plausibility bounds (:func:`prism.dataset_utils.outlier_detection.apply_hard_bounds`)
    are applied by the caller *before* this function runs (and before Table 1),
    not inside it — so the baseline descriptive table and every downstream step
    describe the same corrected cohort, instead of Table 1 reflecting the raw
    aberrant values while matching/GLMM use the corrected ones.

    Steps:
      1. Drop :data:`COLUMNS_TO_DROP` when present.
      2. Optionally restrict the search to ``candidate_columns``.
      3. Keep numeric/boolean columns with missing rate <= ``max_missing``.
      4. Keep ``{prefix}*`` questionnaire columns with response rate
         >= ``min_questionnaire_response_rate``, label-encoded.
      5. Drop every other non-reserved column from the output.
      6. Detect (and, when ``outlier_auto_repair`` is True, median-repair)
         statistically implausible cells in ``outlier_columns`` (defaults to
         every retained numeric covariate), before imputation runs (when
         ``detect_outliers`` is True). Note: many real biomarkers (CRP, GGT,
         cortisol, serology titers...) are legitimately right-skewed or
         bimodal — a plain median/MAD rule over-flags them. Callers should
         narrow ``outlier_columns`` to the covariates that actually matter
         (e.g. the matching feature set) rather than checking every numeric
         column in a wide EHR extract.
      7. Fill empty cells with 0 in all retained covariates (when ``impute`` is True).

    Args:
        df: Wide patient-level input (e.g. RDS extract).
        max_missing: Maximum NaN fraction per retained numeric covariate.
        reserved_columns: Identifier / outcome columns always kept when present.
        extra_exclude: Additional columns excluded from covariate selection.
        candidate_columns: If set, only these names are eligible (non-questionnaire).
        questionnaire_prefix: Column name prefix for pivoted ``fact_answer`` fields.
        min_questionnaire_response_rate: Minimum share of members with an answer.
        include_questionnaire: Whether to apply the questionnaire selection step.
        impute: Whether to fill empty cells with 0 in retained covariates.
        max_iter: Unused (kept for API compatibility).
        random_state: Unused (kept for API compatibility).
        detect_outliers: Whether to run cell-level outlier detection on the
            numeric covariates (report-only unless ``outlier_auto_repair``).
        outlier_auto_repair: Whether flagged cells are actually replaced with
            the column median. When False (default), detection only produces
            an audit report and ``df`` is left untouched — zero analytical
            impact.
        outlier_threshold: Modified z-score cutoff (absolute value) used by
            :func:`prism.dataset_utils.outlier_detection.detect_outlier_cells`.
        outlier_min_unique_values: Minimum distinct non-null values required
            for a column to be eligible for outlier detection.
        outlier_columns: Restrict detection to these columns (intersected with
            the retained numeric covariates). ``None`` (default) checks every
            retained numeric covariate — see the scope caveat above.
        member_id_col: Row identifier column used in the outlier audit report.
        physician_col: Optional physician identifier column used in the
            outlier audit report.

    Returns:
        ``(df_slim, covariate_names, outlier_report)`` where ``covariate_names``
        is the list of imputed feature columns passed to the analysis /
        matching pipeline, and ``outlier_report`` is one row per flagged cell
        (empty-but-columned when detection is disabled or nothing was flagged).
    """
    drop_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_present:
        df = df.drop(columns=drop_present)
        LOGGER.info(
            "Dropped %d excluded column(s): %s",
            len(drop_present),
            ", ".join(drop_present),
        )

    reserved_present = [c for c in reserved_columns if c in df.columns]
    q_extra = tuple(
        c for c in df.columns if str(c).startswith(questionnaire_prefix)
    )
    combined_extra = tuple(dict.fromkeys((*extra_exclude, *q_extra)))

    if candidate_columns is not None:
        candidates = [
            c for c in candidate_columns
            if c in df.columns
            and c not in reserved_present
            and not str(c).startswith(questionnaire_prefix)
        ]
        q_in_df = [c for c in df.columns if str(c).startswith(questionnaire_prefix)]
        work_cols = list(dict.fromkeys(reserved_present + candidates + q_in_df))
        df_work = df[work_cols].copy()
    else:
        df_work = df.copy()

    numeric_selected, df_numeric = select_low_missing_covariates(
        df=df_work,
        max_missing=max_missing,
        reserved_columns=reserved_columns,
        extra_exclude=combined_extra,
    )
    qa_selected: list[str] = []
    if include_questionnaire:
        qa_selected, df_numeric = select_questionnaire_covariates(
            df_numeric,
            column_prefix=questionnaire_prefix,
            min_response_rate=min_questionnaire_response_rate,
            reserved_columns=reserved_columns,
        )

    selected = list(dict.fromkeys(numeric_selected + qa_selected))
    if not selected:
        raise ValueError(
            "Wide preprocess produced 0 covariates "
            f"(numeric max_missing={max_missing}, "
            f"questionnaire min_response_rate={min_questionnaire_response_rate}); "
            "review the dataset or thresholds."
        )

    keep_cols = list(dict.fromkeys(reserved_present + selected))
    out = df_numeric[keep_cols].copy()
    n_dropped = df.shape[1] - out.shape[1]
    LOGGER.info(
        "Wide preprocess: %d patients x %d columns kept "
        "(%d numeric + %d questionnaire covariates, %d reserved); "
        "%d columns dropped.",
        out.shape[0],
        out.shape[1],
        len(numeric_selected),
        len(qa_selected),
        len(keep_cols) - len(selected),
        n_dropped,
    )

    outlier_report = pd.DataFrame(columns=list(OUTLIER_REPORT_COLUMNS))
    detect_columns = (
        [c for c in outlier_columns if c in numeric_selected]
        if outlier_columns is not None
        else numeric_selected
    )
    if detect_outliers and detect_columns:
        out, outlier_report = null_and_repair_outlier_cells(
            out,
            detect_columns,
            threshold=outlier_threshold,
            min_unique_values=outlier_min_unique_values,
            member_id_col=member_id_col,
            physician_col=physician_col,
            repair="median" if outlier_auto_repair else "none",
        )
        if not outlier_report.empty:
            LOGGER.info(
                "Outlier detection: %d cell(s) flagged across %d column(s) "
                "(threshold=%.2f, auto_repair=%s).",
                len(outlier_report),
                outlier_report["column"].nunique(),
                outlier_threshold,
                outlier_auto_repair,
            )
        else:
            LOGGER.info("Outlier detection: 0 cells flagged, no manifest written.")

    if impute:
        out = zero_impute(out, columns=selected)
    return out, selected, outlier_report


def zero_impute(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Fill empty / NaN cells with 0 in the selected numeric columns."""
    if not columns:
        return df.copy()

    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    if not cols:
        return out

    sub = out[cols].astype(float)
    n_missing_before = int(sub.isna().sum().sum())
    if n_missing_before == 0:
        LOGGER.info("Zero imputation: no empty cells in selected covariates, skipping.")
        return out

    LOGGER.info(
        "Zero imputation: filling %d empty cells across %d columns x %d rows.",
        n_missing_before,
        len(cols),
        sub.shape[0],
    )
    out.loc[:, cols] = sub.fillna(0.0)
    return out


DEFAULT_MICE_N_NEAREST_FEATURES = 20


def _is_discrete_mice_column(name: str) -> bool:
    """Questionnaire codes and clinical binaries that must stay integer levels."""
    s = str(name)
    return s.startswith(DEFAULT_QUESTIONNAIRE_PREFIX) or s.startswith("is_")


def _round_discrete_imputed(
    series: pd.Series,
    lo: float,
    hi: float,
) -> pd.Series:
    """Round to nearest integer then clip to observed discrete bounds."""
    return series.astype(float).round().clip(lower=float(lo), upper=float(hi))


def round_discrete_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Project ``qa__*`` / ``is_*`` columns onto integer levels after imputation.

    Bounds are taken from values already near integers when available (typical
    for factorized questionnaire codes and 0/1 flags); otherwise floor/ceil of
    the column range. Continuous columns are left unchanged.
    """
    out = df.copy()
    rounded_cols: list[str] = []
    for col in columns:
        if col not in out.columns or not _is_discrete_mice_column(col):
            continue
        s = out[col].astype(float)
        valid = s.dropna()
        if valid.empty:
            continue
        near_int = valid[np.isclose(valid.to_numpy(), np.round(valid.to_numpy()), atol=1e-6)]
        if near_int.empty:
            lo = float(np.floor(valid.min()))
            hi = float(np.ceil(valid.max()))
        else:
            lo = float(near_int.min())
            hi = float(near_int.max())
        out[col] = _round_discrete_imputed(s, lo, hi)
        rounded_cols.append(col)
    if rounded_cols:
        LOGGER.info(
            "Discrete post-MICE round: %d column(s) projected to integer levels (%s).",
            len(rounded_cols),
            ", ".join(rounded_cols[:8]) + ("..." if len(rounded_cols) > 8 else ""),
        )
    return out


def mice_impute(
    df: pd.DataFrame,
    columns: Sequence[str],
    max_iter: int = 10,
    random_state: int = 0,
    sample_posterior: bool = False,
    n_nearest_features: int | None = DEFAULT_MICE_N_NEAREST_FEATURES,
) -> pd.DataFrame:
    """Run MICE (IterativeImputer) on a subset of columns, in place on a copy.

    Args:
        df: Input DataFrame.
        columns: Numeric columns to impute. Non-numeric columns are passed
            through unchanged.
        max_iter: ``IterativeImputer.max_iter``.
        random_state: RNG seed.
        sample_posterior: Forwarded to ``IterativeImputer``. ``False`` is
            deterministic.
        n_nearest_features: Cap on predictors per target column. ``None`` uses
            all other columns. When set, clipped to ``min(n, p - 1)``.

    Returns:
        Copy of ``df`` with NaNs in ``columns`` replaced by MICE estimates.
        Imputed values are clipped to each column's observed (non-NaN) min/max.
        ``qa__*`` and ``is_*`` columns are then rounded to integer levels.
    """
    if not columns:
        return df.copy()

    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df.copy()

    sub = df[cols].astype(float)
    n_missing_before = int(sub.isna().sum().sum())
    if n_missing_before == 0:
        LOGGER.info("MICE: no NaN in selected covariates, skipping.")
        return df.copy()

    observed_min = sub.min(skipna=True)
    observed_max = sub.max(skipna=True)

    p = len(cols)
    nearest: int | None
    if n_nearest_features is None:
        nearest = None
    else:
        nearest = max(1, min(int(n_nearest_features), max(p - 1, 1)))

    LOGGER.info(
        "MICE: imputing %d NaNs across %d columns x %d rows "
        "(max_iter=%d, n_nearest_features=%s).",
        n_missing_before,
        p,
        sub.shape[0],
        max_iter,
        nearest,
    )
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=sample_posterior,
        n_nearest_features=nearest,
    )
    imputed = np.asarray(imputer.fit_transform(sub.to_numpy()), dtype=float)
    imputed_df = pd.DataFrame(imputed, index=sub.index, columns=cols)
    for col in cols:
        lo = observed_min[col]
        hi = observed_max[col]
        if pd.notna(lo) and pd.notna(hi):
            imputed_df[col] = imputed_df[col].clip(lower=float(lo), upper=float(hi))
            if _is_discrete_mice_column(col):
                imputed_df[col] = _round_discrete_imputed(
                    imputed_df[col], float(lo), float(hi)
                )

    out = df.copy()
    out.loc[:, cols] = imputed_df
    LOGGER.info(
        "MICE: done. Remaining NaN in imputed columns: %d",
        int(out[cols].isna().sum().sum()),
    )
    return out
