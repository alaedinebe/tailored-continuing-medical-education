# -*- coding: utf-8 -*-
"""Run automatic validation checks after a PRISM analysis."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.prism.validation.matching_quality import balance_table_by_covariate, compute_pair_smd_stats
from src.prism.validation.clinical_gate import write_clinical_review_gate
from src.prism.validation.plots import plot_matching_balance_heatmap
from src.prism.validation.report import CheckResult, announce_console, write_reports
from src.prism.validation.thresholds import resolve_thresholds

LOGGER = logging.getLogger(__name__)

LEAKAGE_FORBIDDEN = frozenset({
    "n_recos",
    "n_target_recos",
    "_predicted_prob",
    "_residual",
    "_obs_id",
    "member_pseudo_id",
    "person_id",
})


def _add(
    checks: list[CheckResult],
    check_id: str,
    status: str,
    *,
    value: Any = None,
    threshold: Any = None,
    message: str = "",
    recommendation: str = "",
) -> None:
    checks.append(
        CheckResult(
            id=check_id,
            status=status,
            value=value,
            threshold=threshold,
            message=message,
            recommendation=recommendation,
        )
    )


def _compare_upper(
    checks: list[CheckResult],
    check_id: str,
    value: float | None,
    warn_threshold: float,
    *,
    fail_threshold: float | None = None,
    message: str = "",
) -> None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        _add(checks, check_id, "INFO", value=value, message="No finite value to evaluate.")
        return
    if fail_threshold is not None and value > fail_threshold:
        _add(checks, check_id, "FAIL", value=round(value, 4), threshold=fail_threshold, message=message)
    elif value > warn_threshold:
        _add(checks, check_id, "WARN", value=round(value, 4), threshold=warn_threshold, message=message)
    else:
        _add(checks, check_id, "PASS", value=round(value, 4), threshold=warn_threshold)


def _compare_lower(
    checks: list[CheckResult],
    check_id: str,
    value: float | None,
    warn_threshold: float,
    *,
    fail_threshold: float | None = None,
    message: str = "",
) -> None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        _add(checks, check_id, "INFO", value=value, message="No finite value to evaluate.")
        return
    if fail_threshold is not None and value < fail_threshold:
        _add(checks, check_id, "FAIL", value=round(value, 4), threshold=fail_threshold, message=message)
    elif value < warn_threshold:
        _add(checks, check_id, "WARN", value=round(value, 4), threshold=warn_threshold, message=message)
    else:
        _add(checks, check_id, "PASS", value=round(value, 4), threshold=warn_threshold)


def _check_leakage(
    checks: list[CheckResult],
    matching_cols: list[str],
    outcome_col: str,
    physician_col: str,
) -> None:
    forbidden_present = [
        c
        for c in matching_cols
        if c in LEAKAGE_FORBIDDEN
        or c == outcome_col
        or c == physician_col
        or str(c).startswith("outcome_")
    ]
    if forbidden_present:
        _add(
            checks,
            "leakage.matching_columns_clean",
            "FAIL",
            value=forbidden_present,
            message="Forbidden columns found in matching covariates.",
        )
    else:
        _add(checks, "leakage.matching_columns_clean", "PASS", value=True)


def _check_method_consistency(
    checks: list[CheckResult],
    df: pd.DataFrame,
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [
        ("discordance_rate_euclidean", "discordance_rate_mahalanobis", "euclidean", "mahalanobis"),
        ("discordance_rate_euclidean", "discordance_rate_rf_matching", "euclidean", "rf"),
        ("discordance_rate_euclidean", "ensemble_matching", "euclidean", "ensemble"),
    ]
    for col_a, col_b, name_a, name_b in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        a = pd.to_numeric(df[col_a], errors="coerce")
        b = pd.to_numeric(df[col_b], errors="coerce")
        valid = a.notna() & b.notna()
        if valid.sum() < 3:
            continue
        spearman = float(a[valid].corr(b[valid], method="spearman"))
        delta = (a - b).abs()
        rows.append(
            {
                "method_a": name_a,
                "method_b": name_b,
                "spearman": spearman,
                "mean_abs_delta": float(delta[valid].mean()),
                "p90_abs_delta": float(delta[valid].quantile(0.9)),
                "max_abs_delta": float(delta[valid].max()),
            }
        )
        if name_a == "euclidean" and name_b == "mahalanobis":
            _compare_lower(
                checks,
                "methods.spearman_euclidean_mahalanobis",
                spearman,
                thresholds["min_spearman_euclidean_mahalanobis"],
                fail_threshold=thresholds["critical_min_spearman_euclidean_mahalanobis"],
            )
            _compare_upper(
                checks,
                "methods.delta_euclidean_mahalanobis_mean",
                float(delta[valid].mean()),
                thresholds["max_mean_delta_euclidean_mahalanobis"],
            )
            _compare_upper(
                checks,
                "methods.delta_euclidean_mahalanobis_p90",
                float(delta[valid].quantile(0.9)),
                thresholds["max_p90_delta_euclidean_mahalanobis"],
            )
            _compare_upper(
                checks,
                "methods.delta_euclidean_mahalanobis_max",
                float(delta[valid].max()),
                thresholds["max_physician_delta_euclidean_mahalanobis"],
            )
        if name_a == "euclidean" and name_b == "rf":
            _compare_lower(
                checks,
                "methods.spearman_euclidean_rf",
                spearman,
                thresholds["min_spearman_euclidean_rf"],
            )
            _compare_upper(
                checks,
                "methods.delta_euclidean_rf_mean",
                float(delta[valid].mean()),
                thresholds["max_mean_delta_euclidean_rf"],
            )
            _compare_upper(
                checks,
                "methods.delta_euclidean_rf_max",
                float(delta[valid].max()),
                thresholds["max_physician_delta_euclidean_rf"],
            )
    if "discordance_rate_rf_matching" in df.columns and "discordance_rate_euclidean" in df.columns:
        gap = (
            pd.to_numeric(df["discordance_rate_rf_matching"], errors="coerce")
            - pd.to_numeric(df["discordance_rate_euclidean"], errors="coerce")
        ).mean()
        _compare_upper(
            checks,
            "methods.rf_minus_euclidean_mean",
            float(gap) if pd.notna(gap) else None,
            thresholds["max_rf_minus_euclidean_mean"],
            message="RF discordance much higher than Euclidean (possible circularity).",
        )
    if "discordance_rate_learning" in df.columns and "discordance_rate_euclidean" in df.columns:
        gap = (
            pd.to_numeric(df["discordance_rate_learning"], errors="coerce")
            - pd.to_numeric(df["discordance_rate_euclidean"], errors="coerce")
        ).mean()
        _compare_upper(
            checks,
            "methods.learning_minus_euclidean_mean",
            float(gap) if pd.notna(gap) else None,
            thresholds["max_learning_minus_euclidean_mean"],
        )
    return pd.DataFrame(rows)


def _check_discordance_sanity(
    checks: list[CheckResult],
    df: pd.DataFrame,
    thresholds: dict[str, Any],
) -> None:
    if "prescription_rate" not in df.columns:
        return
    p = pd.to_numeric(df["prescription_rate"], errors="coerce")
    bern = 2 * p * (1 - p)
    disc_cols = [c for c in df.columns if c.startswith("discordance_rate_")]
    if "ensemble_matching" in df.columns:
        disc_cols.append("ensemble_matching")
    primary = "discordance_rate_euclidean" if "discordance_rate_euclidean" in df.columns else disc_cols[0]
    for col in disc_cols:
        d = pd.to_numeric(df[col], errors="coerce")
        if d.isna().all():
            _add(checks, f"discordance.{col}.present", "WARN", message=f"{col} is all NaN.")
            continue
        if ((d < 0) | (d > 1)).any():
            _add(checks, f"discordance.{col}_in_bounds", "FAIL", message=f"{col} outside [0, 1].")
        else:
            _add(checks, f"discordance.{col}_in_bounds", "PASS", value=True)
        ratio = d / bern
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        if col == primary:
            _compare_upper(
                checks,
                "discordance.bernoulli_ratio_p90",
                float(ratio.quantile(0.9)),
                thresholds["max_bernoulli_ratio_p90"],
            )
            _compare_upper(
                checks,
                "discordance.bernoulli_ratio_max",
                float(ratio.max()),
                thresholds["max_bernoulli_ratio_any_physician"],
                fail_threshold=thresholds["critical_max_bernoulli_ratio"],
            )
            below = float((ratio < 1.0).mean())
            _add(
                checks,
                "discordance.below_bernoulli_fraction",
                "INFO",
                value=round(below, 3),
                message="Fraction of physicians with discordance below Bernoulli baseline.",
            )
            _compare_upper(
                checks,
                "discordance.max_rate",
                float(d.max()),
                thresholds["max_discordance_rate"],
            )


def run_validation_report(
    *,
    results_dir: str | Path,
    config: dict[str, Any],
    results: dict[str, Any],
    df: pd.DataFrame | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Execute validation checks and write reports under ``results_dir/validation/``."""
    log = logger or LOGGER
    validation_cfg = config.get("validation") or {}
    if validation_cfg.get("enabled", True) is False:
        log.info("Validation disabled by config.")
        return {}

    results_dir = Path(results_dir)
    profile, thresholds = resolve_thresholds(config)
    checks: list[CheckResult] = []
    artifacts: dict[str, str] = {}
    out_dir = results_dir / "validation"

    ctx = results.get("validation_context") or {}
    basis = ctx.get("matching_basis") or {}
    run_id = results_dir.name

    # --- Cohort / basis ---
    n_raw = int(basis.get("n_raw_rows", len(df) if df is not None else 0))
    n_clean = int(basis.get("n_clean_rows", 0))
    n_physicians = int(basis.get("n_physicians_eligible", 0))
    if n_raw > 0 and n_clean > 0:
        dropna_frac = 1.0 - (n_clean / n_raw)
        _compare_upper(
            checks,
            "basis.dropna_fraction",
            dropna_frac,
            thresholds["max_dropna_fraction"],
            fail_threshold=thresholds["critical_max_dropna_fraction"],
        )
    _compare_lower(
        checks,
        "basis.physicians_eligible",
        float(n_physicians),
        float(thresholds["min_physicians_eligible"]),
        fail_threshold=float(thresholds["critical_min_physicians_eligible"]),
    )
    _add(checks, "info.basis.n_raw_rows", "INFO", value=n_raw)
    _add(checks, "info.basis.n_clean_rows", "INFO", value=n_clean)
    _add(checks, "info.basis.n_physicians_eligible", "INFO", value=n_physicians)

    # --- Manifest / leakage ---
    manifest_path = results_dir / "analysis_columns_used.csv"
    if manifest_path.is_file():
        _add(checks, "exports.analysis_columns_used", "PASS", value=str(manifest_path.name))
        manifest = pd.read_csv(manifest_path)
        matching_cols = manifest.loc[manifest["usage"] == "matching", "column_name"].astype(str).tolist()
        _compare_lower(
            checks,
            "covariates.matching_count",
            float(len(matching_cols)),
            float(thresholds["min_matching_covariates"]),
            fail_threshold=1.0,
        )
        _check_leakage(
            checks,
            matching_cols,
            str(config.get("outcome_col", "recommendation")),
            str(config.get("physician_col", "professional_id")),
        )
    else:
        _add(checks, "exports.analysis_columns_used", "WARN", message="analysis_columns_used.csv missing.")

    # --- Matching balance (SMD) ---
    df_match = ctx.get("df_match")
    pairs_by_method: dict[str, dict[str, Any]] = ctx.get("matching_pairs") or {}
    balance_frames: list[pd.DataFrame] = []
    pair_quality_rows: list[dict[str, object]] = []
    pair_quality_df: pd.DataFrame | None = None

    if isinstance(df_match, pd.DataFrame) and pairs_by_method:
        for method, payload in pairs_by_method.items():
            ref = np.asarray(payload.get("ref_indices", []), dtype=int)
            match = np.asarray(payload.get("match_indices", []), dtype=int)
            feature_cols = list(payload.get("feature_cols", []))
            stats = compute_pair_smd_stats(df_match, feature_cols, ref, match)
            pair_quality_rows.append({"method": method, **stats})
            balance_frames.append(balance_table_by_covariate(df_match, feature_cols, ref, match, method))

            if stats["n_pairs"] == 0:
                _add(checks, f"matching.{method}.pairs_formed", "FAIL", value=0)
                continue
            _add(checks, f"matching.{method}.pairs_formed", "PASS", value=stats["n_pairs"])
            _compare_upper(
                checks,
                f"matching.{method}.mean_abs_pair_smd",
                stats["mean_abs_smd"],
                thresholds["max_mean_pair_smd"],
                fail_threshold=thresholds["critical_max_mean_pair_smd"],
            )
            _compare_upper(
                checks,
                f"matching.{method}.worst_pair_max_smd",
                stats["worst_pair_max_smd"],
                thresholds["max_worst_pair_smd"],
                fail_threshold=thresholds["critical_max_worst_pair_smd"],
            )
            _compare_upper(
                checks,
                f"matching.{method}.frac_pairs_worst_smd_gt_2",
                stats["frac_pairs_worst_smd_gt_2"],
                thresholds["max_frac_pairs_worst_smd_gt_2"],
            )

        if pair_quality_rows:
            pair_quality_df = pd.DataFrame(pair_quality_rows)
            pq_path = out_dir / "pair_quality_by_method.csv"
            out_dir.mkdir(parents=True, exist_ok=True)
            pair_quality_df.to_csv(pq_path, index=False)
            artifacts["pair_quality"] = str(pq_path.relative_to(results_dir))
        if balance_frames:
            bal = pd.concat(balance_frames, ignore_index=True)
            bal_path = out_dir / "matching_balance_by_method.csv"
            bal.to_csv(bal_path, index=False)
            artifacts["matching_balance"] = str(bal_path.relative_to(results_dir))
            if validation_cfg.get("export_balance_heatmap", True):
                heatmap_path = out_dir / "matching_balance_heatmap.png"
                plotted = plot_matching_balance_heatmap(
                    bal,
                    heatmap_path,
                    logger=log,
                )
                if plotted is not None:
                    artifacts["matching_balance_heatmap"] = str(
                        heatmap_path.relative_to(results_dir)
                    )

    imputation = ctx.get("imputation") or {}
    if imputation:
        _compare_upper(
            checks,
            "covariates.imputed_cell_fraction",
            imputation.get("imputed_cell_fraction"),
            thresholds["max_imputed_cell_fraction"],
        )

    outliers = ctx.get("outliers") or {}
    if outliers:
        _compare_upper(
            checks,
            "covariates.outlier_cell_fraction",
            outliers.get("flagged_cell_fraction"),
            thresholds["max_outlier_cell_fraction"],
            fail_threshold=thresholds["critical_max_outlier_cell_fraction"],
            message="Fraction of covariate cells flagged as statistically implausible "
            "(median/MAD modified z-score); see outlier_detection_manifest for detail.",
        )
        _add(
            checks,
            "info.covariates.n_outlier_cells_flagged",
            "INFO",
            value=outliers.get("n_flagged_cells"),
        )
        by_method = outliers.get("by_method") or {}
        if by_method.get("hard_bound"):
            _add(
                checks,
                "info.covariates.n_hard_bound_corrections",
                "INFO",
                value=by_method["hard_bound"],
                message="Cells outside explicit clinical plausibility bounds "
                "(always corrected — see analysis.outlier_detection.hard_bounds).",
            )

    # --- Discordance + method consistency ---
    ip = results.get("intra_physician_variability")
    if isinstance(ip, pd.DataFrame) and not ip.empty:
        consistency = _check_method_consistency(checks, ip, thresholds)
        if not consistency.empty:
            cpath = out_dir / "method_consistency_matrix.csv"
            out_dir.mkdir(parents=True, exist_ok=True)
            consistency.to_csv(cpath, index=False)
            artifacts["method_consistency"] = str(cpath.relative_to(results_dir))
        _check_discordance_sanity(checks, ip, thresholds)
        if (results_dir / "intra_physician_variability.csv").is_file():
            _add(checks, "exports.intra_physician_variability", "PASS")
        else:
            _add(checks, "exports.intra_physician_variability", "WARN", message="CSV not found on disk.")

    # --- GLMM ---
    glmm = results.get("glmm") or {}
    if glmm and glmm.get("error") is None:
        _add(checks, "glmm.converged", "PASS", value=True)
        _add(checks, "info.glmm.n_observations", "INFO", value=glmm.get("n_observations"))
        if n_clean and glmm.get("n_observations"):
            overlap = min(int(glmm["n_observations"]), n_clean) / max(n_clean, 1)
            _add(checks, "info.glmm.cohort_overlap_matching", "INFO", value=round(overlap, 3))
    elif config.get("glmm", {}).get("enabled"):
        _add(checks, "glmm.converged", "WARN", message="GLMM enabled but no result stored.")

    # --- Synthetic ground truth ---
    gt_path = results_dir / "synthetic_ground_truth_physicians.csv"
    if gt_path.is_file() and isinstance(ip, pd.DataFrame):
        _add(checks, "synthetic.ground_truth_present", "INFO", value=True)
        try:
            gt = pd.read_csv(gt_path)
            if "physician_id" in gt.columns and "prescription_rate_true" in gt.columns:
                rate_col = "prescription_rate" if "prescription_rate" in ip.columns else None
                if rate_col:
                    merged = ip.merge(
                        gt,
                        left_on="physician",
                        right_on="physician_id",
                        how="inner",
                    )
                    if len(merged) >= 3 and "discordance_rate_euclidean" in merged.columns:
                        if "true_discordance" in merged.columns:
                            corr = merged["discordance_rate_euclidean"].corr(
                                merged["true_discordance"], method="spearman"
                            )
                            corr_val = round(float(corr), 3) if pd.notna(corr) else None
                            if corr_val is not None:
                                _compare_lower(
                                    checks,
                                    "synthetic.discordance_rank_correlation",
                                    corr_val,
                                    thresholds.get(
                                        "min_synthetic_discordance_rank_correlation", 0.30
                                    ),
                                    message="Low recovery of synthetic physician discordance ranking.",
                                )
                            else:
                                _add(
                                    checks,
                                    "synthetic.discordance_rank_correlation",
                                    "INFO",
                                    message="Could not compute rank correlation.",
                                )
        except Exception as exc:
            _add(checks, "synthetic.ground_truth_read", "WARN", message=str(exc))

    # --- Exports ---
    plot_path = results_dir / "plots" / "method_comparison" / "comparison_3panels.png"
    if plot_path.is_file():
        _add(checks, "exports.method_comparison_plot", "PASS")
    else:
        _add(checks, "exports.method_comparison_plot", "WARN", message="comparison_3panels.png missing.")

    clinical_dir = results_dir / "clinical_reviews"
    if clinical_dir.is_dir():
        n_pair_files = len(list(clinical_dir.glob("*_pairs.csv")))
        if n_pair_files > 0:
            _add(checks, "exports.clinical_pairs", "PASS", value=n_pair_files)
        else:
            _add(checks, "exports.clinical_pairs", "WARN", message="No clinical_reviews/*_pairs.csv files.")
    else:
        _add(checks, "exports.clinical_pairs", "WARN", message="clinical_reviews/ directory missing.")

    if (results_dir / "config_used.yaml").is_file():
        _add(checks, "exports.config_used", "PASS")
    else:
        _add(checks, "exports.config_used", "WARN")

    # --- Clinical review gate ---
    gate = write_clinical_review_gate(
        results_dir,
        pair_quality=pair_quality_df,
        thresholds=thresholds,
        primary_method=str(validation_cfg.get("clinical_review_primary_method", "euclidean")),
    )
    artifacts["clinical_review_gate"] = "clinical_reviews/validation_gate.json"
    gate_status = gate["status"]
    if gate_status == "fail":
        _add(
            checks,
            "clinical_review.smd_gate",
            "FAIL",
            value=gate.get("worst_pair_max_smd"),
            threshold=gate.get("fail_threshold"),
            message=gate.get("message", ""),
            recommendation="Read validation_gate.json before opening *_pair_reviewer.html.",
        )
    elif gate_status == "warn":
        _add(
            checks,
            "clinical_review.smd_gate",
            "WARN",
            value=gate.get("worst_pair_max_smd"),
            threshold=gate.get("warn_threshold"),
            message=gate.get("message", ""),
        )
    elif gate_status == "pass":
        _add(
            checks,
            "clinical_review.smd_gate",
            "PASS",
            value=gate.get("worst_pair_max_smd"),
            threshold=gate.get("warn_threshold"),
        )
    else:
        _add(checks, "clinical_review.smd_gate", "INFO", message=gate.get("message", ""))

    # --- Write + announce ---
    json_path = write_reports(
        out_dir,
        run_id=run_id,
        profile=profile,
        checks=checks,
        artifacts=artifacts,
        extra_meta={"thresholds_profile": profile, "validation_context": basis},
    )
    artifacts["validation_report"] = str(json_path.relative_to(results_dir))

    if validation_cfg.get("announce_console", True):
        announce_console(
            log,
            run_id=run_id,
            profile=profile,
            checks=checks,
            report_path=json_path,
        )

    global_status = next(
        (c.status for c in checks if c.status == "FAIL"),
        next((c.status for c in checks if c.status == "WARN"), "PASS"),
    )
    if validation_cfg.get("fail_on_critical") and any(c.status == "FAIL" for c in checks):
        raise RuntimeError(f"Validation failed with critical checks (see {json_path}).")

    return {
        "global_status": global_status,
        "profile": profile,
        "report_path": str(json_path),
        "checks": [c.__dict__ for c in checks],
    }
