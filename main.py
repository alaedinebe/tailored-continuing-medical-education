# -*- coding: utf-8 -*-
"""
Orchestrator for the Prism SCORE2 pipeline.

Runtime flow:
1) load config
2) resolve experiment metadata and output folders
3) generate one or many synthetic cohorts (single SCORE2 mode or multi-rule batch mode)
4) run analysis and export outputs for each experiment
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

# Add src to path BEFORE local imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.prism.analysis.analysis_pipeline import Analysis
from src.prism.analysis.analysis_pipeline import METHOD_DISPLAY_NAMES
from src.prism.analysis.baseline_table import (
    DEFAULT_OUTCOME_COL,
    DEFAULT_PHYSICIAN_COL,
    generate_baseline_table_one,
)
from src.prism.analysis.strobe import (
    StrobeStage,
    render_strobe_diagram,
    stages_to_dataframe,
)
from src.prism.snapshot_utils import SnapshotSaver, make_snapshot_saver
from src.prism.dataset_utils.imputation import (
    DEFAULT_MAX_MISSING,
    DEFAULT_QUESTIONNAIRE_PREFIX,
    DEFAULT_RESERVED_COLUMNS,
    preprocess_wide_cohort_for_analysis,
    questionnaire_response_detail,
    questionnaire_response_rates,
)
from src.prism.dataset_utils.outlier_detection import (
    DEFAULT_MIN_UNIQUE_VALUES,
    DEFAULT_MODIFIED_ZSCORE_THRESHOLD,
    REPORT_COLUMNS as OUTLIER_REPORT_COLUMNS,
    apply_hard_bounds,
)
from src.prism.dataset_utils.synthetic_generator import (
    PROGRESSIVE_RULE_DEFAULT_N_PASSES,
    REALISTIC_HETEROGENEOUS_STRATEGY,
    REALISTIC_MATCHING_FEATURE_COLUMNS,
    REALISTIC_PRESCRIPTION_RATE_BOUNDS,
    REALISTIC_RULE_NAME,
    REALISTIC_RULE_THREE_GROUPS_OUTCOME_COL,
    REALISTIC_RULE_THREE_GROUPS_STRATEGY,
    SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS,
    SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS,
    SCORE2_ONLY_STRATEGY,
    build_progressive_rule_plan,
    expected_progressive_rule_count,
    generate_progressive_rule_experiment,
    generate_realistic_heterogeneous_patients,
    generate_realistic_rule_three_groups_patients,
    generate_score2_five_groups_heter_patients,
)
from src.prism.experiment_paths import (
    DEFAULT_EXPERIMENTS_DATE_FORMAT,
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_SCRATCH_DIRNAME_PREFIX,
    make_experiment_dir,
)
from src.prism.logs_utils.python_logger import get_simple_logger


DEFAULT_CONFIG_PATH = "configs/article_stats.yaml"
DEFAULT_EXPERIMENT_NAME = "run"
DEFAULT_RANDOM_STATE = 42
DEFAULT_RANDOM_STATE_OFFSET = 100
DEFAULT_OUTCOME_COL = "recommendation"
SCORE2_OUTCOME_COL = "outcome_score2_five_groups_heter_patients"
AUTO_FIXED_EFFECTS_TOKEN = "auto"
STATIN_RELEVANT_FIXED_EFFECTS_TOKEN = "statin_relevant"
DEFAULT_FIXED_EFFECTS = [
    "age",
    "biomarker_hba1c_ngsp_blood",
    "non_hdl_cholesterol",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "estimated_glomerular_filtration_rate",
    "score_questionnaire_HAD",
    "is_smoker",
    "is_male",
    "has_car",
]

# Clinically curated covariates driving the statin prescription decision (ESC/EAS
# 2021 dyslipidaemia guidelines). Used when ``analysis.fixed_effects`` is set to
# ``"statin_relevant"``: a much smaller set than ``"auto"`` so the GLMM
# converges in seconds rather than minutes. Engineered binaries ``is_male`` and
# ``is_current_smoker`` are derived on the fly from ``gender`` and ``smoker``.
# Columns not present in the dataset are silently skipped.
#
# Selection rationale - we deliberately drop covariates that create severe
# multicollinearity (all empirical VIFs of this list verified < 5):
#   * total_cholesterol  -> redundant with LDL + HDL + 0.2*TG (Friedewald);
#   * SCORE2             -> deterministic combination of age, sex, smoker,
#                           total_chol, HDL, SBP -> already captured here;
#   * diastolic BP       -> r=0.91 with systolic BP.
# LDL is the canonical statin target (ESC/EAS); SBP is the BP component
# carrying most of the CV risk weight.
STATIN_RELEVANT_FIXED_EFFECTS: tuple[str, ...] = (
    "age",
    "is_male",
    "bmi",
    "is_current_smoker",
    "biomarker_ldl_cholesterol_calculated_serum",
    "biomarker_hdl_cholesterol_serum",
    "biomarker_triglyceride_serum",
    "biomarker_systolic_blood_pressure_sitting",
    "biomarker_hba1c_ngsp_blood",
    "biomarker_glomerular_filtration_rate_cdk_epi_serum",
    "biomarker_high_sensitivity_c_reactive_protein_serum",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prescription variability pipeline (synthetic cohorts)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config.",
    )
    return parser.parse_args()




def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def resolve_experiment_name(cfg: dict[str, Any]) -> str:
    """Resolve the experiment name from config (or prompt), sanitized for path usage."""
    project_cfg = cfg.get("project", {})
    ask_user = bool(project_cfg.get("ask_user_for_experiment_name", False))
    default_name = str(project_cfg.get("default_experiment_name", DEFAULT_EXPERIMENT_NAME))
    exp_name = (input("Experiment name (optional): ").strip() if ask_user else default_name) or default_name
    return exp_name.replace(" ", "_")


def infer_results_dir(cfg: dict[str, Any], experiment_name: str) -> str:
    """Create and return the run output directory."""
    paths_cfg = cfg["paths"]
    results_dir = make_experiment_dir(
        experiment_name,
        experiments_root=paths_cfg.get("experiments_root", DEFAULT_EXPERIMENTS_ROOT),
        date_format=paths_cfg.get("experiments_date_format", DEFAULT_EXPERIMENTS_DATE_FORMAT),
        dirname_prefix=paths_cfg.get("scratch_dirname_prefix", DEFAULT_SCRATCH_DIRNAME_PREFIX),
    )
    return str(results_dir)


def save_effective_config(cfg: dict[str, Any], results_dir: str) -> None:
    """Persist the effective config for reproducibility."""
    config_path = os.path.join(results_dir, "config_used.yaml")
    with open(config_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(cfg, file_obj, sort_keys=False, allow_unicode=True)


@dataclass
class GeneratedDataset:
    """Container for one generated cohort and associated metadata."""

    df: pd.DataFrame
    strategy_name: str
    generation_strategy: str
    expected_summary: str
    outcome_col: str
    experiment_index: int
    pass_index: int
    window_size: int
    window_start: int
    variables: list[str]
    rule_name: str
    ground_truth_df: pd.DataFrame | None = None


def _resolve_generation_params(cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Extrait les paramètres de génération synthétique depuis la configuration.

    Args:
        cfg: Configuration YAML chargée en mémoire.

    Returns:
        Tuple `(n_patients, n_physicians, effective_seed)` où `effective_seed`
        inclut l'offset éventuel défini dans `dataset.score2_five_groups`.
    """
    ds_cfg = cfg.get("dataset", {})
    score2_cfg = ds_cfg.get("score2_five_groups", {})
    n_patients = int(score2_cfg.get("n_patients", SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS))
    n_physicians = int(score2_cfg.get("n_physicians", SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS))
    random_state = int(cfg.get("project", {}).get("random_state", DEFAULT_RANDOM_STATE))
    state_offset = int(score2_cfg.get("random_state_offset", DEFAULT_RANDOM_STATE_OFFSET))
    return n_patients, n_physicians, random_state + state_offset






def build_generated_datasets(
    cfg: dict[str, Any],
    logger: Any,
    snapshots_root: str | None = None,
) -> list[GeneratedDataset]:
    """Generate one synthetic cohort or a multi-rule batch of cohorts."""
    ds_cfg = cfg.get("dataset", {})
    generation_strategy = ds_cfg.get("generation_strategy", SCORE2_ONLY_STRATEGY)

    n_patients, n_physicians, effective_seed = _resolve_generation_params(cfg)

    batch_cfg = ds_cfg.get("multi_rule_experiments", {})
    batch_enabled = bool(batch_cfg.get("enabled", False))
    if batch_enabled:
        analysis_cfg = cfg.get("analysis", {})
        raw_fe = analysis_cfg.get("fixed_effects", DEFAULT_FIXED_EFFECTS)
        if isinstance(raw_fe, str) and raw_fe.strip().lower() == AUTO_FIXED_EFFECTS_TOKEN:
            logger.warning(
                "fixed_effects='auto' is incompatible with multi_rule_experiments; "
                "falling back to DEFAULT_FIXED_EFFECTS for the rule plan."
            )
            raw_fe = DEFAULT_FIXED_EFFECTS
        fixed_effects = [str(c) for c in raw_fe]
        n_passes = int(batch_cfg.get("n_passes", PROGRESSIVE_RULE_DEFAULT_N_PASSES))
        rule_plan = build_progressive_rule_plan(fixed_effects=fixed_effects, n_passes=n_passes)
        n_experiments_expected = expected_progressive_rule_count(len(fixed_effects), n_passes=n_passes)
        if len(rule_plan) != n_experiments_expected:
            raise RuntimeError(
                f"Inconsistent progressive plan size: got {len(rule_plan)}, expected {n_experiments_expected}."
            )
        datasets: list[GeneratedDataset] = []
        for exp_idx, rule_definition in enumerate(rule_plan, start=1):
            df, metadata = generate_progressive_rule_experiment(
                rule_definition=rule_definition,
                experiment_index=exp_idx,
                fixed_effects=fixed_effects,
                n_patients=n_patients,
                expected_n_pros=n_physicians,
                random_state=effective_seed,
                logger=logger,
            )
            datasets.append(
                GeneratedDataset(
                    df=df,
                    strategy_name=str(metadata["strategy_name"]),
                    generation_strategy=str(metadata["generation_strategy"]),
                    expected_summary=str(metadata["expected_summary"]),
                    outcome_col=str(metadata["outcome_col"]),
                    experiment_index=exp_idx,
                    pass_index=int(metadata.get("pass_index", 0)),
                    window_size=int(metadata.get("window_size", 0)),
                    window_start=int(metadata.get("window_start", 0)),
                    variables=[str(v) for v in metadata.get("variables", [])],
                    rule_name=str(metadata.get("rule_name", metadata["strategy_name"])),
                )
            )
        logger.info("✅ Generated %d progressive-rule synthetic cohorts.", len(datasets))
        return datasets

    if generation_strategy == REALISTIC_HETEROGENEOUS_STRATEGY:
        shared_prescription_rate = bool(ds_cfg.get("shared_prescription_rate", False))
        df, ground_truth_df = generate_realistic_heterogeneous_patients(
            n_patients=n_patients,
            expected_n_pros=n_physicians,
            random_state=effective_seed,
            logger=logger,
            shared_prescription_rate=shared_prescription_rate,
        )
        if snapshots_root:
            gt_path = os.path.join(snapshots_root, "synthetic_ground_truth_physicians.csv")
            ground_truth_df.to_csv(gt_path, index=False)
            logger.info("Saved synthetic ground truth to %s", gt_path)
        logger.info(
            "✅ Synthetic cohort '%s' generated with shape %s.",
            REALISTIC_HETEROGENEOUS_STRATEGY,
            df.shape,
        )
        if shared_prescription_rate:
            shared_p = float(ground_truth_df["prescription_rate_true"].iloc[0])
            expected_summary = (
                f"Shared Bernoulli prescription rate p={shared_p:.4f} (drawn from "
                f"U{list(REALISTIC_PRESCRIPTION_RATE_BOUNDS)}) for all physicians "
                f"({n_patients} patients, {n_physicians} physicians)."
            )
        else:
            expected_summary = (
                f"Realistic heterogeneous physicians: independent Bernoulli prescription "
                f"with p ~ U{list(REALISTIC_PRESCRIPTION_RATE_BOUNDS)} "
                f"({n_patients} patients, {n_physicians} physicians)."
            )
        return [
            GeneratedDataset(
                df=df,
                strategy_name=REALISTIC_HETEROGENEOUS_STRATEGY,
                generation_strategy=REALISTIC_HETEROGENEOUS_STRATEGY,
                expected_summary=expected_summary,
                outcome_col=DEFAULT_OUTCOME_COL,
                experiment_index=1,
                pass_index=0,
                window_size=0,
                window_start=0,
                variables=list(REALISTIC_MATCHING_FEATURE_COLUMNS),
                rule_name=REALISTIC_HETEROGENEOUS_STRATEGY,
                ground_truth_df=ground_truth_df,
            )
        ]

    if generation_strategy == REALISTIC_RULE_THREE_GROUPS_STRATEGY:
        df, ground_truth_df = generate_realistic_rule_three_groups_patients(
            n_patients=n_patients,
            expected_n_pros=n_physicians,
            random_state=effective_seed,
            logger=logger,
        )
        if snapshots_root:
            gt_path = os.path.join(snapshots_root, "synthetic_ground_truth_physicians.csv")
            ground_truth_df.to_csv(gt_path, index=False)
            logger.info("Saved synthetic ground truth to %s", gt_path)
        logger.info(
            "✅ Synthetic cohort '%s' generated with shape %s.",
            REALISTIC_RULE_THREE_GROUPS_STRATEGY,
            df.shape,
        )
        expected_summary = (
            "Statin indication rule (3 pathways + very-high-risk shortcut) on 6 matching "
            "covariates. Physicians: P1-P6 random Bernoulli (p~U[0.1,0.6]), P7-P13 "
            "80%/10% adherence, P14-P19 100%/0% adherence."
        )
        return [
            GeneratedDataset(
                df=df,
                strategy_name=REALISTIC_RULE_THREE_GROUPS_STRATEGY,
                generation_strategy=REALISTIC_RULE_THREE_GROUPS_STRATEGY,
                expected_summary=expected_summary,
                outcome_col=REALISTIC_RULE_THREE_GROUPS_OUTCOME_COL,
                experiment_index=1,
                pass_index=0,
                window_size=0,
                window_start=0,
                variables=list(REALISTIC_MATCHING_FEATURE_COLUMNS),
                rule_name=REALISTIC_RULE_NAME,
                ground_truth_df=ground_truth_df,
            )
        ]

    if generation_strategy != SCORE2_ONLY_STRATEGY:
        logger.warning(
            "Unsupported synthetic strategy '%s'; forcing '%s'.",
            generation_strategy,
            SCORE2_ONLY_STRATEGY,
        )

    df = generate_score2_five_groups_heter_patients(
        n_patients=n_patients,
        expected_n_pros=n_physicians,
        random_state=effective_seed,
        logger=logger,
    )
    logger.info("✅ Synthetic cohort '%s' generated with shape %s.", SCORE2_ONLY_STRATEGY, df.shape)

    raw_csv_path = cfg["paths"].get("dataset_csv", "dataset.csv")
    if bool(ds_cfg.get("save_raw_csv", False)):
        os.makedirs(os.path.dirname(raw_csv_path) or ".", exist_ok=True)
        df.to_csv(raw_csv_path, index=False)
        logger.warning("⚠️ Raw dataset saved to %s (disable save_raw_csv if not needed).", raw_csv_path)

    return [
        GeneratedDataset(
            df=df,
            strategy_name=SCORE2_ONLY_STRATEGY,
            generation_strategy=SCORE2_ONLY_STRATEGY,
            expected_summary="SCORE2-only cohort with five physician behavior groups.",
            outcome_col=SCORE2_OUTCOME_COL,
            experiment_index=1,
            pass_index=0,
            window_size=0,
            window_start=0,
            variables=[],
            rule_name="score2",
        )
    ]


def _add_statin_relevant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered binaries used by the ``statin_relevant`` covariate preset.

    Two columns are derived from the raw cohort and only created if their
    source column is present:

      * ``is_male``           = 1 if ``gender`` is ``"Male"``, 0 if female-coded,
                                NaN otherwise (so MICE / dropna can act on it).
      * ``is_current_smoker`` = 1 if ``smoker`` is ``"Current smoker"``, else 0
                                (NaN preserved for genuinely missing values).
    """
    out = df.copy()
    if "gender" in out.columns and "is_male" not in out.columns:
        gender = out["gender"].astype("string").str.strip().str.lower()
        is_male = pd.Series(np.nan, index=out.index, dtype="float64")
        is_male.loc[gender.eq("male")] = 1.0
        is_male.loc[gender.isin(["female"])] = 0.0
        out["is_male"] = is_male
    if "smoker" in out.columns and "is_current_smoker" not in out.columns:
        smoker = out["smoker"].astype("string").str.strip().str.lower()
        is_current = pd.Series(np.nan, index=out.index, dtype="float64")
        is_current.loc[smoker.eq("current smoker")] = 1.0
        is_current.loc[smoker.isin(["never smoker", "former smoker"])] = 0.0
        out["is_current_smoker"] = is_current
    return out


def _analysis_reserved_columns(cfg: dict[str, Any]) -> tuple[str, ...]:
    """Columns always kept through wide preprocess (ids, outcome, reco metadata)."""
    analysis_cfg = cfg.get("analysis", {})
    reserved = set(DEFAULT_RESERVED_COLUMNS)
    reserved.add(analysis_cfg.get("physician_col", "professional_id"))
    reserved.add(analysis_cfg.get("outcome_col", DEFAULT_OUTCOME_COL))
    reserved.add("n_target_recos")
    reserved.add("age")
    return tuple(sorted(reserved))


def _resolve_fixed_effects_and_impute(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    logger: Any,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Wide preprocess: drop columns with >20% NaN, repair outliers, optionally zero-fill.

    When ``analysis.imputation.method`` is ``mice``, empty cells are left as NaN
    here; MICE runs later in ``Analysis`` on matching ∪ GLMM columns only.
    When method is ``zero`` (default), retained covariates are zero-filled.

    All retained covariates are passed to the matching / discordance methods via
    ``fixed_effects_override``. Presets:

      * ``"auto"``            — every numeric column in the cohort (default).
      * ``"statin_relevant"`` — engineered binaries added first, then same filter.
      * explicit list         — only listed names are eligible for the 20% filter.
    """
    analysis_cfg = cfg.get("analysis", {})
    raw = analysis_cfg.get("fixed_effects", DEFAULT_FIXED_EFFECTS)
    imputation_cfg = analysis_cfg.get("imputation", {}) or {}
    outlier_cfg = analysis_cfg.get("outlier_detection", {}) or {}
    random_state = int(cfg.get("project", {}).get("random_state", DEFAULT_RANDOM_STATE))
    max_iter = int(imputation_cfg.get("max_iter", 10))
    imputation_enabled = bool(imputation_cfg.get("enabled", True))
    imputation_method = str(imputation_cfg.get("method", "zero")).strip().lower()
    if imputation_method not in {"zero", "mice"}:
        raise ValueError(
            f"Unsupported analysis.imputation.method '{imputation_method}'. "
            "Expected 'zero' or 'mice'."
        )
    # Zero-fill only in preprocess; MICE is deferred to Analysis after FS.
    do_impute = imputation_enabled and imputation_method == "zero"
    if imputation_enabled and imputation_method == "mice":
        logger.info(
            "Imputation method=mice: skipping zero-fill in wide preprocess; "
            "MICE will run on matching ∪ GLMM columns after feature selection."
        )
    detect_outliers = bool(outlier_cfg.get("enabled", True))
    outlier_auto_repair = bool(outlier_cfg.get("auto_repair", False))
    outlier_threshold = float(outlier_cfg.get("threshold", DEFAULT_MODIFIED_ZSCORE_THRESHOLD))
    outlier_min_unique_values = int(outlier_cfg.get("min_unique_values", DEFAULT_MIN_UNIQUE_VALUES))
    max_missing = float(analysis_cfg.get("auto_max_missing", DEFAULT_MAX_MISSING))
    reserved = _analysis_reserved_columns(cfg)

    # Scope outlier detection to the covariates that actually matter (matching /
    # GLMM), not every numeric column in a wide EHR extract: many real biomarkers
    # (CRP, GGT, cortisol, serology titers...) are legitimately right-skewed or
    # bimodal, and a plain median/MAD rule massively over-flags them if applied
    # broadly. Defaults to the manually curated matching feature list when
    # available; falls back to every retained numeric covariate otherwise (noisier,
    # logged explicitly so it isn't a silent surprise).
    feature_selection_cfg = analysis_cfg.get("feature_selection", {}) or {}
    outlier_columns_cfg = outlier_cfg.get("columns", "auto")
    outlier_columns: list[str] | None
    if isinstance(outlier_columns_cfg, list):
        outlier_columns = [str(c) for c in outlier_columns_cfg]
    elif (
        str(outlier_columns_cfg).strip().lower() == "auto"
        and str(feature_selection_cfg.get("method", "")).strip().lower() == "manual"
        and feature_selection_cfg.get("columns")
    ):
        outlier_columns = [str(c) for c in feature_selection_cfg["columns"]]
    else:
        outlier_columns = None
        if detect_outliers:
            logger.info(
                "Outlier detection: no manual feature_selection.columns to scope to; "
                "checking every retained numeric covariate (noisier audit report)."
            )

    candidate_columns: list[str] | None = None
    df_in = df
    # Always derive is_male / is_current_smoker when sources exist: needed for
    # manual feature_selection (and harmless for statin_relevant / auto).
    df_in = _add_statin_relevant_features(df_in)
    if isinstance(raw, str) and raw.strip().lower() == STATIN_RELEVANT_FIXED_EFFECTS_TOKEN:
        candidate_columns = list(STATIN_RELEVANT_FIXED_EFFECTS)
    elif isinstance(raw, str) and raw.strip().lower() == AUTO_FIXED_EFFECTS_TOKEN:
        candidate_columns = None
    elif isinstance(raw, list):
        candidate_columns = [str(c) for c in raw]
    else:
        candidate_columns = [str(raw)]

    q_cfg = analysis_cfg.get("questionnaire", {}) or {}
    questionnaire_prefix = str(q_cfg.get("column_prefix", "qa__"))
    min_q_response = float(q_cfg.get("min_response_rate", 0.20))
    include_questionnaire = bool(q_cfg.get("enabled", True))

    df_out, covariates, outlier_report = preprocess_wide_cohort_for_analysis(
        df_in,
        max_missing=max_missing,
        reserved_columns=reserved,
        extra_exclude=(SCORE2_OUTCOME_COL,),
        candidate_columns=candidate_columns,
        questionnaire_prefix=questionnaire_prefix,
        min_questionnaire_response_rate=min_q_response,
        include_questionnaire=include_questionnaire,
        impute=do_impute,
        max_iter=max_iter,
        random_state=random_state,
        detect_outliers=detect_outliers,
        outlier_auto_repair=outlier_auto_repair,
        outlier_threshold=outlier_threshold,
        outlier_min_unique_values=outlier_min_unique_values,
        outlier_columns=outlier_columns,
        physician_col=str(analysis_cfg.get("physician_col", "professional_id")),
    )
    # ``age`` is a reserved column (always kept regardless of missingness) and is
    # consumed by the matching pipeline as a covariate. Declare it explicitly in
    # the covariate list so ``config["fixed_effects"]`` reflects what matching
    # actually uses, instead of silently diverging.
    if "age" in df_out.columns and "age" not in covariates and pd.api.types.is_numeric_dtype(df_out["age"]):
        covariates = covariates + ["age"]
    n_qa = sum(1 for c in covariates if str(c).startswith(questionnaire_prefix))
    logger.info(
        "Analysis input: %d covariates after wide preprocess "
        "(%d numeric, %d questionnaire; numeric max_missing=%.0f%%, "
        "questionnaire min_response=%.0f%%), table shape %s.",
        len(covariates),
        len(covariates) - n_qa,
        n_qa,
        max_missing * 100,
        min_q_response * 100,
        df_out.shape,
    )
    mad_scoped_columns = (
        outlier_columns
        if outlier_columns is not None
        else [c for c in covariates if not str(c).startswith(questionnaire_prefix)]
    )
    outlier_summary = {
        "report": outlier_report,
        "columns_checked": mad_scoped_columns,
        "threshold": outlier_threshold,
        "auto_repair": outlier_auto_repair,
    }
    return df_out, covariates, outlier_summary


def build_analysis_config(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    strategy_name: str,
    generation_strategy: str,
    outcome_col_hint: str,
    expected_summary: str,
    fixed_effects_override: list[str] | None = None,
    *,
    questionnaire_response_rates: dict[str, float] | None = None,
    questionnaire_response_detail: list[dict[str, Any]] | None = None,
    questionnaire_prefix: str | None = None,
) -> dict[str, Any]:
    """Build analysis config for one generated dataset."""
    analysis_cfg = cfg.get("analysis", {})
    if outcome_col_hint in df.columns:
        outcome_col = outcome_col_hint
    elif SCORE2_OUTCOME_COL in df.columns:
        outcome_col = SCORE2_OUTCOME_COL
    else:
        outcome_col = analysis_cfg.get("outcome_col", DEFAULT_OUTCOME_COL)
    if fixed_effects_override is not None:
        fixed_effects = list(fixed_effects_override)
    else:
        fixed_effects = analysis_cfg.get("fixed_effects", DEFAULT_FIXED_EFFECTS)
    configured_targets = analysis_cfg.get("dataset_targets", {})
    dynamic_targets = dict(configured_targets) if isinstance(configured_targets, dict) else {}
    dynamic_targets[strategy_name] = {"expected_summary": expected_summary}
    q_cfg = analysis_cfg.get("questionnaire", {}) or {}
    runtime_analysis_cfg = {
        "outcome_col": outcome_col,
        "physician_col": analysis_cfg.get("physician_col", "professional_id"),
        "fixed_effects": fixed_effects,
        "min_patients_per_physician": analysis_cfg.get("min_patients_per_physician", 30),
        "gray_zone_bounds": tuple(analysis_cfg.get("gray_zone_bounds", [0.3, 0.7])),
        "generation_strategy": generation_strategy,
        "dataset_targets": dynamic_targets,
        "questionnaire_prefix": (
            questionnaire_prefix
            if questionnaire_prefix is not None
            else str(q_cfg.get("column_prefix", DEFAULT_QUESTIONNAIRE_PREFIX))
        ),
        "questionnaire_response_rates": dict(questionnaire_response_rates or {}),
        "questionnaire_response_detail": list(questionnaire_response_detail or []),
    }
    # Preserve optional RF hyperparameters from YAML so Analysis can parse/validate them.
    if "rf_n_estimators" in analysis_cfg:
        runtime_analysis_cfg["rf_n_estimators"] = analysis_cfg["rf_n_estimators"]
    # Sélection de features par variance (optionnelle) appliquée avant le matching.
    if "feature_selection" in analysis_cfg:
        runtime_analysis_cfg["feature_selection"] = analysis_cfg["feature_selection"]
    if "glmm" in analysis_cfg:
        runtime_analysis_cfg["glmm"] = analysis_cfg["glmm"]
    if "methods" in analysis_cfg:
        runtime_analysis_cfg["methods"] = analysis_cfg["methods"]
    if "matching" in analysis_cfg:
        runtime_analysis_cfg["matching"] = analysis_cfg["matching"]
    if "bernoulli_residual" in analysis_cfg:
        runtime_analysis_cfg["bernoulli_residual"] = analysis_cfg["bernoulli_residual"]
    if "validation" in analysis_cfg:
        runtime_analysis_cfg["validation"] = analysis_cfg["validation"]
    elif "validation" in cfg:
        runtime_analysis_cfg["validation"] = cfg["validation"]
    # Propagate imputation settings (incl. random_state) for deferred MICE in Analysis.
    imputation_runtime = dict(analysis_cfg.get("imputation", {}) or {})
    if "random_state" not in imputation_runtime:
        imputation_runtime["random_state"] = int(
            cfg.get("project", {}).get("random_state", DEFAULT_RANDOM_STATE)
        )
    runtime_analysis_cfg["imputation"] = imputation_runtime
    runtime_analysis_cfg["dataset_source"] = cfg.get("dataset", {}).get(
        "source", generation_strategy
    )
    return runtime_analysis_cfg


def _extract_mean_delta_vs_manual(
    results: dict[str, Any],
    strategy_name: str,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    """Calcule l'écart moyen de chaque méthode versus la méthode manuelle.

    Args:
        results: Dictionnaire de sortie de `Analysis.run_full_analysis()`.
        strategy_name: Nom de la stratégie analysée (pour traçabilité).
        logger: Logger optionnel pour les avertissements.

    Returns:
        Une liste de lignes prêtes pour agrégation inter-expériences avec les
        clés `strategy_name`, `method` et `mean_delta_vs_manual`.
    """
    if "intra_physician_variability" not in results:
        if logger is not None:
            logger.warning("Skipping delta extraction for '%s': no intra_physician_variability results.", strategy_name)
        return []
    df = results["intra_physician_variability"]
    if "discordance_rate_manual" not in df.columns:
        if logger is not None:
            logger.warning("Skipping delta extraction for '%s': missing discordance_rate_manual.", strategy_name)
        return []
    manual = pd.to_numeric(df["discordance_rate_manual"], errors="coerce")
    method_columns = [c for c in df.columns if c.startswith("discordance_rate_") and c != "discordance_rate_manual"]
    if "ensemble_matching" in df.columns:
        method_columns.append("ensemble_matching")
    rows: list[dict[str, Any]] = []
    for method_col in method_columns:
        cur = pd.to_numeric(df[method_col], errors="coerce")
        delta_mean = float((cur - manual).mean(skipna=True))
        rows.append(
            {
                "strategy_name": strategy_name,
                "method": method_col,
                "mean_delta_vs_manual": delta_mean,
            }
        )
    return rows


def _save_cross_experiment_outputs(rows: list[dict[str, Any]], results_dir: str, logger: Any) -> None:
    """Sauvegarde les tableaux et le graphique d'agrégation inter-expériences.

    Args:
        rows: Lignes détaillées par expérience et par méthode.
        results_dir: Dossier racine de l'exécution courante.
        logger: Logger applicatif.
    """
    if not rows:
        logger.warning("No cross-experiment rows available for aggregation.")
        return
    out_dir = Path(results_dir) / "plots" / "cross_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed_df = pd.DataFrame(rows)
    aggregated_df = (
        detailed_df.groupby("method", as_index=False)["mean_delta_vs_manual"]
        .mean()
        .rename(columns={"mean_delta_vs_manual": "mean_delta_vs_manual_over_experiments"})
        .sort_values("mean_delta_vs_manual_over_experiments", ascending=False)
    )
    detailed_path = out_dir / "mean_delta_vs_manual_per_experiment_and_method.csv"
    aggregated_path = out_dir / "mean_delta_vs_manual_aggregated_by_method.csv"
    detailed_df.to_csv(detailed_path, index=False)
    aggregated_df.to_csv(aggregated_path, index=False)

    x_labels = [METHOD_DISPLAY_NAMES.get(m, m) for m in aggregated_df["method"].tolist()]
    y_values = aggregated_df["mean_delta_vs_manual_over_experiments"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x_labels, y_values, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Methods")
    ax.set_ylabel("Mean(method - manual) over experiments")
    ax.set_title("Cross-experiment mean delta vs Manual Pairing")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_delta_vs_manual_by_method.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cross-experiment outputs in %s", out_dir)




def run_single_dataset_analysis(
    cfg: dict[str, Any],
    generated: GeneratedDataset,
    results_dir: str,
    logger: Any,
    snapshots: SnapshotSaver,
) -> dict[str, Any] | None:
    """Persist one dataset, run analysis, and return results."""
    df = generated.df
    if df.empty:
        raise ValueError("Dataset is empty after generation. Cannot run the pipeline.")

    logger.info("✅ Dataset '%s' loaded with shape %s.", generated.strategy_name, df.shape)
    snapshots.save(df, "df_raw_head")

    strobe_stages: list[StrobeStage] = [
        StrobeStage(label="Cohorte source", n=int(len(df))),
    ]

    analysis_cfg = cfg.get("analysis", {}) or {}
    outlier_cfg = analysis_cfg.get("outlier_detection", {}) or {}
    hard_bounds_cfg = outlier_cfg.get("hard_bounds", {}) or {}
    hard_bounds: dict[str, dict[str, float]] = {
        str(col): {str(k): float(v) for k, v in (bound or {}).items()}
        for col, bound in hard_bounds_cfg.items()
    }
    bounds_report = pd.DataFrame(columns=list(OUTLIER_REPORT_COLUMNS))
    if hard_bounds:
        # Applied before Table 1 (not inside wide preprocess) so the baseline
        # descriptive table and the rest of the pipeline describe the same
        # corrected cohort, instead of Table 1 reflecting the raw aberrant
        # values while matching/GLMM use the corrected ones.
        df, bounds_report = apply_hard_bounds(
            df,
            hard_bounds,
            physician_col=str(analysis_cfg.get("physician_col", DEFAULT_PHYSICIAN_COL)),
        )
        if not bounds_report.empty:
            logger.info(
                "Hard bounds: %d cell(s) corrected across %d column(s) before Table 1 (%s).",
                len(bounds_report),
                bounds_report["column"].nunique(),
                ", ".join(sorted(bounds_report["column"].unique())),
            )

    generate_baseline_table_one(
        df,
        Path(results_dir),
        outcome_col=str(analysis_cfg.get("outcome_col", DEFAULT_OUTCOME_COL)),
        outcome_display="recommendation",
        physician_col=analysis_cfg.get("physician_col", DEFAULT_PHYSICIAN_COL),
    )
    strobe_stages.append(
        StrobeStage(label="Population Table 1 (descriptive)", n=int(len(df)))
    )

    if not bool(cfg.get("analysis", {}).get("enabled", True)):
        logger.info("Analysis disabled by config; generation/export completed.")
        _maybe_write_strobe_diagram(
            cfg=cfg,
            results_dir=results_dir,
            stages=strobe_stages,
            logger=logger,
        )
        return None

    # Capture questionnaire completion before encode / impute (blank stays blank).
    q_cfg_pre = (cfg.get("analysis", {}) or {}).get("questionnaire", {}) or {}
    questionnaire_prefix = str(q_cfg_pre.get("column_prefix", DEFAULT_QUESTIONNAIRE_PREFIX))
    qa_rates = questionnaire_response_rates(df, column_prefix=questionnaire_prefix)
    qa_detail = questionnaire_response_detail(df, column_prefix=questionnaire_prefix)
    if qa_rates:
        logger.info(
            "Questionnaire pre-impute response rates: %d columns "
            "(mean %.1f%%, min %.1f%%, max %.1f%%).",
            len(qa_rates),
            100.0 * float(sum(qa_rates.values()) / len(qa_rates)),
            100.0 * float(min(qa_rates.values())),
            100.0 * float(max(qa_rates.values())),
        )

    df_for_analysis, resolved_fixed_effects, outlier_summary = _resolve_fixed_effects_and_impute(
        df=df, cfg=cfg, logger=logger
    )
    snapshots.save(df_for_analysis, "df_preprocessed_head")

    mad_report = outlier_summary.pop("report")
    mad_columns_checked = outlier_summary.pop("columns_checked")
    outlier_report = pd.concat([bounds_report, mad_report], ignore_index=True)
    snapshots.save_manifest(outlier_report, "outlier_detection_manifest", full=True)
    if outlier_report.empty:
        logger.info("Outlier detection: 0 cells flagged, no outlier_detection_manifest written.")

    n_checked_columns = len(set(mad_columns_checked) | set(hard_bounds.keys()))
    total_cells_checked = int(len(df_for_analysis) * n_checked_columns)
    outlier_summary.update(
        {
            "n_flagged_cells": int(len(outlier_report)),
            "total_cells_checked": total_cells_checked,
            "flagged_cell_fraction": (
                float(len(outlier_report) / total_cells_checked) if total_cells_checked else 0.0
            ),
            "by_column": (
                {str(k): int(v) for k, v in outlier_report["column"].value_counts().items()}
                if not outlier_report.empty
                else {}
            ),
            "by_method": (
                {str(k): int(v) for k, v in outlier_report["method"].value_counts().items()}
                if not outlier_report.empty
                else {}
            ),
            "hard_bounds": hard_bounds,
        }
    )

    analysis = Analysis(
        df=df_for_analysis,
        results_dir=results_dir,
        LOGGER=logger,
        snapshots=snapshots,
        outlier_stats=outlier_summary,
        config=build_analysis_config(
            cfg=cfg,
            df=df_for_analysis,
            strategy_name=generated.strategy_name,
            generation_strategy=generated.generation_strategy,
            outcome_col_hint=generated.outcome_col,
            expected_summary=generated.expected_summary,
            fixed_effects_override=resolved_fixed_effects,
            questionnaire_response_rates=qa_rates,
            questionnaire_response_detail=qa_detail.to_dict(orient="records"),
            questionnaire_prefix=questionnaire_prefix,
        ),
    )
    results = analysis.run_full_analysis()
    logger.info("✅ Analysis '%s' completed with %d results keys", generated.strategy_name, len(results))

    matching_basis_ctx = (
        (results.get("validation_context") or {}).get("matching_basis") or {}
        if isinstance(results, dict)
        else {}
    )
    n_pre_matching = matching_basis_ctx.get("n_raw_rows")
    n_clean = matching_basis_ctx.get("n_clean_rows")
    if n_pre_matching is not None and n_clean is not None:
        n_incomplete = int(n_pre_matching) - int(n_clean)
        strobe_stages.append(
            StrobeStage(
                label="Base de matching (complete-case)",
                n=int(n_clean),
                excluded_label="données incomplètes (dropna covariables)",
                n_excluded=n_incomplete,
            )
        )
        intra = results.get("intra_physician_variability")
        if isinstance(intra, pd.DataFrame) and not intra.empty and "n_patients" in intra.columns:
            n_on_eligible_physicians = int(intra["n_patients"].sum())
            n_low_volume = int(n_clean) - n_on_eligible_physicians
            min_patients = int(
                (cfg.get("analysis", {}) or {}).get("min_patients_per_physician", 30)
            )
            strobe_stages.append(
                StrobeStage(
                    label=(
                        "Sous-population métriques / appariement\n"
                        f"(médecins ≥ {min_patients} patients)"
                    ),
                    n=n_on_eligible_physicians,
                    excluded_label=f"médecin à faible effectif (< {min_patients})",
                    n_excluded=n_low_volume,
                )
            )

    _maybe_write_strobe_diagram(
        cfg=cfg,
        results_dir=results_dir,
        stages=strobe_stages,
        logger=logger,
    )
    return results


def _maybe_write_strobe_diagram(
    *,
    cfg: dict[str, Any],
    results_dir: str,
    stages: list[StrobeStage],
    logger: Any,
    medication_label: str | None = None,
) -> None:
    """Write STROBE PNG + CSV under ``plots/strobe/`` when enabled in config."""
    strobe_cfg = (cfg.get("analysis", {}) or {}).get("strobe_diagram", {}) or {}
    if not bool(strobe_cfg.get("enabled", True)):
        return
    if not stages:
        return
    out_dir = Path(results_dir) / "plots" / "strobe"
    out_dir.mkdir(parents=True, exist_ok=True)
    title = "Flux de sélection des patients (STROBE)"
    if medication_label:
        title = f"{title} — {medication_label}"
    png_path = out_dir / "strobe_diagram.png"
    render_strobe_diagram(stages, png_path, title=title)
    csv_path = out_dir / "strobe_flow.csv"
    stages_to_dataframe(stages).to_csv(csv_path, index=False)
    logger.info("STROBE flow diagram saved to %s (flow CSV: %s)", png_path, csv_path)


def run_pipeline(cfg: dict[str, Any]) -> str:
    """Run one complete execution (single cohort or multi-rule batch)."""
    experiment_name = resolve_experiment_name(cfg)
    logger = get_simple_logger(
        app_name=cfg.get("logging", {}).get("app_name", "main"),
        log_level=cfg.get("logging", {}).get("level", "info"),
        nom_experience=experiment_name,
    )

    results_dir = infer_results_dir(cfg, experiment_name)
    save_effective_config(cfg, results_dir)
    snapshots = make_snapshot_saver(os.path.join(results_dir, "snapshots"), cfg)
    generated_datasets = build_generated_datasets(
        cfg, logger, snapshots_root=results_dir
    )
    cross_experiment_rows: list[dict[str, Any]] = []
    show_progress = bool(cfg.get("dataset", {}).get("multi_rule_experiments", {}).get("enabled", False))
    progress = tqdm(
        generated_datasets,
        desc="Experiments",
        unit="exp",
        disable=not show_progress,
        dynamic_ncols=True,
    )

    for idx, generated in enumerate(progress, start=1):
        if len(generated_datasets) == 1:
            current_results_dir = results_dir
            current_snapshots = snapshots
        else:
            safe_strategy = generated.strategy_name.replace("/", "_")
            current_results_dir = os.path.join(results_dir, f"{idx:03d}_{safe_strategy}")
            current_snapshots = make_snapshot_saver(
                os.path.join(current_results_dir, "snapshots"),
                cfg,
            )
            os.makedirs(current_results_dir, exist_ok=True)
        logger.info(
            "Running experiment %d/%d with strategy '%s'.",
            idx,
            len(generated_datasets),
            generated.strategy_name,
        )
        results = run_single_dataset_analysis(
            cfg, generated, current_results_dir, logger, current_snapshots
        )
        if results is not None:
            extracted_rows = _extract_mean_delta_vs_manual(
                results, generated.strategy_name, logger=logger
            )
            if not extracted_rows:
                logger.warning(
                    "No cross-experiment delta rows extracted for strategy '%s'.",
                    generated.strategy_name,
                )
            for row in extracted_rows:
                cross_experiment_rows.append(
                    {
                        **row,
                        "experiment_index": generated.experiment_index,
                        "rule_name": generated.rule_name,
                        "pass_index": generated.pass_index,
                        "window_size": generated.window_size,
                        "window_start": generated.window_start,
                        "variables": "|".join(generated.variables),
                    }
                )
        if show_progress:
            progress.set_postfix_str(generated.strategy_name)

    if show_progress:
        progress.close()

    if len(generated_datasets) > 1:
        _save_cross_experiment_outputs(cross_experiment_rows, results_dir, logger)
        logger.info(
            "🏁 Multi-rule batch completed: %d experiments under %s.",
            len(generated_datasets),
            results_dir,
        )
    logger.info("🏁 Pipeline completed. Results saved in: %s", results_dir)
    return results_dir


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    cfg = load_config(args.config)
    print("🚀 Starting prescription variability pipeline (synthetic mode)...")
    try:
        run_pipeline(cfg)
    except Exception as exc:
        print(f"❌ Pipeline failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
