# -*- coding: utf-8 -*-
"""Synthetic dataset generation for SCORE2 and progressive multi-rule experiments."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

SCORE2_ONLY_STRATEGY = "score2_five_groups_heter_patients"
REALISTIC_HETEROGENEOUS_STRATEGY = "realistic_heterogeneous_patients"
REALISTIC_RULE_THREE_GROUPS_STRATEGY = "realistic_rule_three_groups"
REALISTIC_RULE_THREE_GROUPS_OUTCOME_COL = "outcome_realistic_rule_three_groups"
REALISTIC_RULE_NAME = "statin_three_pathways"
REALISTIC_RULE_DESCRIPTION = (
    "Pathway A: score_score2>=7 AND (DBP>=90 OR homocysteine>=12.5 OR weight>=92); "
    "Pathway B: hba1c>=7; Pathway C: cystatin>=1.2 AND score_score2>=6; "
    "Pathway D: score_score2>=9.5"
)
RULE_THREE_GROUPS_ADHERENCE: tuple[tuple[str, int, int, float, float], ...] = (
    ("random", 0, 6, -1.0, -1.0),
    ("partial_80", 6, 13, 0.80, 0.10),
    ("perfect_100", 13, 19, 1.00, 0.00),
)
RULE_PREVALENCE_WARN_BOUNDS = (0.25, 0.55)
REALISTIC_MATCHING_FEATURE_COLUMNS: tuple[str, ...] = (
    "score_score2",
    "biomarker_hba1c_ngsp_blood",
    "biomarker_cystatin_c_serum",
    "biomarker_weight",
    "biomarker_homocysteine_serum",
    "biomarker_diastolic_blood_pressure_sitting",
)
REALISTIC_PRESCRIPTION_RATE_BOUNDS = (0.1, 0.6)
MULTI_RULE_STRATEGY_PREFIX = "synthetic_rule"
SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS = 10_000 # vient du fichier de config, sinon valeur par défaut
SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS = 20
SCORE2_FIVE_CATEGORIES_CLUSTER_PROBS = ((1.0, 0.0), (0.9, 0.05), (0.8, 0.1), (0.7, 0.2), (0.5, 0.5))
PROGRESSIVE_RULE_DEFAULT_N_PASSES = 2

# Minimum support required per physician at generation time.
MIN_PATIENTS_PER_PHYSICIAN_BASE = 90

AGE_MEAN, AGE_STD, AGE_MIN, AGE_MAX = 60.0, 12.0, 40, 90
HBA1C_MEAN, HBA1C_STD, HBA1C_MIN, HBA1C_MAX = 6.5, 1.5, 4.0, 12.0
NON_HDL_MMOL_MEAN, NON_HDL_MMOL_STD, NON_HDL_MMOL_MIN, NON_HDL_MMOL_MAX = 3.6, 0.95, 1.0, 8.0
HDL_MMOL_MEAN, HDL_MMOL_STD, HDL_MMOL_MIN, HDL_MMOL_MAX = 1.35, 0.38, 0.4, 2.8
SBP_MEAN, SBP_STD, SBP_MIN, SBP_MAX = 130.0, 20.0, 90, 200
LDL_MMOL_MEAN, LDL_MMOL_STD, LDL_MMOL_MIN, LDL_MMOL_MAX = 1.3, 0.35, 0.4, 2.2
DBP_MEAN, DBP_STD, DBP_MIN, DBP_MAX = 82.0, 11.0, 60, 115
HAD_MEAN_A, HAD_MEAN_B, HAD_COMPONENT_STD, HAD_MIN, HAD_MAX = 6.0, 15.0, 4.0, 2, 38
HAS_CAR_PROB, SMOKER_PROB, MALE_PROB = 0.25, 0.2, 0.6
EGFR_MEAN, EGFR_STD, EGFR_MIN, EGFR_MAX = 90.0, 25.0, 15.0, 140.0

SCORE2_DEFAULT_RISK_REGION = "Low"
_SCORE2_SCALES: dict[tuple[str, str, str], tuple[float, float]] = {
    ("Low", "young", "male"): (-0.5699, 0.7476),
    ("Low", "young", "female"): (-0.7380, 0.7019),
    ("Moderate", "young", "male"): (-0.1565, 0.8009),
    ("Moderate", "young", "female"): (-0.3143, 0.7701),
    ("High", "young", "male"): (0.3207, 0.9360),
    ("High", "young", "female"): (0.5710, 0.9369),
    ("Very high", "young", "male"): (0.5836, 0.8294),
    ("Very high", "young", "female"): (0.9412, 0.8329),
    ("Low", "old", "male"): (-0.34, 1.19),
    ("Low", "old", "female"): (-0.52, 1.01),
    ("Moderate", "old", "male"): (0.01, 1.25),
    ("Moderate", "old", "female"): (-0.1, 1.1),
    ("High", "old", "male"): (0.08, 1.15),
    ("High", "old", "female"): (0.38, 1.09),
    ("Very high", "old", "male"): (0.05, 0.7),
    ("Very high", "old", "female"): (0.38, 0.69),
}


def _validate_counts(n_patients: int, expected_n_pros: int) -> None:
    """Valide les volumes minimaux demandés pour la génération."""
    if expected_n_pros < 1:
        raise ValueError("expected_n_pros must be >= 1.")
    if n_patients < 1:
        raise ValueError("n_patients must be >= 1.")


def _ensure_min_support(n_patients: int, expected_n_pros: int, logger: Optional[Any]) -> int:
    """Ajuste `n_patients` pour garantir un support minimum par médecin."""
    min_total = expected_n_pros * MIN_PATIENTS_PER_PHYSICIAN_BASE
    if n_patients >= min_total:
        return n_patients
    if logger:
        logger.warning(
            "n_patients (%d) < n_physicians * %d (= %d); adjusting to %d.",
            n_patients,
            MIN_PATIENTS_PER_PHYSICIAN_BASE,
            min_total,
            min_total,
        )
    return min_total


def _validate_score2_rule_columns(df: pd.DataFrame) -> None:
    """Vérifie les colonnes requises par les règles de risque SCORE2."""
    required = {"age", "is_smoker", "systolic_blood_pressure", "non_hdl_cholesterol", "hdl_cholesterol", "is_male"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing columns for SCORE2-style rule: %s." % ", ".join(missing))


def _professional_ids(expected_n_pros: int) -> list[str]:
    return [f"P{i}" for i in range(1, expected_n_pros + 1)]


def _build_universal_base_cohort(
    n_patients: int, expected_n_pros: int, random_state: int
) -> tuple[pd.DataFrame, list[str]]:
    """Construit une cohorte synthétique de base avant application des règles.

    Returns:
        Un tuple `(df, professionals)` avec les covariables patient simulées et
        la liste ordonnée des identifiants médecins.
    """
    _validate_counts(n_patients, expected_n_pros)
    rng = np.random.default_rng(random_state)
    professionals = _professional_ids(expected_n_pros)
    base_assignment = np.repeat(professionals, MIN_PATIENTS_PER_PHYSICIAN_BASE)
    if n_patients <= base_assignment.size:
        professional_col = rng.permutation(base_assignment)[:n_patients]
    else:
        remaining = n_patients - int(base_assignment.size)
        professional_col = np.concatenate([base_assignment, rng.choice(professionals, size=remaining, replace=True)])
        rng.shuffle(professional_col)

    had_group_b = rng.binomial(1, 0.5, n_patients).astype(bool)
    had_raw = np.where(
        had_group_b,
        rng.normal(HAD_MEAN_B, HAD_COMPONENT_STD, n_patients),
        rng.normal(HAD_MEAN_A, HAD_COMPONENT_STD, n_patients),
    )

    df = pd.DataFrame(
        {
            "member_pseudo_id": np.arange(1, n_patients + 1, dtype=np.int32),
            "professional_id": professional_col,
            "age": np.clip(rng.normal(AGE_MEAN, AGE_STD, n_patients), AGE_MIN, AGE_MAX).astype(np.int32),
            "biomarker_hba1c_ngsp_blood": np.round(np.clip(rng.normal(HBA1C_MEAN, HBA1C_STD, n_patients), HBA1C_MIN, HBA1C_MAX), 2).astype(np.float64),
            "non_hdl_cholesterol": np.round(np.clip(rng.normal(NON_HDL_MMOL_MEAN, NON_HDL_MMOL_STD, n_patients), NON_HDL_MMOL_MIN, NON_HDL_MMOL_MAX), 2).astype(np.float64),
            "hdl_cholesterol": np.round(np.clip(rng.normal(HDL_MMOL_MEAN, HDL_MMOL_STD, n_patients), HDL_MMOL_MIN, HDL_MMOL_MAX), 2).astype(np.float64),
            "ldl_cholesterol": np.round(np.clip(rng.normal(LDL_MMOL_MEAN, LDL_MMOL_STD, n_patients), LDL_MMOL_MIN, LDL_MMOL_MAX), 2).astype(np.float64),
            "systolic_blood_pressure": np.clip(rng.normal(SBP_MEAN, SBP_STD, n_patients), SBP_MIN, SBP_MAX).astype(np.int32),
            "diastolic_blood_pressure": np.round(np.clip(rng.normal(DBP_MEAN, DBP_STD, n_patients), DBP_MIN, DBP_MAX)).astype(np.int32),
            "estimated_glomerular_filtration_rate": np.round(np.clip(rng.normal(EGFR_MEAN, EGFR_STD, n_patients), EGFR_MIN, EGFR_MAX), 2).astype(np.float64),
            "score_questionnaire_HAD": np.clip(np.round(had_raw).astype(np.int32), HAD_MIN, HAD_MAX),
            "is_smoker": rng.binomial(1, SMOKER_PROB, n_patients).astype(bool),
            "is_male": rng.binomial(1, MALE_PROB, n_patients).astype(bool),
            "has_car": rng.binomial(1, HAS_CAR_PROB, n_patients).astype(bool),
        }
    )
    return df, professionals


def _score2_scale_arrays(risk_region: str, age: np.ndarray, is_male: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.zeros(len(age), dtype=float)
    s2 = np.zeros(len(age), dtype=float)
    young = age < 70.0
    for sex_key, sex_bool in (("male", True), ("female", False)):
        for age_key, age_mask in (("young", young), ("old", ~young)):
            mask = (is_male == sex_bool) & age_mask
            key = (risk_region, age_key, sex_key)
            if key not in _SCORE2_SCALES:
                raise ValueError("Unknown SCORE2 risk region: %r" % risk_region)
            s1[mask], s2[mask] = _SCORE2_SCALES[key]
    return s1, s2


def score2_ten_year_risk_percent_vectorized(
    df: pd.DataFrame, risk_region: str = SCORE2_DEFAULT_RISK_REGION, diabetes: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """Calcule le risque cardiovasculaire SCORE2 à 10 ans de façon vectorisée.

    Args:
        df: Données patient contenant les variables cliniques nécessaires.
        risk_region: Région de risque SCORE2 (`Low`, `Moderate`, `High`, `Very high`).
        diabetes: Vecteur binaire optionnel aligné avec `df`.

    Returns:
        Tableau `numpy` des risques en pourcentage, un par patient.
    """
    _validate_score2_rule_columns(df)
    n = len(df)
    age = df["age"].to_numpy(dtype=float)
    is_male = df["is_male"].to_numpy(dtype=bool)
    smk = df["is_smoker"].to_numpy(dtype=float)
    sbp = df["systolic_blood_pressure"].to_numpy(dtype=float)
    non_hdl = df["non_hdl_cholesterol"].to_numpy(dtype=float)
    hdl = df["hdl_cholesterol"].to_numpy(dtype=float)
    tc = non_hdl + hdl
    dm = np.zeros(n, dtype=float) if diabetes is None else np.asarray(diabetes, dtype=float).ravel()
    if dm.shape[0] != n:
        raise ValueError("diabetes array must match DataFrame length.")
    s1, s2 = _score2_scale_arrays(risk_region, age, is_male)
    risk_pct = np.full(n, np.nan, dtype=float)
    young = age < 70.0
    my, fy, mo, fo = is_male & young, (~is_male) & young, is_male & (~young), (~is_male) & (~young)

    if np.any(my):
        i = my
        xx = 0.3742*(age[i]-60.0)/5.0 + 0.6012*smk[i] + 0.2777*(sbp[i]-120.0)/20.0 + 0.6457*dm[i] + 0.1458*(tc[i]-6.0) + (-0.2698)*(hdl[i]-1.3)/0.5 + (-0.0755)*(age[i]-60.0)/5.0*smk[i] + (-0.0255)*(age[i]-60.0)/5.0*(sbp[i]-120.0)/20.0 + (-0.0281)*(age[i]-60.0)/5.0*(tc[i]-6.0) + 0.0426*(age[i]-60.0)/5.0*(hdl[i]-1.3)/0.5 + (-0.0983)*(age[i]-60.0)/5.0*dm[i]
        x2 = np.clip(1.0 - np.power(0.9605, np.exp(xx)), 1e-12, 1.0 - 1e-12)
        risk_pct[i] = (1.0 - np.exp(-np.exp(s1[i] + s2[i] * np.log(-np.log(1.0 - x2))))) * 100.0
    if np.any(fy):
        i = fy
        xx = 0.4648*(age[i]-60.0)/5.0 + 0.7744*smk[i] + 0.3131*(sbp[i]-120.0)/20.0 + 0.8096*dm[i] + 0.1002*(tc[i]-6.0) + (-0.2606)*(hdl[i]-1.3)/0.5 + (-0.1088)*(age[i]-60.0)/5.0*smk[i] + (-0.0277)*(age[i]-60.0)/5.0*(sbp[i]-120.0)/20.0 + (-0.0226)*(age[i]-60.0)/5.0*(tc[i]-6.0) + 0.0613*(age[i]-60.0)/5.0*(hdl[i]-1.3)/0.5 + (-0.1272)*(age[i]-60.0)/5.0*dm[i]
        x2 = np.clip(1.0 - np.power(0.9776, np.exp(xx)), 1e-12, 1.0 - 1e-12)
        risk_pct[i] = (1.0 - np.exp(-np.exp(s1[i] + s2[i] * np.log(-np.log(1.0 - x2))))) * 100.0
    if np.any(mo):
        i = mo
        xx = 0.0634*(age[i]-73.0) + 0.4245*dm[i] + 0.3524*smk[i] + 0.0094*(sbp[i]-150.0) + 0.0850*(tc[i]-6.0) + (-0.3564)*(hdl[i]-1.4) + (-0.0174)*(age[i]-73.0)*dm[i] + (-0.0247)*(age[i]-73.0)*smk[i] + (-0.0005)*(age[i]-73.0)*(sbp[i]-150.0) + 0.0073*(age[i]-73.0)*(tc[i]-6.0) + 0.0091*(age[i]-73.0)*(hdl[i]-1.4)
        x2 = np.clip(1.0 - np.power(0.7576, np.exp(xx - 0.0929)), 1e-12, 1.0 - 1e-12)
        risk_pct[i] = (1.0 - np.exp(-np.exp(s1[i] + s2[i] * np.log(-np.log(1.0 - x2))))) * 100.0
    if np.any(fo):
        i = fo
        xx = 0.0789*(age[i]-73.0) + 0.6010*dm[i] + 0.4921*smk[i] + 0.0102*(sbp[i]-150.0) + 0.0605*(tc[i]-6.0) + (-0.3040)*(hdl[i]-1.4) + (-0.0107)*(age[i]-73.0)*dm[i] + (-0.0255)*(age[i]-73.0)*smk[i] + (-0.0004)*(age[i]-73.0)*(sbp[i]-150.0) + (-0.0009)*(age[i]-73.0)*(tc[i]-6.0) + 0.0154*(age[i]-73.0)*(hdl[i]-1.4)
        x2 = np.clip(1.0 - np.power(0.8082, np.exp(xx - 0.229)), 1e-12, 1.0 - 1e-12)
        risk_pct[i] = (1.0 - np.exp(-np.exp(s1[i] + s2[i] * np.log(-np.log(1.0 - x2))))) * 100.0
    return risk_pct


def _score2_at_least_moderate_risk_mask(age: np.ndarray, risk_pct: np.ndarray) -> np.ndarray:
    a = age.astype(float)
    r = np.asarray(risk_pct, dtype=float)
    return ((a < 50.0) & (r >= 2.5)) | ((a >= 50.0) & (a <= 69.0) & (r >= 5.0)) | ((a > 69.0) & (r >= 7.5))


def _calculate_score2_moderate_or_high_vectorized(df: pd.DataFrame) -> pd.Series:
    risk_pct = score2_ten_year_risk_percent_vectorized(df, risk_region=SCORE2_DEFAULT_RISK_REGION)
    return pd.Series(_score2_at_least_moderate_risk_mask(df["age"].to_numpy(dtype=float), risk_pct), index=df.index)


def build_progressive_rule_plan(
    fixed_effects: list[str],
    n_passes: int = PROGRESSIVE_RULE_DEFAULT_N_PASSES,
) -> list[dict[str, Any]]:
    """
    Build the exact progressive rule schedule:
      exp 1 = SCORE2
      pass 1 windows size 1..N, sliding
      pass 2 idem, with different thresholds at generation time.
    """
    plan: list[dict[str, Any]] = [{"rule_type": "score2", "window_size": 0, "window_start": 0, "pass_index": 0, "variables": []}]
    n_features = len(fixed_effects)
    for pass_index in range(1, int(n_passes) + 1):
        for window_size in range(1, n_features + 1):
            for window_start in range(0, n_features - window_size + 1):
                variables = fixed_effects[window_start : window_start + window_size]
                plan.append(
                    {
                        "rule_type": "window_threshold",
                        "window_size": window_size,
                        "window_start": window_start,
                        "pass_index": pass_index,
                        "variables": variables,
                    }
                )
    return plan


def expected_progressive_rule_count(n_fixed_effects: int, n_passes: int = PROGRESSIVE_RULE_DEFAULT_N_PASSES) -> int:
    """Retourne le nombre théorique d'expériences du plan progressif."""
    return 1 + int(n_passes) * (n_fixed_effects * (n_fixed_effects + 1) // 2)


def _sample_threshold_for_column(values: np.ndarray, rng: np.random.Generator) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    col_min = float(np.min(arr))
    col_max = float(np.max(arr))
    if col_max <= col_min:
        return col_min
    unique_vals = np.unique(arr)
    if unique_vals.size <= 2:
        return float(rng.choice(unique_vals))
    return float(rng.uniform(col_min, col_max))


def _build_group_probability_maps(professionals: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    group_boundaries = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20))
    p_high: dict[str, float] = {}
    p_low: dict[str, float] = {}
    for group_index, (lo, hi) in enumerate(group_boundaries):
        hi_p, lo_p = SCORE2_FIVE_CATEGORIES_CLUSTER_PROBS[group_index]
        for pid in professionals[lo:hi]:
            p_high[pid] = hi_p
            p_low[pid] = lo_p
    return p_high, p_low


def _sanitize_strategy_token(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower()


def get_generation_rule_mask(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """Récupère le masque d'éligibilité de la règle ayant généré le dataset.

    Args:
        df: Dataset synthétique.
        strategy: Nom de la stratégie de génération.

    Returns:
        Tableau booléen indiquant les patients dans la zone "traitable" selon la
        règle de génération.
    """
    if strategy == SCORE2_ONLY_STRATEGY:
        _validate_score2_rule_columns(df)
        return _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)
    if strategy == REALISTIC_RULE_THREE_GROUPS_STRATEGY:
        if "generation_rule_mask" in df.columns:
            return df["generation_rule_mask"].to_numpy(dtype=bool)
        return _compute_statin_indication_mask(df)
    if "generation_rule_mask" in df.columns:
        return df["generation_rule_mask"].to_numpy(dtype=bool)
    raise ValueError(
        "Unknown synthetic generation strategy: %r. Supported: %s or datasets containing 'generation_rule_mask'."
        % (strategy, SCORE2_ONLY_STRATEGY)
    )


def generate_progressive_rule_experiment(
    rule_definition: dict[str, Any],
    experiment_index: int,
    fixed_effects: list[str],
    n_patients: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS,
    expected_n_pros: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS,
    random_state: int = 42,
    logger: Optional[Any] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one synthetic experiment for the given progressive rule definition."""
    _validate_counts(n_patients, expected_n_pros)
    if expected_n_pros != 20:
        raise ValueError(
            "generate_progressive_rule_experiment requires expected_n_pros=20 (got %d)."
            % expected_n_pros
        )
    n_patients = _ensure_min_support(n_patients, expected_n_pros, logger)
    base_seed = int(random_state) + int(experiment_index) * 1_009
    df, professionals = _build_universal_base_cohort(n_patients, expected_n_pros, base_seed)
    df["score2_ten_year_cvd_risk_percent"] = np.round(
        score2_ten_year_risk_percent_vectorized(df, risk_region=SCORE2_DEFAULT_RISK_REGION), 2
    )
    rule_type = str(rule_definition.get("rule_type", "window_threshold"))
    pass_index = int(rule_definition.get("pass_index", 0))
    window_size = int(rule_definition.get("window_size", 0))
    window_start = int(rule_definition.get("window_start", 0))
    variables = [str(c) for c in rule_definition.get("variables", [])]

    threshold_rng = np.random.default_rng(base_seed + 20_003 + pass_index * 7_001 + window_size * 199 + window_start * 17)
    thresholds: dict[str, float] = {}
    if rule_type == "score2":
        eligibility_mask = _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)
        rule_name = "score2"
        rule_description = "SCORE2 moderate/high risk"
        generation_strategy_name = SCORE2_ONLY_STRATEGY
    else:
        if not variables:
            raise ValueError("Window-threshold rule must include at least one variable.")
        masks: list[np.ndarray] = []
        for col in variables:
            if col not in df.columns:
                raise ValueError(f"Missing fixed effect column in generated dataset: '{col}'.")
            threshold = _sample_threshold_for_column(df[col].to_numpy(), threshold_rng)
            thresholds[col] = threshold
            masks.append(df[col].to_numpy(dtype=float) <= threshold)
        eligibility_mask = np.logical_and.reduce(masks) if masks else np.zeros(len(df), dtype=bool)
        vars_token = "_".join(_sanitize_strategy_token(col) for col in variables)
        rule_name = f"w{window_size}_{vars_token}_pass{pass_index}"
        threshold_str = ", ".join(f"{k} <= {v:.4g}" for k, v in thresholds.items())
        rule_description = f"All true among: {threshold_str}"
        generation_strategy_name = f"{MULTI_RULE_STRATEGY_PREFIX}_{rule_name}"

    p_high, p_low = _build_group_probability_maps(professionals)

    p_vec = np.where(
        eligibility_mask,
        df["professional_id"].map(p_high).to_numpy(dtype=float),
        df["professional_id"].map(p_low).to_numpy(dtype=float),
    )
    outcome_rng = np.random.default_rng(base_seed + 20_000)
    strategy_name = f"exp_{experiment_index:03d}_{rule_name}"
    outcome_col = f"outcome_{strategy_name}"
    df[outcome_col] = outcome_rng.binomial(1, p_vec).astype(np.int32)
    df["recommendation"] = df[outcome_col]
    df["generation_rule_mask"] = np.asarray(eligibility_mask, dtype=bool)
    df["generation_rule_name"] = rule_name
    df["generation_rule_description"] = rule_description
    df["generation_rule_pass_index"] = pass_index
    df["generation_rule_window_size"] = window_size

    expected_summary = (
        f"Rule '{rule_name}' ({rule_description}) with fixed five-group prescription probabilities "
        "(100/0, 90/5, 80/10, 70/20, 50/50)."
    )
    if logger:
        logger.info(
            "Synthetic experiment %03d generated with strategy '%s' (%s).",
            experiment_index,
            strategy_name,
            rule_description,
        )
    return df, {
        "strategy_name": strategy_name,
        "generation_strategy": generation_strategy_name,
        "expected_summary": expected_summary,
        "outcome_col": outcome_col,
        "rule_name": rule_name,
        "rule_description": rule_description,
        "rule_thresholds": thresholds,
        "pass_index": pass_index,
        "window_size": window_size,
        "window_start": window_start,
        "variables": variables,
    }


def _simulate_realistic_matching_covariates(
    n_patients: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate the six matching covariates with mild cross-correlation."""
    z = rng.standard_normal((n_patients, len(REALISTIC_MATCHING_FEATURE_COLUMNS)))
    # Shared latent factor for mild clinical correlation.
    latent = rng.standard_normal(n_patients)
    z[:, 0] = 0.55 * latent + 0.45 * z[:, 0]
    z[:, 1] = 0.40 * latent + 0.60 * z[:, 1]
    z[:, 5] = 0.35 * latent + 0.65 * z[:, 5]

    score_score2 = np.clip(5.0 + 2.5 * z[:, 0], 1.0, 25.0)
    hba1c = np.clip(HBA1C_MEAN + HBA1C_STD * z[:, 1], HBA1C_MIN, HBA1C_MAX)
    cystatin = np.clip(1.0 + 0.25 * z[:, 2], 0.5, 2.5)
    weight = np.clip(75.0 + 14.0 * z[:, 3], 45.0, 130.0)
    homocysteine = np.clip(10.0 + 3.0 * z[:, 4], 5.0, 25.0)
    dbp = np.clip(DBP_MEAN + DBP_STD * z[:, 5], DBP_MIN, DBP_MAX)

    return pd.DataFrame(
        {
            "score_score2": np.round(score_score2, 1).astype(np.float64),
            "biomarker_hba1c_ngsp_blood": np.round(hba1c, 2).astype(np.float64),
            "biomarker_cystatin_c_serum": np.round(cystatin, 2).astype(np.float64),
            "biomarker_weight": np.round(weight, 1).astype(np.float64),
            "biomarker_homocysteine_serum": np.round(homocysteine, 1).astype(np.float64),
            "biomarker_diastolic_blood_pressure_sitting": np.round(dbp).astype(np.int32),
        }
    )


def _compute_statin_indication_mask(df: pd.DataFrame) -> np.ndarray:
    """Return the boolean statin-indication mask from the six matching covariates."""
    modificateur = (
        (df["biomarker_diastolic_blood_pressure_sitting"].to_numpy(dtype=float) >= 90)
        | (df["biomarker_homocysteine_serum"].to_numpy(dtype=float) >= 12.5)
        | (df["biomarker_weight"].to_numpy(dtype=float) >= 92.0)
    )
    score = df["score_score2"].to_numpy(dtype=float)
    voie_a = (score >= 7.0) & modificateur
    voie_b = df["biomarker_hba1c_ngsp_blood"].to_numpy(dtype=float) >= 7.0
    voie_c = (
        (df["biomarker_cystatin_c_serum"].to_numpy(dtype=float) >= 1.20)
        & (score >= 6.0)
    )
    voie_d = score >= 9.5
    return voie_a | voie_b | voie_c | voie_d


def _assign_professional_column(
    n_patients: int,
    professionals: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign patients to physicians with a minimum support per physician."""
    base_assignment = np.repeat(professionals, MIN_PATIENTS_PER_PHYSICIAN_BASE)
    if n_patients <= base_assignment.size:
        return rng.permutation(base_assignment)[:n_patients]
    remaining = n_patients - int(base_assignment.size)
    professional_col = np.concatenate(
        [base_assignment, rng.choice(professionals, size=remaining, replace=True)]
    )
    rng.shuffle(professional_col)
    return professional_col


def generate_realistic_rule_three_groups_patients(
    n_patients: int = 3730,
    expected_n_pros: int = 19,
    random_state: int = 42,
    logger: Optional[Any] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic cohort with a known statin rule and three physician adherence groups."""
    _validate_counts(n_patients, expected_n_pros)
    if expected_n_pros != 19:
        raise ValueError(
            "realistic_rule_three_groups requires expected_n_pros=19 to keep fixed groups P1..P19 (got %d)."
            % expected_n_pros
        )
    n_patients = _ensure_min_support(n_patients, expected_n_pros, logger)
    rng = np.random.default_rng(random_state)
    professionals = _professional_ids(expected_n_pros)
    professional_col = _assign_professional_column(n_patients, professionals, rng)

    covariates = _simulate_realistic_matching_covariates(n_patients, rng)
    df = pd.DataFrame(
        {
            "member_pseudo_id": np.arange(1, n_patients + 1, dtype=np.int32),
            "professional_id": professional_col,
        }
    )
    df = pd.concat([df, covariates], axis=1)

    indication_mask = _compute_statin_indication_mask(df)
    prevalence = float(indication_mask.mean())
    if logger and not (RULE_PREVALENCE_WARN_BOUNDS[0] <= prevalence <= RULE_PREVALENCE_WARN_BOUNDS[1]):
        logger.warning(
            "Rule prevalence %.3f outside expected bounds %s.",
            prevalence,
            RULE_PREVALENCE_WARN_BOUNDS,
        )

    p_low, p_high = REALISTIC_PRESCRIPTION_RATE_BOUNDS
    random_rates = {
        pid: float(rng.uniform(p_low, p_high))
        for pid in professionals[: RULE_THREE_GROUPS_ADHERENCE[0][2]]
    }

    truth_rows: list[dict[str, object]] = []
    recommendation = np.zeros(n_patients, dtype=np.int32)

    for group_name, lo, hi, p_if_indicated, p_if_not_indicated in RULE_THREE_GROUPS_ADHERENCE:
        for physician in professionals[lo:hi]:
            phy_mask = df["professional_id"].to_numpy() == physician
            idx = np.where(phy_mask)[0]
            n_phy = idx.size
            if n_phy == 0:
                continue

            rule_phy = indication_mask[phy_mask]
            if group_name == "random":
                p_phy = random_rates[physician]
                p_vec = np.full(n_phy, p_phy, dtype=float)
                p_hi = p_lo = p_phy
            else:
                p_hi = p_if_indicated
                p_lo = p_if_not_indicated
                p_vec = np.where(rule_phy, p_hi, p_lo).astype(float)

            y_phy = rng.binomial(1, p_vec).astype(np.int32)
            recommendation[idx] = y_phy
            rule_prev = float(rule_phy.mean())
            truth_rows.append(
                {
                    "physician": physician,
                    "adherence_group": group_name,
                    "p_if_indicated": round(p_hi, 4),
                    "p_if_not_indicated": round(p_lo, 4),
                    "prescription_rate_true": round(float(p_hi * rule_prev + p_lo * (1.0 - rule_prev)), 4),
                    "prescription_rate_observed": round(float(y_phy.mean()), 4),
                    "rule_prevalence_observed": round(rule_prev, 4),
                    "n_patients": n_phy,
                }
            )

    df["generation_rule_mask"] = np.asarray(indication_mask, dtype=bool)
    df["generation_rule_name"] = REALISTIC_RULE_NAME
    df["generation_rule_description"] = REALISTIC_RULE_DESCRIPTION
    df[REALISTIC_RULE_THREE_GROUPS_OUTCOME_COL] = recommendation
    df["recommendation"] = recommendation
    ground_truth_df = pd.DataFrame(truth_rows)

    if logger:
        logger.info(
            "Synthetic dataset '%s' generated: %d patients, %d physicians "
            "(rule prevalence=%.3f; groups: P1-P6 random, P7-P13 80/10, P14-P19 100/0).",
            REALISTIC_RULE_THREE_GROUPS_STRATEGY,
            len(df),
            expected_n_pros,
            prevalence,
        )
    return df, ground_truth_df


def generate_realistic_heterogeneous_patients(
    n_patients: int = 3730,
    expected_n_pros: int = 19,
    random_state: int = 42,
    logger: Optional[Any] = None,
    shared_prescription_rate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic cohort mirroring real statin-matching experiments.

    By default each physician receives an independent prescription probability
    drawn uniformly in [0.1, 0.6]. When ``shared_prescription_rate`` is True,
    a single probability is drawn once and shared by all physicians. Every patient
    is prescribed independently via Bernoulli(p). Covariates do not drive the
    outcome; discordance is measured later by the analysis matching methods.

    Returns:
        ``(patient_df, ground_truth_df)`` where ``ground_truth_df`` holds the
        injected prescription rate and the empirical rate after random draws.
    """
    _validate_counts(n_patients, expected_n_pros)
    n_patients = _ensure_min_support(n_patients, expected_n_pros, logger)
    rng = np.random.default_rng(random_state)
    professionals = _professional_ids(expected_n_pros)

    base_assignment = np.repeat(professionals, MIN_PATIENTS_PER_PHYSICIAN_BASE)
    if n_patients <= base_assignment.size:
        professional_col = rng.permutation(base_assignment)[:n_patients]
    else:
        remaining = n_patients - int(base_assignment.size)
        professional_col = np.concatenate(
            [base_assignment, rng.choice(professionals, size=remaining, replace=True)]
        )
        rng.shuffle(professional_col)

    covariates = _simulate_realistic_matching_covariates(n_patients, rng)
    df = pd.DataFrame(
        {
            "member_pseudo_id": np.arange(1, n_patients + 1, dtype=np.int32),
            "professional_id": professional_col,
        }
    )
    df = pd.concat([df, covariates], axis=1)

    p_low, p_high = REALISTIC_PRESCRIPTION_RATE_BOUNDS
    truth_rows: list[dict[str, object]] = []
    recommendation = np.zeros(n_patients, dtype=np.int32)
    shared_p = float(rng.uniform(p_low, p_high)) if shared_prescription_rate else None

    for physician in professionals:
        mask = df["professional_id"].to_numpy() == physician
        idx = np.where(mask)[0]
        n_phy = idx.size
        if n_phy == 0:
            continue

        target_p = shared_p if shared_p is not None else float(rng.uniform(p_low, p_high))
        y_phy = rng.binomial(1, target_p, size=n_phy).astype(np.int32)
        recommendation[idx] = y_phy

        truth_rows.append(
            {
                "physician": physician,
                "prescription_rate_true": round(target_p, 4),
                "prescription_rate_observed": round(float(y_phy.mean()), 4),
                "n_patients": n_phy,
            }
        )

    df["recommendation"] = recommendation
    df["outcome_realistic_heterogeneous_patients"] = recommendation
    ground_truth_df = pd.DataFrame(truth_rows)

    if logger:
        if shared_prescription_rate:
            logger.info(
                "Synthetic dataset '%s' generated: %d patients, %d physicians "
                "(shared Bernoulli prescription, p=%.4f drawn from U%s).",
                REALISTIC_HETEROGENEOUS_STRATEGY,
                len(df),
                expected_n_pros,
                shared_p,
                REALISTIC_PRESCRIPTION_RATE_BOUNDS,
            )
        else:
            logger.info(
                "Synthetic dataset '%s' generated: %d patients, %d physicians "
                "(independent Bernoulli prescription, p ~ U%s per physician).",
                REALISTIC_HETEROGENEOUS_STRATEGY,
                len(df),
                expected_n_pros,
                REALISTIC_PRESCRIPTION_RATE_BOUNDS,
            )

    return df, ground_truth_df


def generate_score2_five_groups_heter_patients(
    n_patients: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS,
    expected_n_pros: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS,
    random_state: int = 42,
    logger: Optional[Any] = None,
    ) -> pd.DataFrame:
    """Generate a synthetic dataset with 20 physicians and 5 SCORE2-based categories."""
    _validate_counts(n_patients, expected_n_pros)
    if expected_n_pros != 20:
        raise ValueError(
            "score2_five_groups_heter_patients requires expected_n_pros=20 to keep fixed groups P1..P20 (got %d)."
            % expected_n_pros
        )
    n_patients = _ensure_min_support(n_patients, expected_n_pros, logger)
    df, professionals = _build_universal_base_cohort(n_patients, expected_n_pros, random_state)
    df["score2_ten_year_cvd_risk_percent"] = np.round(
        score2_ten_year_risk_percent_vectorized(df, risk_region=SCORE2_DEFAULT_RISK_REGION), 2
    )
    score2_indication = _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)

    group_boundaries = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20))
    p_high: dict[str, float] = {}
    p_low: dict[str, float] = {}
    for group_index, (lo, hi) in enumerate(group_boundaries):
        hi_p, lo_p = SCORE2_FIVE_CATEGORIES_CLUSTER_PROBS[group_index]
        for pid in professionals[lo:hi]:
            p_high[pid] = hi_p
            p_low[pid] = lo_p

    rng = np.random.default_rng(random_state + 40)
    p_vec = np.where(
        score2_indication,
        df["professional_id"].map(p_high).to_numpy(dtype=float),
        df["professional_id"].map(p_low).to_numpy(dtype=float),
    )
    outcome_col = "outcome_score2_five_groups_heter_patients"
    df[outcome_col] = rng.binomial(1, p_vec).astype(np.int32)
    df["recommendation"] = df[outcome_col]

    if logger:
        logger.info(
            "Synthetic dataset '%s' generated: %d patients, %d physicians "
            "(5 physician categories under SCORE2: group1 100%%/0%%, group2 90%%/5%%, "
            "group3 80%%/10%%, group4 70%%/20%%, group5 50%%/50%%).",
            SCORE2_ONLY_STRATEGY,
            len(df),
            expected_n_pros,
        )
    return df
