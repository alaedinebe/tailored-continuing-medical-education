# -*- coding: utf-8 -*-
"""Validation threshold profiles for synthetic PRISM run audits."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "synthetic_signal": {
        "min_physicians_eligible": 5,
        "critical_min_physicians_eligible": 2,
        "max_dropna_fraction": 0.25,
        "critical_max_dropna_fraction": 0.50,
        "min_matching_covariates": 3,
        "max_zero_variance_covariates": 2,
        "max_imputed_cell_fraction": 0.20,
        "max_outlier_cell_fraction": 0.01,
        "critical_max_outlier_cell_fraction": 0.05,
        "max_mean_pair_smd": 1.00,
        "critical_max_mean_pair_smd": 1.50,
        "max_worst_pair_smd": 4.00,
        "critical_max_worst_pair_smd": 6.00,
        "max_frac_pairs_worst_smd_gt_2": 0.50,
        "min_spearman_euclidean_mahalanobis": 0.65,
        "critical_min_spearman_euclidean_mahalanobis": 0.50,
        "min_spearman_euclidean_rf": 0.65,
        "max_mean_delta_euclidean_mahalanobis": 0.08,
        "max_p90_delta_euclidean_mahalanobis": 0.15,
        "max_physician_delta_euclidean_mahalanobis": 0.20,
        "max_mean_delta_euclidean_rf": 0.10,
        "max_physician_delta_euclidean_rf": 0.40,
        "max_rf_minus_euclidean_mean": 0.08,
        "max_learning_minus_euclidean_mean": 0.08,
        "max_bernoulli_ratio_p90": 1.35,
        "max_bernoulli_ratio_any_physician": 1.50,
        "critical_max_bernoulli_ratio": 2.00,
        "max_discordance_rate": 0.80,
        "min_synthetic_discordance_rank_correlation": 0.30,
    },
    "synthetic_null": {
        "min_physicians_eligible": 5,
        "critical_min_physicians_eligible": 2,
        "max_dropna_fraction": 0.25,
        "critical_max_dropna_fraction": 0.50,
        "min_matching_covariates": 3,
        "max_zero_variance_covariates": 2,
        "max_imputed_cell_fraction": 0.20,
        "max_outlier_cell_fraction": 0.01,
        "critical_max_outlier_cell_fraction": 0.05,
        "max_mean_pair_smd": 1.00,
        "critical_max_mean_pair_smd": 1.50,
        "max_worst_pair_smd": 4.00,
        "critical_max_worst_pair_smd": 6.00,
        "max_frac_pairs_worst_smd_gt_2": 0.50,
        "min_spearman_euclidean_mahalanobis": 0.60,
        "critical_min_spearman_euclidean_mahalanobis": 0.45,
        "min_spearman_euclidean_rf": 0.60,
        "max_mean_delta_euclidean_mahalanobis": 0.10,
        "max_p90_delta_euclidean_mahalanobis": 0.18,
        "max_physician_delta_euclidean_mahalanobis": 0.25,
        "max_mean_delta_euclidean_rf": 0.12,
        "max_physician_delta_euclidean_rf": 0.45,
        "max_rf_minus_euclidean_mean": 0.10,
        "max_learning_minus_euclidean_mean": 0.10,
        "max_bernoulli_ratio_p90": 1.15,
        "max_bernoulli_ratio_any_physician": 1.30,
        "critical_max_bernoulli_ratio": 1.60,
        "max_discordance_rate": 0.55,
        "min_synthetic_discordance_rank_correlation": 0.20,
    },
}

SYNTHETIC_NULL_STRATEGIES = frozenset({
    "realistic_heterogeneous_patients",
})
SYNTHETIC_SIGNAL_STRATEGIES = frozenset({
    "score2_five_groups_heter_patients",
    "realistic_rule_three_groups",
})


def resolve_profile_name(generation_strategy: str, dataset_source: str | None = None) -> str:
    """Pick a threshold profile from generation strategy."""
    del dataset_source
    if generation_strategy in SYNTHETIC_NULL_STRATEGIES:
        return "synthetic_null"
    if generation_strategy in SYNTHETIC_SIGNAL_STRATEGIES:
        return "synthetic_signal"
    if "heter" in generation_strategy:
        return "synthetic_signal"
    if "homo" in generation_strategy:
        return "synthetic_null"
    return "synthetic_signal"


def resolve_thresholds(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Merge YAML overrides onto the selected profile defaults."""
    validation_cfg = config.get("validation") or {}
    profile_name = str(validation_cfg.get("calibration_profile", "auto")).strip().lower()
    if profile_name in {"", "auto"}:
        profile_name = resolve_profile_name(
            str(config.get("generation_strategy", "")),
            config.get("dataset_source"),
        )
    if profile_name not in DEFAULT_PROFILES:
        profile_name = "synthetic_signal"
    base = deepcopy(DEFAULT_PROFILES[profile_name])
    overrides = validation_cfg.get("thresholds") or {}
    if isinstance(overrides, dict):
        base.update(overrides)
    return profile_name, base
