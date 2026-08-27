# -*- coding: utf-8 -*-
"""
Prism analysis module: quantification of intra-physician prescription variability.

Le pipeline principal (`run_full_analysis`) prépare une base matching, exécute les
méthodes activées via ``config["methods"]``, fusionne optionnellement un score
composite, ajuste un GLMM (OLRE) si activé, puis produit figures multi-méthodes
et exports CSV.

Des méthodes historiques restent disponibles individuellement dans la classe.
"""
from __future__ import annotations

import os
import json
import logging
import textwrap
import time
import warnings
import random
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple
from contextlib import contextmanager, redirect_stdout

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpatheffects
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

import statsmodels.genmod.bayes_mixed_glm as _smbm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from adjustText import adjust_text

from src.prism.experiment_paths import DEFAULT_EXPERIMENTS_ROOT
from src.prism.dataset_utils.imputation import (
    DEFAULT_MICE_N_NEAREST_FEATURES,
    mice_impute,
    round_discrete_columns,
)
FEATURE_SELECTION_METHODS = frozenset({"variance", "anova", "manual"})

from src.prism.analysis.matching_core import (
    _discordance_rate_from_nn_local,
    _glmm_fixed_effect_weights,
    _greedy_nearest_neighbor_pairs,
    _n_discordant_from_nn_local,
    match_reuse_stats,
    nearest_neighbor_with_replacement,
)
from src.prism.analysis.plotting import (
    _color_for_physician,
    _scatter_hue_physician_categorical,
    _scatter_hue_physician_continuous,
    _truncate_physician_id_display,
    format_physician_display_label,
    physician_color_map,
    physician_face_colors,
    plot_questionnaire_completion,
    questionnaire_display_label,
    USAGE_BOTH,
    USAGE_GLMM,
    USAGE_MATCHING,
)
from src.prism.dataset_utils.synthetic_generator import (
    get_generation_rule_mask,
)
from src.prism.analysis.pair_covariate_diagnostics import write_pair_covariate_diagnostics
from src.prism.clinical_review import build_clinical_review_html_safe, paired_review_html_path
from src.prism.snapshot_utils import SnapshotSaver
from src.prism.validation.run_validation import run_validation_report

# -----------------------------------------------------------------------------
# Vue d'ensemble rapide du pipeline principal
# -----------------------------------------------------------------------------
# 1) run_matching_basis_prep()
#    - Construit le même sous-ensemble complet de lignes que l’ancienne entrée GLMM
#      (sans ajuster de modèle).
#
# 2) Random Forest proximity / learned RF weights / mutual information
#    - Trois discordances par médecin sur ce sous-ensemble.
#
# 3) run_ensemble_matching_analysis()
#    - Moyenne des trois scores uniquement ; colonnes intermédiaires retirées.
#
# 4) plot_glmm_vs_matching_comparison()
#    - Figure synthétique pour ``ensemble_matching`` (nom historique conservé).
# -----------------------------------------------------------------------------

# Colonnes des trois méthodes conservées avant agrégation finale.
THREE_METHOD_DISCORDANCE_COLS: tuple[str, ...] = (
    "discordance_rate_rf_matching",
    "discordance_rate_learning",
    "discordance_rate_mutual_info",
)

# Labels des trois groupes de médecins par tertile de taux de prescription
# (ordre canonique low → medium → high, utilisé en x-axis de la figure finale).
PRESCRIBER_GROUP_LABELS: tuple[str, str, str] = (
    "low_prescribers",
    "medium_prescribers",
    "high_prescribers",
)

# Marqueur vert (cercle) sur les panneaux continus : ordonnée fixe dans [0, 0.2]
# pour signaler la prévalence d'indication clinique sur l'axe des taux.
INDICATION_PREVALENCE_MARKER_Y: float = 0.1
INDICATION_PREVALENCE_MARKER_RADIUS: float = 0.05
INDICATION_PREVALENCE_MARKER_COLOR: str = "#b8f0b8"
INDICATION_PREVALENCE_MARKER_ALPHA: float = 0.3
INDICATION_PREVALENCE_MARKER_LABEL: str = "Clinical indication prevalence"

# Courbe de référence Bernoulli D = 2p(1-p) superposée aux panneaux pour
# situer chaque médecin par rapport à la discordance attendue si ses décisions
# étaient des Bernoulli indépendants de paramètre p = taux de prescription.
BERNOULLI_REFERENCE_LABEL: str = "Bernoulli D = 2p(1−p)"
BERNOULLI_REFERENCE_COLOR: str = "#222831"
BERNOULLI_REFERENCE_LINESTYLE: str = "--"
BERNOULLI_REFERENCE_LINEWIDTH: float = 1.6


def _bernoulli_discordance(p: np.ndarray | float) -> np.ndarray | float:
    """Discordance attendue pour des décisions Bernoulli i.i.d. : ``D = 2 p (1 - p)``."""
    arr = np.asarray(p, dtype=float)
    return 2.0 * arr * (1.0 - arr)


BERNOULLI_RESIDUAL_SE_LABEL: str = "±1 SE on D (m matched pairs)"
GAP_RATIO_SE_LABEL: str = "±1 SE on ratio (from D)"


def _discordance_standard_error(
    discordance: float,
    n_pairs: object,
) -> float | None:
    """Standard error of discordance ``D`` from ``m`` matched pairs: ``sqrt(D(1-D)/m)``.

    Returns ``None`` when ``n_pairs`` is missing, non-finite, or ``<= 0``, or when
    ``discordance`` is non-finite. ``D`` is clipped to ``[0, 1]`` before the formula.
    """
    n = _as_optional_int(n_pairs)
    if n is None or n <= 0 or not np.isfinite(discordance):
        return None
    d = float(np.clip(float(discordance), 0.0, 1.0))
    se = np.sqrt(d * (1.0 - d) / float(n))
    if not np.isfinite(se):
        return None
    return float(se)


def _gap_ratio_standard_error(
    discordance: float,
    prescription_rate: float,
    n_pairs: object,
) -> float | None:
    """Standard error of ``100 × D / (2p(1−p))`` with ``p`` held at its observed value.

    Propagates ``SE(D)`` as ``100 / (2p(1−p)) × SE(D)``.
    """
    se_d = _discordance_standard_error(discordance, n_pairs)
    if se_d is None or not np.isfinite(prescription_rate):
        return None
    bern = float(_bernoulli_discordance(prescription_rate))
    if not np.isfinite(bern) or bern <= 0.0:
        return None
    se_ratio = 100.0 / bern * se_d
    if not np.isfinite(se_ratio):
        return None
    return float(se_ratio)






def _draw_bernoulli_reference_continuous(ax, *, label_once: bool = True) -> None:
    """Trace la courbe ``D = 2p(1-p)`` sur un axe ``x`` continu dans ``[0, 1]``."""
    xs = np.linspace(0.0, 1.0, 200)
    ys = _bernoulli_discordance(xs)
    ax.plot(
        xs,
        ys,
        color=BERNOULLI_REFERENCE_COLOR,
        linestyle=BERNOULLI_REFERENCE_LINESTYLE,
        linewidth=BERNOULLI_REFERENCE_LINEWIDTH,
        zorder=6,
        label=BERNOULLI_REFERENCE_LABEL if label_once else None,
    )


def _draw_bernoulli_reference_tertile(
    ax,
    df: pd.DataFrame,
    *,
    labels: list[str],
    rate_col: str = "prescription_rate",
    group_col: str = "prescriber_group",
) -> None:
    """Place la courbe Bernoulli sur le panneau tertile.

    Pour chaque tertile, on calcule ``p = moyenne(prescription_rate)`` du groupe,
    puis on positionne le point à ``D = 2p(1-p)`` sur l'abscisse du tertile
    (0/1/2). Les points sont reliés par un trait pour matérialiser la courbe.
    """
    if rate_col not in df.columns or group_col not in df.columns:
        return
    means = (
        df.dropna(subset=[rate_col, group_col])
        .groupby(group_col, observed=False)[rate_col]
        .mean()
    )
    xs: list[float] = []
    ys: list[float] = []
    for i, lbl in enumerate(labels):
        if lbl in means.index:
            p = float(means.loc[lbl])
            if np.isfinite(p):
                xs.append(float(i))
                ys.append(float(_bernoulli_discordance(p)))
    if not xs:
        return
    ax.plot(
        xs,
        ys,
        color=BERNOULLI_REFERENCE_COLOR,
        linestyle=BERNOULLI_REFERENCE_LINESTYLE,
        linewidth=BERNOULLI_REFERENCE_LINEWIDTH,
        marker="D",
        markersize=7,
        markerfacecolor=BERNOULLI_REFERENCE_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=6,
        label=BERNOULLI_REFERENCE_LABEL,
    )

# Display names pour les sorties figures / agrégations.
METHOD_PLOT_SPECS = [
    ("discordance_rate_manual", "Manual Pairing"),
    ("discordance_rate_euclidean", "Euclidean Distance Pairing"),
    ("discordance_rate_mahalanobis", "Mahalanobis Distance Pairing"),
    ("discordance_rate_learning", "Learned Weights Euclidean Distance Pairing"),
    ("discordance_rate_rf_matching", "Random Forest Proximity Pairing"),
    ("discordance_rate_mutual_info", "Mutual Information"),
    ("overdispersion_local", "Generalized Mixed Model"),
    ("ensemble_matching", "Ensemble Method"),
]
METHOD_DISPLAY_NAMES = dict(METHOD_PLOT_SPECS)

# Subdirectory under results_dir / "plots" / ...
PLOT_SUBDIR_METHOD_COMPARISON = "method_comparison"
PLOT_SUBDIR_BERNOULLI_RESIDUAL = "bernoulli_residual"
PLOT_SUBDIR_PHYSICIAN_CASELOAD = "physician_caseload"
PLOT_SUBDIR_QUESTIONNAIRE = "questionnaire"
PATIENTS_PER_PHYSICIAN_CUTOFF_LINE_LABEL = "min patients per physician"
CLINICAL_REVIEW_PAIRS_PREVIEW_MAX = 20

GLMM_FIXED_EFFECTS_MATCHING = "matching"
GLMM_FIXED_EFFECTS_STATIN_RELEVANT = "statin_relevant"
GLMM_FIXED_EFFECTS_AUTO = "auto"

STATIN_RELEVANT_GLMM_COVARIATES: tuple[str, ...] = (
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

LEGACY_DEFAULT_METHOD_FLAGS: dict[str, bool] = {
    "euclidean": False,
    "mahalanobis": False,
    "rf_matching": True,
    "learning": True,
    "mutual_info": True,
    "ensemble": True,
    "manual_pairing": False,
}


@contextmanager
def _live_glmm_progress(logger: logging.Logger, log_every_seconds: float = 2.0):
    """Stream live progress of statsmodels' VB optimizer to ``logger``.

    ``BinomialBayesMixedGLM.fit_vb`` does not expose a callback, but it
    delegates to ``scipy.optimize.minimize`` via the ``minimize`` symbol
    imported in :mod:`statsmodels.genmod.bayes_mixed_glm`. We monkey-patch
    that symbol for the duration of the ``with`` block and inject:

      * a wrapped objective/jacobian that caches the latest ``-ELBO`` and
        ``|grad|`` (no extra evaluations);
      * a callback that prints one log line per iteration (rate-limited to
        ``log_every_seconds`` to stay readable on big problems).

    The original ``minimize`` is always restored on exit.
    """
    original_minimize = _smbm.minimize

    def patched_minimize(fun, x0, jac=None, **kwargs):
        state = {
            "n_iter": 0,
            "n_fcalls": 0,
            "n_gcalls": 0,
            "t0": time.perf_counter(),
            "t_last_log": -float("inf"),
            "last_f": float("nan"),
            "last_gnorm": float("nan"),
            "best_f": float("inf"),
        }

        def wrapped_fun(x):
            v = float(fun(x))
            state["n_fcalls"] += 1
            state["last_f"] = v
            if v < state["best_f"]:
                state["best_f"] = v
            return v

        if jac is not None:
            def wrapped_jac(x):
                g = jac(x)
                state["n_gcalls"] += 1
                try:
                    state["last_gnorm"] = float(np.linalg.norm(g))
                except (TypeError, ValueError):
                    state["last_gnorm"] = float("nan")
                return g
        else:
            wrapped_jac = None

        user_callback = kwargs.pop("callback", None)

        def progress_callback(xk, *_args, **_kwargs):
            state["n_iter"] += 1
            elapsed = time.perf_counter() - state["t0"]
            should_log = (
                state["n_iter"] == 1
                or (elapsed - state["t_last_log"]) >= log_every_seconds
            )
            if should_log:
                state["t_last_log"] = elapsed
                logger.info(
                    "  GLMM fit_vb | iter %4d | -ELBO=%.4f (best=%.4f) "
                    "| |grad|=%.3e | f-calls=%d | g-calls=%d | t=%.1fs",
                    state["n_iter"],
                    state["last_f"],
                    state["best_f"],
                    state["last_gnorm"],
                    state["n_fcalls"],
                    state["n_gcalls"],
                    elapsed,
                )
                for h in logger.handlers:
                    try:
                        h.flush()
                    except Exception:  # pragma: no cover - best effort
                        pass
            if user_callback is not None:
                user_callback(xk)

        result = original_minimize(
            wrapped_fun,
            x0,
            jac=wrapped_jac,
            callback=progress_callback,
            **kwargs,
        )

        elapsed = time.perf_counter() - state["t0"]
        logger.info(
            "  GLMM fit_vb | DONE in %.1fs | iters=%d | f-calls=%d | g-calls=%d "
            "| -ELBO=%.4f | success=%s | message=%s",
            elapsed,
            state["n_iter"],
            state["n_fcalls"],
            state["n_gcalls"],
            state["last_f"],
            getattr(result, "success", "?"),
            getattr(result, "message", "?"),
        )
        return result

    _smbm.minimize = patched_minimize
    try:
        yield
    finally:
        _smbm.minimize = original_minimize


def _n_pairs_column_name(score_col: str) -> str:
    """Column storing per-physician pair counts for a discordance score column."""
    return f"n_pairs_{score_col}"


def _n_discordant_column_name(score_col: str) -> str:
    """Column storing per-physician discordant-pair counts for a score column."""
    return f"n_discordant_{score_col}"


def _coverage_column_name(score_col: str) -> str:
    """Column storing per-physician matching coverage (covered / panel size)."""
    return f"coverage_{score_col}"


def _n_covered_column_name(score_col: str) -> str:
    """Column storing per-physician count of patients with an admissible NN match."""
    return f"n_covered_{score_col}"


def _n_patients_reused_column_name(score_col: str) -> str:
    """Column storing per-physician count of match targets used more than once."""
    return f"n_patients_reused_{score_col}"


def _n_reuse_assignments_column_name(score_col: str) -> str:
    """Column storing per-physician extra NN assignments due to reuse."""
    return f"n_reuse_assignments_{score_col}"


def _reuse_rate_column_name(score_col: str) -> str:
    """Column storing per-physician fraction of pair slots filled by reuse."""
    return f"reuse_rate_{score_col}"


def _max_reuse_count_column_name(score_col: str) -> str:
    """Column storing per-physician maximum times a single patient was reused as match."""
    return f"max_reuse_count_{score_col}"


def _as_optional_int(value: object) -> int | None:
    """Coerce ``value`` to ``int``, returning ``None`` for missing/non-finite input."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _n_pairs_for_score(row: object, score_col: str) -> int | None:
    """Read ``n_pairs_{score_col}`` from a Series/Mapping row, if present."""
    col = _n_pairs_column_name(score_col)
    try:
        if col not in row:  # type: ignore[operator]
            return None
        return _as_optional_int(row[col])  # type: ignore[index]
    except (TypeError, KeyError, ValueError):
        return None


def _n_discordant_for_score(row: object, score_col: str) -> int | None:
    """Read ``n_discordant_{score_col}`` from a Series/Mapping row, if present."""
    col = _n_discordant_column_name(score_col)
    try:
        if col not in row:  # type: ignore[operator]
            return None
        return _as_optional_int(row[col])  # type: ignore[index]
    except (TypeError, KeyError, ValueError):
        return None


def _pair_count_series(
    df: pd.DataFrame, score_col: str, *, column_name_fn: Callable[[str], str]
) -> pd.Series:
    """Return a numeric series for ``column_name_fn(score_col)``, or NA if missing."""
    col = column_name_fn(score_col)
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index)


def _physician_label_with_counts(
    physician_id: object,
    n_prescriptions: object,
    n_patients: object,
    n_pairs: object = None,
    n_discordant: object = None,
) -> str:
    """Return ``"<id> X/n"`` (optionally ``" (d/k)"`` or ``" (k pairs)"``) for a label.

    ``X`` is the number of target-medication prescriptions issued by the
    physician; ``n`` is the total number of patients they followed. When
    both ``n_discordant`` and ``n_pairs`` are provided, append ``" (d/k)"``.
    When only ``n_pairs`` is provided, append ``" (k pairs)"``. Missing values
    gracefully degrade to ``"<id>"`` or ``"<id> (X)"`` / ``"<id> (/n)"``.
    """
    base = _truncate_physician_id_display(physician_id)

    x = _as_optional_int(n_prescriptions)
    n = _as_optional_int(n_patients)
    if x is None and n is None:
        label = base
    elif x is not None and n is not None:
        label = f"{base} {x}/{n}"
    elif x is None:
        label = f"{base} (/{n})"
    else:
        label = f"{base} ({x})"

    d = _as_optional_int(n_discordant)
    k = _as_optional_int(n_pairs)
    if d is not None and k is not None:
        label = f"{label} ({d}/{k})"
    elif k is not None:
        label = f"{label} ({k} pairs)"
    return label


def _adjust_physician_point_labels(
    ax,
    texts: list,
    xs,
    ys,
    *,
    objects=None,
) -> None:
    """Repel physician labels from points using adjustText 1.3 API.

    Must be called **after** final axis limits are set: adjustText converts
    display-space offsets into data coordinates using the current axes scales.

    Prefer explicit ``xs``/``ys`` over ``objects=`` PathCollections when each
    physician is drawn as its own single-point scatter: those collections often
    report non-finite window extents and crash KDTree inside adjustText.
    """
    if not texts:
        return
    with open(os.devnull, "w") as f, redirect_stdout(f), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adjust_text(
            texts,
            x=np.asarray(xs, dtype=float),
            y=np.asarray(ys, dtype=float),
            objects=objects,
            ax=ax,
            expand=(1.4, 1.8),
            force_text=(0.5, 1.0),
            force_static=(0.5, 1.0),
            force_pull=(0.005, 0.005),
            min_arrow_len=6,
            arrowprops=dict(
                arrowstyle="-",
                color="0.45",
                lw=0.6,
                shrinkA=1,
                shrinkB=6,
            ),
            ensure_inside_axes=True,
            expand_axes=False,
            time_lim=3.0,
        )


def _physician_color_map_from_df(
    df: pd.DataFrame | None,
    color_map: Mapping | None = None,
) -> dict:
    """Experiment-wide physician colours: explicit map, else roster of ``df``."""
    if color_map:
        return {str(k): v for k, v in color_map.items()}
    if df is None or df.empty or "physician" not in df.columns:
        return {}
    return physician_color_map(df["physician"])




def _plot_tertile_panel_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    score_col: str = "ensemble_matching",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    point_size: float = 8.0,
    label_fontsize: int = 7,
    color_map: Mapping | None = None,
) -> bool:
    """Draw the tertile-by-prescriber violin + scatter panel on ``ax``.

    Returns ``True`` when something was drawn (i.e. the input has the expected
    columns and at least one physician with an assigned tertile), ``False``
    otherwise so the caller can skip / fall back.
    """
    if (
        score_col not in df.columns
        or "prescriber_group" not in df.columns
        or "physician" not in df.columns
    ):
        return False

    labels = list(PRESCRIBER_GROUP_LABELS)
    n_patients_series = (
        pd.to_numeric(df["n_patients"], errors="coerce")
        if "n_patients" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    n_prescriptions_series = (
        pd.to_numeric(df["n_prescriptions"], errors="coerce")
        if "n_prescriptions" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    n_pairs_series = _pair_count_series(df, score_col, column_name_fn=_n_pairs_column_name)
    n_discordant_series = _pair_count_series(
        df, score_col, column_name_fn=_n_discordant_column_name
    )
    df_plot = pd.DataFrame({
        "physician": df["physician"].astype(str),
        "Group": df["prescriber_group"].astype(str),
        "Score": pd.to_numeric(df[score_col], errors="coerce"),
        "NPatients": n_patients_series.values,
        "NPrescriptions": n_prescriptions_series.values,
        "NPairs": n_pairs_series.values,
        "NDiscordant": n_discordant_series.values,
    })
    df_plot = df_plot[df_plot["Group"].isin(labels)].copy()
    if df_plot.empty:
        return False
    resolved_colors = _physician_color_map_from_df(df, color_map)

    if df_plot["Score"].nunique(dropna=True) > 1:
        sns.violinplot(
            data=df_plot,
            x="Group",
            y="Score",
            order=labels,
            inner=None,
            color="lightgray",
            alpha=0.5,
            ax=ax,
        )
    _scatter_hue_physician_categorical(
        ax,
        df_plot,
        "Group",
        "Score",
        "physician",
        order=labels,
        color_map=resolved_colors,
        size=point_size,
        linewidth=1.0,
        jitter=0.12,
    )
    x_dict = {group: i for i, group in enumerate(labels)}
    texts = [
        ax.text(
            x=x_dict.get(row["Group"], 0),
            y=row["Score"],
            s=_physician_label_with_counts(
                row["physician"],
                row["NPrescriptions"],
                row["NPatients"],
                row["NPairs"],
                row["NDiscordant"],
            ),
            fontsize=label_fontsize,
            color="black",
            alpha=0.8,
        )
        for _, row in df_plot.iterrows()
        if pd.notna(row["Score"]) and row["Group"] in x_dict
    ]
    with open(os.devnull, "w") as f, redirect_stdout(f), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adjust_text(texts, ax=ax, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2))

    _draw_bernoulli_reference_tertile(ax, df, labels=labels)

    counts = df_plot.groupby("Group", observed=False)["physician"].nunique()
    xtick_labels = [f"{label}\n(n={int(counts.get(label, 0))})" for label in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(xtick_labels)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Prescriber tertile (by prescription rate)" if show_xlabel else "", fontsize=12)
    ax.set_ylabel("Discordance rate" if show_ylabel else "", fontsize=12)

    finite_scores = pd.to_numeric(df_plot["Score"], errors="coerce").to_numpy(dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size > 0:
        observed_max = float(np.max(finite_scores))
        y_upper = max(min(1.0, observed_max + 0.05), 0.55)
    else:
        y_upper = 0.55
    ax.set_ylim(0.0, y_upper)
    return True












BERNOULLI_RESIDUAL_ZERO_LABEL: str = "Bernoulli baseline (Δ = 0)"
PERFECT_CONCORDANCE_RESIDUAL_LABEL: str = "Perfect concordance (D = 0)"
PERFECT_CONCORDANCE_GAP_MIN_PRESCRIPTION_RATE: float = 0.05
BERNOULLI_BASELINE_RATIO_PCT_LABEL: str = "Bernoulli baseline (100%)"


def _draw_perfect_concordance_residual_reference(ax, *, label_once: bool = True) -> None:
    """Trace ``Δ = −2p(1−p)`` : résidu si la discordance observée ``D`` vaut 0."""
    xs = np.linspace(0.0, 1.0, 200)
    ys = -_bernoulli_discordance(xs)
    ax.plot(
        xs,
        ys,
        color=BERNOULLI_REFERENCE_COLOR,
        linestyle=BERNOULLI_REFERENCE_LINESTYLE,
        linewidth=BERNOULLI_REFERENCE_LINEWIDTH,
        zorder=6,
        label=PERFECT_CONCORDANCE_RESIDUAL_LABEL if label_once else None,
    )


def _draw_bernoulli_residual_se_bars(
    ax,
    df_plot: pd.DataFrame,
    *,
    color_map: Mapping | None = None,
) -> bool:
    """Vertical ``±1 SE`` bars on ``Δ``; ``p`` held at the observed x-coordinate."""
    drawn = False
    for _, row in df_plot.iterrows():
        se = _discordance_standard_error(float(row["Y"]), row["NPairs"])
        if se is None:
            continue
        delta = float(row["Delta"])
        color = _color_for_physician(row["physician"], color_map or {})
        ax.errorbar(
            float(row["X"]),
            delta,
            yerr=se,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=3.0,
            capthick=1.0,
            alpha=0.85,
            zorder=4,
            label=BERNOULLI_RESIDUAL_SE_LABEL if not drawn else None,
        )
        drawn = True
    return drawn


def _plot_bernoulli_residual_panel_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    score_col: str = "ensemble_matching",
    x_col: str = "prescription_rate",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    point_size: float = 8.0,
    label_fontsize: int = 7,
    color_map: Mapping | None = None,
) -> bool:
    """Scatter of ``discordance − 2p(1−p)`` vs prescription rate for each physician.

    Each point is one physician at ``(p, D − 2p(1−p))`` where ``p`` is the
    prescription rate and ``D`` the composite discordance score. A horizontal
    reference at ``Δ = 0`` marks the Bernoulli baseline; a dashed curve at
    ``Δ = −2p(1−p)`` marks perfect concordance (``D = 0``). When ``n_pairs``
    is present, a vertical ``±1 SE`` bar on ``D`` (``p`` fixed) is drawn at
    each point.

    Returns ``True`` when at least one physician was plotted, ``False`` otherwise.
    """
    if (
        score_col not in df.columns
        or x_col not in df.columns
        or "physician" not in df.columns
    ):
        return False

    n_patients_series = (
        pd.to_numeric(df["n_patients"], errors="coerce")
        if "n_patients" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    n_prescriptions_series = (
        pd.to_numeric(df["n_prescriptions"], errors="coerce")
        if "n_prescriptions" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    n_pairs_series = _pair_count_series(df, score_col, column_name_fn=_n_pairs_column_name)
    n_discordant_series = _pair_count_series(
        df, score_col, column_name_fn=_n_discordant_column_name
    )
    df_plot = pd.DataFrame({
        "physician": df["physician"].astype(str),
        "X": pd.to_numeric(df[x_col], errors="coerce"),
        "Y": pd.to_numeric(df[score_col], errors="coerce"),
        "NPatients": n_patients_series.values,
        "NPrescriptions": n_prescriptions_series.values,
        "NPairs": n_pairs_series.values,
        "NDiscordant": n_discordant_series.values,
    })
    df_plot = df_plot.dropna(subset=["X", "Y"]).copy()
    if df_plot.empty:
        return False

    bernoulli_expected = _bernoulli_discordance(df_plot["X"].to_numpy(dtype=float))
    df_plot["Delta"] = df_plot["Y"].to_numpy(dtype=float) - np.asarray(
        bernoulli_expected, dtype=float
    )
    se_ys: list[float] = []
    for _, row in df_plot.iterrows():
        se = _discordance_standard_error(float(row["Y"]), row["NPairs"])
        if se is None:
            continue
        delta = float(row["Delta"])
        se_ys.extend([delta - se, delta + se])
    resolved_colors = _physician_color_map_from_df(df, color_map)

    se_drawn = _draw_bernoulli_residual_se_bars(ax, df_plot, color_map=resolved_colors)
    _scatter_hue_physician_continuous(
        ax,
        df_plot,
        "X",
        "Delta",
        "physician",
        color_map=resolved_colors,
        size=point_size,
        linewidth=1.0,
    )

    # Final limits before labels: adjustText converts display offsets using the
    # current axes scales; changing lims afterwards would recompress them.
    ax.set_xlim(0.0, 1.0)
    finite_deltas = df_plot["Delta"].to_numpy(dtype=float)
    finite_deltas = finite_deltas[np.isfinite(finite_deltas)]
    curve_ys = -_bernoulli_discordance(np.linspace(0.0, 1.0, 200))
    extra = [
        ys
        for ys in (
            finite_deltas,
            curve_ys,
            np.asarray(se_ys, dtype=float),
        )
        if np.size(ys) > 0
    ]
    reference_ys = np.concatenate(extra) if extra else curve_ys
    if reference_ys.size > 0:
        delta_max = float(np.max(np.abs(reference_ys)))
        y_pad = max(0.05, delta_max * 0.15)
        y_upper = min(1.0, float(np.max(reference_ys)) + y_pad)
        y_lower = max(-1.0, float(np.min(reference_ys)) - y_pad)
        if y_lower >= 0.0:
            y_lower = -y_pad
        if y_upper <= 0.0:
            y_upper = y_pad
    else:
        y_lower, y_upper = -0.1, 0.1
    ax.set_ylim(y_lower, y_upper)

    ax.axhline(
        0.0,
        color=BERNOULLI_REFERENCE_COLOR,
        linestyle=BERNOULLI_REFERENCE_LINESTYLE,
        linewidth=BERNOULLI_REFERENCE_LINEWIDTH,
        zorder=6,
        label=BERNOULLI_RESIDUAL_ZERO_LABEL,
    )
    _draw_perfect_concordance_residual_reference(ax)

    if se_drawn:
        se_handles = [
            handle
            for handle, lab in zip(*ax.get_legend_handles_labels())
            if lab == BERNOULLI_RESIDUAL_SE_LABEL
        ]
        if se_handles:
            ax.legend(
                handles=se_handles,
                loc="upper right",
                fontsize=8,
                framealpha=0.85,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Prescription rate" if show_xlabel else "", fontsize=12)
    ax.set_ylabel("Discordance − 2p(1−p)" if show_ylabel else "", fontsize=12)

    label_path_effects = [
        mpatheffects.withStroke(linewidth=2.0, foreground="white"),
    ]
    texts = [
        ax.text(
            x=float(row["X"]),
            y=float(row["Delta"]),
            s=_physician_label_with_counts(
                row["physician"],
                row["NPrescriptions"],
                row["NPatients"],
                row["NPairs"],
                row["NDiscordant"],
            ),
            fontsize=label_fontsize,
            color="black",
            alpha=0.8,
            zorder=10,
            path_effects=label_path_effects,
        )
        for _, row in df_plot.iterrows()
    ]
    _adjust_physician_point_labels(
        ax,
        texts,
        df_plot["X"].to_numpy(dtype=float),
        df_plot["Delta"].to_numpy(dtype=float),
        # Per-physician PathCollections return non-finite window extents, so
        # repulsion uses explicit (x, y) point coordinates instead of objects=.
        objects=None,
    )
    return True


def plot_medication_bernoulli_residual(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Discordance excess vs. Bernoulli baseline",
    score_col: str = "ensemble_matching",
    x_col: str = "prescription_rate",
    suptitle: str | None = None,
    logger: Optional[logging.Logger] = None,
    color_map: Mapping | None = None,
) -> Optional[Path]:
    """Save a single-medication figure of ``D − 2p(1−p)`` vs prescription rate."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    drawn = _plot_bernoulli_residual_panel_on_ax(
        ax,
        df,
        title=title,
        score_col=score_col,
        x_col=x_col,
        color_map=color_map,
    )
    if not drawn:
        plt.close(fig)
        if logger is not None:
            logger.warning(
                "plot_medication_bernoulli_residual skipped: no renderable physicians."
            )
        return None

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info("Saved Bernoulli residual plot to %s", out_path)
    return out_path




def plot_multi_method_bernoulli_residual_overview(
    df: pd.DataFrame,
    score_cols: list[str],
    out_path: str | Path,
    *,
    x_col: str = "prescription_rate",
    suptitle: str = (
        "Discordance excess vs. Bernoulli baseline (D − 2p(1−p)) — all methods"
    ),
    logger: Optional[logging.Logger] = None,
    color_map: Mapping | None = None,
) -> Optional[Path]:
    """Render one ``D − 2p(1−p)`` panel per discordance method in a single figure."""
    if not score_cols:
        if logger is not None:
            logger.warning(
                "plot_multi_method_bernoulli_residual_overview skipped: no score columns."
            )
        return None

    n_panels = len(score_cols)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0 * n_cols, 7.0 * n_rows),
        squeeze=False,
    )
    axes_flat = axes_grid.flatten()
    resolved_colors = _physician_color_map_from_df(df, color_map)
    drawn = 0
    for ax, score_col in zip(axes_flat, score_cols):
        title = METHOD_DISPLAY_NAMES.get(score_col, score_col)
        ok = _plot_bernoulli_residual_panel_on_ax(
            ax,
            df,
            title=title,
            score_col=score_col,
            x_col=x_col,
            show_xlabel=True,
            show_ylabel=True,
            color_map=resolved_colors,
        )
        if ok:
            drawn += 1
        else:
            ax.axis("off")
            ax.set_title(
                f"{title}\n(no prescription-rate data)",
                fontsize=14,
                fontweight="bold",
            )
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    if drawn == 0:
        plt.close(fig)
        if logger is not None:
            logger.warning(
                "plot_multi_method_bernoulli_residual_overview: nothing renderable "
                "across %d method(s).",
                n_panels,
            )
        return None

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info("Saved multi-method Bernoulli residual overview to %s", out_path)
    return out_path


def _compute_perfect_concordance_gap_frame(
    df: pd.DataFrame,
    *,
    score_col: str,
    x_col: str = "prescription_rate",
    min_prescription_rate: float = PERFECT_CONCORDANCE_GAP_MIN_PRESCRIPTION_RATE,
) -> pd.DataFrame:
    """Return per-physician vertical gap from the Bernoulli residual point to ``D = 0``.

    Each row includes ``gap_to_perfect_concordance`` = ``Δ − (−2p(1−p))`` = ``D`` and
    ``gap_to_bell_ratio_pct`` = ``100 × D / (2p(1−p))``.
    Physicians with ``prescription_rate < min_prescription_rate`` are excluded.
    """
    if (
        score_col not in df.columns
        or x_col not in df.columns
        or "physician" not in df.columns
    ):
        return pd.DataFrame()

    n_patients_series = (
        pd.to_numeric(df["n_patients"], errors="coerce")
        if "n_patients" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    n_prescriptions_series = (
        pd.to_numeric(df["n_prescriptions"], errors="coerce")
        if "n_prescriptions" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    n_pairs_series = _pair_count_series(df, score_col, column_name_fn=_n_pairs_column_name)
    n_discordant_series = _pair_count_series(
        df, score_col, column_name_fn=_n_discordant_column_name
    )
    out = pd.DataFrame({
        "physician": df["physician"].astype(str),
        "prescription_rate": pd.to_numeric(df[x_col], errors="coerce"),
        "discordance": pd.to_numeric(df[score_col], errors="coerce"),
        "n_patients": n_patients_series.values,
        "n_prescriptions": n_prescriptions_series.values,
        "n_pairs": n_pairs_series.values,
        "n_discordant": n_discordant_series.values,
    })
    out = out.dropna(subset=["prescription_rate", "discordance"]).copy()
    if out.empty:
        return out
    out = out[out["prescription_rate"] >= min_prescription_rate].copy()
    if out.empty:
        return out

    bernoulli_expected = np.asarray(
        _bernoulli_discordance(out["prescription_rate"].to_numpy(dtype=float)),
        dtype=float,
    )
    delta = out["discordance"].to_numpy(dtype=float) - bernoulli_expected
    out["gap_to_perfect_concordance"] = delta - (-bernoulli_expected)
    out["bernoulli_expected"] = bernoulli_expected
    gap_values = out["gap_to_perfect_concordance"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_pct = np.where(
            bernoulli_expected > 0,
            100.0 * gap_values / bernoulli_expected,
            np.nan,
        )
    out["gap_to_bell_ratio_pct"] = ratio_pct
    out = out.dropna(subset=["gap_to_bell_ratio_pct"]).copy()
    return out


def _plot_perfect_concordance_gap_barchart_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    score_col: str,
    x_col: str = "prescription_rate",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    label_fontsize: int = 7,
    color_map: Mapping | None = None,
) -> bool:
    """Bar chart of vertical gap (point → perfect-concordance curve) per physician."""
    gap_df = _compute_perfect_concordance_gap_frame(
        df, score_col=score_col, x_col=x_col
    )
    if gap_df.empty:
        return False

    gap_df = gap_df.sort_values(
        "gap_to_perfect_concordance", ascending=True
    ).reset_index(drop=True)
    labels = [
        _physician_label_with_counts(
            row["physician"],
            row["n_prescriptions"],
            row["n_patients"],
            row["n_pairs"],
            row["n_discordant"],
        )
        for _, row in gap_df.iterrows()
    ]
    resolved_colors = _physician_color_map_from_df(df, color_map)
    bar_colors = physician_face_colors(gap_df["physician"], resolved_colors)
    x_pos = np.arange(len(gap_df))
    ax.bar(
        x_pos,
        gap_df["gap_to_perfect_concordance"].to_numpy(dtype=float),
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Physician" if show_xlabel else "", fontsize=12)
    ax.set_ylabel(
        "Gap to perfect concordance (D)" if show_ylabel else "",
        fontsize=12,
    )
    ax.set_ylim(bottom=0.0)
    return True


def plot_perfect_concordance_gap_barchart(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Vertical gap to perfect concordance (D)",
    score_col: str = "ensemble_matching",
    x_col: str = "prescription_rate",
    suptitle: str | None = None,
    logger: Optional[logging.Logger] = None,
    color_map: Mapping | None = None,
) -> Optional[Path]:
    """Save a bar chart of point-to-bell-curve vertical gap per physician."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    drawn = _plot_perfect_concordance_gap_barchart_on_ax(
        ax,
        df,
        title=title,
        score_col=score_col,
        x_col=x_col,
        color_map=color_map,
    )
    if not drawn:
        plt.close(fig)
        if logger is not None:
            logger.warning(
                "plot_perfect_concordance_gap_barchart skipped: no renderable physicians."
            )
        return None

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info("Saved perfect concordance gap bar chart to %s", out_path)
    return out_path


def plot_multi_method_perfect_concordance_gap_overview(
    df: pd.DataFrame,
    score_cols: list[str],
    out_path: str | Path,
    *,
    x_col: str = "prescription_rate",
    suptitle: str = "Vertical gap to perfect concordance (D) — all methods",
    logger: Optional[logging.Logger] = None,
    color_map: Mapping | None = None,
) -> Optional[Path]:
    """Render one perfect-concordance gap bar chart per discordance method."""
    if not score_cols:
        if logger is not None:
            logger.warning(
                "plot_multi_method_perfect_concordance_gap_overview skipped: no score columns."
            )
        return None

    n_panels = len(score_cols)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes_grid.flatten()
    resolved_colors = _physician_color_map_from_df(df, color_map)
    drawn = 0
    for ax, score_col in zip(axes_flat, score_cols):
        title = METHOD_DISPLAY_NAMES.get(score_col, score_col)
        ok = _plot_perfect_concordance_gap_barchart_on_ax(
            ax,
            df,
            title=title,
            score_col=score_col,
            x_col=x_col,
            show_xlabel=True,
            show_ylabel=True,
            color_map=resolved_colors,
        )
        if ok:
            drawn += 1
        else:
            ax.axis("off")
            ax.set_title(
                f"{title}\n(no prescription-rate data)",
                fontsize=14,
                fontweight="bold",
            )
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    if drawn == 0:
        plt.close(fig)
        if logger is not None:
            logger.warning(
                "plot_multi_method_perfect_concordance_gap_overview: nothing renderable "
                "across %d method(s).",
                n_panels,
            )
        return None

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info("Saved multi-method perfect concordance gap overview to %s", out_path)
    return out_path


def _plot_perfect_concordance_gap_ratio_barchart_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    score_col: str,
    x_col: str = "prescription_rate",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    label_fontsize: int = 7,
    color_map: Mapping | None = None,
) -> bool:
    """Bar chart of ``100 × D / (2p(1−p))`` per physician (gap relative to bell-curve span)."""
    gap_df = _compute_perfect_concordance_gap_frame(
        df, score_col=score_col, x_col=x_col
    )
    if gap_df.empty:
        return False

    gap_df = gap_df.sort_values("gap_to_bell_ratio_pct", ascending=True).reset_index(drop=True)
    labels = [
        _physician_label_with_counts(
            row["physician"],
            row["n_prescriptions"],
            row["n_patients"],
            row["n_pairs"],
            row["n_discordant"],
        )
        for _, row in gap_df.iterrows()
    ]
    resolved_colors = _physician_color_map_from_df(df, color_map)
    bar_colors = physician_face_colors(gap_df["physician"], resolved_colors)
    x_pos = np.arange(len(gap_df))
    values = gap_df["gap_to_bell_ratio_pct"].to_numpy(dtype=float)
    se_ys: list[float] = []
    se_drawn = False
    for i, (_, row) in enumerate(gap_df.iterrows()):
        se = _gap_ratio_standard_error(
            float(row["discordance"]),
            float(row["prescription_rate"]),
            row["n_pairs"],
        )
        if se is None:
            continue
        val = float(row["gap_to_bell_ratio_pct"])
        se_ys.extend([val - se, val + se])
        ax.errorbar(
            x_pos[i],
            values[i],
            yerr=se,
            fmt="none",
            ecolor=bar_colors[i],
            elinewidth=1.0,
            capsize=3.0,
            capthick=1.0,
            alpha=0.85,
            zorder=4,
            label=GAP_RATIO_SE_LABEL if not se_drawn else None,
        )
        se_drawn = True

    ax.bar(
        x_pos,
        values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    ax.axhline(
        100.0,
        color=BERNOULLI_REFERENCE_COLOR,
        linestyle=BERNOULLI_REFERENCE_LINESTYLE,
        linewidth=BERNOULLI_REFERENCE_LINEWIDTH,
        zorder=6,
        label=BERNOULLI_BASELINE_RATIO_PCT_LABEL,
    )

    if se_drawn:
        se_handles = [
            handle
            for handle, lab in zip(*ax.get_legend_handles_labels())
            if lab == GAP_RATIO_SE_LABEL
        ]
        if se_handles:
            ax.legend(
                handles=se_handles,
                loc="upper right",
                fontsize=8,
                framealpha=0.85,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Physician" if show_xlabel else "", fontsize=12)
    ax.set_ylabel(
        "Gap / |bell curve| (%)" if show_ylabel else "",
        fontsize=12,
    )
    y_max = float(np.max(values)) if values.size > 0 else 100.0
    if se_ys:
        y_max = max(y_max, float(np.max(se_ys)))
    y_pad = max(5.0, y_max * 0.1)
    ax.set_ylim(bottom=0.0, top=max(110.0, y_max + y_pad))
    return True


def plot_perfect_concordance_gap_ratio_barchart(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Gap to perfect concordance as % of |bell curve|",
    score_col: str = "ensemble_matching",
    x_col: str = "prescription_rate",
    suptitle: str | None = None,
    logger: Optional[logging.Logger] = None,
    color_map: Mapping | None = None,
) -> Optional[Path]:
    """Save a bar chart of gap-to-bell-curve ratio (%) per physician."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    drawn = _plot_perfect_concordance_gap_ratio_barchart_on_ax(
        ax,
        df,
        title=title,
        score_col=score_col,
        x_col=x_col,
        color_map=color_map,
    )
    if not drawn:
        plt.close(fig)
        if logger is not None:
            logger.warning(
                "plot_perfect_concordance_gap_ratio_barchart skipped: no renderable physicians."
            )
        return None

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info("Saved perfect concordance gap ratio bar chart to %s", out_path)
    return out_path


def plot_multi_method_perfect_concordance_gap_ratio_overview(
    df: pd.DataFrame,
    score_cols: list[str],
    out_path: str | Path,
    *,
    x_col: str = "prescription_rate",
    suptitle: str = "Gap / |bell curve| (%) — all methods",
    logger: Optional[logging.Logger] = None,
    color_map: Mapping | None = None,
) -> Optional[Path]:
    """Render one gap-ratio (%) bar chart per discordance method."""
    if not score_cols:
        if logger is not None:
            logger.warning(
                "plot_multi_method_perfect_concordance_gap_ratio_overview skipped: "
                "no score columns."
            )
        return None

    n_panels = len(score_cols)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes_grid.flatten()
    resolved_colors = _physician_color_map_from_df(df, color_map)
    drawn = 0
    for ax, score_col in zip(axes_flat, score_cols):
        title = METHOD_DISPLAY_NAMES.get(score_col, score_col)
        ok = _plot_perfect_concordance_gap_ratio_barchart_on_ax(
            ax,
            df,
            title=title,
            score_col=score_col,
            x_col=x_col,
            show_xlabel=True,
            show_ylabel=True,
            color_map=resolved_colors,
        )
        if ok:
            drawn += 1
        else:
            ax.axis("off")
            ax.set_title(
                f"{title}\n(no prescription-rate data)",
                fontsize=14,
                fontweight="bold",
            )
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    if drawn == 0:
        plt.close(fig)
        if logger is not None:
            logger.warning(
                "plot_multi_method_perfect_concordance_gap_ratio_overview: nothing renderable "
                "across %d method(s).",
                n_panels,
            )
        return None

    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info(
            "Saved multi-method perfect concordance gap ratio overview to %s",
            out_path,
        )
    return out_path


# Forest plot: odds ratio de prescription par médecin, ajusté sur le profil
# patient via l'effet aléatoire du GLMM (``random_effect_bias``), coloré par
# l'identité du médecin (palette partagée de l'expérience).
PLOT_SUBDIR_PHYSICIAN_ODDS: str = "physician_odds"
PRESCRIPTION_ODDS_RATIO_REFERENCE_LABEL: str = "OR = 1 (average physician)"
# Candidate tick positions for the log-scaled odds-ratio axis; filtered down to
# the data range so the reader can situate each physician precisely instead of
# relying on the sparse default 10^-1 / 10^0 / 10^1 log ticks.
PRESCRIPTION_ODDS_RATIO_CANDIDATE_TICKS: tuple[float, ...] = (
    0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7,
    1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0,
)


def _odds_ratio_log_axis_bounds_and_ticks(
    x_min: float, x_max: float
) -> tuple[float, float, list[float]]:
    """Return ``(xlim_low, xlim_high, ticks)`` covering ``[x_min, x_max]`` on a log axis.

    The axis bounds always contain the data (with a small margin) even when no
    candidate tick lands exactly there; ``1.0`` (the no-effect reference) is
    always included among the ticks so the reader keeps a fixed anchor point.
    """
    lo = min(x_min, 1.0) / 1.15
    hi = max(x_max, 1.0) * 1.15
    ticks = [t for t in PRESCRIPTION_ODDS_RATIO_CANDIDATE_TICKS if lo <= t <= hi]
    if 1.0 not in ticks:
        ticks.append(1.0)
    return lo, hi, sorted(set(ticks))


def _compute_physician_prescription_odds_ratio_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per physician with a profile-adjusted prescription odds ratio.

    ``prescription_odds_ratio = exp(random_effect_bias)``: the multiplicative
    change in prescription odds for that physician relative to an "average"
    physician, holding the GLMM fixed-effect covariates (patient profile) constant.
    Physicians without a usable ``random_effect_bias`` are excluded.
    """
    if "physician" not in df.columns or "random_effect_bias" not in df.columns:
        return pd.DataFrame()

    n_patients_series = (
        pd.to_numeric(df["n_patients"], errors="coerce")
        if "n_patients" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    n_prescriptions_series = (
        pd.to_numeric(df["n_prescriptions"], errors="coerce")
        if "n_prescriptions" in df.columns
        else pd.Series([pd.NA] * len(df), index=df.index)
    )
    out = pd.DataFrame({
        "physician": df["physician"].astype(str),
        "random_effect_bias": pd.to_numeric(df["random_effect_bias"], errors="coerce"),
        "n_patients": n_patients_series.values,
        "n_prescriptions": n_prescriptions_series.values,
    })
    out = out.dropna(subset=["random_effect_bias"]).copy()
    if out.empty:
        return out

    out["prescription_odds_ratio"] = np.exp(out["random_effect_bias"].to_numpy(dtype=float))
    out = out.sort_values("prescription_odds_ratio", ascending=True).reset_index(drop=True)
    return out


def _plot_physician_prescription_odds_ratio_forest_on_ax(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    label_fontsize: int = 9,
    color_map: Mapping | None = None,
) -> bool:
    """Horizontal forest plot of profile-adjusted prescription odds ratio per physician."""
    or_df = _compute_physician_prescription_odds_ratio_frame(df)
    if or_df.empty:
        return False

    labels = [
        _physician_label_with_counts(row["physician"], row["n_prescriptions"], row["n_patients"])
        for _, row in or_df.iterrows()
    ]
    y_pos = np.arange(len(or_df))
    odds_ratios = or_df["prescription_odds_ratio"].to_numpy(dtype=float)
    n_patients = pd.to_numeric(or_df["n_patients"], errors="coerce").to_numpy(dtype=float)

    resolved_colors = _physician_color_map_from_df(df, color_map)
    point_colors = physician_face_colors(or_df["physician"], resolved_colors)

    has_n_patients = np.isfinite(n_patients).any()
    if has_n_patients:
        finite_n = n_patients[np.isfinite(n_patients)]
        n_min, n_max = float(np.min(finite_n)), float(np.max(finite_n))
        if n_max > n_min:
            sizes = 30.0 + 90.0 * (np.nan_to_num(n_patients, nan=n_min) - n_min) / (n_max - n_min)
        else:
            sizes = np.full_like(odds_ratios, 60.0)
    else:
        sizes = np.full_like(odds_ratios, 60.0)

    ax.hlines(
        y_pos,
        xmin=1.0,
        xmax=odds_ratios,
        color="#b0b0b0",
        linewidth=1.2,
        zorder=2,
    )
    ax.scatter(
        odds_ratios,
        y_pos,
        c=point_colors,
        s=sizes,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.axvline(
        1.0,
        color=BERNOULLI_REFERENCE_COLOR,
        linestyle=BERNOULLI_REFERENCE_LINESTYLE,
        linewidth=BERNOULLI_REFERENCE_LINEWIDTH,
        zorder=4,
        label=PRESCRIPTION_ODDS_RATIO_REFERENCE_LABEL,
    )
    ax.set_xscale("log")
    x_min = float(np.min(odds_ratios))
    x_max = float(np.max(odds_ratios))
    x_lo, x_hi, x_ticks = _odds_ratio_log_axis_bounds_and_ticks(x_min, x_max)
    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.grid(True, axis="x", which="major", linestyle=":", linewidth=0.7, alpha=0.6, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel("Odds ratio (GLMM random effect, covariate-adjusted)", fontsize=12)
    ax.set_ylabel("Physician", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(-1, len(or_df))

    return True


def plot_physician_prescription_odds_ratio_forest(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Prescription odds ratio vs. average physician (profile-adjusted)",
    color_map: Mapping | None = None,
    suptitle: str | None = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Save a forest plot of profile-adjusted prescription odds ratio per physician."""
    sns.set_theme(style="whitegrid", context="notebook")
    or_df = _compute_physician_prescription_odds_ratio_frame(df)
    n_physicians = len(or_df)
    fig_height = max(6.0, 0.35 * n_physicians)
    fig, ax = plt.subplots(1, 1, figsize=(10.0, fig_height))
    drawn = _plot_physician_prescription_odds_ratio_forest_on_ax(
        ax,
        df,
        title=title,
        color_map=color_map,
    )
    if not drawn:
        plt.close(fig)
        if logger is not None:
            logger.warning(
                "plot_physician_prescription_odds_ratio_forest skipped: "
                "no physician with a usable random_effect_bias."
            )
        return None

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info("Saved physician prescription odds ratio forest plot to %s", out_path)
    return out_path


def _patients_per_physician_count_frame(
    df: pd.DataFrame,
    *,
    physician_col: str = "physician",
    n_patients_col: str = "n_patients",
) -> pd.DataFrame:
    """One row per physician with a finite non-negative caseload."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[physician_col, n_patients_col])
    if n_patients_col not in df.columns:
        return pd.DataFrame(columns=[physician_col, n_patients_col])
    out = df.copy()
    out[n_patients_col] = pd.to_numeric(out[n_patients_col], errors="coerce")
    out = out.loc[np.isfinite(out[n_patients_col]) & (out[n_patients_col] >= 0)].copy()
    if out.empty:
        return pd.DataFrame(columns=[physician_col, n_patients_col])
    out[n_patients_col] = out[n_patients_col].astype(int)
    if physician_col not in out.columns:
        out[physician_col] = [f"P{i + 1}" for i in range(len(out))]
    return out[[physician_col, n_patients_col]].drop_duplicates(subset=[physician_col])


def _plot_patients_per_physician_distribution_on_axes(
    ax_hist,
    ax_bars,
    df: pd.DataFrame,
    *,
    min_patients: int = 30,
    physician_col: str = "physician",
    n_patients_col: str = "n_patients",
) -> bool:
    """Histogram + ranked caseload bars, with the minimum-patient cutoff marked."""
    counts_df = _patients_per_physician_count_frame(
        df, physician_col=physician_col, n_patients_col=n_patients_col
    )
    if counts_df.empty:
        return False

    values = counts_df[n_patients_col].to_numpy(dtype=int)
    below = values < int(min_patients)
    n_below = int(below.sum())
    n_kept = int((~below).sum())
    color_below = "#C44E52"
    color_kept = "#4C72B0"

    max_n = int(values.max())
    binwidth = 1 if max_n <= 80 else max(2, int(np.ceil(max_n / 40)))
    bin_edges = np.arange(0, max_n + binwidth + 1, binwidth)
    hist_series: list[np.ndarray] = []
    hist_colors: list[str] = []
    hist_labels: list[str] = []
    if n_below:
        hist_series.append(values[below])
        hist_colors.append(color_below)
        hist_labels.append(f"Below cutoff (n={n_below})")
    if n_kept:
        hist_series.append(values[~below])
        hist_colors.append(color_kept)
        hist_labels.append(f"At or above cutoff (n={n_kept})")
    stacked = len(hist_series) > 1
    ax_hist.hist(
        hist_series if stacked else hist_series[0],
        bins=bin_edges,
        stacked=stacked,
        color=hist_colors if stacked else hist_colors[0],
        label=hist_labels if stacked else hist_labels[0],
        edgecolor="white",
        linewidth=0.4,
    )
    ax_hist.axvline(
        float(min_patients),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=PATIENTS_PER_PHYSICIAN_CUTOFF_LINE_LABEL,
    )
    ax_hist.set_xlabel("Patients per physician")
    ax_hist.set_ylabel("Number of physicians")
    ax_hist.set_title(
        "Caseload distribution (before minimum-patient filter)",
        fontsize=13,
        fontweight="bold",
    )
    ax_hist.legend(frameon=True, loc="best")

    ranked = counts_df.sort_values(n_patients_col, ascending=True, kind="mergesort")
    bar_colors = [color_below if n < min_patients else color_kept for n in ranked[n_patients_col]]
    y_pos = np.arange(len(ranked))
    ax_bars.barh(
        y_pos,
        ranked[n_patients_col].to_numpy(dtype=int),
        color=bar_colors,
        edgecolor="white",
        linewidth=0.3,
        height=0.85,
    )
    ax_bars.axvline(
        float(min_patients),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=PATIENTS_PER_PHYSICIAN_CUTOFF_LINE_LABEL,
    )
    ax_bars.set_xlabel("Patients per physician")
    ax_bars.set_ylabel("Physician")
    ax_bars.set_title("Patients per physician (ranked)", fontsize=13, fontweight="bold")
    ax_bars.set_yticks(y_pos)
    if len(ranked) <= 40:
        ax_bars.set_yticklabels(
            [format_physician_display_label(pid) for pid in ranked[physician_col]]
        )
    else:
        ax_bars.set_yticklabels([])
        ax_bars.set_ylabel(f"Physicians ranked by caseload (n={len(ranked)})")
    ax_bars.set_xlim(left=0)
    return True


def plot_patients_per_physician_distribution(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    min_patients: int = 30,
    physician_col: str = "physician",
    n_patients_col: str = "n_patients",
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Save the pre-cutoff distribution of patients per physician."""
    counts_df = _patients_per_physician_count_frame(
        df, physician_col=physician_col, n_patients_col=n_patients_col
    )
    if counts_df.empty:
        if logger is not None:
            logger.warning(
                "plot_patients_per_physician_distribution skipped: no physician caseloads."
            )
        return None

    sns.set_theme(style="whitegrid", context="notebook")
    n_physicians = len(counts_df)
    bar_height = max(4.5, min(0.28 * n_physicians, 16.0))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 5.5 + bar_height),
        gridspec_kw={"height_ratios": [1.0, max(1.2, bar_height / 5.5)]},
    )
    drawn = _plot_patients_per_physician_distribution_on_axes(
        axes[0],
        axes[1],
        counts_df,
        min_patients=min_patients,
        physician_col=physician_col,
        n_patients_col=n_patients_col,
    )
    if not drawn:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Patients per physician before ≥{int(min_patients)} filter",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info("Saved patients-per-physician distribution to %s", out_path)
    return out_path


class Analysis:
    """
    Orchestrator for the full variability analysis pipeline.

    ``run_full_analysis`` prépare un sous-ensemble complet de lignes (sans GLMM),
    calcule trois discordances par matching (RF proximity, RF learned weights,
    mutual information), puis ne conserve que leur moyenne dans ``ensemble_matching``.

    Attributes:
        df: Patient-level DataFrame (copy); must contain outcome, physician ID, and
            fixed-effect columns.
        results_dir: Directory for reports, plots, and output CSV.
        config: Dict with outcome_col, physician_col, fixed_effects, min_patients_per_physician,
            and optional dataset_targets for reporting.
        results: Dict holding ``matching_basis``, ``intra_physician_variability``, etc.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        results_dir: Optional[str] = None,
        LOGGER=None,
        config: Optional[dict] = None,
        snapshots: Optional[SnapshotSaver] = None,
        outlier_stats: Optional[dict] = None,
    ):
        """
        Initialize the analysis with data and configuration.

        Arguments:
            df: Raw patient-level DataFrame (outcome, physician, covariates).
            results_dir: Output directory for reports and plots; default "exp".
            LOGGER: Logger instance (required).
            config: Override for column names and thresholds; see _default_config().
            snapshots: Optional numbered head-CSV exporter (see ``snapshot_utils``).
            outlier_stats: Optional scalar summary from the upstream cell-level
                outlier detection step (see ``dataset_utils.outlier_detection``),
                surfaced to the post-run validation report.
        """
        if LOGGER is None:
            raise ValueError("LOGGER must be provided.")
        self.df = df.copy()
        self.LOGGER = LOGGER
        self.results_dir = Path(results_dir) if results_dir else Path(DEFAULT_EXPERIMENTS_ROOT)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or self._default_config()
        self.results = {}
        self._snapshots = snapshots
        self._outlier_stats = outlier_stats
        # Cache de la sélection de features par variance (top-k). Calculé une
        # seule fois pour rester identique entre la préparation du sous-ensemble
        # et toutes les méthodes de matching.
        self._selected_matching_features: list[str] | None = None
        self._matching_features_snapshotted: bool = False
        self._matching_features_selected_snapshotted: bool = False
        self._analysis_imputed: bool = False

    def _save_snapshot(self, df: pd.DataFrame | None, name: str) -> None:
        """Persist ``df.head()`` when snapshot export is enabled."""
        if self._snapshots is None or df is None:
            return
        self._snapshots.save(df, name)

    def _save_snapshot_manifest(
        self, df: pd.DataFrame | None, name: str, *, full: bool = False
    ) -> None:
        """Persist a small metadata table in full (feature lists, rankings)."""
        if self._snapshots is None or df is None:
            return
        self._snapshots.save_manifest(df, name, full=full)

    def _eligible_matching_covariate_columns(self, df: pd.DataFrame) -> list[str]:
        """All numeric/boolean columns in *df* usable for matching (post-preprocess)."""
        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        exclude = {
            outcome_col,
            physician_col,
            "_predicted_prob",
            "_residual",
            "_obs_id",
            "member_pseudo_id",
            "person_id",
            "checkup_id",
            "approximate_birth_date",
            # Recommendation metadata: directly derived from the outcome
            # (recommendation = n_target_recos > 0), so using them as matching
            # covariates leaks the outcome and collapses the discordance metric.
            "n_recos",
            "n_target_recos",
        }
        exclude.update(c for c in df.columns if str(c).startswith("outcome_"))
        return [
            c
            for c in df.columns
            if c not in exclude
            and (
                pd.api.types.is_numeric_dtype(df[c])
                or pd.api.types.is_bool_dtype(df[c])
            )
        ]

    def _snapshot_matching_features_before_selection(
        self,
        df: pd.DataFrame,
        eligible: list[str],
    ) -> None:
        """Export candidate covariates and a head preview before variance selection."""
        if self._matching_features_snapshotted or not eligible:
            return
        self._save_snapshot_manifest(
            pd.DataFrame({"feature": eligible}),
            "matching_features_candidates",
            full=True,
        )
        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        preview_cols = [
            c for c in [outcome_col, physician_col, *eligible] if c in df.columns
        ]
        self._save_snapshot(
            df[preview_cols],
            "matching_features_pre_selection_head",
        )

    def _snapshot_matching_features_after_selection(
        self,
        ranking: pd.DataFrame,
        method: str,
    ) -> None:
        """Export the feature ranking table once selection has run."""
        if self._matching_features_snapshotted or ranking.empty:
            return
        self._save_snapshot_manifest(
            ranking,
            f"matching_features_{method}_ranking",
            full=True,
        )

    def _snapshot_matching_features_selected(self, selected: list[str]) -> None:
        """Export the final covariate list passed to matching methods."""
        if self._matching_features_selected_snapshotted or not selected:
            return
        fs_cfg = self.config.get("feature_selection") or {}
        if fs_cfg.get("enabled"):
            method = str(fs_cfg.get("method", "variance")).strip().lower()
            if method == "manual":
                top_k = len(selected)
            else:
                top_k = int(fs_cfg.get("top_k", len(selected)))
        else:
            method = "all"
            top_k = len(selected)
        manifest = pd.DataFrame(
            {
                "rank": np.arange(1, len(selected) + 1),
                "feature": selected,
                "selection_method": method,
                "top_k": top_k,
            }
        )
        self._save_snapshot_manifest(manifest, "matching_features_selected", full=True)
        self._matching_features_selected_snapshotted = True

    def _matching_covariate_columns(self, df: pd.DataFrame) -> list[str]:
        """Return matching covariates, optionally filtered by top-k feature selection."""
        eligible = self._eligible_matching_covariate_columns(df)
        if not self._matching_features_snapshotted:
            self._snapshot_matching_features_before_selection(df, eligible)

        fs_cfg = self.config.get("feature_selection") or {}
        if fs_cfg.get("enabled"):
            method = str(fs_cfg.get("method", "variance")).strip().lower()
            if method not in FEATURE_SELECTION_METHODS:
                supported = ", ".join(sorted(FEATURE_SELECTION_METHODS))
                raise ValueError(
                    f"Unsupported feature_selection.method '{method}'. "
                    f"Expected one of: {supported}."
                )
            if self._selected_matching_features is None:
                if method == "manual":
                    ranking = self._manual_ranking_table(df, eligible, fs_cfg)
                    self._selected_matching_features = list(
                        ranking.loc[ranking["selected"], "feature"]
                    )
                else:
                    top_k = int(fs_cfg.get("top_k", 25))
                    standardize = bool(fs_cfg.get("standardize", True))
                    if method == "variance":
                        ranking = self._variance_ranking_table(
                            df,
                            eligible,
                            standardize=standardize,
                            top_k=top_k,
                        )
                    else:
                        ranking = self._anova_ranking_table(
                            df,
                            eligible,
                            standardize=standardize,
                            top_k=top_k,
                        )
                    self._selected_matching_features = list(
                        ranking.loc[ranking["selected"], "feature"]
                    )
                if not self._matching_features_snapshotted:
                    self._snapshot_matching_features_after_selection(ranking, method)
                if method == "manual":
                    self._log_manual_feature_selection(
                        selected=self._selected_matching_features,
                        n_candidates=len(eligible),
                        n_requested=int(ranking.shape[0]),
                    )
                else:
                    top_k = int(fs_cfg.get("top_k", 25))
                    standardize = bool(fs_cfg.get("standardize", True))
                    self._log_feature_selection(
                        method=method,
                        selected=self._selected_matching_features,
                        n_candidates=len(eligible),
                        top_k=top_k,
                        standardize=standardize,
                    )
            eligible_set = set(eligible)
            eligible = [
                c for c in self._selected_matching_features if c in eligible_set
            ]

        if not self._matching_features_selected_snapshotted:
            self._snapshot_matching_features_selected(eligible)
        if not self._matching_features_snapshotted:
            self._matching_features_snapshotted = True
        return eligible

    def _method_enabled(self, name: str) -> bool:
        """Return whether a matching / ensemble step is active in config."""
        methods_cfg = self.config.get("methods")
        if methods_cfg is None:
            return bool(LEGACY_DEFAULT_METHOD_FLAGS.get(name, False))
        return bool(methods_cfg.get(name, False))

    def _glmm_enabled(self) -> bool:
        glmm_cfg = self.config.get("glmm") or {}
        return bool(glmm_cfg.get("enabled", False))

    def _resolve_glmm_covariates(self) -> list[str]:
        """Resolve GLMM fixed effects from config (matching preset, statin, auto, or list)."""
        glmm_cfg = self.config.get("glmm") or {}
        raw = glmm_cfg.get("fixed_effects", GLMM_FIXED_EFFECTS_MATCHING)
        if isinstance(raw, list):
            eligible = set(self._eligible_matching_covariate_columns(self.df))
            cols = [str(c) for c in raw if str(c) in eligible]
            if not cols:
                raise ValueError(
                    "CRITICAL: GLMM fixed_effects list is empty after leakage-safe column filtering."
                )
            self._glmm_covariates_resolved = cols
            return cols
        if not isinstance(raw, str):
            raise ValueError(f"Unsupported glmm.fixed_effects type: {type(raw)!r}")
        preset = raw.strip().lower()
        if preset == GLMM_FIXED_EFFECTS_MATCHING:
            cols = self._matching_covariate_columns(self.df)
        elif preset == GLMM_FIXED_EFFECTS_STATIN_RELEVANT:
            cols = [c for c in STATIN_RELEVANT_GLMM_COVARIATES if c in self.df.columns]
            if not cols:
                raise ValueError(
                    "CRITICAL: statin_relevant GLMM preset found no matching columns in the dataset."
                )
        elif preset == GLMM_FIXED_EFFECTS_AUTO:
            eligible = set(self._eligible_matching_covariate_columns(self.df))
            cols = [c for c in self.config.get("fixed_effects", []) if c in eligible]
            if not cols:
                raise ValueError(
                    "CRITICAL: auto GLMM preset found no leakage-safe fixed_effects columns."
                )
        else:
            raise ValueError(
                f"Unsupported glmm.fixed_effects preset '{raw}'. "
                f"Expected matching, statin_relevant, auto, or an explicit list."
            )
        if not cols:
            raise ValueError("CRITICAL: The resolved GLMM covariate list is empty.")
        self._glmm_covariates_resolved = cols
        return cols

    def _active_plot_score_columns(self) -> list[str]:
        """Score columns present in intra_physician_variability and eligible for plots."""
        df = self.results.get("intra_physician_variability")
        if df is None or df.empty:
            return []
        specs = METHOD_PLOT_SPECS
        if not self._method_enabled("manual_pairing"):
            specs = [(c, n) for c, n in specs if c != "discordance_rate_manual"]
        return [col for col, _ in specs if col in df.columns]

    def _bernoulli_residual_score_columns(self) -> list[str]:
        """Columns for Bernoulli residual plots (discordance rates + ensemble by default)."""
        df = self.results.get("intra_physician_variability")
        if df is None or df.empty:
            return []
        br_cfg = self.config.get("bernoulli_residual") or {}
        methods = br_cfg.get("methods", "all")
        discordance_cols = [c for c in df.columns if c.startswith("discordance_rate_")]
        available = discordance_cols + (["ensemble_matching"] if "ensemble_matching" in df.columns else [])
        if methods == "all" or methods is None:
            return [c for c in available if c in df.columns]
        if isinstance(methods, list):
            return [str(c) for c in methods if str(c) in df.columns]
        return []

    def _save_analysis_columns_manifest(self) -> None:
        """Export GLMM and matching covariate lists used in this run."""
        glmm_columns = list(getattr(self, "_glmm_covariates_resolved", None) or [])
        if not glmm_columns and self._glmm_enabled():
            try:
                glmm_columns = self._resolve_glmm_covariates()
            except ValueError:
                glmm_columns = []
        matching_columns = self._matching_covariate_columns(self.df)
        rows: list[dict[str, object]] = []
        for rank, column_name in enumerate(glmm_columns, start=1):
            rows.append({"usage": "glmm", "column_name": column_name, "rank": rank})
        for rank, column_name in enumerate(matching_columns, start=1):
            rows.append({"usage": "matching", "column_name": column_name, "rank": rank})
        manifest = pd.DataFrame(rows, columns=["usage", "column_name", "rank"])
        out_path = self.results_dir / "analysis_columns_used.csv"
        manifest.to_csv(out_path, index=False)
        self.LOGGER.info(
            "Saved analysis columns manifest to %s (glmm=%d, matching=%d)",
            out_path,
            len(glmm_columns),
            len(matching_columns),
        )

    def plot_questionnaire_completion_used(self) -> None:
        """Bar chart + CSV of pre-impute response rates for qa__* used in matching/GLMM."""
        prefix = str(self.config.get("questionnaire_prefix", "qa__"))
        matching_columns = self._matching_covariate_columns(self.df)
        glmm_columns: list[str] = list(getattr(self, "_glmm_covariates_resolved", None) or [])
        if not glmm_columns and self._glmm_enabled():
            try:
                glmm_columns = self._resolve_glmm_covariates()
            except ValueError:
                glmm_columns = []

        matching_qa = {c for c in matching_columns if str(c).startswith(prefix)}
        glmm_qa = {c for c in glmm_columns if str(c).startswith(prefix)}
        used_qa = sorted(matching_qa | glmm_qa)
        if not used_qa:
            self.LOGGER.info(
                "Questionnaire completion plot skipped: no %s* columns in matching/GLMM.",
                prefix,
            )
            return

        detail_rows = {
            str(r["column_name"]): r
            for r in (self.config.get("questionnaire_response_detail") or [])
            if isinstance(r, dict) and "column_name" in r
        }
        rates = dict(self.config.get("questionnaire_response_rates") or {})
        n_members_fallback = int(len(self.df))

        rows: list[dict[str, object]] = []
        for col in used_qa:
            in_matching = col in matching_qa
            in_glmm = col in glmm_qa
            if in_matching and in_glmm:
                usage = USAGE_BOTH
            elif in_matching:
                usage = USAGE_MATCHING
            else:
                usage = USAGE_GLMM

            detail = detail_rows.get(col)
            if detail is not None:
                rate = float(detail.get("response_rate", rates.get(col, 0.0)))
                n_answered = int(detail.get("n_answered", 0))
                n_members = int(detail.get("n_members", n_members_fallback))
            else:
                rate = float(rates.get(col, 0.0))
                n_members = n_members_fallback
                n_answered = int(round(rate * n_members))

            rows.append(
                {
                    "column_name": col,
                    "display_label": questionnaire_display_label(col),
                    "response_rate": rate,
                    "response_rate_pct": 100.0 * rate,
                    "n_answered": n_answered,
                    "n_members": n_members,
                    "used_in_matching": in_matching,
                    "used_in_glmm": in_glmm,
                    "usage": usage,
                }
            )

        summary = pd.DataFrame(rows)
        csv_path = self.results_dir / "questionnaire_completion_used.csv"
        summary[
            [
                "column_name",
                "response_rate",
                "n_answered",
                "n_members",
                "used_in_matching",
                "used_in_glmm",
            ]
        ].to_csv(csv_path, index=False)
        self.LOGGER.info(
            "Saved questionnaire completion CSV to %s (%d questions).",
            csv_path,
            len(summary),
        )

        plot_dir = self.results_dir / "plots" / PLOT_SUBDIR_QUESTIONNAIRE
        plot_path = plot_dir / "completion_used.png"
        plot_questionnaire_completion(summary, plot_path)
        self.LOGGER.info("Saved questionnaire completion plot to %s", plot_path)

    @staticmethod
    def _median_fill_for_ranking(
        df: pd.DataFrame,
        cols: list[str],
    ) -> pd.DataFrame:
        """Column-wise median fill for feature-selection scoring only.

        Entirely-NaN columns fall back to 0 so downstream scorers stay finite.
        Does not mutate the caller's DataFrame.
        """
        if not cols:
            return df.copy()
        out = df[cols].astype(float).copy()
        for col in cols:
            median = out[col].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            out[col] = out[col].fillna(float(median))
        return out

    @staticmethod
    def _scale_features_for_selection(
        sub: pd.DataFrame,
        *,
        standardize: bool,
    ) -> pd.DataFrame:
        """Min-max scale columns to [0, 1] when ``standardize`` is True.

        Callers should median-fill ranking copies first; residual NaNs (e.g.
        constant columns after scaling) are zero-filled so scorers stay finite.
        """
        if not standardize:
            return sub.fillna(0.0)
        col_min = sub.min()
        col_range = (sub.max() - col_min).replace(0, np.nan)
        scaled = (sub - col_min) / col_range
        # Constant columns have zero range -> NaN after scaling; treat as 0 so
        # downstream scorers (variance, f_classif) receive finite inputs.
        return scaled.fillna(0.0)

    @staticmethod
    def _apply_top_k_selection_flag(
        table: pd.DataFrame,
        candidate_cols: list[str],
        top_k: int | None,
    ) -> pd.DataFrame:
        if top_k is None or len(candidate_cols) <= top_k:
            table["selected"] = True
        else:
            table["selected"] = table["rank"] <= top_k
        return table

    def _variance_ranking_table(
        self,
        df: pd.DataFrame,
        candidate_cols: list[str],
        *,
        standardize: bool = True,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Build a variance ranking table with an optional top-k ``selected`` flag."""
        empty = pd.DataFrame(columns=["feature", "variance", "rank", "selected"])
        if not candidate_cols:
            return empty

        filled = self._median_fill_for_ranking(df, candidate_cols)
        sub = self._scale_features_for_selection(
            filled,
            standardize=standardize,
        )
        variances = sub.var(ddof=0).fillna(0.0)
        ranked = variances.sort_values(ascending=False)
        table = pd.DataFrame(
            {
                "feature": ranked.index.astype(str),
                "variance": ranked.values,
            }
        )
        table["rank"] = np.arange(1, len(table) + 1)
        return self._apply_top_k_selection_flag(table, candidate_cols, top_k)

    def _manual_ranking_table(
        self,
        df: pd.DataFrame,
        eligible: list[str],
        fs_cfg: dict,
    ) -> pd.DataFrame:
        """Build a manual selection table from ``feature_selection.columns`` in YAML."""
        raw_columns = fs_cfg.get("columns") or []
        if not isinstance(raw_columns, list) or not raw_columns:
            raise ValueError(
                "feature_selection.columns must be a non-empty list when method='manual'."
            )

        requested = [str(c) for c in raw_columns]
        eligible_set = set(eligible)
        df_columns = set(df.columns)
        missing = [c for c in requested if c not in df_columns]
        not_eligible = [
            c for c in requested if c in df_columns and c not in eligible_set
        ]
        if missing:
            self.LOGGER.warning(
                "Manual feature selection: %d column(s) absent from dataset: %s",
                len(missing),
                missing,
            )
        if not_eligible:
            self.LOGGER.warning(
                "Manual feature selection: %d column(s) present but not eligible "
                "for matching: %s",
                len(not_eligible),
                not_eligible,
            )

        selected = [c for c in requested if c in eligible_set]
        if not selected:
            raise ValueError(
                "Manual feature selection produced 0 covariates. "
                "Check feature_selection.columns against the preprocessed dataset."
            )

        return pd.DataFrame(
            {
                "feature": requested,
                "rank": np.arange(1, len(requested) + 1),
                "in_dataset": [c in df_columns for c in requested],
                "eligible": [c in eligible_set for c in requested],
                "selected": [c in eligible_set for c in requested],
            }
        )

    def _log_manual_feature_selection(
        self,
        *,
        selected: list[str],
        n_candidates: int,
        n_requested: int,
    ) -> None:
        self.LOGGER.info(
            "Feature selection (manual): %d/%d covariables conservées "
            "(%d demandées dans le YAML).",
            len(selected),
            n_candidates,
            n_requested,
        )
        self.LOGGER.debug("Features retenues (manual): %s", selected)

    def _anova_ranking_table(
        self,
        df: pd.DataFrame,
        candidate_cols: list[str],
        *,
        standardize: bool = True,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """Build an ANOVA F-score ranking table with an optional top-k ``selected`` flag."""
        empty = pd.DataFrame(columns=["feature", "f_score", "p_value", "rank", "selected"])
        if not candidate_cols:
            return empty

        outcome_col = self.config["outcome_col"]
        if outcome_col not in df.columns:
            raise ValueError(
                f"Feature selection (anova) requires outcome column '{outcome_col}'."
            )

        y_num = pd.to_numeric(df[outcome_col], errors="coerce")
        valid_y = y_num.notna()
        if not valid_y.any():
            return empty

        y = y_num.loc[valid_y].astype(int).to_numpy()
        unique_y = np.unique(y)
        if unique_y.size < 2:
            raise ValueError(
                f"Feature selection (anova) requires a binary outcome with two classes; "
                f"found {unique_y.tolist()} in '{outcome_col}'."
            )

        # Median-fill per column on a ranking copy (not joint complete-case), so
        # deferred MICE / sparse NaNs do not empty the ANOVA sample.
        filled = self._median_fill_for_ranking(df.loc[valid_y], candidate_cols)
        sub = self._scale_features_for_selection(filled, standardize=standardize)
        x_values = sub.to_numpy(dtype=float)
        if not np.all(np.isfinite(x_values)):
            raise ValueError(
                "Feature selection (anova) received non-finite values in X after scaling."
            )
        f_scores, p_values = f_classif(x_values, y)
        table = pd.DataFrame(
            {
                "feature": candidate_cols,
                "f_score": f_scores,
                "p_value": p_values,
            }
        )
        table = table.sort_values(
            ["f_score", "feature"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        table["rank"] = np.arange(1, len(table) + 1)
        return self._apply_top_k_selection_flag(table, candidate_cols, top_k)

    def _log_feature_selection(
        self,
        *,
        method: str,
        selected: list[str],
        n_candidates: int,
        top_k: int,
        standardize: bool,
    ) -> None:
        self.LOGGER.info(
            "Feature selection (%s, standardize=%s): %d/%d covariables conservées (top %d).",
            method,
            standardize,
            len(selected),
            n_candidates,
            top_k,
        )
        self.LOGGER.debug("Features retenues (%s): %s", method, selected)

    def _rank_features_by_variance(
        self,
        df: pd.DataFrame,
        candidate_cols: list[str],
        top_k: int = 25,
        standardize: bool = True,
    ) -> list[str]:
        """Classe *candidate_cols* par variance décroissante et garde le top-k.

        Avec ``standardize=True`` (défaut), chaque colonne est ramenée sur
        l'échelle [0, 1] avant le calcul de variance. Cela rend la sélection
        invariante à l'unité de mesure (sinon les biomarqueurs à grande
        amplitude écraseraient systématiquement les flags binaires et les
        scores de questionnaire). Les colonnes constantes ont une variance
        nulle et sont donc écartées en priorité.
        """
        ranking = self._variance_ranking_table(
            df,
            candidate_cols,
            standardize=standardize,
            top_k=top_k,
        )
        selected = list(ranking.loc[ranking["selected"], "feature"])
        self._log_feature_selection(
            method="variance",
            selected=selected,
            n_candidates=len(candidate_cols),
            top_k=top_k,
            standardize=standardize,
        )
        return selected

    def _default_config(self) -> dict:
        """
        Default analysis configuration: column names and GLMM/matching parameters.

        Returns:
            dict: outcome_col, physician_col, fixed_effects list, min_patients_per_physician,
                and optional dataset_targets for expected variance/coefficients in reports.
        """
        return {
            "outcome_col": "recommendation",
            "physician_col": "professional_id",
            "fixed_effects": [
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
            ],
            "min_patients_per_physician": 30,
            "generation_strategy": "score2_five_groups_heter_patients",
            "dataset_targets": {},
        }

    # --------------------------------------------------------------------------
    # Pipeline principal : préparation sans GLMM
    # --------------------------------------------------------------------------

    def _ensure_analysis_covariates_imputed(self) -> None:
        """Run deferred MICE on matching ∪ GLMM columns before any dropna.

        Feature selection (when enabled) scores a median-filled *copy* and does
        not mutate ``self.df``. MICE then fills only the columns that matching
        and/or GLMM actually use. Idempotent via ``_analysis_imputed``.
        """
        if self._analysis_imputed:
            return

        imputation_cfg = self.config.get("imputation") or {}
        enabled = bool(imputation_cfg.get("enabled", True))
        method = str(imputation_cfg.get("method", "zero")).strip().lower()
        if not enabled or method != "mice":
            self._analysis_imputed = True
            return

        matching_cols = self._matching_covariate_columns(self.df)
        glmm_cols: list[str] = []
        if self._glmm_enabled():
            try:
                glmm_cols = self._resolve_glmm_covariates()
            except ValueError as exc:
                self.LOGGER.warning(
                    "MICE: GLMM covariate resolution failed (%s); "
                    "imputing matching columns only.",
                    exc,
                )

        columns = list(dict.fromkeys([*matching_cols, *glmm_cols]))
        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            self.LOGGER.warning("MICE: no matching/GLMM columns to impute; skipping.")
            self._analysis_imputed = True
            return

        fs_cfg = self.config.get("feature_selection") or {}
        glmm_cfg = self.config.get("glmm") or {}
        fs_disabled = not bool(fs_cfg.get("enabled", False))
        glmm_auto = (
            str(glmm_cfg.get("fixed_effects", "")).strip().lower()
            == GLMM_FIXED_EFFECTS_AUTO
        )
        if fs_disabled or glmm_auto:
            self.LOGGER.warning(
                "MICE: large covariate union (%d columns) — feature_selection "
                "enabled=%s, glmm.fixed_effects=%r. Consider scoping FS/GLMM; "
                "n_nearest_features will be capped at min(configured, p-1).",
                len(columns),
                not fs_disabled,
                glmm_cfg.get("fixed_effects"),
            )

        n_missing_before = int(self.df[columns].isna().sum().sum())
        total_cells = int(len(self.df) * len(columns)) if columns else 0
        max_iter = int(imputation_cfg.get("max_iter", 10))
        random_state = int(imputation_cfg.get("random_state", 0))
        n_nearest = imputation_cfg.get(
            "n_nearest_features", DEFAULT_MICE_N_NEAREST_FEATURES
        )
        n_nearest_features = None if n_nearest is None else int(n_nearest)

        if n_missing_before == 0:
            self.LOGGER.info(
                "MICE: no NaN in %d matching∪GLMM columns; skipping.",
                len(columns),
            )
            validation_ctx = self.results.setdefault("validation_context", {})
            validation_ctx["imputation"] = {
                "method": "mice",
                "imputed_cell_fraction": 0.0,
                "imputed_cells": 0,
                "total_cells": total_cells,
                "columns": columns,
            }
            self._analysis_imputed = True
            return

        try:
            self.df = mice_impute(
                self.df,
                columns,
                max_iter=max_iter,
                random_state=random_state,
                n_nearest_features=n_nearest_features,
            )
        except Exception as exc:
            self.LOGGER.warning(
                "MICE failed (%s); falling back to column medians on %d columns.",
                exc,
                len(columns),
            )

        # Fallback for residual NaNs (non-convergence, degenerate columns).
        residual = int(self.df[columns].isna().sum().sum())
        if residual > 0:
            self.LOGGER.warning(
                "MICE left %d NaN(s); filling with column medians before dropna.",
                residual,
            )
            for col in columns:
                n_miss = int(self.df[col].isna().sum())
                if n_miss == 0:
                    continue
                median = self.df[col].median(skipna=True)
                if pd.isna(median):
                    median = 0.0
                self.df[col] = self.df[col].fillna(float(median))

        # Median fallback (and rare non-rounded paths) can leave .5 on qa__/is_*.
        self.df = round_discrete_columns(self.df, columns)

        n_missing_after = int(self.df[columns].isna().sum().sum())
        imputed_cells = n_missing_before - n_missing_after
        validation_ctx = self.results.setdefault("validation_context", {})
        validation_ctx["imputation"] = {
            "method": "mice",
            "imputed_cell_fraction": (
                float(imputed_cells / total_cells) if total_cells else 0.0
            ),
            "imputed_cells": int(imputed_cells),
            "total_cells": total_cells,
            "columns": columns,
        }
        self.LOGGER.info(
            "MICE: filled %d cells across %d matching∪GLMM columns "
            "(fraction=%.4f).",
            imputed_cells,
            len(columns),
            float(imputed_cells / total_cells) if total_cells else 0.0,
        )
        self._analysis_imputed = True

    def run_matching_basis_prep(self) -> None:
        """
        Valide outcome / médecin / covariables, applique les règles de complétude
        du sous-ensemble de matching (sans scaler ni modèle), puis initialise
        ``intra_physician_variability`` avec les médecins ayant assez de patients.

        Le libellé d'erreur « Not enough complete data for matching » signale
        une fenêtre temporelle trop peu peuplée.
        """
        self.LOGGER.info("=== préparation du jeu de données pour le matching (sans GLMM) ===")
        self._ensure_analysis_covariates_imputed()

        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        covariates = self._matching_covariate_columns(self.df)
        min_patients = self.config["min_patients_per_physician"]

        cols_needed = [outcome_col, physician_col] + covariates
        missing_cols = [c for c in cols_needed if c not in self.df.columns]
        if missing_cols:
            error_msg = f"CRITICAL: Missing columns for matching: {missing_cols}"
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        df_clean = self.df[cols_needed].dropna().copy()
        if len(df_clean) < 50:
            error_msg = f"CRITICAL: Not enough complete data for matching. Found {len(df_clean)} rows, minimum is 50."
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        if pd.api.types.is_bool_dtype(df_clean[outcome_col]):
            df_clean[outcome_col] = df_clean[outcome_col].astype(int)
        else:
            outcome_num = pd.to_numeric(df_clean[outcome_col], errors="coerce")
            if outcome_num.isna().any():
                n_bad = int(outcome_num.isna().sum())
                bad_examples = df_clean.loc[outcome_num.isna(), outcome_col].astype(str).head(5).tolist()
                error_msg = (
                    f"CRITICAL: Outcome column '{outcome_col}' contains {n_bad} non-numeric values. "
                    f"Examples: {bad_examples}"
                )
                self.LOGGER.error(error_msg)
                raise ValueError(error_msg)
            unique_vals = sorted(pd.unique(outcome_num))
            unique_set = set(unique_vals)
            if unique_set.issubset({0, 1}):
                df_clean[outcome_col] = outcome_num.astype(int)
            elif len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                self.LOGGER.warning(
                    "Outcome column '%s' is binary but not encoded as {0,1}. Applying mapping: %s",
                    outcome_col,
                    mapping,
                )
                df_clean[outcome_col] = outcome_num.map(mapping).astype(int)
            else:
                error_msg = (
                    f"CRITICAL: Outcome column '{outcome_col}' must be binary. "
                    f"Found values: {unique_vals}"
                )
                self.LOGGER.error(error_msg)
                raise ValueError(error_msg)
        if df_clean[outcome_col].nunique() < 2:
            error_msg = f"CRITICAL: Outcome column '{outcome_col}' has a single class after cleaning."
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        for c in covariates:
            if c in df_clean.columns and pd.api.types.is_bool_dtype(df_clean[c]):
                df_clean[c] = df_clean[c].astype(int)

        if not covariates:
            raise ValueError(
                "CRITICAL: No numeric covariate columns available for matching after preprocess."
            )

        physician_metrics: list[dict[str, object]] = []
        pre_cutoff_metrics: list[dict[str, object]] = []
        grouped = df_clean.groupby(physician_col)
        for physician, group in grouped:
            n_patients = len(group)
            y_obs = group[outcome_col].values
            n_prescriptions = int(np.sum(y_obs))
            row = {
                "physician": physician,
                "n_patients": n_patients,
                "n_prescriptions": n_prescriptions,
                "prescription_rate": round(float(np.mean(y_obs)), 4),
            }
            pre_cutoff_metrics.append(row)
            if n_patients < min_patients:
                continue
            physician_metrics.append(row)

        df_metrics = pd.DataFrame(physician_metrics)
        df_pre_cutoff = pd.DataFrame(pre_cutoff_metrics)
        plot_patients_per_physician_distribution(
            df_pre_cutoff,
            self.results_dir
            / "plots"
            / PLOT_SUBDIR_PHYSICIAN_CASELOAD
            / "patients_per_physician_pre_cutoff.png",
            min_patients=int(min_patients),
            logger=self.LOGGER,
        )
        self._save_snapshot(df_pre_cutoff, "patients_per_physician_pre_cutoff_head")
        self.results["matching_basis"] = {
            "df_subset": df_clean,
            "patients_per_physician_pre_cutoff": df_pre_cutoff,
        }
        self.results["intra_physician_variability"] = df_metrics
        prior_validation = self.results.get("validation_context") or {}
        self.results["validation_context"] = {
            "matching_basis": {
                "n_raw_rows": int(len(self.df)),
                "n_clean_rows": int(len(df_clean)),
                "n_physicians_eligible": int(len(df_metrics)),
                "n_covariates": int(len(covariates)),
                "n_physicians_excluded": int(self.df[physician_col].nunique() - len(df_metrics)),
                "n_extreme_prescribers": int(
                    ((df_metrics["prescription_rate"] <= 0.05) | (df_metrics["prescription_rate"] >= 0.95)).sum()
                ),
            },
            "outliers": self._outlier_stats,
        }
        if "imputation" in prior_validation:
            self.results["validation_context"]["imputation"] = prior_validation["imputation"]
        self.LOGGER.info(
            "Matching basis: %d patients x %d columns (%d covariates).",
            len(df_clean),
            len(cols_needed),
            len(covariates),
        )
        self._save_snapshot(df_clean, "matching_basis_subset_head")
        self._save_snapshot(df_metrics, "intra_physician_metrics_head")

    def _snapshot_matching_covariates(self) -> None:
        """Export the patient-level matrix used for distance / RF matching."""
        if not self._has_matching_subset():
            return
        try:
            _, _, _, df_match = self._prepare_matching_data()
            self._save_snapshot(df_match, "matching_covariates_head")
        except Exception as exc:
            self.LOGGER.warning("Snapshot matching_covariates_head skipped: %s", exc)

    # --------------------------------------------------------------------------
    # PIPELINE METHOD 1: Parametric modelling (GLMM + OLRE)
    # --------------------------------------------------------------------------

    def run_glmm_analysis(self) -> None:
        """
        Fit a binomial GLMM, compute global and per-physician variability (OLRE), and save a master report.
        """
        self.LOGGER.info("=== STARTING GLMM & OLRE PIPELINE ===")
        self._ensure_analysis_covariates_imputed()

        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        fixed_effects = self._resolve_glmm_covariates()
        min_patients = self.config["min_patients_per_physician"]

        # On vérifie d'abord que toutes les colonnes indispensables existent.
        cols_needed = [outcome_col, physician_col] + fixed_effects
        missing_cols = [c for c in cols_needed if c not in self.df.columns]
        if missing_cols:
            error_msg = f"CRITICAL: Missing columns for GLMM: {missing_cols}"
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # On travaille sur un sous-ensemble complet (dropna) pour garantir
        # que le modèle statistique reçoit des entrées valides.
        df_clean = self.df[cols_needed].dropna().copy()
        if len(df_clean) < 50:
            error_msg = f"CRITICAL: Not enough complete data for GLMM. Found {len(df_clean)} rows, minimum is 50."
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # Normalisation robuste de l'outcome en binaire 0/1.
        # On tolère les booléens et les colonnes numériques. Pour les colonnes
        # non booléennes contenant exactement 2 modalités numériques (ex: {1,2}),
        # on applique un mapping stable vers {0,1} avec warning explicite.
        if pd.api.types.is_bool_dtype(df_clean[outcome_col]):
            df_clean[outcome_col] = df_clean[outcome_col].astype(int)
        else:
            outcome_num = pd.to_numeric(df_clean[outcome_col], errors="coerce")
            if outcome_num.isna().any():
                n_bad = int(outcome_num.isna().sum())
                bad_examples = df_clean.loc[outcome_num.isna(), outcome_col].astype(str).head(5).tolist()
                error_msg = (
                    f"CRITICAL: Outcome column '{outcome_col}' contains {n_bad} non-numeric values. "
                    f"Examples: {bad_examples}"
                )
                self.LOGGER.error(error_msg)
                raise ValueError(error_msg)
            unique_vals = sorted(pd.unique(outcome_num))
            unique_set = set(unique_vals)
            if unique_set.issubset({0, 1}):
                df_clean[outcome_col] = outcome_num.astype(int)
            elif len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                self.LOGGER.warning(
                    "Outcome column '%s' is binary but not encoded as {0,1}. Applying mapping: %s",
                    outcome_col,
                    mapping,
                )
                df_clean[outcome_col] = outcome_num.map(mapping).astype(int)
            else:
                error_msg = (
                    f"CRITICAL: Outcome column '{outcome_col}' must be binary. "
                    f"Found values: {unique_vals}"
                )
                self.LOGGER.error(error_msg)
                raise ValueError(error_msg)
        if df_clean[outcome_col].nunique() < 2:
            error_msg = f"CRITICAL: Outcome column '{outcome_col}' has a single class after cleaning."
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)
        # Standardisation uniquement pour les variables numériques "continues"
        # (évite d'écraser les variables binaires à 2 modalités).
        # Les booléens des covariables sont castés en int pour éviter les surprises
        # de typage lors du fit (tout en gardant leur nature binaire).
        for c in fixed_effects:
            if c in df_clean.columns and pd.api.types.is_bool_dtype(df_clean[c]):
                df_clean[c] = df_clean[c].astype(int)
        numeric_cols = [
            c for c in fixed_effects
            if c in df_clean.columns
            and pd.api.types.is_numeric_dtype(df_clean[c])
            and df_clean[c].nunique() > 2
        ]
        if numeric_cols:
            scaler = StandardScaler()
            df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

        if not fixed_effects:
            raise ValueError("CRITICAL: The 'fixed_effects' list is empty. At least one fixed effect is required.")

        # Fin de la préparation des données pour le GLMM.
        # ------------------------------------------------------------
        # Formule GLMM: effets fixes + intercept, avec effet aléatoire médecin.
        formula = f"{outcome_col} ~ {' + '.join(fixed_effects)}"
        vc_formulas = {"physician_re": f"0 + C({physician_col})"}

        seed_value = 42
        np.random.seed(seed_value)
        random.seed(seed_value)

        # Ajustement du GLMM bayésien variationnel.
        # On est dans le bloc 2: Model fitting, post préparation des données.
        n_obs = len(df_clean)
        n_phys = int(df_clean[physician_col].nunique())
        n_fe = len(fixed_effects)
        self.LOGGER.info(
            "Fitting binomial GLMM via VB on %d patients x %d fixed effects "
            "with %d physician random effects... live progress below.",
            n_obs,
            n_fe,
            n_phys,
        )
        try:
            model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas=vc_formulas, data=df_clean)
            with _live_glmm_progress(self.LOGGER):
                result = model.fit_vb()
        except Exception as exc:
            error_msg = f"CRITICAL: GLMM fitting failed for formula '{formula}'. Error: {exc}"
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg) from exc

        # Fin de l'ajustement du GLMM bayésien variationnel.
        # ------------------------------------------------------------

        summary_text = str(result.summary())
        # Poids "globaux" dérivés des coefficients fixes (utilisés pour reporting).
        fixed_effect_weights = _glmm_fixed_effect_weights(result)

        # Hétérogénéité inter-médecins de l'effet aléatoire.
        # ATTENTION: dans statsmodels, `vcp_mean` porte la moyenne a posteriori du
        # *log* de l'écart-type de chaque composante de variance, pas une variance
        # (cf. summary(): "Variance parameters are modeled as log standard
        # deviations", et la colonne "SD" du tableau vaut exp(Post. Mean)).
        # On expose donc les trois grandeurs explicitement, sans valeur de repli
        # trompeuse: 0.0 en log-échelle voudrait dire SD = 1.
        physician_re_log_sd = None
        physician_re_sd = None
        physician_re_variance = None
        if hasattr(result, 'vcp_mean') and result.vcp_mean is not None:
            vcp = np.array(result.vcp_mean).flatten()
            if len(vcp) > 0 and not np.isnan(vcp[0]):
                physician_re_log_sd = float(vcp[0])
                physician_re_sd = float(np.exp(physician_re_log_sd))
                physician_re_variance = float(physician_re_sd ** 2)
        if physician_re_variance is None:
            self.LOGGER.warning(
                "GLMM physician random-effect scale unavailable (vcp_mean missing or NaN); "
                "log_sd/sd/variance exported as null."
            )
        else:
            self.LOGGER.info(
                "GLMM physician random effect: log_sd=%.4f, sd=%.4f, variance=%.4f.",
                physician_re_log_sd,
                physician_re_sd,
                physician_re_variance,
            )

        # Probabilité prédite patient par patient.
        if hasattr(result, "predict"):
            p_pred = result.predict()
        else:
            lin_pred = model.predict(result.params)
            p_pred = 1 / (1 + np.exp(-lin_pred))

        if hasattr(p_pred, "flatten"):
            p_pred = p_pred.flatten()
        elif isinstance(p_pred, (pd.Series, pd.DataFrame)):
            p_pred = p_pred.values.flatten()
        p_pred = np.asarray(p_pred, dtype=float).reshape(-1)
        if p_pred.shape[0] != len(df_clean):
            error_msg = (
                f"CRITICAL: Predicted probability length mismatch: got {p_pred.shape[0]}, "
                f"expected {len(df_clean)}."
            )
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)
        if not np.all(np.isfinite(p_pred)):
            self.LOGGER.warning("Non-finite predicted probabilities detected; replacing with 0.5.")
            p_pred = np.where(np.isfinite(p_pred), p_pred, 0.5)
        p_pred = np.clip(p_pred, 1e-10, 1 - 1e-10)
        df_clean["_predicted_prob"] = p_pred

        # Extraction des effets aléatoires individuels médecin.
        # On est dans le bloc 3: Extract global variance, predicted probabilities, and per-physician random effects.
        # Selon la structure de summary() (qui varie parfois), on tente plusieurs chemins.
        def _normalize_physician_label(raw_label: object) -> str:
            label = str(raw_label).strip()
            label = label.strip("'").strip('"')
            if label.startswith("T."):
                label = label[2:]
            if "[" in label and "]" in label:
                label = label.split("[")[-1].split("]")[0]
                label = label.strip("'").strip('"')
                if label.startswith("T."):
                    label = label[2:]
            return label

        physician_labels = [str(p) for p in pd.unique(df_clean[physician_col])]
        physician_bias = {p: 0.0 for p in physician_labels}
        try:
            # Chemin 1 (prioritaire): utiliser la correspondance explicite vc_names <-> vc_mean.
            vc_names = getattr(model, "vc_names", None)
            vc_mean = getattr(result, "vc_mean", None)
            mapped = 0
            if vc_names is not None and vc_mean is not None:
                vc_names_list = list(vc_names)
                vc_vals = np.asarray(vc_mean).flatten()
                if len(vc_names_list) == len(vc_vals):
                    for name, val in zip(vc_names_list, vc_vals):
                        key = _normalize_physician_label(name)
                        if key in physician_bias and np.isfinite(val):
                            physician_bias[key] = float(val)
                            mapped += 1
                else:
                    self.LOGGER.warning(
                        "Random-effect mapping mismatch: %d vc_names vs %d vc_mean values.",
                        len(vc_names_list),
                        len(vc_vals),
                    )

            # Chemin 2: parser la table de summary si le mapping explicite est incomplet.
            summary_tables = result.summary().tables
            if mapped < len(physician_bias) and len(summary_tables) > 1:
                summary_df = summary_tables[1]
                re_rows = [idx for idx in summary_df.index if 'physician_re' in idx]
                if re_rows:
                    post_mean_col = next((c for c in summary_df.columns if 'mean' in str(c).lower()), None)
                    for row_name in re_rows:
                        clean_name = _normalize_physician_label(row_name)
                        if post_mean_col is not None:
                            try:
                                val = float(summary_df.loc[row_name, post_mean_col])
                                if clean_name in physician_bias and np.isfinite(val):
                                    physician_bias[clean_name] = val
                            except (KeyError, TypeError, ValueError):
                                continue

            # Chemin 3 (ultime fallback): si rien n'a pu être aligné nominalement mais les tailles
            # correspondent, on applique un zip heuristique en conservant l'ordre d'apparition.
            if all(v == 0.0 for v in physician_bias.values()):
                vcp = np.asarray(getattr(result, "vc_mean", np.array([]))).flatten()
                if len(vcp) == len(physician_labels):
                    self.LOGGER.warning(
                        "Using heuristic fallback for physician random effects (order-based zip)."
                    )
                    for p, v in zip(physician_labels, vcp):
                        if np.isfinite(v):
                            physician_bias[p] = float(v)
        except Exception as e:
            self.LOGGER.warning(f"Failed to parse physician random effects robustly, keeping defaults. Error: {e}")

        # Calcul métriques locales par médecin:
        # - taux de prescription observé,
        # - biais aléatoire estimé,
        # - overdispersion locale via résidus de déviance signés.
        #
        # Les résidus de déviance sont la métrique de référence pour les GLM
        # binomiaux (voir McCullagh & Nelder, 1989). Ils sont sensiblement plus
        # stables que les résidus de Pearson aux probabilités prédites
        # extrêmes, ce qui évite que les médecins à taux de prescription très
        # bas (ou très haut) se voient attribuer un OLRE artificiellement
        # gonflé par un seul cas rare.
        physician_metrics = []
        all_deviance_residuals: list[float] = []
        deviance_residuals_by_physician: dict[str, np.ndarray] = {}
        all_actual_values: list[float] = []
        all_fitted_values: list[float] = []
        actual_fitted_by_physician: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        epsilon = 1e-10
        grouped = df_clean.groupby(physician_col)
        for physician, group in grouped:
            n_patients = len(group)
            u_j = physician_bias.get(str(physician), 0.0)
            p_avg = group["_predicted_prob"].values
            p_avg_clipped = np.clip(p_avg, epsilon, 1 - epsilon)
            z_adjusted = np.log(p_avg_clipped / (1 - p_avg_clipped)) + u_j
            p_specific = 1 / (1 + np.exp(-z_adjusted))
            y_obs = group[outcome_col].values

            # Résidus de déviance signés pour une réponse binomiale Bernoulli.
            # Convention : 0*log(0/x) = 0 (limite). Le clip évite log(0).
            p_safe = np.clip(p_specific, epsilon, 1 - epsilon)
            term_y1 = np.where(y_obs > 0, y_obs * np.log(np.clip(y_obs, epsilon, None) / p_safe), 0.0)
            term_y0 = np.where(
                y_obs < 1,
                (1 - y_obs) * np.log(np.clip(1 - y_obs, epsilon, None) / (1 - p_safe)),
                0.0,
            )
            deviance_squared = 2.0 * (term_y1 + term_y0)
            deviance_squared = np.clip(deviance_squared, 0.0, None)
            deviance_residuals = np.sign(y_obs - p_specific) * np.sqrt(deviance_squared)

            finite_af = np.isfinite(y_obs) & np.isfinite(p_specific)
            if np.any(finite_af):
                y_obs_finite = np.asarray(y_obs[finite_af], dtype=float)
                p_specific_finite = np.asarray(p_specific[finite_af], dtype=float)
                all_actual_values.extend(y_obs_finite.tolist())
                all_fitted_values.extend(p_specific_finite.tolist())
                actual_fitted_by_physician[str(physician)] = (
                    y_obs_finite.copy(),
                    p_specific_finite.copy(),
                )

            deviance_residuals = deviance_residuals[np.isfinite(deviance_residuals)]
            if deviance_residuals.size:
                all_deviance_residuals.extend(deviance_residuals.tolist())
                deviance_residuals_by_physician[str(physician)] = deviance_residuals.copy()
            if n_patients < min_patients:
                continue
            overdispersion = (
                float(np.mean(deviance_residuals ** 2)) if deviance_residuals.size else np.nan
            )
            physician_metrics.append(
                {
                    "physician": physician,
                    "n_patients": n_patients,
                    "prescription_rate": round(y_obs.mean(), 4),
                    "random_effect_bias": round(u_j, 4),
                    "overdispersion_local": round(overdispersion, 4),
                }
            )

        df_metrics = pd.DataFrame(physician_metrics)
        # On centralise les sorties GLMM pour qu'elles soient réutilisables
        # par toutes les étapes aval (matching + graphiques + export final).
        self.results["glmm"] = {
            "model": result,
            "summary": summary_text,
            "fixed_effect_weights": fixed_effect_weights,
            # Voir le commentaire au calcul: log_sd est la valeur brute statsmodels,
            # sd = exp(log_sd), variance = sd**2. `variance_physician` porte bien
            # une variance depuis ce correctif (auparavant: le log-SD).
            "log_sd_physician": physician_re_log_sd,
            "sd_physician": physician_re_sd,
            "variance_physician": physician_re_variance,
            "df_with_residuals": df_clean,
            "n_physicians": df_clean[physician_col].nunique(),
            "n_observations": len(df_clean),
            "fixed_effects_used": fixed_effects,
        }
        merge_cols = ["physician", "overdispersion_local", "random_effect_bias"]
        self._merge_intra_physician_result(df_metrics[merge_cols])

        glmm_dir = self.results_dir / "glmm"
        glmm_dir.mkdir(parents=True, exist_ok=True)
        master_report = {
            "metadata": {
                "n_observations": len(df_clean),
                "n_physicians": df_clean[physician_col].nunique(),
                "fixed_effects": fixed_effects,
                "min_patients_threshold": min_patients,
            },
            "dataset_targets": self.config.get("dataset_targets", {}),
            "glmm_global_results": {
                # `empirical_variance_absolute` (retiré) contenait en réalité le
                # log-écart-type; il est remplacé par les trois champs explicites.
                "physician_re_log_sd": physician_re_log_sd,
                "physician_re_sd": physician_re_sd,
                "physician_re_variance": physician_re_variance,
                "model_summary_raw": summary_text,
                "fixed_effect_weights": fixed_effect_weights,
            },
            "physician_metrics": df_metrics.to_dict(orient="records"),
        }
        master_file_path = glmm_dir / "glmm_master_report.json"
        with open(master_file_path, "w", encoding="utf-8") as f:
            json.dump(master_report, f, indent=4, ensure_ascii=False)

        def _safe_physician_label(label: str) -> str:
            safe_label = "".join(
                ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
                for ch in str(label)
            )
            return safe_label or "unknown_physician"

        def _calibration_curve_points(
            actual_arr: np.ndarray,
            fitted_arr: np.ndarray,
            n_bins: int = 10,
            min_bin_size: int = 5,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if actual_arr.size == 0 or fitted_arr.size == 0:
                return np.array([]), np.array([]), np.array([])
            cal_df = pd.DataFrame({"actual": actual_arr, "fitted": fitted_arr}).dropna()
            if cal_df.empty:
                return np.array([]), np.array([]), np.array([])
            try:
                cal_df["bin"] = pd.qcut(cal_df["fitted"], q=min(n_bins, len(cal_df)), duplicates="drop")
            except ValueError:
                cal_df["bin"] = pd.cut(cal_df["fitted"], bins=min(n_bins, len(cal_df)), duplicates="drop")
            grouped_cal = (
                cal_df.groupby("bin", observed=False)
                .agg(mean_fitted=("fitted", "mean"), observed_rate=("actual", "mean"), n=("actual", "size"))
                .dropna()
            )
            grouped_cal = grouped_cal[grouped_cal["n"] >= min_bin_size]
            if grouped_cal.empty:
                grouped_cal = (
                    cal_df.groupby("bin", observed=False)
                    .agg(mean_fitted=("fitted", "mean"), observed_rate=("actual", "mean"), n=("actual", "size"))
                    .dropna()
                )
            if grouped_cal.empty:
                return np.array([]), np.array([]), np.array([])
            return (
                grouped_cal["mean_fitted"].to_numpy(dtype=float),
                grouped_cal["observed_rate"].to_numpy(dtype=float),
                grouped_cal["n"].to_numpy(dtype=float),
            )

        if all_deviance_residuals:
            deviance_residuals_arr = np.asarray(all_deviance_residuals, dtype=float)
            plt.figure(figsize=(10, 6))
            sns.histplot(deviance_residuals_arr, bins=50, stat="density", kde=True, color="#4C72B0", alpha=0.35)
            plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
            plt.title("Distribution of deviance residuals (all observations)")
            plt.xlabel("Deviance residual")
            plt.ylabel("Density")
            plt.tight_layout()
            deviance_plot_path = glmm_dir / "deviance_residuals_distribution.png"
            plt.savefig(deviance_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.results["glmm"]["deviance_residuals_plot"] = str(deviance_plot_path)

        if deviance_residuals_by_physician:
            by_physician_dir = glmm_dir / "deviance_residuals_by_physician"
            by_physician_dir.mkdir(parents=True, exist_ok=True)
            by_physician_paths: dict[str, str] = {}
            for physician_label, residuals_arr in deviance_residuals_by_physician.items():
                safe_physician_label = _safe_physician_label(physician_label)
                plt.figure(figsize=(10, 6))
                sns.histplot(
                    residuals_arr,
                    bins=30,
                    stat="density",
                    kde=bool(residuals_arr.size > 1),
                    color="#55A868",
                    alpha=0.35,
                )
                plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
                plt.title(f"Deviance residuals - physician {physician_label} (n={residuals_arr.size})")
                plt.xlabel("Deviance residual")
                plt.ylabel("Density")
                plt.tight_layout()
                physician_plot_path = by_physician_dir / f"deviance_residuals_{safe_physician_label}.png"
                plt.savefig(physician_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                by_physician_paths[physician_label] = str(physician_plot_path)
            self.results["glmm"]["deviance_residuals_plots_by_physician"] = by_physician_paths

        if all_actual_values and all_fitted_values:
            actual_arr = np.asarray(all_actual_values, dtype=float)
            fitted_arr = np.asarray(all_fitted_values, dtype=float)
            x_jitter = np.random.uniform(-0.03, 0.03, size=actual_arr.size)
            plt.figure(figsize=(10, 6))
            plt.scatter(actual_arr + x_jitter, fitted_arr, alpha=0.25, s=14, color="#C44E52", edgecolors="none")
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.02, 1.02)
            plt.xticks([0, 1], ["0", "1"])
            plt.xlabel("Actual")
            plt.ylabel("Fitted")
            plt.title("Actual vs fitted probabilities (all observations)")
            plt.grid(axis="y", alpha=0.2)
            plt.tight_layout()
            actual_fitted_plot_path = glmm_dir / "actual_vs_fitted.png"
            plt.savefig(actual_fitted_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.results["glmm"]["actual_vs_fitted_plot"] = str(actual_fitted_plot_path)

        if actual_fitted_by_physician:
            by_physician_af_dir = glmm_dir / "actual_vs_fitted_by_physician"
            by_physician_af_dir.mkdir(parents=True, exist_ok=True)
            by_physician_af_paths: dict[str, str] = {}
            for physician_label, (actual_arr, fitted_arr) in actual_fitted_by_physician.items():
                if actual_arr.size == 0 or fitted_arr.size == 0:
                    continue
                safe_physician_label = _safe_physician_label(physician_label)
                x_jitter = np.random.uniform(-0.03, 0.03, size=actual_arr.size)
                plt.figure(figsize=(10, 6))
                plt.scatter(actual_arr + x_jitter, fitted_arr, alpha=0.35, s=14, color="#8172B2", edgecolors="none")
                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.02, 1.02)
                plt.xticks([0, 1], ["0", "1"])
                plt.xlabel("Actual")
                plt.ylabel("Fitted")
                plt.title(f"Actual vs fitted - physician {physician_label} (n={actual_arr.size})")
                plt.grid(axis="y", alpha=0.2)
                plt.tight_layout()
                physician_af_path = by_physician_af_dir / f"actual_vs_fitted_{safe_physician_label}.png"
                plt.savefig(physician_af_path, dpi=300, bbox_inches="tight")
                plt.close()
                by_physician_af_paths[physician_label] = str(physician_af_path)
            self.results["glmm"]["actual_vs_fitted_plots_by_physician"] = by_physician_af_paths

        if all_actual_values and all_fitted_values:
            actual_arr = np.asarray(all_actual_values, dtype=float)
            fitted_arr = np.asarray(all_fitted_values, dtype=float)
            x_curve, y_curve, n_curve = _calibration_curve_points(actual_arr, fitted_arr, n_bins=10, min_bin_size=10)
            if x_curve.size and y_curve.size:
                marker_size = np.clip(12 + np.sqrt(n_curve) * 2.5, 16, 70)
                plt.figure(figsize=(8, 8))
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Perfect calibration")
                plt.scatter(x_curve, y_curve, s=marker_size, alpha=0.85, color="#4C72B0")
                plt.plot(x_curve, y_curve, color="#4C72B0", linewidth=1.4)
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.xlabel("Mean fitted probability (bin)")
                plt.ylabel("Observed outcome rate (bin)")
                plt.title("Calibration curve (all observations)")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                calib_plot_path = glmm_dir / "calibration_curve.png"
                plt.savefig(calib_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                self.results["glmm"]["calibration_curve_plot"] = str(calib_plot_path)

        if actual_fitted_by_physician:
            by_physician_calib_dir = glmm_dir / "calibration_curve_by_physician"
            by_physician_calib_dir.mkdir(parents=True, exist_ok=True)
            by_physician_calib_paths: dict[str, str] = {}
            for physician_label, (actual_arr, fitted_arr) in actual_fitted_by_physician.items():
                x_curve, y_curve, n_curve = _calibration_curve_points(actual_arr, fitted_arr, n_bins=6, min_bin_size=5)
                if not (x_curve.size and y_curve.size):
                    continue
                safe_physician_label = _safe_physician_label(physician_label)
                marker_size = np.clip(14 + np.sqrt(n_curve) * 2.2, 16, 64)
                plt.figure(figsize=(8, 8))
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Perfect calibration")
                plt.scatter(x_curve, y_curve, s=marker_size, alpha=0.85, color="#55A868")
                plt.plot(x_curve, y_curve, color="#55A868", linewidth=1.4)
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.xlabel("Mean fitted probability (bin)")
                plt.ylabel("Observed outcome rate (bin)")
                plt.title(f"Calibration curve - physician {physician_label} (n={actual_arr.size})")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                physician_calib_path = by_physician_calib_dir / f"calibration_curve_{safe_physician_label}.png"
                plt.savefig(physician_calib_path, dpi=300, bbox_inches="tight")
                plt.close()
                by_physician_calib_paths[physician_label] = str(physician_calib_path)
            self.results["glmm"]["calibration_curve_plots_by_physician"] = by_physician_calib_paths

    # --------------------------------------------------------------------------
    # PIPELINE METHODS 2–6: Matching-based discordance
    # --------------------------------------------------------------------------

    def _prepare_matching_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prépare les entrées communes pour toutes les méthodes de matching.

        Retour:
            X_all: matrice de covariables numériques pour le matching.
            Y_all: vecteur outcome binaire.
            physicians_all: vecteur id médecin.
            df_match: DataFrame consolidé (outcome, physician, covariables).
        """
        # On ne garde que les lignes déjà retenues par la préparation (GLMM ou matching_basis),
        # pour que le matching travaille exactement sur la même population.
        basis = self._matching_basis_df()
        if basis is None:
            raise RuntimeError("Matching subset not prepared; call run_matching_basis_prep() or run_glmm_analysis() first.")
        df_source = self.df.loc[basis.index].copy()
        # Noms des deux colonnes "cibles" : le résultat clinique et l'identifiant du médecin.
        outcome_col, physician_col = self.config["outcome_col"], self.config["physician_col"]
        # Colonnes à EXCLURE du matching :
        # - outcome_col et physician_col (ce ne sont pas des covariables de matching),
        # - colonnes techniques internes générées dans le pipeline,
        # - identifiants patients qui ne doivent pas guider l'appariement.
        exclude_cols = [outcome_col, physician_col, "_predicted_prob", "_residual", "_obs_id", "member_pseudo_id", "person_id"]
        # On exclut aussi toute colonne "outcome_*" pour éviter d'introduire d'autres outcomes
        # dans la distance de matching.
        exclude_cols = exclude_cols + [c for c in df_source.columns if str(c).startswith("outcome_")]
        matching_cols = self._matching_covariate_columns(df_source)

        # Si on n'a toujours aucune variable de matching, on arrête explicitement avec un message clair.
        if len(matching_cols) == 0:
            raise ValueError(
                "No numeric or boolean covariate columns available for matching. "
                "Ensure the dataset has at least one matching covariate (e.g. fixed_effects) in addition to outcome and physician ID."
            )

        # DataFrame de travail minimal : outcome, médecin, et covariables de matching.
        df_match = df_source[[outcome_col, physician_col] + matching_cols].copy()
        imputed_cells = 0
        total_cells = int(len(df_match) * len(matching_cols)) if matching_cols else 0
        for col in matching_cols:
            # Les booléens sont convertis en float pour homogénéiser les calculs de distance.
            if pd.api.types.is_bool_dtype(df_match[col]):
                df_match[col] = df_match[col].astype(float)
            # Imputation simple : on remplace les NaN par la médiane de la colonne.
            # Cela évite les erreurs de matching liées aux valeurs manquantes.
            n_missing = int(df_match[col].isna().sum())
            if n_missing > 0:
                imputed_cells += n_missing
                df_match[col] = df_match[col].fillna(df_match[col].median())
                self.LOGGER.warning(
                    "WARNING: %s column has %d missing values, filled with median.",
                    col,
                    n_missing,
                )
        validation_ctx = self.results.setdefault("validation_context", {})
        validation_ctx["df_match"] = df_match
        # Preserve MICE stats when deferred imputation already ran; only report
        # residual median fills when method is zero or MICE left gaps.
        prior_imputation = validation_ctx.get("imputation") or {}
        if str(prior_imputation.get("method", "")).lower() == "mice" and imputed_cells == 0:
            pass  # keep MICE imputed_cell_fraction
        elif str(prior_imputation.get("method", "")).lower() == "mice" and imputed_cells > 0:
            prior_cells = int(prior_imputation.get("imputed_cells", 0))
            prior_total = int(prior_imputation.get("total_cells", total_cells) or total_cells)
            combined = prior_cells + imputed_cells
            validation_ctx["imputation"] = {
                **prior_imputation,
                "imputed_cells": combined,
                "imputed_cell_fraction": (
                    float(combined / prior_total) if prior_total else 0.0
                ),
                "residual_median_cells": imputed_cells,
            }
        else:
            validation_ctx["imputation"] = {
                "method": "median_fallback",
                "imputed_cell_fraction": (
                    float(imputed_cells / total_cells) if total_cells else 0.0
                ),
                "imputed_cells": imputed_cells,
                "total_cells": total_cells,
            }

        # Sortie au format attendu par les méthodes de matching :
        # X_all = matrice des covariables, Y_all = outcome, physicians_all = IDs médecins.
        X_all = df_match[matching_cols].values
        Y_all = df_match[outcome_col].values
        physicians_all = df_match[physician_col].values
        return X_all, Y_all, physicians_all, df_match

    def _record_matching_pairs_for_validation(
        self,
        df_match: pd.DataFrame,
        ref_indices: np.ndarray,
        match_indices: np.ndarray,
        method_name: str,
    ) -> None:
        """Store pair indices for post-run SMD / balance validation."""
        if len(ref_indices) == 0 or len(match_indices) == 0:
            return
        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        feature_cols = [
            c for c in df_match.columns if c not in {outcome_col, physician_col}
        ]
        validation_ctx = self.results.setdefault("validation_context", {})
        pairs_store = validation_ctx.setdefault("matching_pairs", {})
        pairs_store[method_name] = {
            "ref_indices": np.asarray(ref_indices, dtype=int).tolist(),
            "match_indices": np.asarray(match_indices, dtype=int).tolist(),
            "feature_cols": feature_cols,
        }

    def _export_matched_pairs_to_csv(self, df_source: pd.DataFrame, ref_indices: np.ndarray, match_indices: np.ndarray, method_name: str) -> None:
        """
        Exporte les paires A/B retenues par une méthode de matching.

        Le CSV produit sert surtout à la revue clinique: comparer deux patients
        supposés "similaires" et vérifier la cohérence de la décision.
        """
        if len(ref_indices) == 0 or len(match_indices) == 0:
            return
        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        exclude_for_export = {
            outcome_col,
            physician_col,
            "n_recos",
            "n_target_recos",
            "member_pseudo_id",
            "person_id",
        } | {c for c in df_source.columns if str(c).startswith("outcome_")}
        feature_cols = [c for c in df_source.columns if c not in exclude_for_export]
        df_A = df_source.iloc[ref_indices].reset_index(drop=True)
        df_B = df_source.iloc[match_indices].reset_index(drop=True)
        physician_id = df_A[physician_col].rename("physician_id")
        id_frames: list[pd.Series] = []
        for id_col in ("member_pseudo_id", "person_id"):
            if id_col in df_source.columns:
                id_frames.append(df_A[id_col].rename(f"patient_A_{id_col}"))
                id_frames.append(df_B[id_col].rename(f"patient_B_{id_col}"))
        patient_A_outcome = df_A[outcome_col].rename("patient_A_outcome")
        patient_B_outcome = df_B[outcome_col].rename("patient_B_outcome")
        df_A_feat = df_A[feature_cols].add_suffix("_A")
        df_B_feat = df_B[feature_cols].add_suffix("_B")
        paired = pd.concat(
            [physician_id, *id_frames, patient_A_outcome, patient_B_outcome, df_A_feat, df_B_feat],
            axis=1,
        )
        out_dir = self.results_dir / "clinical_reviews"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{method_name}_pairs.csv"
        paired.to_csv(out_path, index=False)
        self.LOGGER.info("Clinical validation export: %s (%d pairs)", out_path, len(paired))
        medication = str(self.config.get("medication_key", "statin"))
        build_clinical_review_html_safe(
            out_path,
            paired_review_html_path(out_path),
            medication=medication,
        )
        self._record_matching_pairs_for_validation(
            df_source, ref_indices, match_indices, method_name
        )

    def _matching_basis_df(self) -> pd.DataFrame | None:
        """DataFrame index servant de référence commune au matching (GLMM ou préparation seule)."""
        mb = self.results.get("matching_basis")
        if isinstance(mb, dict):
            sub = mb.get("df_subset")
            if isinstance(sub, pd.DataFrame) and len(sub.index):
                return sub
        glmm = self.results.get("glmm")
        if isinstance(glmm, dict) and glmm.get("error") is None:
            legacy = glmm.get("df_with_residuals")
            if isinstance(legacy, pd.DataFrame) and len(legacy.index):
                return legacy
        return None

    def _has_matching_subset(self) -> bool:
        """True si une base patient indexée est disponible pour les méthodes de matching."""
        return self._matching_basis_df() is not None

    def _standardize_matrix(self, X: np.ndarray) -> np.ndarray:
        """Standardisation colonne par colonne (z-score) avec epsilon de stabilité."""
        X = np.asarray(X, dtype=float)
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)

    def _merge_intra_physician_result(self, df_res: pd.DataFrame) -> None:
        """Fusionne un score par médecin dans la table agrégée principale."""
        if df_res.empty:
            return
        if "intra_physician_variability" not in self.results:
            self.results["intra_physician_variability"] = df_res.copy()
            return
        self.results["intra_physician_variability"] = pd.merge(
            self.results["intra_physician_variability"], df_res, on="physician", how="left"
        )

    def _resolve_matching_caliper(self, metric_col: str) -> float | None:
        """Return the distance caliper for ``metric_col``, or None when disabled."""
        matching_cfg = self.config.get("matching") or {}
        if not matching_cfg.get("caliper_enabled", False):
            return None
        if metric_col == "discordance_rate_rf_matching":
            return float(matching_cfg.get("caliper_rf", 0.3))
        if metric_col == "discordance_rate_mahalanobis":
            return float(
                matching_cfg.get(
                    "caliper_mahalanobis",
                    matching_cfg.get("caliper_sd", 0.2),
                )
            )
        return float(matching_cfg.get("caliper_sd", 0.2))

    def _min_coverage_warn_threshold(self) -> float:
        matching_cfg = self.config.get("matching") or {}
        try:
            return float(matching_cfg.get("min_coverage_warn", 0.5))
        except (TypeError, ValueError):
            return 0.5

    def _log_nn_reuse_summary(
        self,
        physicians_all: np.ndarray,
        ref_flat: np.ndarray,
        match_flat: np.ndarray,
        metric_col: str,
        total_ties: int,
        caliper: float | None,
        coverage_by_physician: list[float],
    ) -> None:
        """Log reuse, caliper, and coverage diagnostics after NN pairing."""
        if ref_flat.size == 0 or match_flat.size == 0:
            caliper_label = "none" if caliper is None else f"{caliper:.4g}"
            self.LOGGER.info(
                "pairing=nn_with_replacement metric=%s caliper=%s: no eligible pairs",
                metric_col,
                caliper_label,
            )
            return
        df_pairs = pd.DataFrame(
            {
                "physician": physicians_all[ref_flat],
                "match_idx": match_flat,
            }
        )
        reuse_rates: list[float] = []
        for _, grp in df_pairs.groupby("physician", sort=False):
            reuse = match_reuse_stats(grp["match_idx"].to_numpy(dtype=int))
            rate = reuse["reuse_rate"]
            if isinstance(rate, (int, float)) and np.isfinite(rate):
                reuse_rates.append(float(rate))
        median_reuse = float(np.median(reuse_rates)) if reuse_rates else 0.0
        max_reuse = float(np.max(reuse_rates)) if reuse_rates else 0.0
        cov_arr = np.asarray(coverage_by_physician, dtype=float)
        cov_arr = cov_arr[np.isfinite(cov_arr)]
        median_cov = float(np.median(cov_arr)) if cov_arr.size else float("nan")
        min_cov = float(np.min(cov_arr)) if cov_arr.size else float("nan")
        caliper_label = "none" if caliper is None else f"{caliper:.4g}"
        self.LOGGER.info(
            "pairing=nn_with_replacement metric=%s caliper=%s n_pairs=%d "
            "reuse_rate_median=%.4f reuse_rate_max=%.4f "
            "coverage_median=%.4f coverage_min=%.4f n_ties=%d",
            metric_col,
            caliper_label,
            int(len(ref_flat)),
            median_reuse,
            max_reuse,
            median_cov,
            min_cov,
            int(total_ties),
        )
        if total_ties > 0:
            self.LOGGER.debug(
                "pairing=nn_with_replacement metric=%s: %d row(s) with distance ties (argmin tie-break)",
                metric_col,
                int(total_ties),
            )

    def _distance_pairs_by_physician(
        self,
        y_all: np.ndarray,
        physicians_all: np.ndarray,
        metric_col: str,
        local_distance_builder: Callable[[np.ndarray], np.ndarray],
        clip_to_unit_interval: bool = False,
        caliper: float | None = None,
        apply_config_caliper: bool = True,
        ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Moteur commun:
        - calcule une matrice de distance locale par médecin,
        - construit des paires nearest-neighbor avec remise (argmin par ligne),
        - applique un caliper optionnel sur la distance du match retenu,
        - calcule un taux de discordance outcome dans les paires admissibles.

        Note mémoire: on évite volontairement toute matrice globale NxN
        pour limiter l'empreinte quand N est grand.
        """
        if apply_config_caliper and caliper is None:
            caliper = self._resolve_matching_caliper(metric_col)
        min_coverage_warn = self._min_coverage_warn_threshold()
        all_ref_indices: list[np.ndarray] = []
        all_match_indices: list[np.ndarray] = []
        discordance_results: list[dict[str, object]] = []
        reuse_detail_rows: list[dict[str, object]] = []
        coverage_by_physician: list[float] = []
        total_ties = 0
        for physician in np.unique(physicians_all):
            idx = np.where(physicians_all == physician)[0]
            if len(idx) < self.config["min_patients_per_physician"]:
                continue
            local_dists = np.asarray(local_distance_builder(idx), dtype=float)
            if local_dists.shape != (len(idx), len(idx)):
                self.LOGGER.warning(
                    "Invalid local distance shape for physician %s: expected (%d, %d), got %s. Skipping.",
                    physician, len(idx), len(idx), tuple(local_dists.shape)
                )
                continue
            if not np.all(np.isfinite(local_dists)):
                np.nan_to_num(local_dists, copy=False, nan=1e12, posinf=1e12, neginf=1e12)
            y_local = y_all[idx]
            nn_local, diag = nearest_neighbor_with_replacement(local_dists, caliper=caliper)
            total_ties += int(diag.get("n_ties", 0))
            coverage = float(diag.get("coverage", float("nan")))
            coverage_by_physician.append(coverage)
            if np.isfinite(coverage) and coverage < min_coverage_warn:
                self.LOGGER.warning(
                    "Low matching coverage for physician %s metric=%s: coverage=%.3f (threshold=%.3f)",
                    physician,
                    metric_col,
                    coverage,
                    min_coverage_warn,
                )
            valid = nn_local >= 0
            if np.any(valid):
                ref_local = np.where(valid)[0]
                all_ref_indices.append(idx[ref_local])
                all_match_indices.append(idx[nn_local[valid]])
                local_matches = nn_local[valid]
                counts = np.bincount(local_matches)
                for local_j in np.where(counts > 1)[0]:
                    reuse_detail_rows.append(
                        {
                            "method": metric_col,
                            "physician": str(physician),
                            "patient_index": int(idx[int(local_j)]),
                            "n_times_as_match": int(counts[int(local_j)]),
                        }
                    )
            reuse = match_reuse_stats(nn_local[valid] if np.any(valid) else np.asarray([], dtype=np.intp))
            score = _discordance_rate_from_nn_local(y_local, nn_local)
            if clip_to_unit_interval and np.isfinite(score):
                score = float(np.clip(score, 0.0, 1.0))
            n_covered = int(diag.get("n_finite_rows", 0))
            discordance_results.append({
                "physician": physician,
                metric_col: score,
                _n_pairs_column_name(metric_col): n_covered,
                _n_covered_column_name(metric_col): n_covered,
                _coverage_column_name(metric_col): coverage,
                _n_discordant_column_name(metric_col): _n_discordant_from_nn_local(
                    y_local, nn_local
                ),
                _n_patients_reused_column_name(metric_col): int(reuse["n_patients_reused"]),
                _n_reuse_assignments_column_name(metric_col): int(reuse["n_reuse_assignments"]),
                _reuse_rate_column_name(metric_col): float(reuse["reuse_rate"]),
                _max_reuse_count_column_name(metric_col): int(reuse["max_reuse_count"]),
            })
        if all_ref_indices:
            ref_flat = np.concatenate(all_ref_indices)
            match_flat = np.concatenate(all_match_indices)
        else:
            ref_flat = np.asarray([], dtype=int)
            match_flat = np.asarray([], dtype=int)
        self._log_nn_reuse_summary(
            physicians_all,
            ref_flat,
            match_flat,
            metric_col,
            total_ties,
            caliper,
            coverage_by_physician,
        )
        df_res = pd.DataFrame(discordance_results)
        if reuse_detail_rows:
            detail = self.results.setdefault("matching_patient_reuse_detail", [])
            if isinstance(detail, list):
                detail.extend(reuse_detail_rows)
        return df_res, ref_flat, match_flat

    def _export_matching_patient_reuse_artifacts(self) -> None:
        """Write global and per-reused-patient NN reuse tables under results_dir."""
        detail = self.results.get("matching_patient_reuse_detail")
        if isinstance(detail, list) and detail:
            detail_path = self.results_dir / "matching_patient_reuse_detail.csv"
            pd.DataFrame(detail).to_csv(detail_path, index=False)
            self.LOGGER.info("Saved matching patient reuse detail to %s", detail_path)

        ip = self.results.get("intra_physician_variability")
        if not isinstance(ip, pd.DataFrame) or ip.empty:
            return

        summary_rows: list[dict[str, object]] = []
        prefix = "n_patients_reused_"
        for col in ip.columns:
            if not str(col).startswith(prefix):
                continue
            method = str(col)[len(prefix) :]
            n_pairs_col = _n_pairs_column_name(method)
            n_reuse_col = _n_reuse_assignments_column_name(method)
            max_reuse_col = _max_reuse_count_column_name(method)
            if n_pairs_col not in ip.columns:
                continue
            n_pairs = float(pd.to_numeric(ip[n_pairs_col], errors="coerce").fillna(0).sum())
            n_patients_reused = float(pd.to_numeric(ip[col], errors="coerce").fillna(0).sum())
            n_reuse_assignments = (
                float(pd.to_numeric(ip[n_reuse_col], errors="coerce").fillna(0).sum())
                if n_reuse_col in ip.columns
                else 0.0
            )
            max_reuse_vals = (
                pd.to_numeric(ip[max_reuse_col], errors="coerce")
                if max_reuse_col in ip.columns
                else pd.Series(dtype=float)
            )
            max_reuse_count = (
                float(max_reuse_vals.max()) if not max_reuse_vals.empty and max_reuse_vals.notna().any() else float("nan")
            )
            summary_rows.append(
                {
                    "method": method,
                    "n_pairs": int(n_pairs),
                    "n_patients_reused": int(n_patients_reused),
                    "n_reuse_assignments": int(n_reuse_assignments),
                    "reuse_rate": n_reuse_assignments / n_pairs if n_pairs > 0 else float("nan"),
                    "max_reuse_count": max_reuse_count,
                }
            )

        if not summary_rows:
            return
        summary_path = self.results_dir / "matching_patient_reuse_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        self.LOGGER.info("Saved matching patient reuse summary to %s", summary_path)

    def run_mahalanobis_matching_analysis(self) -> None:
        # Mahalanobis = distance euclidienne dans un espace "blanchi" par covariance,
        # utile quand les covariables sont corrélées entre elles.
        self.LOGGER.info("Starting Discordance Analysis (Mahalanobis matching)...")
        if not self._has_matching_subset():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        X_std = self._standardize_matrix(X_all)
        cols_ok = np.std(X_all, axis=0) > 1e-9
        X_clean = X_std[:, cols_ok]
        if X_clean.shape[1] == 0:
            self.LOGGER.warning("Mahalanobis matching skipped: all covariates have zero variance.")
            return
        if X_clean.shape[1] != X_std.shape[1]:
            dropped_idx = np.flatnonzero(~cols_ok)
            feature_cols = [
                c
                for c in df_match.columns
                if c not in {self.config["outcome_col"], self.config["physician_col"]}
            ]
            dropped = [
                f"{feature_cols[i]} (col {i})" if i < len(feature_cols) else f"col {i}"
                for i in dropped_idx
            ]
            self.LOGGER.warning(
                "Mahalanobis: %d covariate(s) dropped for zero variance: %s",
                dropped_idx.size,
                ", ".join(dropped),
            )

        # Pseudo-inverse pour robustesse numérique même en covariance mal conditionnée.
        cov_inv = np.linalg.pinv(np.cov(X_clean, rowvar=False))
        df_res, ref_flat, match_flat = self._distance_pairs_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_mahalanobis",
            local_distance_builder=lambda idx: cdist(
                X_clean[idx], X_clean[idx], metric="mahalanobis", VI=cov_inv
            ),
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(df_match, ref_flat, match_flat, "mahalanobis")
        self._merge_intra_physician_result(df_res)

    def run_manual_matching_analysis(self) -> None:
        # "Gold standard" basé sur la règle de génération synthétique:
        # compare les patients jugés comparables par design.
        self.LOGGER.info("Starting Discordance Analysis (Manual Clinical Thresholds)...")
        if not self._has_matching_subset():
            return
        # Important: use the analysis-aligned source table (not df_match) so batch-specific
        # metadata columns such as generation_rule_mask remain available.
        basis = self._matching_basis_df()
        if basis is None:
            return
        df_source = self.df.loc[basis.index].copy()
        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        if outcome_col not in df_source.columns or physician_col not in df_source.columns:
            self.LOGGER.warning(
                "Manual matching skipped: missing required columns outcome='%s' or physician='%s'.",
                outcome_col,
                physician_col,
            )
            return
        y_all = df_source[outcome_col].to_numpy()
        physicians_all = df_source[physician_col].to_numpy()
        generation_strategy = self.config.get("generation_strategy", "")
        try:
            eligible_mask = get_generation_rule_mask(df_source, generation_strategy)
        except ValueError as exc:
            self.LOGGER.warning(
                "Manual matching skipped: could not retrieve generation rule for strategy '%s' (%s).",
                generation_strategy, exc
            )
            return
        if len(eligible_mask) != len(df_source):
            self.LOGGER.warning(
                "Manual matching skipped: generation rule mask length mismatch (%d vs %d).",
                len(eligible_mask),
                len(df_source),
            )
            return
        # Matrice booléenne NxN: True si les deux patients sont dans la zone éligible.
        global_match = eligible_mask[:, None] & eligible_mask[None, :]
        np.fill_diagonal(global_match, False)
        discordance_results = []
        for physician in np.unique(physicians_all):
            # Indices des patients suivis par ce médecin.
            idx = np.where(physicians_all == physician)[0]
            # On ignore les médecins avec trop peu de patients (stabilité statistique).
            if len(idx) < self.config["min_patients_per_physician"]:
                self.LOGGER.warning("WARNING: Physician %s has less than %d patients, skipping.", physician, self.config["min_patients_per_physician"])
                continue
            # Sous-matrice de matching restreinte aux patients de ce médecin.
            # local_match[i, j] = True si i et j (chez ce médecin) sont comparables
            # selon la règle clinique "manuelle".
            local_match = global_match[np.ix_(idx, idx)]
            # Outcomes des patients de ce médecin (vecteur binaire).
            y_local = y_all[idx]
            # Nombre de paires valides:
            # - local_match est symétrique (i,j) et (j,i),
            # - donc on divise par 2 pour compter chaque paire une seule fois.
            n_pairs_found = float(np.sum(local_match) / 2)
            n_pairs_col = _n_pairs_column_name("discordance_rate_manual")
            n_discordant_col = _n_discordant_column_name("discordance_rate_manual")
            n_pairs_int = int(n_pairs_found)
            if n_pairs_found == 0:
                # Aucun couple comparable trouvé => score non défini (NaN).
                discordance_results.append({
                    "physician": physician,
                    "discordance_rate_manual": np.nan,
                    n_pairs_col: 0,
                    n_discordant_col: 0,
                })
                self.LOGGER.warning("WARNING: No comparable pairs found for physician %s, discordance rate set to NaN.", physician)
            else:
                # Discordance = proportion de paires comparables où les outcomes diffèrent.
                # (y_local[:, None] != y_local) construit la matrice NxN des désaccords.
                # On intersecte avec local_match pour garder uniquement les paires valides.
                # /2 pour la symétrie, puis /n_pairs_found pour obtenir une proportion [0,1].
                n_discordant_int = int(
                    np.sum(local_match & (y_local[:, None] != y_local)) / 2
                )
                discordance_results.append({
                    "physician": physician,
                    "discordance_rate_manual": np.sum(local_match & (y_local[:, None] != y_local)) / 2 / n_pairs_found,
                    n_pairs_col: n_pairs_int,
                    n_discordant_col: n_discordant_int,
                })
        self._merge_intra_physician_result(pd.DataFrame(discordance_results))

    def _resolve_rf_n_estimators(self, default_n_estimators: int = 300) -> int:
        """Read and validate ``rf_n_estimators`` from config (shared by RF methods)."""
        raw_rf_n_estimators = self.config.get("rf_n_estimators", default_n_estimators)
        rf_n_estimators = default_n_estimators
        try:
            parsed_n_estimators = float(raw_rf_n_estimators)
            if np.isfinite(parsed_n_estimators) and parsed_n_estimators > 0 and parsed_n_estimators.is_integer():
                rf_n_estimators = int(parsed_n_estimators)
            else:
                raise ValueError
        except (TypeError, ValueError):
            self.LOGGER.warning(
                "Invalid rf_n_estimators=%r in config; using default %d.",
                raw_rf_n_estimators,
                default_n_estimators,
            )
        return rf_n_estimators

    def run_random_forest_matching_analysis(self) -> None:
        # Similarité RF-proximity: fraction d'arbres où deux patients partagent la même feuille.
        self.LOGGER.info("Starting Discordance Analysis (Random Forest Matching)...")
        if not self._has_matching_subset():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Random Forest matching skipped: outcome has only one class.")
            return
        rf_n_estimators = self._resolve_rf_n_estimators()
        self.LOGGER.info("Random Forest matching parameter: n_estimators=%d", rf_n_estimators)
        target_rf = np.asarray(Y_all, dtype=int)
        min_leaf = max(5, int(0.01 * len(df_match)))
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=42, n_jobs=-1, max_depth=8, min_samples_leaf=min_leaf
        ).fit(X_all, target_rf)
        # leaf_indices[i, t] = identifiant de feuille du patient i dans l'arbre t.
        leaf_indices = rf.apply(X_all)
        n_trees = leaf_indices.shape[1]

        def _rf_local_dists(idx: np.ndarray) -> np.ndarray:
            local_leaves = leaf_indices[idx]
            local_prox = np.zeros((len(idx), len(idx)), dtype=float)
            for tree_idx in range(n_trees):
                leaves = local_leaves[:, tree_idx]
                local_prox += leaves[:, None] == leaves
            local_prox /= n_trees
            np.fill_diagonal(local_prox, -1.0)
            return 1.0 - local_prox

        df_res, ref_flat, match_flat = self._distance_pairs_by_physician(
            Y_all, physicians_all, "discordance_rate_rf_matching", local_distance_builder=_rf_local_dists
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(df_match, ref_flat, match_flat, "rf_matching")
        self._merge_intra_physician_result(df_res)

    def run_learning_matching_analysis(self) -> None:
        # Variante "learning": pondère l'espace par importances RF puis distance euclidienne.
        self.LOGGER.info("Starting Discordance Analysis (Learning Matching via RF weights)...")
        if not self._has_matching_subset():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Learning matching skipped: outcome has only one class.")
            return
        X_std = self._standardize_matrix(X_all)
        target_learning = np.asarray(Y_all, dtype=int)
        rf_n_estimators = self._resolve_rf_n_estimators()
        min_leaf = max(5, int(0.01 * len(df_match)))
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=42, n_jobs=-1, max_depth=8, min_samples_leaf=min_leaf
        ).fit(X_std, target_learning)
        # Garde-fous si les importances sont dégénérées/non finies.
        importances = np.array(rf.feature_importances_, dtype=float)
        if (not np.all(np.isfinite(importances))) or importances.sum() <= 0:
            self.LOGGER.warning(
                "Learning matching: RF importances invalid (non-finite or sum <= 0); using equal weights."
            )
            importances = np.ones_like(importances, dtype=float)
        importances = np.maximum(importances, 1e-6)
        importances = importances / importances.sum()
        X_weighted = X_std * np.sqrt(importances)
        df_res, ref_flat, match_flat = self._distance_pairs_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_learning",
            local_distance_builder=lambda idx: cdist(
                X_weighted[idx], X_weighted[idx], metric="euclidean"
            ),
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(df_match, ref_flat, match_flat, "learning")
        self._merge_intra_physician_result(df_res)

    def run_euclidean_matching_analysis(self) -> None:
        # Distance euclidienne avec pondération uniforme fixe.
        self.LOGGER.info("Starting Discordance Analysis (Euclidean matching)...")
        if not self._has_matching_subset():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Euclidean matching skipped: outcome has only one class.")
            return
        n_features = X_all.shape[1]
        weights = np.ones(n_features, dtype=float) / n_features
        X_std = self._standardize_matrix(X_all)
        X_weighted = X_std * np.sqrt(weights)
        df_res, ref_flat, match_flat = self._distance_pairs_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_euclidean",
            local_distance_builder=lambda idx: cdist(
                X_weighted[idx], X_weighted[idx], metric="euclidean"
            ),
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(df_match, ref_flat, match_flat, "euclidean")
        self._merge_intra_physician_result(df_res)

    def run_mutual_info_analysis(self) -> None:
        # Poidage des covariables par information mutuelle (feature -> outcome).
        self.LOGGER.info("Starting Heterogeneity Analysis (Mutual Information)...")
        if not self._has_matching_subset():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Mutual information analysis skipped: outcome has only one class.")
            return
        X_std = self._standardize_matrix(X_all)
        try:
            mi_weights = mutual_info_classif(X_std, Y_all, random_state=42)
        except Exception as exc:
            self.LOGGER.warning("Mutual information weighting failed (%s); using equal weights.", exc)
            mi_weights = np.ones(X_std.shape[1], dtype=float)
        mi_weights = np.asarray(mi_weights, dtype=float).ravel()
        if mi_weights.size != X_std.shape[1] or (not np.all(np.isfinite(mi_weights))) or float(np.sum(mi_weights)) <= 0.0:
            mi_weights = np.ones(X_std.shape[1], dtype=float)
        mi_weights = np.maximum(mi_weights, 1e-9)
        mi_weights = mi_weights / np.sum(mi_weights)
        X_weighted = X_std * np.sqrt(mi_weights)
        df_res, ref_flat, match_flat = self._distance_pairs_by_physician(
            Y_all.astype(int),
            physicians_all,
            "discordance_rate_mutual_info",
            local_distance_builder=lambda idx: cdist(
                X_weighted[idx], X_weighted[idx], metric="euclidean"
            ),
            clip_to_unit_interval=True,
        )
        self._merge_intra_physician_result(df_res)
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(df_match, ref_flat, match_flat, "mutual_info")

    def run_ensemble_matching_analysis(self) -> None:
        """Moyenne des trois discordances RF proximity / learned weights / mutual info uniquement."""
        self.LOGGER.info(
            "Computing composite discordance (mean of RF proximity, learned weights, mutual information)..."
        )
        if "intra_physician_variability" not in self.results:
            return
        df = self.results["intra_physician_variability"].copy()
        cols = list(THREE_METHOD_DISCORDANCE_COLS)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            self.LOGGER.warning(
                "Composite discordance skipped; missing columns: %s",
                ", ".join(missing),
            )
            return
        df["ensemble_matching"] = df[cols].mean(axis=1, skipna=False)
        # Ensemble has no own matching; use the integer mean of component pair counts.
        pair_cols = [_n_pairs_column_name(c) for c in cols if _n_pairs_column_name(c) in df.columns]
        if pair_cols:
            pair_mean = df[pair_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            df[_n_pairs_column_name("ensemble_matching")] = pair_mean.round().astype("Int64")
            df[_n_covered_column_name("ensemble_matching")] = pair_mean.round().astype("Int64")
        coverage_cols = [_coverage_column_name(c) for c in cols if _coverage_column_name(c) in df.columns]
        if coverage_cols:
            df[_coverage_column_name("ensemble_matching")] = df[coverage_cols].apply(
                pd.to_numeric, errors="coerce"
            ).mean(axis=1, skipna=True)
        discordant_cols = [
            _n_discordant_column_name(c) for c in cols if _n_discordant_column_name(c) in df.columns
        ]
        if discordant_cols:
            discordant_mean = (
                df[discordant_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            )
            df[_n_discordant_column_name("ensemble_matching")] = discordant_mean.round().astype(
                "Int64"
            )
        methods_cfg = self.config.get("methods") or {}
        drop_cols = bool(methods_cfg.get("drop_component_columns", False))
        if drop_cols:
            df.drop(columns=cols, inplace=True)
        self.results["intra_physician_variability"] = df

    def assign_prescriber_tertile_groups(self) -> None:
        """Assigne chaque médecin à un tertile selon ``prescription_rate``.

        Ajoute la colonne ``prescriber_group`` à ``intra_physician_variability`` avec
        les valeurs ``low_prescribers`` / ``medium_prescribers`` / ``high_prescribers``.
        """
        self.LOGGER.info("Assigning physicians to prescription-rate tertile groups...")
        if "intra_physician_variability" not in self.results:
            return
        df = self.results["intra_physician_variability"].copy()
        if df.empty or "prescription_rate" not in df.columns:
            self.LOGGER.warning(
                "Prescriber tertile assignment skipped: prescription_rate column missing or empty."
            )
            return

        rates = pd.to_numeric(df["prescription_rate"], errors="coerce")
        valid_mask = rates.notna()
        n_valid = int(valid_mask.sum())
        df["prescriber_group"] = pd.Series([pd.NA] * len(df), index=df.index, dtype=object)

        labels = list(PRESCRIBER_GROUP_LABELS)
        if n_valid >= 3:
            # Le rang stable garantit trois bacs équilibrés même en présence d'ex æquo.
            ranked = rates[valid_mask].rank(method="first")
            df.loc[valid_mask, "prescriber_group"] = pd.qcut(
                ranked, q=3, labels=labels
            ).astype(object)
        else:
            self.LOGGER.warning(
                "Prescriber tertile assignment fallback: only %d physician(s) with a valid "
                "prescription_rate (need >= 3 for tertiles).",
                n_valid,
            )
        df["prescriber_group"] = pd.Categorical(
            df["prescriber_group"], categories=labels, ordered=True
        )
        self.results["intra_physician_variability"] = df

    def plot_glmm_vs_matching_comparison(self) -> None:
        """Multi-panel figure: GLMM, discordance methods, Spearman heatmap."""
        self.LOGGER.info("Generating multi-method comparison plots...")
        if "intra_physician_variability" not in self.results:
            return
        df = self.results["intra_physician_variability"]
        cols_map = METHOD_DISPLAY_NAMES
        existing_cols = self._active_plot_score_columns()
        if len(existing_cols) < 2:
            self.LOGGER.warning(
                "Comparison plot skipped: need at least two method columns, found %d.",
                len(existing_cols),
            )
            return

        plot_dir = self.results_dir / "plots" / PLOT_SUBDIR_METHOD_COMPARISON
        plot_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")

        corr_matrix = df[existing_cols].corr(method="spearman").fillna(0.0)
        corr_matrix.index = [cols_map[c] for c in corr_matrix.index]
        corr_matrix.columns = [cols_map[c] for c in corr_matrix.columns]

        df_melt = df.copy()
        if "overdispersion_local" in df_melt.columns:
            df_melt["overdispersion_local"] = df_melt["overdispersion_local"].round(2)
        plot_cols = [
            col
            for col in [
                "discordance_rate_manual",
                "discordance_rate_euclidean",
                "discordance_rate_mahalanobis",
                "discordance_rate_learning",
                "discordance_rate_rf_matching",
                "discordance_rate_mutual_info",
                "overdispersion_local",
                "ensemble_matching",
            ]
            if col in df_melt.columns
        ]
        df_long = pd.melt(
            df_melt,
            id_vars=["physician"],
            value_vars=plot_cols,
            var_name="Method",
            value_name="Score",
        )
        method_names = cols_map
        df_long["Method"] = df_long["Method"].map(method_names)
        df_glmm = df_long[df_long["Method"] == "Generalized Mixed Model"]
        df_match = df_long[df_long["Method"] != "Generalized Mixed Model"]
        resolved_colors = _physician_color_map_from_df(df)

        show_delta_panel = (
            self._method_enabled("manual_pairing")
            and "discordance_rate_manual" in df.columns
            and pd.to_numeric(df["discordance_rate_manual"], errors="coerce").notna().any()
        )

        panel_widths: list[float] = [1.0, 2.4]
        if show_delta_panel:
            panel_widths.append(2.0)
        panel_widths.append(1.2)
        n_panels = len(panel_widths)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(max(24.0, 6.0 * n_panels), 8),
            gridspec_kw={"width_ratios": panel_widths},
        )
        axes_list = list(axes) if n_panels > 1 else [axes]
        panel_idx = 0
        ax_glmm = axes_list[panel_idx]
        panel_idx += 1
        ax_match = axes_list[panel_idx]
        panel_idx += 1
        ax_delta = None
        if show_delta_panel:
            ax_delta = axes_list[panel_idx]
            panel_idx += 1
        ax_corr = axes_list[panel_idx]

        def _physician_text_label(physician_id: object) -> str:
            return format_physician_display_label(physician_id)

        if not df_glmm.empty:
            if df_glmm["Score"].nunique() > 1:
                sns.violinplot(
                    data=df_glmm,
                    x="Method",
                    y="Score",
                    inner=None,
                    color="lightgray",
                    alpha=0.5,
                    ax=ax_glmm,
                )
            _scatter_hue_physician_categorical(
                ax_glmm,
                df_glmm,
                "Method",
                "Score",
                "physician",
                order=["Generalized Mixed Model"],
                color_map=resolved_colors,
                size=8.0,
                linewidth=1.0,
            )
            texts_glmm = [
                ax_glmm.text(
                    x=0,
                    y=row["Score"],
                    s=_physician_text_label(row["physician"]),
                    fontsize=7,
                    color="black",
                    alpha=0.8,
                )
                for _, row in df_glmm.iterrows()
                if pd.notna(row["Score"])
            ]
            with open(os.devnull, "w") as f, redirect_stdout(f), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adjust_text(texts_glmm, ax=ax_glmm, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2))
            if df_glmm["Score"].nunique() == 1:
                val = df_glmm["Score"].iloc[0]
                if pd.notna(val) and np.isfinite(val):
                    ax_glmm.set_ylim(max(0.0, float(val) - 1.0), float(val) + 1.0)
                else:
                    ax_glmm.set_ylim(0.0, 1.0)
        ax_glmm.set_title("Generalized Mixed Model", fontsize=14, fontweight="bold")
        ax_glmm.set_xlabel("")
        ax_glmm.set_ylabel("Raw Overdispersion (OLRE)", fontsize=12)
        ax_glmm.tick_params(axis="x", labelsize=12)

        if not df_match.empty:
            order_match = [
                method_names[c]
                for c in plot_cols
                if c != "overdispersion_local" and method_names.get(c) in df_match["Method"].values
            ]
            if df_match["Score"].nunique() > 1:
                sns.violinplot(
                    data=df_match,
                    x="Method",
                    y="Score",
                    order=order_match,
                    inner=None,
                    color="lightgray",
                    alpha=0.5,
                    ax=ax_match,
                )
            _scatter_hue_physician_categorical(
                ax_match,
                df_match,
                "Method",
                "Score",
                "physician",
                order=order_match,
                color_map=resolved_colors,
                size=8.0,
                linewidth=1.0,
            )
            x_dict = {method: i for i, method in enumerate(order_match)}
            texts_match = [
                ax_match.text(
                    x=x_dict.get(row["Method"], 0),
                    y=row["Score"],
                    s=_physician_text_label(row["physician"]),
                    fontsize=7,
                    color="black",
                    alpha=0.8,
                )
                for _, row in df_match.iterrows()
                if pd.notna(row["Score"])
            ]
            with open(os.devnull, "w") as f, redirect_stdout(f), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adjust_text(texts_match, ax=ax_match, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2))
        ax_match.set_title("Pairwise Discordance Methods", fontsize=14, fontweight="bold")
        ax_match.set_xlabel("")
        ax_match.set_ylabel("Discordance Rate (auto-scaled, bounded to [0,1])", fontsize=12)
        ax_match.tick_params(axis="x", labelsize=9)
        plt.setp(ax_match.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
        score_vals = (
            pd.to_numeric(df_match["Score"], errors="coerce").to_numpy(dtype=float)
            if not df_match.empty
            else np.array([], dtype=float)
        )
        finite_scores = score_vals[np.isfinite(score_vals)]
        if finite_scores.size > 0:
            observed_max = float(np.max(finite_scores))
            y_upper = max(0.1, min(1.0, observed_max + 0.05))
        else:
            y_upper = 1.0
        ax_match.set_ylim(0.0, y_upper)

        delta_cols = [
            c
            for c in plot_cols
            if c not in {"discordance_rate_manual", "overdispersion_local"} and c in df.columns
        ]
        if show_delta_panel and ax_delta is not None and "discordance_rate_manual" in df.columns and delta_cols:
            delta_rows = []
            manual_series = pd.to_numeric(df["discordance_rate_manual"], errors="coerce")
            for col in delta_cols:
                cur = pd.to_numeric(df[col], errors="coerce")
                delta = cur - manual_series
                for physician, val in zip(df["physician"], delta):
                    if pd.isna(val):
                        continue
                    delta_rows.append(
                        {
                            "physician": physician,
                            "Method": method_names.get(col, col),
                            "Delta": float(val),
                        }
                    )
            df_delta = pd.DataFrame(delta_rows)
            if not df_delta.empty:
                order_delta = [
                    method_names[c]
                    for c in delta_cols
                    if method_names.get(c) in df_delta["Method"].values
                ]
                if df_delta["Delta"].nunique() > 1:
                    sns.violinplot(
                        data=df_delta,
                        x="Method",
                        y="Delta",
                        order=order_delta,
                        inner=None,
                        color="lightgray",
                        alpha=0.5,
                        ax=ax_delta,
                    )
                _scatter_hue_physician_categorical(
                    ax_delta,
                    df_delta,
                    "Method",
                    "Delta",
                    "physician",
                    order=order_delta,
                    color_map=resolved_colors,
                    size=8.0,
                    linewidth=1.0,
                )
                x_dict_delta = {method: i for i, method in enumerate(order_delta)}
                texts_delta = [
                    ax_delta.text(
                        x=x_dict_delta.get(row["Method"], 0),
                        y=row["Delta"],
                        s=_physician_text_label(row["physician"]),
                        fontsize=7,
                        color="black",
                        alpha=0.8,
                    )
                    for _, row in df_delta.iterrows()
                    if pd.notna(row["Delta"])
                ]
                with open(os.devnull, "w") as f, redirect_stdout(f), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adjust_text(texts_delta, ax=ax_delta, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2))
                mean_delta = (
                    df_delta.groupby("Method", observed=False)["Delta"]
                    .mean()
                    .reindex(order_delta)
                    .dropna()
                )
                if not mean_delta.empty:
                    xticklabels_with_mean = []
                    for method in order_delta:
                        if method in mean_delta.index:
                            xticklabels_with_mean.append(f"{method}\n({mean_delta[method]:+.3f})")
                        else:
                            xticklabels_with_mean.append(method)
                    ax_delta.set_xticks(np.arange(len(order_delta), dtype=float))
                    ax_delta.set_xticklabels(xticklabels_with_mean)
                ax_delta.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
                ax_delta.set_title("Delta vs Manual Pairing", fontsize=14, fontweight="bold")
                ax_delta.set_xlabel("")
                ax_delta.set_ylabel("Method - Manual (Discordance)", fontsize=12)
                ax_delta.tick_params(axis="x", labelsize=9)
                plt.setp(ax_delta.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="RdYlGn",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            ax=ax_corr,
            cbar_kws={"shrink": 0.5},
            square=True,
            annot_kws={"size": 10},
        )
        ax_corr.set_title(
            "Spearman Correlations\n(method vs method)",
            fontsize=12,
            fontweight="bold",
        )
        ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0, fontsize=10)

        plt.suptitle("Distribution of Physician Inconsistency", fontsize=18, fontweight="bold", y=1.02)
        strategy = self.config.get("generation_strategy", "score2_five_groups_heter_patients")
        dataset_targets = self.config.get("dataset_targets", {})
        metrics_for_strategy = dataset_targets.get(strategy) if isinstance(dataset_targets, dict) else None
        if isinstance(metrics_for_strategy, dict):
            expected_summary = metrics_for_strategy.get("expected_summary", "No expected results defined.")
        elif isinstance(metrics_for_strategy, str):
            expected_summary = metrics_for_strategy
        else:
            expected_summary = "No expected results defined."
        if not isinstance(expected_summary, str):
            expected_summary = str(expected_summary)
        annotation_line = f"Expected Targets [Strategy: {strategy}] -> {expected_summary}"
        wrapped_annotation = textwrap.fill(annotation_line, width=100)
        fig.supxlabel(wrapped_annotation, fontweight="bold", fontsize=13, color="#333333")

        plt.tight_layout(rect=(0, 0.06, 1, 0.99))
        plt.savefig(plot_dir / "comparison_3panels.png", dpi=300, bbox_inches="tight")
        plt.close()
        self.LOGGER.info("Multi-panel comparison plot generated successfully.")

    def plot_comparison_by_tertile(self) -> None:
        """Separate figure: discordance / GLMM scores by prescriber tertile, one panel per method."""
        self.LOGGER.info("Generating tertile comparison plot by method...")
        if "intra_physician_variability" not in self.results:
            return
        df = self.results["intra_physician_variability"]
        if "prescriber_group" not in df.columns:
            self.LOGGER.warning("Tertile comparison plot skipped: prescriber_group column missing.")
            return
        score_cols = [
            c
            for c in self._active_plot_score_columns()
            if c != "overdispersion_local"
        ]
        if not score_cols:
            self.LOGGER.warning("Tertile comparison plot skipped: no discordance score columns found.")
            return

        plot_dir = self.results_dir / "plots" / PLOT_SUBDIR_METHOD_COMPARISON
        plot_dir.mkdir(parents=True, exist_ok=True)
        n_panels = len(score_cols)
        n_cols = min(3, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        sns.set_theme(style="whitegrid", context="notebook")
        fig, axes_grid = plt.subplots(
            n_rows,
            n_cols,
            figsize=(7.0 * n_cols, 7.0 * n_rows),
            squeeze=False,
        )
        axes_flat = axes_grid.flatten()
        resolved_colors = _physician_color_map_from_df(df)
        drawn = 0
        for ax, score_col in zip(axes_flat, score_cols):
            title = METHOD_DISPLAY_NAMES.get(score_col, score_col)
            if _plot_tertile_panel_on_ax(
                ax, df, title=title, score_col=score_col, color_map=resolved_colors
            ):
                drawn += 1
            else:
                ax.axis("off")
                ax.set_title(f"{title}\n(no tertile assignment)", fontsize=14, fontweight="bold")
        for ax in axes_flat[n_panels:]:
            ax.axis("off")
        if drawn == 0:
            plt.close(fig)
            self.LOGGER.warning("Tertile comparison plot skipped: no renderable panels.")
            return
        fig.suptitle(
            "Physician inconsistency by prescription-rate tertile",
            fontsize=18,
            fontweight="bold",
            y=1.005,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.99))
        out_path = plot_dir / "comparison_by_tertile.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.LOGGER.info("Tertile comparison plot saved to %s", out_path)

    def plot_bernoulli_residual(self) -> None:
        """One ``D − 2p(1−p)`` figure per active discordance method."""
        self.LOGGER.info("Generating Bernoulli residual plots (D − 2p(1−p))...")
        if "intra_physician_variability" not in self.results:
            return
        df = self.results["intra_physician_variability"]
        if "prescription_rate" not in df.columns:
            self.LOGGER.warning("Bernoulli residual plots skipped: prescription_rate column missing.")
            return
        score_cols = self._bernoulli_residual_score_columns()
        if not score_cols:
            self.LOGGER.warning("Bernoulli residual plots skipped: no eligible score columns.")
            return

        plot_dir = self.results_dir / "plots" / PLOT_SUBDIR_BERNOULLI_RESIDUAL
        plot_dir.mkdir(parents=True, exist_ok=True)
        resolved_colors = _physician_color_map_from_df(df)
        for score_col in score_cols:
            display = METHOD_DISPLAY_NAMES.get(score_col, score_col)
            out_path = plot_dir / f"discordance_minus_bernoulli_{score_col}.png"
            plot_medication_bernoulli_residual(
                df,
                out_path,
                title=f"Discordance excess vs. Bernoulli baseline — {display}",
                score_col=score_col,
                logger=self.LOGGER,
                color_map=resolved_colors,
            )
        plot_multi_method_bernoulli_residual_overview(
            df,
            score_cols,
            plot_dir / "discordance_minus_bernoulli_all_methods.png",
            logger=self.LOGGER,
            color_map=resolved_colors,
        )
        plot_multi_method_perfect_concordance_gap_overview(
            df,
            score_cols,
            plot_dir / "perfect_concordance_gap_all_methods.png",
            logger=self.LOGGER,
            color_map=resolved_colors,
        )
        for score_col in score_cols:
            display = METHOD_DISPLAY_NAMES.get(score_col, score_col)
            plot_perfect_concordance_gap_ratio_barchart(
                df,
                plot_dir / f"perfect_concordance_gap_ratio_{score_col}.png",
                title=f"Gap / |bell curve| (%) — {display}",
                score_col=score_col,
                logger=self.LOGGER,
                color_map=resolved_colors,
            )
        plot_multi_method_perfect_concordance_gap_ratio_overview(
            df,
            score_cols,
            plot_dir / "perfect_concordance_gap_ratio_all_methods.png",
            logger=self.LOGGER,
            color_map=resolved_colors,
        )

    def plot_physician_prescription_odds_ratio(self) -> None:
        """Forest plot of profile-adjusted prescription odds ratio per physician.

        Uses the GLMM random effect (``random_effect_bias``) merged into
        ``intra_physician_variability``; skipped gracefully when the GLMM step
        did not run (or did not converge), since no odds ratio can be derived.
        Point colours follow the experiment-wide physician identity palette.
        """
        self.LOGGER.info("Generating physician prescription odds ratio forest plot...")
        if "intra_physician_variability" not in self.results:
            return
        df = self.results["intra_physician_variability"]
        if "random_effect_bias" not in df.columns:
            self.LOGGER.warning(
                "Physician odds ratio plot skipped: random_effect_bias missing "
                "(GLMM step disabled or not merged)."
            )
            return

        plot_dir = self.results_dir / "plots" / PLOT_SUBDIR_PHYSICIAN_ODDS
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_physician_prescription_odds_ratio_forest(
            df,
            plot_dir / "prescription_odds_ratio_forest.png",
            color_map=_physician_color_map_from_df(df),
            logger=self.LOGGER,
        )

    def run_full_analysis(self) -> dict:
        """Matching basis → active methods → optional ensemble → GLMM merge → figures → export."""
        self.LOGGER.info("=== Starting Analysis Pipeline ===")
        self.run_matching_basis_prep()
        if not self._has_matching_subset():
            self.LOGGER.info("=== Analysis Pipeline Complete (no matching subset) ===")
            return self.results

        self._snapshot_matching_covariates()
        self._save_analysis_columns_manifest()
        self.plot_questionnaire_completion_used()

        method_steps: list[tuple[str, str, str, str]] = [
            ("euclidean", "Euclidean matching", "run_euclidean_matching_analysis", "after_euclidean_matching_head"),
            ("mahalanobis", "Mahalanobis matching", "run_mahalanobis_matching_analysis", "after_mahalanobis_matching_head"),
            ("rf_matching", "RF proximity matching", "run_random_forest_matching_analysis", "after_rf_matching_head"),
            ("learning", "Learned-weights matching", "run_learning_matching_analysis", "after_learning_matching_head"),
            ("mutual_info", "Mutual information matching", "run_mutual_info_analysis", "after_mutual_info_head"),
            ("manual_pairing", "Manual pairing", "run_manual_matching_analysis", "after_manual_matching_head"),
        ]
        for flag, label, method_name, snapshot_name in method_steps:
            if not self._method_enabled(flag):
                continue
            self.LOGGER.info("--- %s ---", label)
            getattr(self, method_name)()
            self._save_snapshot(self.results.get("intra_physician_variability"), snapshot_name)

        if self._method_enabled("ensemble"):
            self.LOGGER.info("--- Composite ensemble score ---")
            self.run_ensemble_matching_analysis()
            self._save_snapshot(
                self.results.get("intra_physician_variability"),
                "after_ensemble_matching_head",
            )

        if self._glmm_enabled():
            self.LOGGER.info("--- GLMM & OLRE ---")
            self.run_glmm_analysis()
            self._save_snapshot(
                self.results.get("intra_physician_variability"),
                "after_glmm_head",
            )

        self.assign_prescriber_tertile_groups()
        self._save_snapshot(
            self.results.get("intra_physician_variability"),
            "after_prescriber_tertiles_head",
        )
        self.plot_glmm_vs_matching_comparison()
        self.plot_comparison_by_tertile()
        self.plot_bernoulli_residual()
        self.plot_physician_prescription_odds_ratio()
        if "intra_physician_variability" in self.results:
            out_path = self.results_dir / "intra_physician_variability.csv"
            self.results["intra_physician_variability"].to_csv(out_path, index=False)
            self._save_snapshot(
                self.results["intra_physician_variability"],
                "intra_physician_variability_final_head",
            )
            self.LOGGER.info("Saved intra_physician_variability to %s", out_path)
        self._export_matching_patient_reuse_artifacts()
        write_pair_covariate_diagnostics(self.results_dir, logger=self.LOGGER)
        self._run_validation_report()
        self.LOGGER.info("=== Analysis Pipeline Complete ===")
        return self.results

    def _run_validation_report(self) -> None:
        """Execute automatic validation checks and export validation/* artifacts."""
        validation_cfg = self.config.get("validation") or {}
        if validation_cfg.get("enabled", True) is False:
            return
        try:
            run_validation_report(
                results_dir=self.results_dir,
                config=self.config,
                results=self.results,
                df=self.df,
                logger=self.LOGGER,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            self.LOGGER.warning("Validation report failed: %s", exc)


__all__ = [
    "Analysis",
    "_greedy_nearest_neighbor_pairs",
    "_glmm_fixed_effect_weights",
    "plot_medication_bernoulli_residual",
    "plot_multi_method_bernoulli_residual_overview",
    "plot_perfect_concordance_gap_barchart",
    "plot_multi_method_perfect_concordance_gap_overview",
    "plot_perfect_concordance_gap_ratio_barchart",
    "plot_multi_method_perfect_concordance_gap_ratio_overview",
    "plot_physician_prescription_odds_ratio_forest",
    "plot_patients_per_physician_distribution",
]
