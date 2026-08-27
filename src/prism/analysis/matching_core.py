# -*- coding: utf-8 -*-
"""Core matching/statistical helpers for Prism analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _glmm_fixed_effect_weights(result) -> dict[str, dict[str, float | None]]:
    """
    Posterior mean (log-odds weight) and posterior SD for each GLMM fixed effect,
    from the variational Bayes summary (Type M rows only; excludes variance parameters).
    """
    try:
        tables = result.summary().tables
        if not tables:
            return {}
        summary_df = tables[0]
        if not isinstance(summary_df, pd.DataFrame) or "Type" not in summary_df.columns:
            return {}
        fe = summary_df[summary_df["Type"] == "M"]
        if fe.empty:
            return {}
        mean_col = "Post. Mean" if "Post. Mean" in fe.columns else None
        sd_col = "Post. SD" if "Post. SD" in fe.columns else None
        if mean_col is None:
            mean_col = next((c for c in fe.columns if "Mean" in str(c)), None)
        out: dict[str, dict[str, float | None]] = {}
        for name in fe.index:
            entry: dict[str, float | None] = {}
            if mean_col and mean_col in fe.columns:
                v = fe.loc[name, mean_col]
                entry["post_mean"] = None if pd.isna(v) else float(v)
            if sd_col and sd_col in fe.columns:
                v = fe.loc[name, sd_col]
                entry["post_sd"] = None if pd.isna(v) else float(v)
            out[str(name)] = entry
        return out
    except Exception:
        return {}


def nearest_neighbor_distances(local_dists: np.ndarray) -> np.ndarray:
    """Return d(i, nn(i)) for each row without applying a caliper."""
    d = np.asarray(local_dists, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        return np.asarray([], dtype=float)
    d = d.copy()
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def match_reuse_stats(local_match_indices: np.ndarray) -> dict[str, int | float]:
    """Reuse statistics for NN match targets on a single physician panel."""
    matches = np.asarray(local_match_indices, dtype=np.intp).ravel()
    empty: dict[str, int | float] = {
        "n_pairs": 0,
        "n_patients_reused": 0,
        "n_reuse_assignments": 0,
        "reuse_rate": float("nan"),
        "max_reuse_count": 0,
    }
    if matches.size == 0:
        return empty
    counts = np.bincount(matches)
    n_pairs = int(matches.size)
    n_reuse_assignments = int(np.sum(np.maximum(counts - 1, 0)))
    return {
        "n_pairs": n_pairs,
        "n_patients_reused": int(np.sum(counts > 1)),
        "n_reuse_assignments": n_reuse_assignments,
        "reuse_rate": n_reuse_assignments / n_pairs if n_pairs > 0 else float("nan"),
        "max_reuse_count": int(np.max(counts)),
    }


def nearest_neighbor_with_replacement(
    local_dists: np.ndarray,
    caliper: float | None = None,
) -> tuple[np.ndarray, dict[str, int | float | None]]:
    """
    Nearest-neighbour matching with replacement (row-wise argmin).

    For each patient i, returns the local index j != i that minimises d[i, j].
    No ``used`` mask: the same j may be chosen by several reference patients.

    When ``caliper`` is set, row i is kept only if d[i, nn(i)] <= caliper.

    Tie-breaking follows ``np.argmin``: the smallest column index wins.

    Arguments:
        local_dists: square distance matrix (n, n). The diagonal is forced to
            +inf so a patient cannot match itself.
        caliper: optional maximum admissible NN distance. Rows above the caliper
            are marked invalid (``nn_local[i] = -1``).

    Returns:
        nn_local: shape (n,), dtype intp. ``nn_local[i]`` is the match for i,
            or -1 when row i has no admissible neighbour.
        diagnostics: ``n_patients``, ``n_ties``, ``n_finite_rows`` (covered rows),
            ``coverage``, ``caliper``, ``median_nn_dist`` (over covered rows).
    """
    d = np.asarray(local_dists, dtype=float)
    empty_diag: dict[str, int | float | None] = {
        "n_patients": 0,
        "n_ties": 0,
        "n_finite_rows": 0,
        "coverage": float("nan"),
        "caliper": caliper,
        "median_nn_dist": float("nan"),
    }
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        return np.asarray([], dtype=np.intp), empty_diag

    n = d.shape[0]
    np.fill_diagonal(d, np.inf)

    nn_local = np.argmin(d, axis=1).astype(np.intp, copy=False)
    chosen_dist = d[np.arange(n), nn_local]
    valid = np.isfinite(chosen_dist)
    if caliper is not None:
        valid &= chosen_dist <= float(caliper)
    nn_local[~valid] = -1

    row_mins = np.min(d, axis=1)
    at_min = (d == row_mins[:, None]) & np.isfinite(row_mins[:, None])
    np.fill_diagonal(at_min, False)
    n_ties = int(np.sum(at_min.sum(axis=1) >= 2))

    n_finite_rows = int(np.sum(valid))
    median_nn = float("nan")
    if n_finite_rows > 0:
        median_nn = float(np.median(chosen_dist[valid]))

    diagnostics: dict[str, int | float | None] = {
        "n_patients": n,
        "n_ties": n_ties,
        "n_finite_rows": n_finite_rows,
        "coverage": n_finite_rows / n if n > 0 else float("nan"),
        "caliper": caliper,
        "median_nn_dist": median_nn,
    }
    return nn_local, diagnostics


def _discordance_rate_from_nn_local(
    y_local: np.ndarray, nn_local: np.ndarray
) -> float:
    """Discordance rate from directed NN pairs; NaN when no valid match."""
    y = np.asarray(y_local)
    nn = np.asarray(nn_local, dtype=np.intp)
    valid = nn >= 0
    if not np.any(valid):
        return float("nan")
    refs = np.where(valid)[0]
    matches = nn[valid]
    return float(np.mean(y[refs] != y[matches]))


def _n_discordant_from_nn_local(y_local: np.ndarray, nn_local: np.ndarray) -> int:
    """Count discordant directed NN pairs; 0 when no valid match."""
    y = np.asarray(y_local)
    nn = np.asarray(nn_local, dtype=np.intp)
    valid = nn >= 0
    if not np.any(valid):
        return 0
    refs = np.where(valid)[0]
    matches = nn[valid]
    return int(np.sum(y[refs] != y[matches]))


def _greedy_nearest_neighbor_pairs(local_dists: np.ndarray) -> list[tuple[int, int]]:
    """
    Greedy one-to-one nearest-neighbour pairing (without replacement).

    Arguments:
        local_dists: square distance matrix (diagonal should be inf).

    Returns:
        List of index pairs (i, j) with i != j and each index used at most once.
        For odd group sizes, one element can remain unpaired.
    """
    d = np.asarray(local_dists, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        return []

    n = d.shape[0]
    used = np.zeros(n, dtype=bool)
    pairs: list[tuple[int, int]] = []

    for i in range(n):
        if used[i]:
            continue
        candidates = np.where(~used)[0]
        candidates = candidates[candidates != i]
        if candidates.size == 0:
            continue
        cand_d = d[i, candidates]
        if not np.any(np.isfinite(cand_d)):
            continue
        j = int(candidates[np.argmin(cand_d)])
        used[i] = True
        used[j] = True
        pairs.append((int(i), j))
    return pairs


def _n_discordant_pairs_from_pairs(
    y_local: np.ndarray, pairs: list[tuple[int, int]]
) -> int:
    """Count pairs whose outcomes differ; 0 when no pairs."""
    if len(pairs) == 0:
        return 0
    y = np.asarray(y_local)
    return int(sum(int(y[i] != y[j]) for i, j in pairs))


def _discordance_rate_from_pairs(y_local: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    """Discordance rate = discordant pairs / total pairs; NaN when no pair."""
    if len(pairs) == 0:
        return np.nan
    y = np.asarray(y_local)
    disc = [float(y[i] != y[j]) for i, j in pairs]
    return float(np.mean(disc))
