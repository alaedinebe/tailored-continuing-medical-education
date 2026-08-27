# -*- coding: utf-8 -*-
"""Clinical review readiness gate based on validation metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_clinical_review_gate(
    results_dir: Path,
    *,
    pair_quality: pd.DataFrame | None,
    thresholds: dict[str, Any],
    primary_method: str = "euclidean",
) -> dict[str, Any]:
    """Write ``clinical_reviews/validation_gate.json`` for human reviewers."""
    clinical_dir = results_dir / "clinical_reviews"
    clinical_dir.mkdir(parents=True, exist_ok=True)

    worst_smd = None
    method_used = primary_method
    if pair_quality is not None and not pair_quality.empty:
        if primary_method in pair_quality["method"].values:
            row = pair_quality.loc[pair_quality["method"] == primary_method].iloc[0]
            worst_smd = float(row["worst_pair_max_smd"])
        else:
            idx = pair_quality["worst_pair_max_smd"].idxmax()
            row = pair_quality.loc[idx]
            worst_smd = float(row["worst_pair_max_smd"])
            method_used = str(row["method"])

    warn_threshold = float(thresholds.get("max_worst_pair_smd", 3.5))
    fail_threshold = float(thresholds.get("critical_max_worst_pair_smd", 5.0))

    if worst_smd is None:
        status = "unknown"
        message = "Pair-level SMD not available; re-run analysis or use backfill with stored pairs."
        review_recommended = True
    elif worst_smd > fail_threshold:
        status = "fail"
        message = (
            f"Worst pair SMD ({worst_smd:.2f}) exceeds critical threshold ({fail_threshold:.2f}). "
            "Interpret clinical pairs with caution."
        )
        review_recommended = True
    elif worst_smd > warn_threshold:
        status = "warn"
        message = (
            f"Worst pair SMD ({worst_smd:.2f}) exceeds warn threshold ({warn_threshold:.2f}). "
            "Prioritize discordant pairs and flag doubtful matchings."
        )
        review_recommended = True
    else:
        status = "pass"
        message = "Pair balance within calibrated limits; clinical review can proceed normally."
        review_recommended = True

    payload = {
        "status": status,
        "review_recommended": review_recommended,
        "primary_method": method_used,
        "worst_pair_max_smd": worst_smd,
        "warn_threshold": warn_threshold,
        "fail_threshold": fail_threshold,
        "message": message,
        "instructions": (
            "Open *_pair_reviewer.html only after reading validation/validation_summary.md. "
            "Mark pairs as 'appariement douteux' when profiles clearly differ on key clinical variables."
        ),
    }
    gate_path = clinical_dir / "validation_gate.json"
    gate_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
