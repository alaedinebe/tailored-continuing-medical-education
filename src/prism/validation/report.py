# -*- coding: utf-8 -*-
"""Validation report writers and console banner."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SEVERITY_ORDER = {"PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}


@dataclass
class CheckResult:
    id: str
    status: str
    value: Any = None
    threshold: Any = None
    message: str = ""
    recommendation: str = ""


def aggregate_status(checks: list[CheckResult]) -> str:
    if not checks:
        return "INFO"
    worst = max(SEVERITY_ORDER.get(c.status, 0) for c in checks)
    for label, rank in SEVERITY_ORDER.items():
        if rank == worst:
            return label
    return "PASS"


def announce_console(
    logger: Any,
    *,
    run_id: str,
    profile: str,
    checks: list[CheckResult],
    report_path: Path,
) -> None:
    """Log a human-readable validation banner."""
    global_status = aggregate_status(checks)
    counts = {k: 0 for k in SEVERITY_ORDER}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1

    border = "═" * 58
    logger.info(border)
    logger.info(" PRISM Run Validation — %s", run_id)
    logger.info(" Profile: %s | Status: %s", profile, global_status)
    logger.info(
        " (%d passed, %d info, %d warn, %d fail)",
        counts.get("PASS", 0),
        counts.get("INFO", 0),
        counts.get("WARN", 0),
        counts.get("FAIL", 0),
    )
    logger.info(border)

    icon = {"PASS": "✓", "INFO": "ℹ", "WARN": "⚠", "FAIL": "✗"}
    for check in checks:
        if check.status == "PASS" and check.id.startswith("info."):
            continue
        prefix = icon.get(check.status, "·")
        value_repr = "" if check.value is None else f" = {check.value}"
        logger.info(" %s  %s%s", prefix, check.id, value_repr)
        if check.status in {"WARN", "FAIL"} and check.message:
            logger.info("     %s", check.message)

    logger.info(" → %s", report_path)
    logger.info(border)


def write_reports(
    out_dir: Path,
    *,
    run_id: str,
    profile: str,
    checks: list[CheckResult],
    artifacts: dict[str, str],
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write JSON + CSV (+ optional Markdown) validation reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    global_status = aggregate_status(checks)
    counts = {k: 0 for k in SEVERITY_ORDER}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "calibration_profile": profile,
        "global_status": global_status,
        "counts": counts,
        "checks": [asdict(c) for c in checks],
        "artifacts": artifacts,
        "meta": extra_meta or {},
    }
    json_path = out_dir / "validation_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = pd.DataFrame([asdict(c) for c in checks])
    summary_path = out_dir / "validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    md_path = out_dir / "validation_summary.md"
    lines = [
        f"# Validation — {run_id}",
        "",
        f"- **Profile:** `{profile}`",
        f"- **Status:** **{global_status}**",
        f"- **Checks:** {counts.get('PASS', 0)} pass, {counts.get('INFO', 0)} info, "
        f"{counts.get('WARN', 0)} warn, {counts.get('FAIL', 0)} fail",
        "",
        "| Status | Check | Value | Threshold | Message |",
        "|---|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.status} | `{check.id}` | {check.value!s} | {check.threshold!s} | {check.message} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path
