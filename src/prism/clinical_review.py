# -*- coding: utf-8 -*-
"""Generate self-contained HTML clinical pair reviewers from *_pairs.csv exports."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / "src" / "prism" / "clinical_pair_reviewer.html"
PLACEHOLDER = "/*__EMBEDDED_DATA__*/"

MEDICATION_LABELS: dict[str, str] = {
    "recommendation": "Recommandation",
}


def _feature_cols(headers: list[str]) -> list[str]:
    return [h[:-2] for h in headers if h.endswith("_A")]


def _dataset_id(records: list[dict], source_name: str) -> str:
    sample = f"{len(records)}|{records[0].get('physician_id', '')}|{source_name}"
    h = 0
    for ch in sample:
        h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
    if h >= 2**31:
        h -= 2**32
    return str(h)


def _normalize_value(val):
    if pd.isna(val):
        return ""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if val.is_integer() else float(val)
    s = str(val).strip()
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return s


def paired_review_html_path(csv_path: Path) -> Path:
    """Derive reviewer HTML path from a ``{method}_pairs.csv`` export."""
    csv_path = Path(csv_path)
    method = csv_path.stem
    if method.endswith("_pairs"):
        method = method[: -len("_pairs")]
    return csv_path.parent / f"{method}_pair_reviewer.html"


def build_payload(csv_path: Path, medication: str = "statin") -> dict:
    """Load a pairs CSV and return the embedded-data payload for the HTML reviewer."""
    df = pd.read_csv(csv_path)
    headers = list(df.columns)
    records = [
        {col: _normalize_value(row[col]) for col in headers}
        for _, row in df.iterrows()
    ]
    discordant = sum(
        1 for r in records
        if r.get("patient_A_outcome") != r.get("patient_B_outcome")
    )
    physicians = sorted({str(r.get("physician_id", "")) for r in records if r.get("physician_id")})
    method = csv_path.stem.replace("_pairs", "") if csv_path.stem.endswith("_pairs") else csv_path.stem
    return {
        "sourceName": csv_path.name,
        "outputCsvName": f"{method}_annotations.csv",
        "datasetId": _dataset_id(records, csv_path.name),
        "medicationLabel": MEDICATION_LABELS.get(medication, medication.replace("_", " ").title()),
        "headers": headers,
        "featureNames": _feature_cols(headers),
        "records": records,
        "meta": {
            "n_pairs": len(records),
            "n_discordant": discordant,
            "n_physicians": len(physicians),
            "physicians": physicians,
        },
    }


def build_clinical_review_html(
    csv_path: Path,
    output_path: Path | None = None,
    *,
    medication: str = "statin",
    template_path: Path | None = None,
    max_pairs_embed: int = 2000,
) -> Path:
    """Write a self-contained HTML reviewer next to the CSV (or at ``output_path``).

    When the CSV is very large, embedding all records can create a massive HTML file.
    In that case, we write the plain template (with drag & drop) instead.
    """
    csv_path = Path(csv_path)
    template = Path(template_path or DEFAULT_TEMPLATE)
    if not template.is_file():
        raise FileNotFoundError(f"Clinical review template not found: {template}")

    text = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in text:
        raise ValueError(f"Placeholder {PLACEHOLDER!r} not found in {template}")

    df = pd.read_csv(csv_path)
    if max_pairs_embed is not None and len(df) > int(max_pairs_embed):
        html = text
    else:
        payload = build_payload(csv_path, medication)
        embedded = "const EMBEDDED_DATA = " + json.dumps(payload, ensure_ascii=False) + ";"
        html = text.replace(PLACEHOLDER, embedded, 1)

    out = Path(output_path) if output_path else paired_review_html_path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "output").mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def build_clinical_review_html_safe(
    csv_path: Path,
    output_path: Path | None = None,
    *,
    medication: str = "statin",
    template_path: Path | None = None,
    max_pairs_embed: int = 2000,
) -> Path | None:
    """Best-effort wrapper for pipeline hooks; logs warnings instead of raising."""
    try:
        out = build_clinical_review_html(
            csv_path,
            output_path,
            medication=medication,
            template_path=template_path,
            max_pairs_embed=max_pairs_embed,
        )
        LOGGER.info("Clinical review HTML: %s", out)
        return out
    except Exception as exc:
        LOGGER.warning("Clinical review HTML generation failed for %s: %s", csv_path, exc)
        return None
