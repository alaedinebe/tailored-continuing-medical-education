# -*- coding: utf-8 -*-
"""Publication-grade Table 1 (baseline cohort characteristics) for Prism.

The table follows the presentation conventions of NEJM / Circulation cohort
papers: variables grouped in clinical sections, ``mean (SD)`` or
``median (IQR)`` for continuous variables, ``no./total no. (%)`` for
categorical variables, one column per exposure group (by default the
prescription outcome), explicit missing-data accounting, between-group
p values and standardized mean differences.

Four artifacts are written under ``<experiment>/data/``: ``.csv`` (display
table), ``_long.csv`` (machine-readable statistics), ``.tex`` (booktabs) and
``.html`` (rendered table).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Statistic kinds
# --------------------------------------------------------------------------- #
MEAN_SD = "mean_sd"
MEDIAN_IQR = "median_iqr"
BINARY = "binary"
CATEGORICAL = "categorical"

# Unit conversions used when a cohort stores lipids in mmol/L instead of mg/dL.
CHOLESTEROL_MMOL_TO_MG_DL = 38.67
TRIGLYCERIDE_MMOL_TO_MG_DL = 88.57
MMOL_CHOLESTEROL_COLUMNS = frozenset(
    {"non_hdl_cholesterol", "hdl_cholesterol", "ldl_cholesterol", "total_cholesterol"}
)

# Default outcome / clustering columns of the Prism pipeline.
DEFAULT_OUTCOME_COL = "recommendation"
DEFAULT_PHYSICIAN_COL = "professional_id"
STRATEGY_COL_CANDIDATES = ("dataset_strategy", "strategy_name")

# A variable is dropped when fewer than this fraction of patients have a value.
DEFAULT_MIN_COMPLETENESS = 0.20

# |SMD| above this value is usually reported as a meaningful imbalance.
SMD_IMBALANCE_THRESHOLD = 0.10

MISSING_PLACEHOLDER = "—"


# --------------------------------------------------------------------------- #
# Variable specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineVariable:
    """One row (or row block) of Table 1.

    Attributes:
        key: Stable identifier used in the machine-readable export.
        name: Clinical name, written in sentence case without statistic.
        kind: One of ``mean_sd``, ``median_iqr``, ``binary``, ``categorical``.
        candidates: Column aliases tried in order (supports both the external
            database schema and the synthetic generators).
        unit: Unit appended to the label; may be overridden per column.
        units_by_column: Column-specific units (e.g. mmol/L vs mg/dL).
        digits: Number of decimals for continuous statistics.
        positive_values: Values counted as events for ``binary`` variables.
        level_order: Preferred display order of ``categorical`` levels.
        max_levels: Levels beyond this count are pooled into ``Other``.
        derive: Optional callable building the series from the whole cohort.
        always_keep: Keep the row even when completeness is below threshold.
    """

    key: str
    name: str
    kind: str
    candidates: tuple[str, ...] = ()
    unit: str = ""
    units_by_column: Mapping[str, str] = field(default_factory=dict)
    digits: int = 1
    positive_values: tuple[str, ...] = ()
    level_order: tuple[str, ...] = ()
    max_levels: int = 6
    derive: Callable[[pd.DataFrame], pd.Series | None] | None = None
    always_keep: bool = False

    def label(self, unit: str) -> str:
        """Return the publication label, statistic descriptor included."""
        unit_suffix = " — %s" % unit if unit else ""
        if self.kind == MEAN_SD:
            return "Mean %s%s (SD)" % (self.name, unit_suffix)
        if self.kind == MEDIAN_IQR:
            return "Median %s%s (IQR)" % (self.name, unit_suffix)
        return "%s%s — no. (%%)" % (self.name[:1].upper() + self.name[1:], unit_suffix)


# --------------------------------------------------------------------------- #
# Column resolution helpers
# --------------------------------------------------------------------------- #
def _first_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first candidate present in ``df``, or None."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return ``df[column]`` coerced to float with unparsable values as NaN."""
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _as_binary(flag: pd.Series, valid: pd.Series) -> pd.Series:
    """Return a 1/0/NaN float series from an event mask and a validity mask."""
    out = pd.Series(np.nan, index=flag.index, dtype="float64")
    out.loc[valid] = flag.loc[valid].astype("float64")
    return out


def _binary_series(series: pd.Series, positive_values: Sequence[str]) -> pd.Series:
    """Coerce booleans, 0/1 codes or free text into a 1/0/NaN event series."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        return _as_binary(numeric > 0, numeric.notna())
    text = series.astype("string").str.strip()
    valid = text.notna() & text.str.len().gt(0)
    wanted = {value.casefold() for value in (positive_values or ("yes", "true", "1"))}
    return _as_binary(text.str.casefold().isin(wanted), valid)


def _categorical_series(series: pd.Series) -> pd.Series:
    """Coerce any dtype into a nullable string series of category levels."""
    if pd.api.types.is_bool_dtype(series):
        return series.map({True: "Yes", False: "No"}).astype("string")
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        levels = set(numeric.dropna().unique().tolist())
        if levels.issubset({0.0, 1.0}):
            return numeric.map({1.0: "Yes", 0.0: "No"}).astype("string")
        formatted = numeric.map(lambda v: "%g" % v if pd.notna(v) else pd.NA)
        return formatted.astype("string")
    text = series.astype("string").str.strip()
    return text.where(text.notna() & text.str.len().gt(0))


def _cholesterol_mg_dl(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    """Return a cholesterol series in mg/dL whatever the source unit."""
    column = _first_column(df, candidates)
    if column is None:
        return None
    series = _numeric(df, column)
    if column in MMOL_CHOLESTEROL_COLUMNS:
        return series * CHOLESTEROL_MMOL_TO_MG_DL
    return series


# --------------------------------------------------------------------------- #
# Derived clinical variables
# --------------------------------------------------------------------------- #
FEMALE_SEX_CANDIDATES = ("gender", "demographic_sex", "sex")
AGE_CANDIDATES = ("age",)
BMI_CANDIDATES = ("bmi", "score_body_mass_index")
SBP_CANDIDATES = ("biomarker_systolic_blood_pressure_sitting", "systolic_blood_pressure")
DBP_CANDIDATES = ("biomarker_diastolic_blood_pressure_sitting", "diastolic_blood_pressure")
HBA1C_CANDIDATES = ("biomarker_hba1c_ngsp_blood",)
LDL_CANDIDATES = ("biomarker_ldl_cholesterol_calculated_serum", "ldl_cholesterol")
EGFR_CANDIDATES = (
    "biomarker_glomerular_filtration_rate_cdk_epi_serum",
    "estimated_glomerular_filtration_rate",
)
SCORE2_CANDIDATES = ("score_score2", "score2_ten_year_cvd_risk_percent")
SMOKING_CANDIDATES = ("smoker", "social_history_smoking_status")

# ESC 2021 age-specific SCORE2 cut-offs (10-year fatal + non-fatal CVD risk, %).
SCORE2_CUTOFFS = ((50.0, 2.5, 7.5), (70.0, 5.0, 10.0), (np.inf, 7.5, 15.0))
SCORE2_LEVELS = ("Low to moderate risk", "High risk", "Very high risk")


def _derive_female_sex(df: pd.DataFrame) -> pd.Series | None:
    """Return female sex as a 1/0/NaN series from any available sex column."""
    column = _first_column(df, FEMALE_SEX_CANDIDATES)
    if column is not None:
        text = df[column].astype("string").str.strip().str.casefold()
        return _as_binary(text.isin({"female", "f", "femme"}), text.notna())
    column = _first_column(df, ("is_male",))
    if column is None:
        return None
    male = _binary_series(df[column], ())
    return _as_binary(male.eq(0.0), male.notna())


def _derive_age_group(df: pd.DataFrame) -> pd.Series | None:
    """Bin age into the SCORE2 age strata used for risk categorization."""
    column = _first_column(df, AGE_CANDIDATES)
    if column is None:
        return None
    age = _numeric(df, column)
    bins = [-np.inf, 50.0, 70.0, np.inf]
    labels = ["<50 yr", "50–69 yr", "≥70 yr"]
    binned = pd.cut(age, bins=bins, labels=labels, right=False)
    return pd.Series(binned, index=df.index).astype("string")


def _derive_bmi_category(df: pd.DataFrame) -> pd.Series | None:
    """Bin body-mass index into the WHO weight categories."""
    column = _first_column(df, BMI_CANDIDATES)
    if column is None:
        return None
    bmi = _numeric(df, column)
    bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
    labels = ["Underweight (<18.5)", "Normal (18.5–24.9)", "Overweight (25.0–29.9)", "Obese (≥30.0)"]
    binned = pd.cut(bmi, bins=bins, labels=labels, right=False)
    return pd.Series(binned, index=df.index).astype("string")


def _derive_elevated_bp(df: pd.DataFrame) -> pd.Series | None:
    """Return blood pressure in the hypertensive range (≥140/90 mm Hg)."""
    sbp_col = _first_column(df, SBP_CANDIDATES)
    dbp_col = _first_column(df, DBP_CANDIDATES)
    if sbp_col is None or dbp_col is None:
        return None
    sbp, dbp = _numeric(df, sbp_col), _numeric(df, dbp_col)
    return _as_binary(sbp.ge(140.0) | dbp.ge(90.0), sbp.notna() & dbp.notna())


def _derive_score2_category(df: pd.DataFrame) -> pd.Series | None:
    """Classify SCORE2 risk using the age-specific ESC thresholds."""
    score_col = _first_column(df, SCORE2_CANDIDATES)
    age_col = _first_column(df, AGE_CANDIDATES)
    if score_col is None or age_col is None:
        return None
    risk, age = _numeric(df, score_col), _numeric(df, age_col)
    category = pd.Series(pd.NA, index=df.index, dtype="string")
    lower_bound = -np.inf
    for upper_age, high_cut, very_high_cut in SCORE2_CUTOFFS:
        stratum = age.ge(lower_bound) & age.lt(upper_age) & risk.notna()
        category.loc[stratum & risk.lt(high_cut)] = SCORE2_LEVELS[0]
        category.loc[stratum & risk.ge(high_cut) & risk.lt(very_high_cut)] = SCORE2_LEVELS[1]
        category.loc[stratum & risk.ge(very_high_cut)] = SCORE2_LEVELS[2]
        lower_bound = upper_age
    return category


def _derive_glycemic_category(df: pd.DataFrame) -> pd.Series | None:
    """Classify HbA1c into the ADA normoglycemia / prediabetes / diabetes bands."""
    column = _first_column(df, HBA1C_CANDIDATES)
    if column is None:
        return None
    hba1c = _numeric(df, column)
    bins = [-np.inf, 5.7, 6.5, np.inf]
    labels = ["Normoglycemia (<5.7)", "Prediabetes (5.7–6.4)", "Diabetes range (≥6.5)"]
    binned = pd.cut(hba1c, bins=bins, labels=labels, right=False)
    return pd.Series(binned, index=df.index).astype("string")


def _derive_diabetes(df: pd.DataFrame) -> pd.Series | None:
    """Combine self-reported diabetes and HbA1c ≥ 6.5% into one indicator."""
    parts: list[pd.Series] = []
    column = _first_column(df, ("declared_diabetes",))
    if column is not None:
        text = df[column].astype("string").str.strip().str.casefold()
        parts.append(_as_binary(~text.str.startswith("no", na=False), text.notna()))
    column = _first_column(df, HBA1C_CANDIDATES)
    if column is not None:
        hba1c = _numeric(df, column)
        parts.append(_as_binary(hba1c.ge(6.5), hba1c.notna()))
    if not parts:
        return None
    stacked = pd.concat(parts, axis=1)
    valid = stacked.notna().any(axis=1)
    return _as_binary(stacked.fillna(0.0).max(axis=1).gt(0.0), valid)


def _derive_ldl_above_target(df: pd.DataFrame) -> pd.Series | None:
    """Return LDL cholesterol above the 3.0 mmol/L (116 mg/dL) ESC threshold."""
    ldl = _cholesterol_mg_dl(df, LDL_CANDIDATES)
    if ldl is None:
        return None
    return _as_binary(ldl.ge(3.0 * CHOLESTEROL_MMOL_TO_MG_DL), ldl.notna())


def _derive_reduced_egfr(df: pd.DataFrame) -> pd.Series | None:
    """Return eGFR below 60 mL/min/1.73 m² (CKD stage 3 or worse)."""
    column = _first_column(df, EGFR_CANDIDATES)
    if column is None:
        return None
    egfr = _numeric(df, column)
    return _as_binary(egfr.lt(60.0), egfr.notna())


def _derive_current_smoker(df: pd.DataFrame) -> pd.Series | None:
    """Return current smoking from a status label or a binary smoker flag."""
    column = _first_column(df, SMOKING_CANDIDATES)
    if column is not None:
        text = df[column].astype("string").str.strip().str.casefold()
        events = text.str.contains("current", na=False) | text.isin({"smoker", "yes", "daily smoker"})
        return _as_binary(events, text.notna())
    column = _first_column(df, ("is_smoker",))
    if column is None:
        return None
    return _binary_series(df[column], ())


def _derive_triglycerides_mg_dl(df: pd.DataFrame) -> pd.Series | None:
    """Return triglycerides in mg/dL whatever the source unit."""
    column = _first_column(df, ("biomarker_triglyceride_serum", "triglycerides"))
    if column is None:
        return None
    series = _numeric(df, column)
    if column == "triglycerides":
        return series * TRIGLYCERIDE_MMOL_TO_MG_DL
    return series


# --------------------------------------------------------------------------- #
# Table 1 layout: clinical sections and their variables
# --------------------------------------------------------------------------- #
TABLE_ONE_SECTIONS: tuple[tuple[str, tuple[BaselineVariable, ...]], ...] = (
    (
        "Demographic characteristics",
        (
            BaselineVariable("age", "age", MEAN_SD, AGE_CANDIDATES, unit="yr", always_keep=True),
            BaselineVariable("age_group", "age group", CATEGORICAL, derive=_derive_age_group,
                             level_order=("<50 yr", "50–69 yr", "≥70 yr"), always_keep=True),
            BaselineVariable("female", "female sex", BINARY, derive=_derive_female_sex, always_keep=True),
            BaselineVariable("ethnicity", "race or ethnic group", CATEGORICAL,
                             ("race", "demographic_ethnicity"), max_levels=5),
            BaselineVariable("education", "educational attainment", CATEGORICAL, ("education",), max_levels=5),
            BaselineVariable("income", "annual household income", CATEGORICAL, ("income",), max_levels=5),
        ),
    ),
    (
        "Anthropometric measures",
        (
            BaselineVariable("bmi", "body-mass index", MEDIAN_IQR, BMI_CANDIDATES, unit="kg/m²"),
            BaselineVariable("bmi_category", "weight category", CATEGORICAL, derive=_derive_bmi_category,
                             level_order=("Underweight (<18.5)", "Normal (18.5–24.9)",
                                          "Overweight (25.0–29.9)", "Obese (≥30.0)")),
            BaselineVariable("weight", "body weight", MEAN_SD, ("weight", "biomarker_weight"), unit="kg"),
            BaselineVariable("height", "height", MEAN_SD, ("height", "biomarker_height"), unit="cm"),
            BaselineVariable("waist", "waist circumference", MEAN_SD,
                             ("biomarker_waist_circumference_midpoint",), unit="cm"),
            BaselineVariable("body_fat", "body fat", MEAN_SD, ("biomarker_body_fat_mass_ratio",), unit="%"),
            BaselineVariable("visceral_fat", "visceral adipose tissue mass", MEDIAN_IQR,
                             ("biomarker_visceral_adipose_tissue_mass",), unit="g", digits=0),
        ),
    ),
    (
        "Vital signs",
        (
            BaselineVariable("sbp", "systolic blood pressure", MEAN_SD, SBP_CANDIDATES, unit="mm Hg"),
            BaselineVariable("dbp", "diastolic blood pressure", MEAN_SD, DBP_CANDIDATES, unit="mm Hg"),
            BaselineVariable("elevated_bp", "blood pressure ≥140/90 mm Hg", BINARY, derive=_derive_elevated_bp),
            BaselineVariable("heart_rate", "resting heart rate", MEAN_SD,
                             ("biomarker_heart_rate_resting",), unit="beats/min"),
        ),
    ),
    (
        "Cardiovascular risk profile",
        (
            BaselineVariable("score2", "SCORE2 10-yr cardiovascular risk", MEDIAN_IQR,
                             SCORE2_CANDIDATES, unit="%"),
            BaselineVariable("score2_category", "SCORE2 risk category", CATEGORICAL,
                             derive=_derive_score2_category, level_order=SCORE2_LEVELS),
            BaselineVariable("smoking_status", "smoking status", CATEGORICAL, SMOKING_CANDIDATES,
                             level_order=("Never smoker", "Former smoker", "Current smoker"), max_levels=5),
            BaselineVariable("current_smoker", "current smoking", BINARY, derive=_derive_current_smoker),
            BaselineVariable("cigarettes_per_day", "cigarettes per day among smokers", MEDIAN_IQR,
                             ("social_history_smoking_cigarettes_per_day",), unit="no.", digits=0),
            BaselineVariable("diabetes", "diabetes (self-reported or HbA1c ≥6.5%)", BINARY,
                             derive=_derive_diabetes),
        ),
    ),
    (
        "Lipid profile",
        (
            BaselineVariable("total_cholesterol", "total cholesterol", MEAN_SD,
                             ("biomarker_total_cholesterol_serum",), unit="mg/dL"),
            BaselineVariable("ldl_cholesterol", "LDL cholesterol", MEAN_SD, LDL_CANDIDATES,
                             unit="mg/dL", units_by_column={"ldl_cholesterol": "mmol/L"}, digits=1),
            BaselineVariable("hdl_cholesterol", "HDL cholesterol", MEAN_SD,
                             ("biomarker_hdl_cholesterol_serum", "hdl_cholesterol"), unit="mg/dL",
                             units_by_column={"hdl_cholesterol": "mmol/L"}),
            BaselineVariable("non_hdl_cholesterol", "non-HDL cholesterol", MEAN_SD,
                             ("biomarker_non_hdl_cholesterol_serum", "non_hdl_cholesterol"), unit="mg/dL",
                             units_by_column={"non_hdl_cholesterol": "mmol/L"}),
            BaselineVariable("ldl_above_target", "LDL cholesterol ≥116 mg/dL (3.0 mmol/L)", BINARY,
                             derive=_derive_ldl_above_target),
            BaselineVariable("triglycerides", "triglycerides", MEDIAN_IQR, derive=_derive_triglycerides_mg_dl,
                             unit="mg/dL", digits=0),
            BaselineVariable("apo_b", "apolipoprotein B", MEAN_SD,
                             ("biomarker_serum_apolipoprotein_b_serum",), unit="g/L", digits=2),
            BaselineVariable("apo_a1", "apolipoprotein A1", MEAN_SD,
                             ("biomarker_apolipoprotein_a1_serum",), unit="g/L", digits=2),
            BaselineVariable("apo_ratio", "apolipoprotein A1:B ratio", MEAN_SD,
                             ("biomarker_apolipoprotein_a1_b_ratio_serum",), digits=2),
            BaselineVariable("lipoprotein_a", "lipoprotein(a)", MEDIAN_IQR,
                             ("biomarker_lipoprotein_a_serum",), unit="mg/dL", digits=0),
            BaselineVariable("atherogenic_index", "atherogenic index", MEAN_SD,
                             ("biomarker_atherogenic_index_serum",), digits=2),
        ),
    ),
    (
        "Glucose metabolism",
        (
            BaselineVariable("hba1c", "glycated hemoglobin", MEAN_SD, HBA1C_CANDIDATES, unit="%", digits=2),
            BaselineVariable("glycemic_category", "glycemic category", CATEGORICAL,
                             derive=_derive_glycemic_category,
                             level_order=("Normoglycemia (<5.7)", "Prediabetes (5.7–6.4)",
                                          "Diabetes range (≥6.5)")),
            BaselineVariable("fasting_glucose", "fasting plasma glucose", MEAN_SD,
                             ("biomarker_glucose_fasting_serum",), unit="mg/dL"),
            BaselineVariable("insulin", "fasting insulin", MEDIAN_IQR, ("biomarker_insulin_serum",),
                             unit="pmol/L", digits=0),
            BaselineVariable("homa", "HOMA index", MEDIAN_IQR, ("biomarker_homa_index_serum",), digits=2),
        ),
    ),
    (
        "Renal function",
        (
            BaselineVariable("creatinine", "serum creatinine", MEAN_SD,
                             ("biomarker_creatinine_serum",), unit="mg/dL", digits=2),
            BaselineVariable("egfr", "estimated GFR (CKD-EPI)", MEAN_SD, EGFR_CANDIDATES,
                             unit="mL/min/1.73 m²"),
            BaselineVariable("reduced_egfr", "estimated GFR <60 mL/min/1.73 m²", BINARY,
                             derive=_derive_reduced_egfr),
            BaselineVariable("cystatin_c", "cystatin C", MEAN_SD, ("biomarker_cystatin_c_serum",),
                             unit="mg/L", digits=2),
        ),
    ),
    (
        "Hepatic, inflammatory and other laboratory measures",
        (
            BaselineVariable("alt", "alanine aminotransferase", MEDIAN_IQR,
                             ("biomarker_alanine_aminotransferase_serum",), unit="U/L", digits=0),
            BaselineVariable("ast", "aspartate aminotransferase", MEDIAN_IQR,
                             ("biomarker_aspartate_aminotransferase_serum",), unit="U/L", digits=0),
            BaselineVariable("ggt", "gamma-glutamyl transferase", MEDIAN_IQR,
                             ("biomarker_gamma_glutamyl_transferase_serum",), unit="U/L", digits=0),
            BaselineVariable("fib4", "FIB-4 index", MEDIAN_IQR, ("score_fib4_score",), digits=2),
            BaselineVariable("hs_crp", "high-sensitivity C-reactive protein", MEDIAN_IQR,
                             ("biomarker_high_sensitivity_c_reactive_protein_serum",
                              "biomarker_c_reactive_protein_serum"), unit="mg/L", digits=2),
            BaselineVariable("homocysteine", "homocysteine", MEDIAN_IQR,
                             ("biomarker_homocysteine_serum",), unit="µmol/L"),
            BaselineVariable("ferritin", "ferritin", MEDIAN_IQR, ("biomarker_ferritin_serum",),
                             unit="µg/L", digits=0),
            BaselineVariable("hemoglobin", "hemoglobin", MEAN_SD, ("biomarker_hemoglobin_blood",),
                             unit="g/dL"),
            BaselineVariable("platelets", "platelet count", MEAN_SD,
                             ("biomarker_platelets_automated_count_blood",), unit="×10⁹/L", digits=0),
            BaselineVariable("tsh", "thyrotropin", MEDIAN_IQR, ("biomarker_tsh_serum",),
                             unit="mIU/L", digits=2),
            BaselineVariable("vitamin_d", "25-hydroxyvitamin D", MEAN_SD,
                             ("biomarker_25_hydroxyvitamin_d3_and_d2_serum",), unit="ng/mL"),
        ),
    ),
    (
        "Lifestyle and functional status",
        (
            BaselineVariable("alcohol", "alcohol consumption", CATEGORICAL,
                             ("declared_alcohol_consumption",), max_levels=6),
            BaselineVariable("physical_activity", "physical activity per week", CATEGORICAL,
                             ("social_history_physical_activity_duration_per_week",), max_levels=6),
            BaselineVariable("poor_health", "self-reported poor health", BINARY, ("declared_low_health",)),
            BaselineVariable("reduced_mobility", "reduced mobility", BINARY,
                             ("social_history_reduced_mobility",)),
            BaselineVariable("epworth", "Epworth sleepiness scale", MEDIAN_IQR,
                             ("score_epworth_sleepiness_scale",), digits=0),
            BaselineVariable("stop_bang", "STOP-BANG score", MEDIAN_IQR, ("score_stop_bang",), digits=0),
            BaselineVariable("had", "HAD questionnaire score", MEAN_SD, ("score_questionnaire_HAD",)),
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Internal row / group containers
# --------------------------------------------------------------------------- #
@dataclass
class _Group:
    """One column of the table."""

    label: str
    index: pd.Index

    @property
    def size(self) -> int:
        return int(len(self.index))

    @property
    def header(self) -> str:
        return "%s (n=%d)" % (self.label, self.size)


@dataclass
class _Row:
    """One rendered line of the table."""

    label: str
    values: list[str]
    indent: int = 0
    missing: str = ""
    p_value: str = ""
    smd: str = ""
    is_section: bool = False


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt_mean_sd(values: pd.Series, digits: int) -> str:
    if values.empty:
        return MISSING_PLACEHOLDER
    sd = values.std(ddof=1)
    sd = 0.0 if pd.isna(sd) else float(sd)
    return "%.*f (%.*f)" % (digits, float(values.mean()), digits, sd)


def _fmt_median_iqr(values: pd.Series, digits: int) -> str:
    if values.empty:
        return MISSING_PLACEHOLDER
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75]).tolist()
    return "%.*f (%.*f–%.*f)" % (digits, median, digits, q1, digits, q3)


def _fmt_count(count: int, total: int) -> str:
    if total <= 0:
        return MISSING_PLACEHOLDER
    return "%d/%d (%.1f)" % (count, total, 100.0 * count / total)


def _fmt_missing(n_missing: int, total: int) -> str:
    if total <= 0:
        return ""
    return "%d (%.1f)" % (n_missing, 100.0 * n_missing / total)


def _fmt_p_value(p_value: float | None) -> str:
    if p_value is None or not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    if p_value < 0.01:
        return "%.3f" % p_value
    return "%.2f" % p_value


def _fmt_smd(smd: float | None) -> str:
    if smd is None or not np.isfinite(smd):
        return ""
    return "%.2f" % smd


def _nan_if_none(value: float | None) -> float:
    """Return ``value`` as a float, mapping None to NaN for the long export."""
    return np.nan if value is None else float(value)


# --------------------------------------------------------------------------- #
# Inferential statistics
# --------------------------------------------------------------------------- #
def _continuous_p_value(samples: list[pd.Series], kind: str) -> tuple[float | None, str]:
    """Return the between-group p value and the name of the test used."""
    usable = [s.to_numpy(dtype=float) for s in samples if len(s) >= 2]
    if len(usable) < 2:
        return None, ""
    try:
        if kind == MEDIAN_IQR:
            if len(usable) == 2:
                _, p_value = stats.mannwhitneyu(*usable, alternative="two-sided")
                return float(p_value), "Mann–Whitney U test"
            _, p_value = stats.kruskal(*usable)
            return float(p_value), "Kruskal–Wallis test"
        if len(usable) == 2:
            _, p_value = stats.ttest_ind(*usable, equal_var=False)
            return float(p_value), "Welch t test"
        _, p_value = stats.f_oneway(*usable)
        return float(p_value), "One-way ANOVA"
    except ValueError:
        return None, ""


def _categorical_p_value(table: np.ndarray) -> tuple[float | None, str]:
    """Return the p value of a levels × groups contingency table."""
    matrix = np.asarray(table, dtype=float)
    matrix = matrix[matrix.sum(axis=1) > 0][:, matrix.sum(axis=0) > 0]
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None, ""
    try:
        _, p_value, _, expected = stats.chi2_contingency(matrix, correction=False)
        if matrix.shape == (2, 2) and (np.asarray(expected) < 5).any():
            _, p_value = stats.fisher_exact(matrix.astype(int))
            return float(p_value), "Fisher exact test"
        return float(p_value), "Pearson chi-square test"
    except ValueError:
        return None, ""


def _continuous_smd(exposed: pd.Series, unexposed: pd.Series) -> float | None:
    """Standardized mean difference of a continuous variable (exposed minus not)."""
    if len(exposed) < 2 or len(unexposed) < 2:
        return None
    pooled = (exposed.var(ddof=1) + unexposed.var(ddof=1)) / 2.0
    if not np.isfinite(pooled) or pooled <= 0.0:
        return None
    return float((exposed.mean() - unexposed.mean()) / np.sqrt(pooled))


def _proportion_smd(
    count_exposed: int, total_exposed: int, count_unexposed: int, total_unexposed: int
) -> float | None:
    """Standardized mean difference of a proportion (exposed minus not)."""
    if total_exposed <= 0 or total_unexposed <= 0:
        return None
    p_exposed, p_unexposed = count_exposed / total_exposed, count_unexposed / total_unexposed
    pooled = (p_exposed * (1.0 - p_exposed) + p_unexposed * (1.0 - p_unexposed)) / 2.0
    if pooled <= 0.0:
        return 0.0 if p_exposed == p_unexposed else None
    return float((p_exposed - p_unexposed) / np.sqrt(pooled))


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #
def _ordered_levels(series: pd.Series, variable: BaselineVariable) -> list[str]:
    """Return display levels: preferred order first, then by frequency."""
    counts = series.value_counts(dropna=True)
    ordered = [level for level in variable.level_order if level in counts.index]
    remaining = [str(level) for level in counts.index if level not in ordered]
    ordered.extend(remaining)
    if len(ordered) > variable.max_levels:
        ordered = ordered[: variable.max_levels]
    return ordered


def _continuous_rows(
    variable: BaselineVariable,
    series: pd.Series,
    unit: str,
    groups: list[_Group],
    long_records: list[dict[str, Any]],
    section: str,
    source: str,
) -> list[_Row]:
    """Build the single row of a continuous variable."""
    samples = [series.loc[group.index].dropna() for group in groups]
    formatter = _fmt_mean_sd if variable.kind == MEAN_SD else _fmt_median_iqr
    values = [formatter(sample, variable.digits) for sample in samples]
    p_value, test = _continuous_p_value(samples[1:], variable.kind) if len(groups) > 2 else (None, "")
    smd = _continuous_smd(samples[2], samples[1]) if len(groups) == 3 else None
    total = samples[0]
    for group, sample in zip(groups, samples):
        long_records.append({
            "section": section, "variable": variable.key, "label": variable.name,
            "source_column": source, "statistic": variable.kind, "level": "",
            "group": group.label, "group_n": group.size, "n_with_data": int(len(sample)),
            "mean": float(sample.mean()) if len(sample) else np.nan,
            "sd": float(sample.std(ddof=1)) if len(sample) > 1 else np.nan,
            "median": float(sample.median()) if len(sample) else np.nan,
            "q1": float(sample.quantile(0.25)) if len(sample) else np.nan,
            "q3": float(sample.quantile(0.75)) if len(sample) else np.nan,
            "count": np.nan, "percent": np.nan,
            "p_value": _nan_if_none(p_value), "test": test, "smd": _nan_if_none(smd),
        })
    return [_Row(
        label=variable.label(unit),
        values=values,
        missing=_fmt_missing(int(len(groups[0].index) - len(total)), groups[0].size),
        p_value=_fmt_p_value(p_value),
        smd=_fmt_smd(smd),
    )]


def _binary_rows(
    variable: BaselineVariable,
    series: pd.Series,
    unit: str,
    groups: list[_Group],
    long_records: list[dict[str, Any]],
    section: str,
    source: str,
) -> list[_Row]:
    """Build the single row of a binary variable."""
    samples = [series.loc[group.index].dropna() for group in groups]
    counts = [int(sample.sum()) for sample in samples]
    totals = [int(len(sample)) for sample in samples]
    values = [_fmt_count(count, total) for count, total in zip(counts, totals)]
    p_value, test = (None, "")
    if len(groups) > 2:
        table = np.array([counts[1:], [t - c for t, c in zip(totals[1:], counts[1:])]], dtype=float)
        p_value, test = _categorical_p_value(table)
    smd = _proportion_smd(counts[2], totals[2], counts[1], totals[1]) if len(groups) == 3 else None
    for group, count, total in zip(groups, counts, totals):
        long_records.append({
            "section": section, "variable": variable.key, "label": variable.name,
            "source_column": source, "statistic": BINARY, "level": "",
            "group": group.label, "group_n": group.size, "n_with_data": total,
            "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan,
            "count": count, "percent": 100.0 * count / total if total else np.nan,
            "p_value": _nan_if_none(p_value), "test": test, "smd": _nan_if_none(smd),
        })
    return [_Row(
        label=variable.label(unit),
        values=values,
        missing=_fmt_missing(groups[0].size - totals[0], groups[0].size),
        p_value=_fmt_p_value(p_value),
        smd=_fmt_smd(smd),
    )]


def _categorical_rows(
    variable: BaselineVariable,
    series: pd.Series,
    unit: str,
    groups: list[_Group],
    long_records: list[dict[str, Any]],
    section: str,
    source: str,
) -> list[_Row]:
    """Build the parent row and one indented row per level."""
    samples = [series.loc[group.index].dropna() for group in groups]
    levels = _ordered_levels(samples[0], variable)
    if not levels:
        return []
    totals = [int(len(sample)) for sample in samples]
    counts_by_level = {
        level: [int((sample == level).sum()) for sample in samples] for level in levels
    }
    table = np.array([counts_by_level[level][1:] for level in levels], dtype=float)
    p_value, test = _categorical_p_value(table) if len(groups) > 2 else (None, "")

    level_smds: dict[str, float | None] = {}
    for level in levels:
        counts = counts_by_level[level]
        level_smds[level] = (
            _proportion_smd(counts[2], totals[2], counts[1], totals[1]) if len(groups) == 3 else None
        )
    finite_smds = [abs(value) for value in level_smds.values() if value is not None and np.isfinite(value)]
    parent_smd = max(finite_smds) if finite_smds else None

    rows = [_Row(
        label=variable.label(unit),
        values=["" for _ in groups],
        missing=_fmt_missing(groups[0].size - totals[0], groups[0].size),
        p_value=_fmt_p_value(p_value),
        smd=_fmt_smd(parent_smd),
    )]
    for level in levels:
        counts = counts_by_level[level]
        rows.append(_Row(
            label=level,
            values=[_fmt_count(count, total) for count, total in zip(counts, totals)],
            indent=1,
            smd=_fmt_smd(level_smds[level]),
        ))
        for group, count, total in zip(groups, counts, totals):
            long_records.append({
                "section": section, "variable": variable.key, "label": variable.name,
                "source_column": source, "statistic": CATEGORICAL, "level": level,
                "group": group.label, "group_n": group.size, "n_with_data": total,
                "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan,
                "count": count, "percent": 100.0 * count / total if total else np.nan,
                "p_value": _nan_if_none(p_value), "test": test,
                "smd": _nan_if_none(level_smds[level]),
            })
    return rows


def _physician_rows(
    df: pd.DataFrame, physician_col: str, groups: list[_Group]
) -> list[_Row]:
    """Build the care-setting rows describing the physician clustering."""
    physicians = df[physician_col]
    n_rows = _Row(
        label="No. of prescribing physicians",
        values=[str(int(physicians.loc[group.index].nunique())) for group in groups],
    )
    per_physician = []
    for group in groups:
        counts = physicians.loc[group.index].value_counts()
        per_physician.append(
            _fmt_median_iqr(counts.astype("float64"), 0) if len(counts) else MISSING_PLACEHOLDER
        )
    return [n_rows, _Row(label="Median patients per physician (IQR)", values=per_physician)]


# --------------------------------------------------------------------------- #
# Group construction
# --------------------------------------------------------------------------- #
def _outcome_group_labels(outcome_display: str) -> tuple[str, str]:
    """Return the (no-event, event) column labels for the outcome strata."""
    lowered = outcome_display.strip().lower() or "treatment"
    capitalized = lowered[:1].upper() + lowered[1:]
    return "No %s prescription" % lowered, "%s prescribed" % capitalized


def _build_groups(
    df: pd.DataFrame,
    stratify_by: str | None,
    outcome_col: str,
    outcome_display: str,
) -> tuple[list[_Group], str | None]:
    """Return the table columns (Total first) and the stratification column."""
    groups = [_Group("All patients", df.index)]
    column = stratify_by
    if column is None:
        column = outcome_col if outcome_col in df.columns else _first_column(df, STRATEGY_COL_CANDIDATES)
    if column is None or column not in df.columns:
        return groups, None

    series = df[column]
    if column == outcome_col:
        events = _binary_series(series, ())
        no_event_label, event_label = _outcome_group_labels(outcome_display)
        groups.append(_Group(no_event_label, df.index[events.eq(0.0).to_numpy()]))
        groups.append(_Group(event_label, df.index[events.eq(1.0).to_numpy()]))
    else:
        for level in series.dropna().unique().tolist():
            groups.append(_Group(str(level), df.index[series.eq(level).to_numpy()]))
    return [group for group in groups if group.size > 0 or group.label == "All patients"], column


# --------------------------------------------------------------------------- #
# Table assembly
# --------------------------------------------------------------------------- #
def _resolve_variable(
    df: pd.DataFrame, variable: BaselineVariable
) -> tuple[pd.Series | None, str, str]:
    """Return the (series, unit, source column) for a variable, if available."""
    if variable.derive is not None:
        series = variable.derive(df)
        return series, variable.unit, "derived"
    column = _first_column(df, variable.candidates)
    if column is None:
        return None, "", ""
    series = df[column]
    if variable.kind in (MEAN_SD, MEDIAN_IQR):
        series = _numeric(df, column)
    elif variable.kind == BINARY:
        series = _binary_series(series, variable.positive_values)
    else:
        series = _categorical_series(series)
    return series, variable.units_by_column.get(column, variable.unit), column


def _prepare_series(variable: BaselineVariable, series: pd.Series) -> pd.Series:
    """Coerce a derived series to the dtype expected by the variable kind."""
    if variable.kind in (MEAN_SD, MEDIAN_IQR):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if variable.kind == BINARY:
        return _binary_series(series, variable.positive_values)
    return _categorical_series(series)


def _build_rows(
    df: pd.DataFrame,
    groups: list[_Group],
    min_completeness: float,
    long_records: list[dict[str, Any]],
) -> list[_Row]:
    """Build every table row, skipping variables that are absent or too sparse."""
    total = len(df)
    rows: list[_Row] = []
    builders = {
        MEAN_SD: _continuous_rows,
        MEDIAN_IQR: _continuous_rows,
        BINARY: _binary_rows,
        CATEGORICAL: _categorical_rows,
    }
    for section, variables in TABLE_ONE_SECTIONS:
        section_rows: list[_Row] = []
        for variable in variables:
            series, unit, source = _resolve_variable(df, variable)
            if series is None:
                continue
            series = _prepare_series(variable, series)
            completeness = float(series.notna().sum()) / total if total else 0.0
            if completeness <= 0.0:
                continue
            if completeness < min_completeness and not variable.always_keep:
                continue
            section_rows.extend(
                builders[variable.kind](
                    variable, series, unit, groups, long_records, section, source
                )
            )
        if section_rows:
            rows.append(_Row(label=section, values=["" for _ in groups], is_section=True))
            rows.extend(section_rows)
    return rows


def _header_labels(groups: list[_Group], with_smd: bool) -> list[str]:
    """Return the column headers of the rendered table."""
    headers = ["Characteristic"] + [group.header for group in groups] + ["Missing — no. (%)"]
    if len(groups) > 2:
        headers.append("P value")
    if with_smd:
        headers.append("SMD")
    return headers


def _row_cells(row: _Row, groups: list[_Group], with_smd: bool) -> list[str]:
    """Return the cells of a row, aligned with :func:`_header_labels`."""
    cells = [row.label, *row.values, row.missing]
    if len(groups) > 2:
        cells.append(row.p_value)
    if with_smd:
        cells.append(row.smd)
    return cells


def _display_frame(rows: list[_Row], groups: list[_Group], with_smd: bool) -> pd.DataFrame:
    """Assemble the rendered table as a DataFrame of strings."""
    records: list[list[str]] = []
    for row in rows:
        cells = _row_cells(row, groups, with_smd)
        if row.indent:
            cells[0] = "    %s" % cells[0]
        records.append(cells)
    return pd.DataFrame(records, columns=_header_labels(groups, with_smd))


def _build_footnotes(
    df: pd.DataFrame,
    long_records: list[dict[str, Any]],
    outcome_col: str,
    outcome_display: str,
    physician_col: str | None,
    with_smd: bool,
) -> list[str]:
    """Return the explanatory notes printed below the table."""
    notes = [
        "Continuous variables are reported as mean (SD) or median (interquartile range); "
        "categorical variables as no. with the characteristic/no. with available data (%).",
        "Percentages are computed among patients with non-missing data; the Missing column "
        "reports the number (%) of patients of the whole cohort without a measurement.",
    ]
    tests = sorted({str(record["test"]) for record in long_records if record.get("test")})
    if tests:
        notes.append(
            "Between-group p values were obtained, as appropriate, with the %s, without "
            "adjustment for multiple comparisons." % ", ".join(tests)
        )
    if with_smd:
        notes.append(
            "SMD denotes the standardized mean difference between the two exposure groups; "
            "|SMD| > %.2f is conventionally considered a meaningful imbalance. For multi-level "
            "variables the largest absolute level-wise SMD is reported." % SMD_IMBALANCE_THRESHOLD
        )
    if outcome_col in df.columns:
        events = _binary_series(df[outcome_col], ())
        n_events, n_valid = int(events.sum()), int(events.notna().sum())
        if n_valid:
            notes.append(
                "A %s was prescribed for %d of %d patients (%.1f%%)."
                % (outcome_display.strip().lower(), n_events, n_valid, 100.0 * n_events / n_valid)
            )
    if physician_col is not None and physician_col in df.columns:
        notes.append(
            "Patients were nested within %d prescribing physicians."
            % int(df[physician_col].nunique())
        )
    notes.append(
        "SCORE2 risk categories use the age-specific ESC 2021 thresholds (high risk: ≥2.5% "
        "before 50 yr, ≥5% at 50–69 yr, ≥7.5% from 70 yr; very high risk: ≥7.5%, ≥10% and "
        "≥15% respectively)."
    )
    notes.append(
        "CKD-EPI denotes Chronic Kidney Disease Epidemiology Collaboration equation, "
        "GFR glomerular filtration rate, HAD Hospital Anxiety and Depression scale, "
        "HDL high-density lipoprotein, HOMA homeostatic model assessment, "
        "IQR interquartile range, LDL low-density lipoprotein and SD standard deviation."
    )
    return notes


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
_LATEX_REPLACEMENTS = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
    ("—", "---"),
    ("–", "--"),
    ("≥", r"$\geq$"),
    ("≤", r"$\leq$"),
    ("<", r"$<$"),
    (">", r"$>$"),
    ("×", r"$\times$"),
    ("²", r"\textsuperscript{2}"),
    ("⁹", r"\textsuperscript{9}"),
    ("µ", r"$\mu$"),
    ("€", r"\texteuro{}"),
    ("’", "'"),
)


def _latex_escape(text: str) -> str:
    """Escape a cell for LaTeX, mapping the unicode glyphs we emit."""
    for source, replacement in _LATEX_REPLACEMENTS:
        text = text.replace(source, replacement)
    return text


def _write_latex(
    path: Path,
    rows: list[_Row],
    groups: list[_Group],
    with_smd: bool,
    caption: str,
    notes: list[str],
) -> None:
    """Write a booktabs LaTeX table with section headers and footnotes."""
    headers = _header_labels(groups, with_smd)
    n_columns = len(headers)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        "\\caption{%s}" % _latex_escape(caption),
        "\\label{tab:baseline}",
        "\\begin{tabular}{%s}" % ("l" + "r" * (n_columns - 1)),
        "\\toprule",
        " & ".join("\\textbf{%s}" % _latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row.is_section:
            lines.append("\\addlinespace")
            lines.append(
                "\\multicolumn{%d}{l}{\\textbf{%s}} \\\\" % (n_columns, _latex_escape(row.label))
            )
            continue
        cells = [_latex_escape(cell) for cell in _row_cells(row, groups, with_smd)]
        if row.indent:
            cells[0] = "\\hspace{1.5em}%s" % cells[0]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if notes:
        lines.append("\\begin{minipage}{\\linewidth}")
        lines.append("\\vspace{0.5em}\\scriptsize")
        for note in notes:
            lines.append("\\par %s" % _latex_escape(note))
        lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_HTML_STYLE = """
body { font-family: 'Times New Roman', Times, serif; margin: 2rem; color: #111; }
h1 { font-size: 1.15rem; }
table { border-collapse: collapse; font-size: 0.88rem; }
th, td { border: 1px solid #444; padding: 4px 10px; vertical-align: middle; }
thead th { background: #d9d9d9; text-align: center; font-weight: bold; }
thead th:first-child { text-align: left; }
td:first-child { text-align: left; }
td { text-align: center; }
tr.section td { background: #f2f2f2; font-weight: bold; text-align: left; }
td.level { padding-left: 2.2rem; }
ol.notes { font-size: 0.78rem; color: #333; max-width: 60rem; }
"""


def _write_html(
    path: Path,
    rows: list[_Row],
    groups: list[_Group],
    with_smd: bool,
    caption: str,
    notes: list[str],
) -> None:
    """Write the table as a standalone HTML page for quick visual review."""
    headers = _header_labels(groups, with_smd)
    n_columns = len(headers)

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>%s</title>" % html_escape(caption),
        "<style>%s</style></head><body>" % _HTML_STYLE,
        "<h1>%s</h1>" % html_escape(caption),
        "<table><thead><tr>%s</tr></thead><tbody>"
        % "".join("<th>%s</th>" % html_escape(header) for header in headers),
    ]
    for row in rows:
        if row.is_section:
            parts.append(
                '<tr class="section"><td colspan="%d">%s</td></tr>'
                % (n_columns, html_escape(row.label))
            )
            continue
        label, *values = _row_cells(row, groups, with_smd)
        cells = ['<td class="level">' if row.indent else "<td>", html_escape(label), "</td>"]
        cells.extend("<td>%s</td>" % html_escape(value) for value in values)
        parts.append("<tr>%s</tr>" % "".join(cells))
    parts.append("</tbody></table>")
    if notes:
        parts.append("<ol class='notes'>%s</ol>" % "".join(
            "<li>%s</li>" % html_escape(note) for note in notes
        ))
    parts.append("</body></html>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_baseline_table_one(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    outcome_display: str = "statin",
    physician_col: str | None = DEFAULT_PHYSICIAN_COL,
    stratify_by: str | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
    caption: str | None = None,
) -> pd.DataFrame:
    """
    Generate Table 1: baseline patient characteristics, NEJM/Circulation style.

    Variables are grouped in clinical sections (demographics, anthropometry,
    vital signs, cardiovascular risk, lipids, glucose metabolism, renal
    function, other laboratory measures, lifestyle) and reported as
    ``mean (SD)``, ``median (IQR)`` or ``no./total no. (%)``. Columns are the
    whole cohort followed by the exposure strata (the prescription outcome by
    default), with missing-data accounting, p values and standardized mean
    differences. Variables absent from ``df`` or measured in fewer than
    ``min_completeness`` of patients are omitted.

    Arguments:
        df: Patient-level DataFrame (one row per patient) with baseline data.
        output_dir: Experiment root; files are written under ``output_dir/data/``.
        outcome_col: Binary prescription outcome used to stratify columns.
        outcome_display: Medication name used in the column headers.
        physician_col: Clustering column; adds the care-setting rows.
        stratify_by: Explicit stratification column overriding ``outcome_col``.
        min_completeness: Minimum fraction of non-missing values to keep a row.
        caption: Table caption; a default is built when omitted.

    Returns:
        The rendered table as a DataFrame of strings, section rows included.
    """
    if df.empty:
        raise ValueError("Cannot build Table 1 from an empty DataFrame.")
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # A unique index keeps the per-group ``.loc`` slicing unambiguous.
    df = df.reset_index(drop=True)

    groups, stratification_col = _build_groups(df, stratify_by, outcome_col, outcome_display)
    with_smd = len(groups) == 3
    long_records: list[dict[str, Any]] = []

    rows = [_Row(label="No. of patients", values=[str(group.size) for group in groups])]
    rows.extend(_build_rows(df, groups, min_completeness, long_records))
    if physician_col is not None and physician_col in df.columns:
        rows.append(_Row(label="Care setting", values=["" for _ in groups], is_section=True))
        rows.extend(_physician_rows(df, physician_col, groups))
    if len(rows) <= 1:
        raise ValueError(
            "No baseline variable could be resolved in the DataFrame; expected at least one of "
            "the columns declared in TABLE_ONE_SECTIONS."
        )

    if caption is None:
        suffix = (
            " according to the %s prescription" % outcome_display.strip().lower()
            if stratification_col == outcome_col
            else ""
        )
        caption = "Table 1. Baseline characteristics of the patients%s." % suffix
    notes = _build_footnotes(
        df, long_records, outcome_col, outcome_display, physician_col, with_smd
    )

    table_df = _display_frame(rows, groups, with_smd)
    table_df.to_csv(data_dir / "table_1_baseline.csv", index=False)
    pd.DataFrame(long_records).to_csv(data_dir / "table_1_baseline_long.csv", index=False)
    _write_latex(data_dir / "table_1_baseline.tex", rows, groups, with_smd, caption, notes)
    _write_html(data_dir / "table_1_baseline.html", rows, groups, with_smd, caption, notes)
    return table_df


__all__ = [
    "BaselineVariable",
    "TABLE_ONE_SECTIONS",
    "generate_baseline_table_one",
]
