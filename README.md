# Measuring Intra-Physician Prescribing Variability to Target Continuing Medical Education: A Comparative Study of Five Patient-Matching Methods

Pipeline for analysing intra-physician prescription variability on synthetic cohorts, using several comparison methods (GLMM + matching).

## Overview

This repository lets you:

- generate synthetic patient cohorts under controlled clinical rules;
- run a multi-method statistical analysis pipeline;
- compare discordance methods across physicians;
- produce reproducible artefacts (CSV files, reports, figures, and the effective configuration).

The application entry point is `main.py`. The analytical core lives in `src/prism/analysis/analysis_pipeline.py`.

## Project structure

```text
.
├── configs/article_stats.yaml  # Single run configuration
├── src/prism/
│   ├── analysis/               # Analysis pipeline, matching, plots
│   ├── dataset_utils/          # Synthetic cohort generation
│   ├── logs_utils/             # Logging
│   └── experiment_paths.py     # Experiment directories
├── main.py                     # CLI orchestrator
└── pyproject.toml              # Poetry dependencies and project metadata
```

Run outputs (`exp/`) are not versioned.

## Requirements

- Python `>=3.12,<3.13`
- Poetry installed locally

## Installation

```bash
poetry install
```

## Running

```bash
poetry run python main.py --config configs/article_stats.yaml
```

Without Poetry:

```bash
python main.py --config configs/article_stats.yaml
```

## Usage examples

- **Single SCORE2 experiment**: `dataset.generation_strategy: score2_five_groups_heter_patients` and `dataset.multi_rule_experiments.enabled: false`.
- **Multi-rule batch**: `dataset.multi_rule_experiments.enabled: true`.
- **Outlier detection**: `analysis.outlier_detection.enabled: true` (default) writes an audit report without changing the data. `auto_repair: true` replaces flagged cells with the column median.
- **Main outputs** (under the experiment folder):
  - `config_used.yaml`
  - `data/dataset.csv`
  - `intra_physician_variability.csv`
  - `plots/method_comparison/comparison_3panels.png`
  - `plots/method_comparison/comparison_by_tertile.png`
  - `plots/bernoulli_residual/discordance_minus_bernoulli_*.png`
  - `plots/bernoulli_residual/discordance_minus_bernoulli_all_methods.png`
  - `plots/bernoulli_residual/perfect_concordance_gap_all_methods.png`
  - `plots/bernoulli_residual/perfect_concordance_gap_ratio_*.png`
  - `plots/bernoulli_residual/perfect_concordance_gap_ratio_all_methods.png`
  - `analysis_columns_used.csv`
  - `snapshots/*matching_features_selected_*`
  - `snapshots/*outlier_detection_manifest*`
  - `validation/validation_report.json`
  - `validation/validation_summary.csv`
  - `validation/matching_balance_heatmap.png`
  - `clinical_reviews/validation_gate.json`
  - `plots/cross_experiments/*` (batch mode)

## Automatic validation (post-run audit)

Every run with `analysis.validation.enabled: true` (default) writes a report under `validation/`:

| File | Contents |
|---|---|
| `validation_report.json` | All checks (PASS / WARN / FAIL) |
| `validation_summary.csv` | Tabular view of the checks |
| `validation_summary.md` | Markdown summary |
| `pair_quality_by_method.csv` | Post-matching SMD by method |
| `matching_balance_by_method.csv` | SMD by covariate |
| `matching_balance_heatmap.png` | Balance heatmap |
| `method_consistency_matrix.csv` | Spearman / Δ between methods |

Before **clinical review** (`clinical_reviews/*_pair_reviewer.html`), read `clinical_reviews/validation_gate.json`.

```yaml
analysis:
  validation:
    enabled: true
    announce_console: true
    fail_on_critical: false
    calibration_profile: auto # synthetic_signal | synthetic_null
```

## Contributing

The contribution process is described in `CONTRIBUTING.md`. Install local hooks with `pre-commit install`.

## Data, secrets, and publication

The default config (`configs/article_stats.yaml`) generates a **synthetic SCORE2 cohort**. No warehouse access is required. Never commit patient extracts or `.env` files.

## License

This project is distributed under the MIT License. See `LICENSE`.

## Contact

Questions and contributions go through GitHub issues and pull requests. The code of conduct is in `CODE_OF_CONDUCT.md`. Report vulnerabilities as described in `SECURITY.md`.

Or, you can directly contact :

- BENANI D. Alaedine
- alaedine.benani@aphp.fr
- (+33) 6 74 38 12 39