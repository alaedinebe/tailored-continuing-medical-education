# Prism-1

Pipeline d'analyse de la variabilité intra-médecin des prescriptions, basé sur des cohortes synthétiques et plusieurs méthodes de comparaison (GLMM + matching).

## Présentation

Ce dépôt permet de:

- générer des jeux de données synthétiques de patients selon des règles cliniques contrôlées;
- exécuter un pipeline d'analyse statistique multi-méthodes;
- comparer les méthodes de discordance entre médecins;
- produire des artefacts reproductibles (CSV, rapports, figures, configuration effective).

Le point d'entrée applicatif est `main.py`, et le coeur analytique se trouve dans `src/prism/analysis/analysis_pipeline.py`.

## Structure du projet

```text
.
├── configs/article_stats.yaml  # Configuration d'exécution (unique)
├── src/prism/
│   ├── analysis/               # Pipeline analytique, matching, visualisations
│   ├── dataset_utils/          # Génération de cohortes synthétiques
│   ├── logs_utils/             # Logging
│   └── experiment_paths.py     # Dossiers d'expérience
├── main.py                     # Orchestrateur CLI
└── pyproject.toml              # Dépendances et configuration Poetry
```

Les sorties d'exécution (`exp/`) et les extraits locaux (`data/`) ne sont pas versionnés.

## Prérequis

- Python `>=3.12,<3.13`
- Poetry installé localement

## Installation

```bash
poetry install
```

## Exécution

```bash
poetry run python main.py --config configs/article_stats.yaml
```

Sans Poetry:

```bash
python main.py --config configs/article_stats.yaml
```

## Exemples d'utilisation

- **Expérience unique SCORE2**: `dataset.generation_strategy: score2_five_groups_heter_patients` et `dataset.multi_rule_experiments.enabled: false`.
- **Batch multi-règles**: `dataset.multi_rule_experiments.enabled: true`.
- **Détection de valeurs aberrantes**: `analysis.outlier_detection.enabled: true` (défaut) calcule un rapport d'audit sans modifier les données. `auto_repair: true` remplace les cellules flaguées par la médiane.
- **Sorties principales** (sous le dossier d'expérience) :
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
  - `plots/cross_experiments/*` (en mode batch)

## Validation automatique (audit post-run)

Chaque exécution avec `analysis.validation.enabled: true` (défaut) produit un rapport sous `validation/` :

| Fichier | Contenu |
|---|---|
| `validation_report.json` | Tous les checks (PASS / WARN / FAIL) |
| `validation_summary.csv` | Tableau lisible des checks |
| `validation_summary.md` | Résumé markdown |
| `pair_quality_by_method.csv` | SMD post-appariement par méthode |
| `matching_balance_by_method.csv` | SMD par covariable |
| `matching_balance_heatmap.png` | Heatmap balance |
| `method_consistency_matrix.csv` | Spearman / Δ entre méthodes |

Avant la **revue clinique** (`clinical_reviews/*_pair_reviewer.html`), lire `clinical_reviews/validation_gate.json`.

```yaml
analysis:
  validation:
    enabled: true
    announce_console: true
    fail_on_critical: false
    calibration_profile: auto # synthetic_signal | synthetic_null
```

## Contribution

Le processus de contribution est décrit dans `CONTRIBUTING.md`. Installez les hooks locaux avec `pre-commit install`.

## Données, secrets et publication

Le mode par défaut (`configs/article_stats.yaml`) génère une **cohorte synthétique SCORE2**. Aucun accès entrepôt n'est requis. Ne versionnez jamais d'extraits patients ni de fichiers `.env`.

**Historique Git :** des clones privés peuvent encore contenir des données cliniques dans d'anciens commits. Ne poussez pas l'historique existant vers un dépôt public. Créez un dépôt neuf (branche orpheline) à partir de l'arbre de travail nettoyé, puis lancez un scan de secrets. Voir `SECURITY.md`.

## Licence

Ce projet est distribué sous licence MIT. Voir `LICENSE`.

## Contact

Les questions et contributions passent par les issues et pull requests du dépôt. Le code de conduite est dans `CODE_OF_CONDUCT.md`. Les vulnérabilités se signalent selon `SECURITY.md`.
