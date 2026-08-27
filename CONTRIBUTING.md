# Guide de contribution

Ce document décrit les bonnes pratiques pour contribuer à `prism-1` de manière fiable et reproductible.

## 1) Avant de commencer

- Utiliser Python `>=3.12,<3.13`.
- Installer les dépendances avec `poetry install`.
- Créer une branche dédiée par sujet (ex: `feat/xxx`, `fix/yyy`, `docs/zzz`).
- Vérifier que les secrets ne sont jamais versionnés (`.env`, clés AWS, tokens).
- Ne pas versionner d'extraits patients, de listes d'identifiants réels, ni de notebooks de labo internes.
- Lire `CODE_OF_CONDUCT.md` et `SECURITY.md`.

## 2) Convention de travail

- Faire des commits petits, cohérents et descriptifs.
- Préférer une modification atomique par commit (code + documentation associée).
- Garder la configuration de référence dans `configs/article_stats.yaml` et documenter tout nouveau paramètre.

## 3) Qualité de code

Installer les hooks (une fois) :

```bash
poetry run pre-commit install
```

Avant de pousser:

```bash
poetry run pre-commit run --all-files
```

## 4) Issues

Pour ouvrir une issue utile:

- décrire le comportement observé et le comportement attendu;
- préciser le contexte (fichier de config, mode single/batch, version Python);
- joindre les traces utiles (erreur, extrait de log, artefacts produits).

## 5) Pull Requests

Chaque Pull Request doit contenir:

- un objectif clair (pourquoi le changement est nécessaire);
- la liste des fichiers impactés;
- un plan de vérification exécuté localement;
- les impacts potentiels sur les résultats analytiques.

Checklist recommandée:

- [ ] code lisible et documenté (docstrings si API/fonctions exposées)
- [ ] documentation mise à jour (`README.md`, config)
- [ ] absence de secrets et de fichiers volumineux non nécessaires

## 6) Documentation

Toute évolution sur:

- la structure du pipeline,
- les méthodes de matching,
- les formats de sortie,

doit être répercutée dans `README.md` et `configs/article_stats.yaml`.
