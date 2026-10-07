# Analyse et traitement des données avec Python

**Auteur : Issa Gueye** — doctorant en mathématiques appliquées (UGB · IPD · CINERI), ingénieur calcul scientifique.

Un cours complet, en français, pour apprendre à **passer de données brutes et imparfaites à des résultats fiables** :
importer, auditer, nettoyer, transformer, explorer, tester, préparer pour la modélisation, réduire la dimension et segmenter.
Tous les exemples s'appuient sur des **données simulées inspirées du Sénégal** (boutiques, météo de Dakar, enquête ménages,
résultats d'étudiants, mobile money) qui contiennent volontairement les défauts rencontrés dans la vraie vie.

[![Exécution des notebooks](https://github.com/iamnewton4596/analyse-traitement-donnees/actions/workflows/notebooks.yml/badge.svg)](https://github.com/iamnewton4596/analyse-traitement-donnees/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![pandas](https://img.shields.io/badge/pandas-3.x-150458)
![Licence](https://img.shields.io/badge/licence-CC%20BY--NC--SA%204.0%20%2F%20MIT-lightgrey)

---

## Contenu du dépôt

```text
analyse-traitement-donnees/
├── cours/            10 chapitres (notebooks exécutés, avec résultats et graphiques)
├── exercices/
│   ├── enonces/      4 TD à faire
│   └── corriges/     les mêmes TD, corrigés et commentés
├── projets/          3 projets intégrateurs + grille d'évaluation + notebook de démarrage
├── memos/            fiches synthétiques (pandas, checklist de nettoyage, stats/visualisation/ML)
├── slides/           support de présentation (Markdown Marp, HTML, PDF)
├── data/             6 jeux de données simulés + dictionnaire des variables
├── src/atd/          fonctions réutilisables (audit de qualité, nettoyage)
├── scripts/          génération des données, construction des notebooks
├── references/       bibliographie commentée et sources de données réelles
└── _sources/         sources des notebooks au format texte (jupytext)
```

## Programme

| # | Chapitre | Notions clés | Colab |
|---|---|---|---|
| 0 | [Introduction](cours/00_introduction.ipynb) | CRISP-DM, types de variables, tidy data, qualité, éthique, environnement | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/00_introduction.ipynb) |
| 1 | [Python scientifique et NumPy](cours/01_numpy.ipynb) | ndarray, vectorisation, broadcasting, indexation booléenne, aléatoire reproductible, moindres carrés | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/01_numpy.ipynb) |
| 2 | [pandas : les fondamentaux](cours/02_pandas_fondamentaux.ipynb) | Series/DataFrame, `loc`/`iloc`, filtres, dtypes, **pandas 3.0** (Copy-on-Write, `str`, `pd.col`) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/02_pandas_fondamentaux.ipynb) |
| 3 | [Import, export, audit de qualité](cours/03_import_export_qualite.ipynb) | CSV piégeux, encodages, Excel/JSON/Parquet, profilage, règles de validation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/03_import_export_qualite.ipynb) |
| 4 | [Nettoyage](cours/04_nettoyage.ipynb) | types, texte, doublons, MCAR/MAR/MNAR, imputation (groupe, KNN, MICE), interpolation, aberrants (IQR, MAD), pipeline de nettoyage | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/04_nettoyage.ipynb) |
| 5 | [Transformer et restructurer](cours/05_transformation.ipynb) | `groupby`/`agg`/`transform`, pivots, `melt`, jointures sûres, `resample`, `rolling` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/05_transformation.ipynb) |
| 6 | [Analyse exploratoire et visualisation](cours/06_eda_visualisation.ipynb) | choix du graphique, distributions, corrélations, Anscombe, facettes, bonnes pratiques | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/06_eda_visualisation.ipynb) |
| 7 | [Statistique inférentielle](cours/07_statistique_inferentielle.ipynb) | TCL, IC, tests (Welch, Mann–Whitney, ANOVA, Kruskal, χ²), tailles d'effet, bootstrap, tests multiples | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/07_statistique_inferentielle.ipynb) |
| 8 | [Feature engineering et pipelines](cours/08_feature_engineering.ipynb) | encodage, mise à l'échelle, transformations, création de variables, **fuite de données**, `ColumnTransformer` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/08_feature_engineering.ipynb) |
| 9 | [ACP, clustering, modélisation](cours/09_acp_clustering_modelisation.ipynb) | ACP (théorie + cercle des corrélations), k-means, silhouette, CAH, régression, classification, validation croisée, métriques | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamnewton4596/analyse-traitement-donnees/blob/main/cours/09_acp_clustering_modelisation.ipynb) |

> 💡 Sur Colab, exécutez d'abord dans une cellule :
> `!git clone https://github.com/iamnewton4596/analyse-traitement-donnees.git && %cd analyse-traitement-donnees`
> pour que les notebooks trouvent les données.

### Travaux dirigés

| TD | Chapitres | Énoncé | Corrigé |
|---|---|---|---|
| 1 — NumPy et pandas | 1–2 | [énoncé](exercices/enonces/TD1_numpy_pandas.ipynb) | [corrigé](exercices/corriges/TD1_numpy_pandas_corrige.ipynb) |
| 2 — Qualité et nettoyage | 3–4 | [énoncé](exercices/enonces/TD2_qualite_nettoyage.ipynb) | [corrigé](exercices/corriges/TD2_qualite_nettoyage_corrige.ipynb) |
| 3 — Transformation, EDA, inférence | 5–7 | [énoncé](exercices/enonces/TD3_transformation_eda_stats.ipynb) | [corrigé](exercices/corriges/TD3_transformation_eda_stats_corrige.ipynb) |
| 4 — Préparation, ACP, clustering | 8–9 | [énoncé](exercices/enonces/TD4_preparation_acp_clustering.ipynb) | [corrigé](exercices/corriges/TD4_preparation_acp_clustering_corrige.ipynb) |

### Projets

1. [Tableau de bord des ventes d'un réseau de boutiques](projets/projet1_ventes/) (avec [notebook de démarrage](projets/projet1_ventes/demarrage.ipynb))
2. [Conditions de vie des ménages : inégalités et accès aux services](projets/projet2_menages/)
3. [Segmentation des clients d'un service de mobile money](projets/projet3_segmentation/)

Grille d'évaluation commune : [projets/README.md](projets/README.md).

### Mémos et slides

- [Mémo pandas 3.x](memos/memo_pandas.md) · [Checklist de nettoyage](memos/memo_nettoyage_checklist.md) · [Statistiques, visualisation et ML](memos/memo_stats_visualisation_ml.md)
- Slides : [PDF](slides/slides_cours.pdf) · [HTML](slides/slides_cours.html) · [source Marp](slides/slides_cours.md)

## Public et prérequis

Étudiants (licence/master), ingénieurs et analystes débutants à intermédiaires. Prérequis : bases de Python
(variables, listes, fonctions, boucles) et de statistique descriptive. Volume indicatif : **30 h de cours/TD + 25 h de projets**.

### Progression conseillée (6 semaines)

| Semaine | Contenu |
|---|---|
| 1 | Chap. 0–2 · TD 1 |
| 2 | Chap. 3–4 · TD 2 · lancement du projet 1 |
| 3 | Chap. 5–6 · rendu du projet 1 |
| 4 | Chap. 7 · TD 3 · lancement du projet 2 |
| 5 | Chap. 8–9 · TD 4 |
| 6 | Projet 2 ou 3 · soutenances |

## Installation

```bash
git clone https://github.com/iamnewton4596/analyse-traitement-donnees.git
cd analyse-traitement-donnees
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt                        # ou : conda env create -f environment.yml && conda activate atd
jupyter lab
```

Le cours est écrit pour **pandas 3.x** (sorti en janvier 2026) ; les différences avec pandas 2 sont signalées dans le texte.

## Pour les enseignants / contributeurs

- Les notebooks sont générés à partir des fichiers texte de `_sources/` (format *jupytext* « percent »), plus faciles à relire et versionner.
- `python scripts/construire.py` reconstruit et **exécute** tous les notebooks ; les énoncés de TD sont produits automatiquement
  à partir des corrigés (cellules marquées `tags=["solution"]`).
- `python scripts/generer_donnees.py` regénère les données à l'identique (graine fixe).
- Une intégration continue GitHub Actions ré-exécute tout à chaque modification.

## Licence

Contenu pédagogique : **CC BY-NC-SA 4.0** · Code : **MIT** — voir [LICENSE](LICENSE).
Merci de citer : *Gueye I. (2026). Analyse et traitement des données avec Python. https://github.com/iamnewton4596/analyse-traitement-donnees*
