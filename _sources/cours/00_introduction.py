# %% [markdown]
# # Chapitre 0 — Introduction à l'analyse et au traitement des données
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# ---
#
# ## Objectifs du chapitre
#
# À la fin de ce chapitre, vous saurez :
#
# 1. définir ce qu'est l'**analyse de données** et la distinguer du *traitement* (préparation) des données ;
# 2. situer chaque étape dans un **cycle de projet** (CRISP-DM) ;
# 3. reconnaître les **types de variables** et les **structures de données** courantes ;
# 4. installer et vérifier votre **environnement de travail** Python.
#
# > 💡 **Idée-clé.** Dans un projet réel, on passe couramment **60 à 80 % du temps** à préparer les données
# > (collecte, nettoyage, mise en forme) et seulement une petite partie à « faire des modèles ».
# > Ce cours insiste donc lourdement sur le *traitement*.

# %% [markdown]
# ## 1. Données, information, connaissance
#
# | Niveau | Exemple | Question |
# |---|---|---|
# | **Donnée** brute | `T100071, 2025-12-12, Diourbel, Café Touba 250g, 1200, 2` | Qu'est-ce qui a été enregistré ? |
# | **Information** | « Diourbel a vendu 4 300 paquets de café en décembre » | Que s'est-il passé ? |
# | **Connaissance** | « Les ventes de café doublent pendant le Magal » | Pourquoi ? Que faire ? |
#
# L'**analyse de données** est l'ensemble des méthodes qui permettent de passer d'un niveau à l'autre :
# **décrire**, **expliquer**, **prédire** et **décider**.
#
# On distingue souvent quatre grandes familles d'analyse (de la plus simple à la plus ambitieuse) :
#
# - **Descriptive** — *Que s'est-il passé ?* (tableaux de bord, statistiques résumées)
# - **Diagnostique** — *Pourquoi ?* (comparaisons de groupes, corrélations, tests)
# - **Prédictive** — *Que va-t-il se passer ?* (modèles statistiques et d'apprentissage)
# - **Prescriptive** — *Que faut-il faire ?* (optimisation, simulation de scénarios)

# %% [markdown]
# ## 2. Le cycle d'un projet de données : CRISP-DM
#
# **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*, 1999–2000) reste le cadre
# de référence le plus utilisé pour organiser un projet de données. Il compte **six phases itératives** :
#
# ```text
#        ┌──────────────────────────┐
#        │ 1. Compréhension métier  │◄──────────────┐
#        └────────────┬─────────────┘               │
#                     ▼                             │
#        ┌──────────────────────────┐               │
#        │ 2. Compréhension données │               │
#        └────────────┬─────────────┘               │
#                     ▼                             │
#        ┌──────────────────────────┐               │
#        │ 3. Préparation données   │◄──┐           │
#        └────────────┬─────────────┘   │           │
#                     ▼                 │           │
#        ┌──────────────────────────┐   │           │
#        │ 4. Modélisation          │───┘           │
#        └────────────┬─────────────┘               │
#                     ▼                             │
#        ┌──────────────────────────┐               │
#        │ 5. Évaluation            │───────────────┘
#        └────────────┬─────────────┘
#                     ▼
#        ┌──────────────────────────┐
#        │ 6. Déploiement           │
#        └──────────────────────────┘
# ```
#
# | Phase | Questions à se poser | Chapitres du cours |
# |---|---|---|
# | 1. Métier | Quel problème ? Quel critère de succès ? | 0 |
# | 2. Données | Quelles sources ? Qualité ? Biais ? | 3, 6 |
# | 3. Préparation | Nettoyage, transformation, variables | 2, 4, 5, 8 |
# | 4. Modélisation | Quelle méthode ? | 7, 9 |
# | 5. Évaluation | Le résultat répond-il à la question ? | 7, 9 |
# | 6. Déploiement | Rapport, tableau de bord, automatisation | Projets |
#
# Le modèle **OSEMN** (*Obtain, Scrub, Explore, Model, iNterpret*) est une variante plus compacte
# souvent citée en data science.

# %% [markdown]
# ## 3. Types de variables
#
# Le **type statistique** d'une variable détermine les résumés, graphiques et tests appropriés.
#
# | Type | Sous-type | Exemple (Sénégal) | Résumés adaptés | Graphiques |
# |---|---|---|---|---|
# | **Qualitative** (catégorielle) | Nominale | région, mode de paiement | effectifs, mode | barres |
# | | Ordinale | niveau d'éducation | médiane, quantiles | barres ordonnées |
# | **Quantitative** | Discrète | taille du ménage | moyenne, médiane, écart-type | histogramme, boîte |
# | | Continue | revenu, température | moyenne, médiane, écart-type, quantiles | histogramme, densité, boîte |
#
# ⚠️ **Piège classique** : un code numérique n'est pas forcément une quantité.
# Un code postal, un identifiant ou un code « 99 = ne sait pas » **ne doivent pas** être moyennés.
#
# **Échelles de mesure (Stevens, 1946)** : nominale → ordinale → d'intervalle (ex. °C, le zéro est
# arbitraire) → de rapport (ex. revenu, le zéro est absolu ; « deux fois plus » a un sens).

# %% [markdown]
# ## 4. Structures de données
#
# - **Données tabulaires** (tableaux lignes × colonnes) — le cœur de ce cours.
# - **Séries temporelles** (une observation indexée par le temps) — chapitre 5.
# - **Données textuelles**, **images**, **graphes**, **données géospatiales** — hors-programme principal,
#   mais les mêmes principes de qualité s'appliquent.
#
# ### Le principe des *tidy data* (Wickham, 2014)
#
# Un jeu de données est **« rangé » (tidy)** quand :
#
# 1. chaque **variable** est une **colonne** ;
# 2. chaque **observation** est une **ligne** ;
# 3. chaque **type d'unité d'observation** forme une **table**.
#
# Presque toutes les fonctions de `pandas`, `seaborn` et `scikit-learn` supposent ce format.
# Le chapitre 5 montre comment passer d'un format « large » à un format « long » (`melt`, `pivot`).

# %% [markdown]
# ## 5. Qualité des données : les dimensions à vérifier
#
# | Dimension | Question | Exemple de défaut |
# |---|---|---|
# | **Complétude** | Manque-t-il des valeurs ? | revenu non déclaré |
# | **Validité** | Les valeurs respectent-elles les règles ? | quantité négative, température −99 |
# | **Exactitude** | Les valeurs reflètent-elles la réalité ? | faute de frappe 999 au lieu de 9 |
# | **Cohérence** | Les mêmes faits sont-ils codés pareil ? | « Dakar », « dakar », « DAKAR » |
# | **Unicité** | Y a-t-il des doublons ? | transaction importée deux fois |
# | **Actualité** | Les données sont-elles à jour ? | référentiel de 2013 |
#
# Les principes **FAIR** (*Findable, Accessible, Interoperable, Reusable* — Wilkinson et al., 2016)
# complètent cette liste du point de vue du partage et de la réutilisation des données.

# %% [markdown]
# ## 6. Éthique et protection des données
#
# Analyser des données sur des personnes impose des obligations :
#
# - **Base légale et finalité** : au Sénégal, la *loi n° 2008-12 sur la protection des données
#   à caractère personnel* et la **CDP** (Commission de Protection des Données Personnelles)
#   encadrent la collecte et le traitement ; en Europe c'est le RGPD.
# - **Minimisation** : ne collecter que ce qui est nécessaire.
# - **Pseudonymisation / anonymisation** : retirer noms, téléphones, numéros d'identité avant l'analyse.
# - **Biais** : un échantillon non représentatif (ex. enquête uniquement en ligne) produit des conclusions fausses
#   pour la population entière.
#
# > Toutes les données de ce cours sont **simulées** : elles ressemblent à des données réelles mais
# > ne concernent aucune personne existante.

# %% [markdown]
# ## 7. Environnement de travail
#
# ### Installation recommandée
#
# ```bash
# # 1. Cloner le dépôt
# git clone https://github.com/iamnewton4596/analyse-traitement-donnees.git
# cd analyse-traitement-donnees
#
# # 2. Créer un environnement (au choix)
# python -m venv .venv && source .venv/bin/activate      # Linux / macOS
# # .venv\Scripts\activate                               # Windows
# pip install -r requirements.txt
# #   — ou —
# conda env create -f environment.yml && conda activate atd
#
# # 3. Lancer Jupyter
# jupyter lab
# ```
#
# Alternative sans installation : ouvrir les notebooks dans **Google Colab** (un bouton est fourni
# dans le README).
#
# ### Vérification
# Exécutez la cellule ci-dessous : elle doit afficher les versions sans erreur.

# %%
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn

print(f"Python       {sys.version.split()[0]}")
for nom, mod in [("numpy", np), ("pandas", pd), ("scipy", scipy), ("matplotlib", matplotlib),
                 ("seaborn", sns), ("scikit-learn", sklearn)]:
    print(f"{nom:<12} {mod.__version__}")

assert int(pd.__version__.split(".")[0]) >= 2, "Ce cours suppose pandas ≥ 2 (idéalement 3.x)."

# %% [markdown]
# ### Les jeux de données du cours
#
# Tous les fichiers sont dans `data/brutes/` et sont décrits dans `data/README.md`.
# Ils sont regénérables à l'identique avec `python scripts/generer_donnees.py`.

# %%
from pathlib import Path

DATA = next(p for p in [Path("data/brutes"), Path("../data/brutes"), Path("../../data/brutes")] if p.exists())
for f in sorted(DATA.glob("*.csv")):
    d = pd.read_csv(f)
    print(f"{f.name:<28} {d.shape[0]:>6} lignes × {d.shape[1]:>2} colonnes")

# %% [markdown]
# Premier coup d'œil à un fichier :

# %%
ventes = pd.read_csv(DATA / "ventes_boutiques.csv")
ventes.head()

# %% [markdown]
# Remarquez déjà plusieurs défauts : des prix écrits « 2 290 FCFA », des régions avec des espaces,
# des dates de formats différents… **Ce sera notre fil rouge.**
#
# ## À retenir
#
# - Un projet de données suit un cycle **itératif** (CRISP-DM) qui commence par la question métier.
# - Le **type** d'une variable conditionne tout le reste de l'analyse.
# - Le format **tidy** (une variable = une colonne, une observation = une ligne) est la cible de la préparation.
# - La **qualité** se mesure selon plusieurs dimensions : complétude, validité, exactitude, cohérence, unicité, actualité.
#
# ## Pour aller plus loin
#
# - Chapman et al. (2000), *CRISP-DM 1.0 : Step-by-step data mining guide*.
# - Wickham H. (2014), « Tidy Data », *Journal of Statistical Software*, 59(10).
# - Wilkinson M. et al. (2016), « The FAIR Guiding Principles… », *Scientific Data*, 3.
#
# ➡️ **Chapitre suivant : Python scientifique et NumPy.**
