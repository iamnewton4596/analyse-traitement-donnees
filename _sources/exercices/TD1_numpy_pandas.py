# %% [markdown]
# # TD 1 — NumPy et pandas : les bases (chapitres 1–2)
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# Durée indicative : 1 h 30. Difficulté : ⭐ facile · ⭐⭐ moyen · ⭐⭐⭐ avancé.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA = next(p for p in [Path("data/brutes"), Path("../data/brutes"), Path("../../data/brutes")] if p.exists())
rng = np.random.default_rng(0)

# %% [markdown]
# ## Exercice 1 ⭐ — Vectorisation
# On donne les prix (FCFA) et les quantités vendues de 5 produits.
# 1. Calculez le chiffre d'affaires (CA) par produit **sans boucle**.
# 2. Calculez la part (%) de chaque produit dans le CA total, arrondie à 1 décimale.
# 3. Affichez les noms des produits dont le CA dépasse 20 000 FCFA.

# %%
noms = np.array(["Riz", "Huile", "Sucre", "Lait", "Pain"])
prix = np.array([12500, 6500, 700, 2300, 150])
qte = np.array([3, 4, 25, 6, 120])

# %% tags=["solution"]
ca = prix * qte
part = (100 * ca / ca.sum()).round(1)
print(ca, part)
print(noms[ca > 20_000])

# %% [markdown]
# ## Exercice 2 ⭐⭐ — Broadcasting
# `notes` contient les notes de 6 étudiants (lignes) dans 4 matières (colonnes).
# 1. Calculez la moyenne de chaque matière et de chaque étudiant.
# 2. Centrez-réduisez **chaque matière** (z-score par colonne, `ddof=1`) en une seule expression.
# 3. Quel étudiant a la meilleure moyenne de z-scores ?

# %%
notes = rng.integers(4, 20, size=(6, 4)).astype(float)
notes

# %% tags=["solution"]
print("moy. matières :", notes.mean(axis=0))
print("moy. étudiants :", notes.mean(axis=1))
z = (notes - notes.mean(axis=0)) / notes.std(axis=0, ddof=1)
print(z.round(2))
print("meilleur étudiant (indice) :", z.mean(axis=1).argmax())

# %% [markdown]
# ## Exercice 3 ⭐⭐ — Filtrage et valeurs manquantes
# Une série de températures contient des `NaN` et des codes d'erreur `-99`.
# Calculez la moyenne, la médiane et le nombre de valeurs **valides** (ni NaN, ni -99, et comprises entre 10 et 45 °C).

# %%
temp = np.array([27.5, 28.1, -99, np.nan, 31.0, 29.4, 55.0, 26.8, np.nan, 30.2])

# %% tags=["solution"]
valide = (temp > 10) & (temp < 45)          # les comparaisons avec NaN renvoient False : NaN exclus d'office
print(f"n valides = {valide.sum()}, moyenne = {temp[valide].mean():.2f}, médiane = {np.median(temp[valide]):.2f}")

# %% [markdown]
# ## Exercice 4 ⭐ — Premiers pas avec pandas
# Chargez `enquete_menages.csv`.
# 1. Combien de lignes et de colonnes ? Quels types ?
# 2. Combien de ménages par milieu ? En proportion ?
# 3. Quel est le revenu médian des ménages de **Ziguinchor** ?

# %% tags=["solution"]
m = pd.read_csv(DATA / "enquete_menages.csv")
print(m.shape); print(m.dtypes)
print(m["milieu"].value_counts()); print(m["milieu"].value_counts(normalize=True).round(3))
print("Revenu médian Ziguinchor :", m.loc[m["region"] == "Ziguinchor", "revenu_mensuel_fcfa"].median())

# %% [markdown]
# ## Exercice 5 ⭐⭐ — loc, iloc, query
# 1. Avec `iloc`, affichez les lignes 10 à 14 (incluses) et les 3 premières colonnes.
# 2. Avec `loc`, affichez `region`, `milieu` et `revenu_mensuel_fcfa` des ménages ruraux de plus de 15 personnes
#    (attention au code 99 !).
# 3. Réécrivez la question 2 avec `query`.

# %% tags=["solution"]
print(m.iloc[10:15, :3])
filtre = (m["milieu"] == "Rural") & (m["taille_menage"] > 15) & (m["taille_menage"] != 99)
print(m.loc[filtre, ["region", "milieu", "revenu_mensuel_fcfa"]])
print(m.query("milieu == 'Rural' and taille_menage > 15 and taille_menage != 99")[["region", "milieu", "revenu_mensuel_fcfa"]].shape)

# %% [markdown]
# ## Exercice 6 ⭐⭐ — Créer des colonnes et Copy-on-Write
# 1. Créez `revenu_par_tete` (revenu / taille, en traitant 99 comme manquant **sans modifier la colonne d'origine**).
# 2. Créez une colonne `niveau_vie` valant `"bas"`, `"moyen"`, `"élevé"` selon les terciles de `revenu_par_tete` (`pd.qcut`).
# 3. Expliquez pourquoi `m["revenu_mensuel_fcfa"][m["region"] == "Dakar"] = 0` ne modifie pas `m` en pandas 3.

# %% tags=["solution"]
taille = m["taille_menage"].where(m["taille_menage"] != 99)      # 99 -> NaN, sur une nouvelle Series
m["revenu_par_tete"] = m["revenu_mensuel_fcfa"] / taille
m["niveau_vie"] = pd.qcut(m["revenu_par_tete"], 3, labels=["bas", "moyen", "élevé"])
print(m["niveau_vie"].value_counts())
# 3. Copy-on-Write : m["revenu_mensuel_fcfa"] renvoie un objet qui se comporte comme une copie ;
#    l'affectation modifie cette copie temporaire, jamais m. Il faut écrire :
#    m.loc[m["region"] == "Dakar", "revenu_mensuel_fcfa"] = 0

# %% [markdown]
# ## Exercice 7 ⭐⭐⭐ — Simulation : le paradoxe des anniversaires
# Par simulation (10 000 tirages), estimez la probabilité qu'au moins deux étudiants d'une classe de 30 aient le même
# jour d'anniversaire (365 jours équiprobables). Comparez à la valeur exacte $1 - \prod_{i=0}^{29}(365-i)/365$.
# Contrainte : **aucune boucle Python sur les élèves**.

# %% tags=["solution"]
tirages = rng.integers(0, 365, size=(10_000, 30))
tri = np.sort(tirages, axis=1)
au_moins_un_doublon = (np.diff(tri, axis=1) == 0).any(axis=1)
exact = 1 - np.prod((365 - np.arange(30)) / 365)
print(f"simulation : {au_moins_un_doublon.mean():.3f} | exact : {exact:.3f}")
