# %% [markdown]
# # Chapitre 2 — pandas : les fondamentaux
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# ## Objectifs
#
# - Manipuler les deux structures de base : **`Series`** et **`DataFrame`**.
# - Sélectionner des lignes et colonnes avec **`loc`**, **`iloc`** et des **masques booléens**.
# - Créer, modifier, trier, compter.
# - Connaître les **nouveautés de pandas 3.0** (Copy-on-Write, type `str`, `pd.col`).

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA = next(p for p in [Path("data/brutes"), Path("../data/brutes"), Path("../../data/brutes")] if p.exists())
pd.set_option("display.max_columns", 20)
print("pandas", pd.__version__)

# %% [markdown]
# ## 1. `Series` : un tableau 1D avec un index

# %%
prix = pd.Series([12500, 6500, 700], index=["Riz 25kg", "Huile 5L", "Sucre 1kg"], name="prix_fcfa")
print(prix)
print("Prix de l'huile :", prix["Huile 5L"])
print(prix * 1.18)   # vectorisé, comme NumPy, l'index est conservé

# %% [markdown]
# L'**index** aligne automatiquement les données lors des opérations :

# %%
stock = pd.Series({"Sucre 1kg": 40, "Riz 25kg": 10, "Pain": 120})
print(prix * stock)   # alignement par étiquette ; NaN là où l'une des deux manque

# %% [markdown]
# ## 2. `DataFrame` : un tableau 2D de colonnes typées
#
# Chaque colonne est une `Series` ; toutes partagent le même index de lignes.

# %%
df = pd.DataFrame({
    "region": ["Dakar", "Thiès", "Kaolack", "Dakar", "Ziguinchor"],
    "milieu": ["Urbain", "Urbain", "Rural", "Urbain", "Rural"],
    "taille_menage": [6, 9, 12, 4, 8],
    "revenu": [350_000, 210_000, 95_000, 520_000, 130_000],
})
df

# %%
df.info()

# %% [markdown]
# > 🆕 **pandas 3.0** : les colonnes de texte ont maintenant le type dédié **`str`** (au lieu de `object`).
# > C'est plus rapide (surtout avec `pyarrow` installé) et plus sûr : on ne peut plus mettre
# > accidentellement un entier dans une colonne de texte.

# %%
print(df.dtypes)
print(df.shape, len(df), list(df.columns))

# %% [markdown]
# ## 3. Lire un vrai fichier

# %%
menages = pd.read_csv(DATA / "enquete_menages.csv")
menages.head()

# %%
menages.describe()                      # résumé des colonnes numériques

# %%
menages.describe(include="str")         # résumé des colonnes texte (pandas 3 : "str")

# %% [markdown]
# ## 4. Sélectionner des colonnes

# %%
menages["region"].head(3)                       # une colonne -> Series
menages[["region", "revenu_mensuel_fcfa"]].head(3)  # liste de colonnes -> DataFrame

# %% [markdown]
# ## 5. Sélectionner des lignes : `loc` vs `iloc`
#
# | Accesseur | Sélectionne par… | Borne de fin | Exemple |
# |---|---|---|---|
# | `.loc[lignes, colonnes]` | **étiquettes** (noms, booléens) | **incluse** | `df.loc[0:4, "region"]` |
# | `.iloc[lignes, colonnes]` | **positions** entières | **exclue** | `df.iloc[0:4, 1]` |

# %%
print(menages.loc[0:2, ["region", "milieu"]])   # lignes d'étiquette 0, 1, 2
print(menages.iloc[0:2, [1, 2]])                # positions 0 et 1

# %% [markdown]
# ### Filtrer avec des conditions (masques booléens)

# %%
urbain_dakar = menages[(menages["region"] == "Dakar") & (menages["milieu"] == "Urbain")]
print(len(urbain_dakar))

sud = menages[menages["region"].isin(["Ziguinchor", "Kolda", "Sédhiou"])]
grands = menages.query("taille_menage >= 15 and taille_menage != 99")   # syntaxe alternative lisible
print(len(sud), len(grands))

# %% [markdown]
# ## 6. Créer et modifier des colonnes

# %%
menages["epargne_fcfa"] = menages["revenu_mensuel_fcfa"] - menages["depenses_mensuelles_fcfa"]
menages["revenu_par_tete"] = menages["revenu_mensuel_fcfa"] / menages["taille_menage"]
menages["est_urbain"] = menages["milieu"] == "Urbain"
menages[["revenu_mensuel_fcfa", "depenses_mensuelles_fcfa", "epargne_fcfa", "revenu_par_tete"]].head()

# %% [markdown]
# `assign` permet d'enchaîner les transformations (style « *method chaining* ») sans modifier l'original.
#
# > 🆕 **pandas 3.0** introduit `pd.col("nom")` pour faire référence à une colonne dans `assign`
# > sans écrire de `lambda`.

# %%
resume = (
    menages
    .assign(revenu_k=pd.col("revenu_mensuel_fcfa") / 1000,
            part_depenses=pd.col("depenses_mensuelles_fcfa") / pd.col("revenu_mensuel_fcfa"))
    .loc[:, ["region", "revenu_k", "part_depenses"]]
    .head()
)
resume

# %% [markdown]
# Équivalent compatible avec pandas 2 : `.assign(revenu_k=lambda d: d["revenu_mensuel_fcfa"] / 1000)`.
#
# ### Modifier des valeurs sous condition : toujours avec `.loc`

# %%
menages.loc[menages["taille_menage"] == 99, "taille_menage"] = np.nan   # 99 = code « ne sait pas »
print(menages["taille_menage"].max())

# %% [markdown]
# > 🆕 **pandas 3.0 — Copy-on-Write (CoW)** : toute sélection se comporte comme une **copie**.
# > L'« affectation en chaîne » ne modifie **jamais** le DataFrame d'origine :
# >
# > ```python
# > df["col"][df["x"] > 5] = 0          # ❌ n'a aucun effet en pandas 3 (avertissement)
# > df.loc[df["x"] > 5, "col"] = 0      # ✅ la bonne façon
# > ```
# >
# > Le célèbre `SettingWithCopyWarning` disparaît : la règle est désormais simple et prévisible.

# %%
sous_df = menages[menages["region"] == "Dakar"]
sous_df.loc[:, "revenu_mensuel_fcfa"] = 0          # modifie la copie seulement…
print(menages.loc[menages["region"] == "Dakar", "revenu_mensuel_fcfa"].head(3))  # …l'original est intact

# %% [markdown]
# ## 7. Trier, compter, valeurs uniques

# %%
menages.sort_values("revenu_mensuel_fcfa", ascending=False).head(3)[["region", "revenu_mensuel_fcfa"]]

# %%
print(menages["region"].value_counts())
print(menages["education_chef"].value_counts(normalize=True).round(3))   # proportions
print(menages["region"].nunique(), menages["region"].unique()[:4])

# %% [markdown]
# ## 8. Statistiques par colonne et par groupe (aperçu)

# %%
print(menages["revenu_mensuel_fcfa"].agg(["mean", "median", "std", "min", "max"]).round(0))

# %%
menages.groupby("milieu")["revenu_mensuel_fcfa"].median()

# %% [markdown]
# Le chapitre 5 détaille `groupby`, les jointures et les tableaux croisés.
#
# ## 9. Types de données (dtypes) à connaître
#
# | dtype pandas | Usage | Valeur manquante |
# |---|---|---|
# | `int64` | entiers (sans manquant) | — |
# | `Int64` (majuscule) | entiers **nullables** | `<NA>` |
# | `float64` | réels | `NaN` |
# | `bool` / `boolean` | vrai/faux / nullable | `<NA>` |
# | `str` (pandas 3) | texte | `NaN` |
# | `category` | peu de modalités répétées (économise la mémoire, ordonnable) | `NaN` |
# | `datetime64[us]` | dates/heures (pandas 3 : résolution micro-seconde par défaut) | `NaT` |

# %%
menages["education_chef"] = pd.Categorical(
    menages["education_chef"],
    categories=["Aucun", "Coranique", "Primaire", "Secondaire", "Supérieur"],
    ordered=True,
)
print(menages["education_chef"].dtype)
print((menages["education_chef"] >= "Secondaire").mean().round(3), "des chefs ont au moins le secondaire")
print(f"Mémoire : {menages.memory_usage(deep=True).sum() / 1e6:.2f} Mo")

# %% [markdown]
# ## À retenir
#
# - `Series` = colonne indexée ; `DataFrame` = table de colonnes partageant un index.
# - **`loc` = étiquettes (fin incluse)**, **`iloc` = positions (fin exclue)**.
# - Filtrer : `df[(cond1) & (cond2)]`, `isin`, `query`.
# - Modifier sous condition : **toujours** `df.loc[masque, "col"] = valeur`.
# - pandas 3.0 : type `str` par défaut, **Copy-on-Write** systématique, `pd.col()` dans `assign`.
#
# **Références** : McKinney W. (2022), *Python for Data Analysis*, 3ᵉ éd., O'Reilly (gratuit en ligne :
# <https://wesmckinney.com/book/>) ; notes de version pandas 3.0 :
# <https://pandas.pydata.org/docs/whatsnew/v3.0.0.html>.
#
# ➡️ **Chapitre suivant : importer, exporter et auditer la qualité des données.**
