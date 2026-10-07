# %% [markdown]
# # Chapitre 5 — Transformer et restructurer les données
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# ## Objectifs
#
# - Agréger avec **`groupby`** (split–apply–combine) : `agg`, `transform`, `filter`.
# - Combiner des tables : **`merge`** (jointures) et **`concat`**.
# - Restructurer : **`pivot_table`**, **`melt`**, **`crosstab`** (format large ↔ long).
# - Travailler avec des **séries temporelles** : `resample`, `rolling`, décalages.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = next(p for p in [Path("data/brutes"), Path("../data/brutes"), Path("../../data/brutes")] if p.exists())
PROPRES = DATA.parent / "propres" / "ventes_propres.parquet"
if not PROPRES.exists():  # le chapitre 4 produit ce fichier ; sinon on le reconstruit
    import sys; sys.path.insert(0, str(DATA.parents[1] / "src"))
    from atd.nettoyage import nettoyer_ventes
    PROPRES.parent.mkdir(exist_ok=True)
    nettoyer_ventes(pd.read_csv(DATA / "ventes_boutiques.csv"),
                    pd.read_csv(DATA / "regions_senegal.csv")["region"])[0].to_parquet(PROPRES)
ventes = pd.read_parquet(PROPRES)
ventes.head(3)

# %% [markdown]
# ## 1. `groupby` : diviser — appliquer — combiner
#
# ```text
#   données ──split──► groupe Dakar ──apply(sum)──► 1 valeur ─┐
#                 └──► groupe Thiès ──apply(sum)──► 1 valeur ─┼─combine─► résultat
#                 └──► …                                       ┘
# ```

# %%
ca_region = ventes.groupby("region", observed=True)["montant"].sum().sort_values(ascending=False)
ca_region.map("{:,.0f}".format)

# %% [markdown]
# ### Plusieurs statistiques, plusieurs colonnes : `agg` avec agrégations nommées

# %%
synthese = ventes.groupby("region", observed=True).agg(
    n_transactions=("id_transaction", "count"),
    ca_total=("montant", "sum"),
    panier_moyen=("montant", "mean"),
    panier_median=("montant", "median"),
    n_boutiques=("boutique", "nunique"),
).sort_values("ca_total", ascending=False)
synthese.round(0)

# %% [markdown]
# ### `transform` : renvoie un résultat **de même taille** que l'entrée
# Idéal pour comparer chaque ligne à son groupe.

# %%
ventes["part_ca_region"] = ventes["montant"] / ventes.groupby("region", observed=True)["montant"].transform("sum")
ventes["ecart_prix_moyen_produit"] = ventes["prix_unitaire"] - ventes.groupby("produit")["prix_unitaire"].transform("mean")
ventes[["region", "produit", "montant", "part_ca_region", "ecart_prix_moyen_produit"]].head()

# %% [markdown]
# ### `filter` : garder des **groupes entiers** selon une condition

# %%
gros_produits = ventes.groupby("produit").filter(lambda g: g["montant"].sum() > 10_000_000)
print(gros_produits["produit"].unique())

# %% [markdown]
# ### Regrouper sur plusieurs clés

# %%
ventes.groupby(["region", "mode_paiement"], observed=True)["montant"].sum().unstack().round(-3).head()

# %% [markdown]
# ## 2. Tableaux croisés : `pivot_table` et `crosstab`

# %%
pd.pivot_table(ventes, values="montant", index="categorie", columns="mode_paiement",
               aggfunc="sum", margins=True, margins_name="Total", observed=True).round(-3)

# %%
# Répartition (en %) des modes de paiement par région — chaque ligne somme à 100
pd.crosstab(ventes["region"], ventes["mode_paiement"], normalize="index").mul(100).round(1)

# %% [markdown]
# ## 3. Format large ↔ format long
#
# - **Long (tidy)** : une ligne par (unité, variable, valeur) — requis par `seaborn`, `groupby`, les modèles.
# - **Large** : une colonne par modalité — pratique pour la lecture humaine et les tableaux.

# %%
large = (ventes.assign(mois=ventes["date"].dt.month)
         .pivot_table(index="region", columns="mois", values="montant", aggfunc="sum", observed=True)
         .round(-3))
large.iloc[:3, :6]

# %%
long = large.reset_index().melt(id_vars="region", var_name="mois", value_name="ca")
long.head()

# %% [markdown]
# `pivot` (sans agrégation) est l'inverse exact de `melt` ; `pivot_table` agrège si plusieurs lignes tombent dans la même case.
#
# ## 4. Combiner des tables
#
# ### 4.1 Jointures avec `merge`
#
# | `how=` | Garde… | Analogue SQL |
# |---|---|---|
# | `"inner"` | les clés présentes **dans les deux** tables | `INNER JOIN` |
# | `"left"` | **toutes** les lignes de gauche | `LEFT JOIN` |
# | `"right"` | toutes les lignes de droite | `RIGHT JOIN` |
# | `"outer"` | toutes les clés des deux | `FULL OUTER JOIN` |

# %%
regions = pd.read_csv(DATA / "regions_senegal.csv")
par_region = synthese.reset_index()
par_region["region"] = par_region["region"].astype(str)

jointure = par_region.merge(regions, on="region", how="left", validate="one_to_one", indicator=True)
print(jointure["_merge"].value_counts())
jointure[["region", "zone", "superficie_km2", "ca_total"]].head()

# %% [markdown]
# > ✅ **Bonnes pratiques de jointure**
# > - `validate="one_to_one"` / `"many_to_one"` : lève une erreur si la clé n'est pas unique là où elle doit l'être
# >   (évite la **multiplication silencieuse des lignes**).
# > - `indicator=True` : colonne `_merge` qui dit d'où vient chaque ligne → repérer les clés orphelines.
# > - Vérifier le **nombre de lignes** avant/après.

# %%
orphelines = regions.merge(par_region, on="region", how="left", indicator=True).query("_merge == 'left_only'")
print("Régions sans aucune vente dans nos données :", orphelines["region"].tolist())

# %% [markdown]
# ### 4.2 Empiler avec `concat`

# %%
t1 = ventes[ventes["date"].dt.quarter == 1]
t2 = ventes[ventes["date"].dt.quarter == 2]
s1 = pd.concat([t1, t2], ignore_index=True)                       # l'un sous l'autre (mêmes colonnes)
print(len(t1), "+", len(t2), "=", len(s1))
cote_a_cote = pd.concat([ca_region.rename("CA"), synthese["n_boutiques"]], axis=1)   # côte à côte (alignement sur l'index)
cote_a_cote.head(3)

# %% [markdown]
# ## 5. Séries temporelles
#
# ### 5.1 Ré-échantillonner : `resample`

# %%
serie = ventes.set_index("date")["montant"]
mensuel = serie.resample("ME").sum()        # "ME" = fin de mois ; "W" = semaine ; "QE" = trimestre
hebdo = serie.resample("W").sum()
mensuel.map("{:,.0f}".format).head()

# %% [markdown]
# ### 5.2 Fenêtres glissantes : `rolling` (lissage)

# %%
quotidien = serie.resample("D").sum()
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(quotidien, color="lightgray", lw=0.8, label="quotidien")
ax.plot(quotidien.rolling(7, center=True).mean(), lw=1.5, label="moyenne mobile 7 j")
ax.plot(quotidien.rolling(30, center=True).mean(), lw=2.5, label="moyenne mobile 30 j")
ax.set_ylabel("CA (FCFA)"); ax.legend(); ax.set_title("Chiffre d'affaires journalier — lissages")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 5.3 Décalages et variations : `shift`, `diff`, `pct_change`

# %%
m = mensuel.to_frame("ca")
m["ca_mois_prec"] = m["ca"].shift(1)
m["variation"] = m["ca"].diff()
m["variation_pct"] = m["ca"].pct_change().mul(100).round(1)
m["ca_cumule"] = m["ca"].cumsum()
m.head(4)

# %% [markdown]
# ### 5.4 Saisonnalité : la météo de Dakar

# %%
meteo = pd.read_csv(DATA / "meteo_dakar.csv", parse_dates=["date"]).set_index("date").asfreq("D")
meteo.loc[meteo["temp_moy_c"] < -50, "temp_moy_c"] = np.nan
meteo["mois"] = meteo.index.month
profil_mensuel = meteo.groupby("mois").agg(temp=("temp_moy_c", "mean"), pluie=("precip_mm", "sum"))
profil_mensuel["pluie"] /= meteo.index.year.nunique()     # cumul moyen par an
profil_mensuel.round(1).T

# %%
fig, ax1 = plt.subplots(figsize=(8, 3.5))
ax1.bar(profil_mensuel.index, profil_mensuel["pluie"], color="tab:blue", alpha=0.6)
ax1.set_ylabel("pluie moyenne (mm/mois)", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(profil_mensuel.index, profil_mensuel["temp"], color="tab:red", marker="o")
ax2.set_ylabel("température moyenne (°C)", color="tab:red")
ax1.set_xticks(range(1, 13)); ax1.set_xlabel("mois")
ax1.set_title("Climatogramme simulé de Dakar (2020–2024) : hivernage juillet–octobre")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Fonctions personnalisées : `apply`, `map` — et pourquoi les éviter quand c'est possible
#
# `apply` exécute une fonction Python **ligne par ligne** : flexible mais lent.
# Préférer les opérations **vectorisées** (`np.where`, `np.select`, `pd.cut`, méthodes `.str`, `.dt`).

# %%
conditions = [ventes["montant"] < 2_000, ventes["montant"] < 20_000]
ventes["taille_panier"] = np.select(conditions, ["petit", "moyen"], default="grand")
ventes["tranche"] = pd.cut(ventes["montant"], bins=[0, 2_000, 20_000, np.inf],
                           labels=["< 2k", "2k–20k", "> 20k"], right=False)
ventes[["montant", "taille_panier", "tranche"]].sample(5, random_state=0)

# %% [markdown]
# ## À retenir
#
# | Besoin | Outil |
# |---|---|
# | Résumer par groupe | `groupby(...).agg(nom=(col, fonction))` |
# | Comparer une ligne à son groupe | `groupby(...).transform(...)` |
# | Tableau croisé | `pivot_table`, `crosstab(normalize=...)` |
# | Large → long / long → large | `melt` / `pivot` |
# | Joindre | `merge(how=..., validate=..., indicator=True)` |
# | Empiler | `concat` |
# | Séries temporelles | `resample`, `rolling`, `shift`, `diff`, `pct_change` |
#
# ➡️ **Chapitre suivant : analyse exploratoire et visualisation.**
