# %% [markdown]
# # Chapitre 3 — Importer, exporter et auditer la qualité des données
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# ## Objectifs
#
# - Lire correctement des fichiers **CSV, Excel, JSON, Parquet** (séparateurs, décimales, encodages, codes manquants).
# - Choisir un **format d'export** adapté.
# - Réaliser un **audit de qualité** systématique avant toute analyse (profilage).

# %%
import io
from pathlib import Path

import numpy as np
import pandas as pd

DATA = next(p for p in [Path("data/brutes"), Path("../data/brutes"), Path("../../data/brutes")] if p.exists())
TMP = Path("_tmp"); TMP.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Le CSV : simple en apparence, piégeux en pratique
#
# Un fichier exporté d'Excel en français utilise souvent `;` comme séparateur et `,` comme décimale.
# Voici un extrait typique (simulé dans une chaîne pour l'exemple) :

# %%
brut = """id;region;revenu;date_enquete;taille
M1;Dakar;350 000,50;15/03/2025;6
M2;Thiès;NSP;16/03/2025;9
M3;Kaolack;95 000;17/03/2025;-
M4;Saint-Louis;;18/03/2025;99
"""
mauvais = pd.read_csv(io.StringIO(brut))
print(mauvais.shape)          # une seule colonne : le séparateur n'est pas reconnu
mauvais.head(2)

# %%
bon = pd.read_csv(
    io.StringIO(brut),
    sep=";",                       # séparateur de colonnes
    decimal=",",                   # virgule décimale
    thousands=" ",                 # espace comme séparateur de milliers
    na_values=["NSP", "-", 99],    # codes à interpréter comme manquants
    parse_dates=["date_enquete"],
    dayfirst=True,                 # 15/03 = 15 mars
)
print(bon.dtypes)
bon

# %% [markdown]
# ⚠️ `na_values=[99]` s'applique à **toutes** les colonnes. Pour cibler une colonne :
# `na_values={"taille": [99]}`.
#
# ### Paramètres de `read_csv` les plus utiles
#
# | Paramètre | Rôle |
# |---|---|
# | `sep` | séparateur (`","`, `";"`, `"\t"`) |
# | `decimal`, `thousands` | format des nombres |
# | `encoding` | `"utf-8"` (défaut), `"latin-1"` / `"cp1252"` pour les vieux fichiers Windows |
# | `na_values`, `keep_default_na` | codes de valeurs manquantes |
# | `dtype` | forcer un type (ex. `{"code_postal": str}` pour garder les zéros initiaux) |
# | `usecols` | ne lire que certaines colonnes (gain mémoire) |
# | `parse_dates`, `dayfirst`, `date_format` | dates |
# | `skiprows`, `nrows`, `header` | lignes d'en-tête parasites, lecture partielle |
# | `chunksize` | lire un très gros fichier par morceaux |
#
# ### Encodages
# Un fichier mal décodé affiche `ThiÃ¨s` au lieu de `Thiès` (UTF-8 lu comme Latin-1), ou lève une
# `UnicodeDecodeError`. Solution : préciser `encoding=`.

# %%
(TMP / "latin1.csv").write_bytes("region,valeur\nThiès,1\nSédhiou,2\n".encode("latin-1"))
try:
    pd.read_csv(TMP / "latin1.csv")
except UnicodeDecodeError as e:
    print("Erreur :", e.reason)
pd.read_csv(TMP / "latin1.csv", encoding="latin-1")

# %% [markdown]
# ### Gros fichiers : lire par morceaux

# %%
total = 0.0
for morceau in pd.read_csv(DATA / "ventes_boutiques.csv", chunksize=1000, usecols=["quantite"]):
    total += morceau["quantite"].clip(lower=0).sum()
print(f"Quantité totale (lue par blocs de 1000 lignes) : {total:,.0f}")

# %% [markdown]
# ## 2. Excel, JSON, Parquet

# %%
menages = pd.read_csv(DATA / "enquete_menages.csv")

# Excel (nécessite openpyxl) — plusieurs feuilles possibles
with pd.ExcelWriter(TMP / "menages.xlsx") as w:
    menages.head(100).to_excel(w, sheet_name="echantillon", index=False)
    menages.groupby("region").size().rename("n").to_excel(w, sheet_name="effectifs")
feuilles = pd.read_excel(TMP / "menages.xlsx", sheet_name=None)   # None = toutes les feuilles -> dict
print(list(feuilles))

# JSON : courant pour les API web
menages.head(3).to_json(TMP / "menages.json", orient="records", force_ascii=False, indent=2)
print((TMP / "menages.json").read_text(encoding="utf-8")[:300])

# %% [markdown]
# Les API renvoient souvent du JSON **imbriqué** : `pd.json_normalize` l'aplatit.

# %%
reponse_api = [
    {"id": 1, "station": {"nom": "Dakar-Yoff", "coord": {"lat": 14.74, "lon": -17.49}}, "mesures": {"temp": 27.1}},
    {"id": 2, "station": {"nom": "Ziguinchor", "coord": {"lat": 12.55, "lon": -16.27}}, "mesures": {"temp": 29.4}},
]
pd.json_normalize(reponse_api, sep="_")

# %% [markdown]
# ### Parquet : le format recommandé pour stocker des données préparées
#
# Format **colonnaire, compressé et typé** (nécessite `pyarrow`). Les types (dates, catégories…) sont
# conservés, contrairement au CSV.

# %%
menages.to_csv(TMP / "menages.csv", index=False)
menages.to_parquet(TMP / "menages.parquet", index=False)
for ext in ["csv", "parquet"]:
    print(f"{ext:<8} {(TMP / f'menages.{ext}').stat().st_size / 1024:8.1f} Ko")

# %% [markdown]
# | Format | Lisible humain | Types conservés | Taille | Usage |
# |---|---|---|---|---|
# | CSV | ✅ | ❌ | grande | échange universel |
# | Excel | ✅ | partiel | moyenne | partage avec non-programmeurs |
# | JSON | ✅ | partiel | grande | API, données imbriquées |
# | Parquet | ❌ | ✅ | **petite** | stockage analytique, gros volumes |
#
# > **Bonne pratique** : ne **jamais** écraser les données brutes. Organiser
# > `data/brutes/` (lecture seule) → script de nettoyage → `data/propres/`.

# %% [markdown]
# ## 3. Audit de qualité (profilage)
#
# Avant toute analyse, on répond systématiquement à ces questions :
#
# 1. **Dimensions** et **types** : chaque colonne a-t-elle le type attendu ?
# 2. **Manquants** : combien, où, selon quel schéma ?
# 3. **Doublons** : lignes ou identifiants répétés ?
# 4. **Domaines de valeurs** : modalités inattendues, valeurs impossibles ?
# 5. **Distributions** : valeurs extrêmes, asymétrie ?

# %%
ventes = pd.read_csv(DATA / "ventes_boutiques.csv")
ventes.info()

# %% [markdown]
# Constat immédiat : `prix_unitaire` et `date` sont du **texte** alors qu'on attend un nombre et une date.
#
# Écrivons une **fonction d'audit réutilisable** (elle est aussi disponible dans `src/atd/qualite.py`) :

# %%
def profil(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau de profilage : une ligne par colonne."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_manquants": df.isna().sum(),
        "pct_manquants": (df.isna().mean() * 100).round(2),
        "n_uniques": df.nunique(),
        "exemple": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    }).sort_values("pct_manquants", ascending=False)


profil(ventes)

# %%
print("Lignes entièrement dupliquées :", ventes.duplicated().sum())
print("Identifiants dupliqués        :", ventes["id_transaction"].duplicated().sum())

# %%
print(ventes["region"].value_counts().to_string())

# %% [markdown]
# On voit 9 « vraies » régions écrites de **plus de 30 façons** (casse, espaces, abréviation `St-Louis`).

# %%
print(ventes["quantite"].describe())
print("Quantités ≤ 0 :", (ventes["quantite"] <= 0).sum(), "| > 50 :", (ventes["quantite"] > 50).sum())

# %%
print("Exemples de formats de date :")
print(ventes["date"].str.replace(r"\d", "9", regex=True).value_counts())

# %% [markdown]
# ### Règles de validation explicites
#
# Formaliser les attentes sous forme de **règles testables** est une excellente pratique
# (c'est l'idée des bibliothèques comme *pandera* ou *Great Expectations*).

# %%
regles = {
    "id unique": ventes["id_transaction"].is_unique,
    "quantité > 0": (ventes["quantite"].dropna() > 0).all(),
    "quantité ≤ 50": (ventes["quantite"].dropna() <= 50).all(),
    "région dans le référentiel": ventes["region"].isin(pd.read_csv(DATA / "regions_senegal.csv")["region"]).all(),
    "prix numérique": pd.api.types.is_numeric_dtype(ventes["prix_unitaire"]),
}
pd.Series(regles, name="respectée ?")

# %% [markdown]
# **Toutes les règles échouent** : le chapitre 4 corrige chacun de ces problèmes.
#
# ## À retenir
#
# - `read_csv` a des paramètres pour presque tous les pièges : `sep`, `decimal`, `thousands`, `encoding`,
#   `na_values`, `dtype`, `parse_dates`.
# - Stocker les données préparées en **Parquet** ; garder les données brutes **intactes**.
# - Toujours **profiler** : types, manquants, doublons, modalités, distributions — et écrire des **règles de validation**.
#
# ➡️ **Chapitre suivant : nettoyage des données.**

# %%
import shutil
shutil.rmtree(TMP, ignore_errors=True)   # ménage des fichiers temporaires
