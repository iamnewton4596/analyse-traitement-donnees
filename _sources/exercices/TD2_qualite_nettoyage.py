# %% [markdown]
# # TD 2 — Qualité et nettoyage des données (chapitres 3–4)
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# Jeu principal : `meteo_dakar.csv` (station météo simulée, 2020–2024) et `enquete_menages.csv`.

# %%
import io
from pathlib import Path

import numpy as np
import pandas as pd

DATA = next(p for p in [Path("data/brutes"), Path("../data/brutes"), Path("../../data/brutes")] if p.exists())

# %% [markdown]
# ## Exercice 1 ⭐ — Lire un CSV « à la française »
# Lisez correctement la chaîne ci-dessous : séparateur `;`, virgule décimale, `ND` et `-` = manquant, dates jour/mois/année.
# Vérifiez que `pluie_mm` est numérique et `date` de type date.

# %%
brut = """date;station;pluie_mm;temp_c
01/08/2024;Dakar-Yoff;12,5;27,1
02/08/2024;Dakar-Yoff;ND;26,8
03/08/2024;Dakar-Yoff;0;-
04/08/2024;Dakar-Yoff;31,2;25,9
"""

# %% tags=["solution"]
df = pd.read_csv(io.StringIO(brut), sep=";", decimal=",", na_values=["ND", "-"], parse_dates=["date"], dayfirst=True)
print(df.dtypes); df

# %% [markdown]
# ## Exercice 2 ⭐⭐ — Audit de la météo
# Chargez `meteo_dakar.csv`. Produisez :
# 1. le nombre et le pourcentage de manquants par colonne ;
# 2. le nombre de **jours absents** du fichier entre la première et la dernière date ;
# 3. le nombre de températures **physiquement impossibles** (< 0 °C ou > 50 °C à Dakar).

# %% tags=["solution"]
meteo = pd.read_csv(DATA / "meteo_dakar.csv", parse_dates=["date"])
print(pd.DataFrame({"n": meteo.isna().sum(), "%": (meteo.isna().mean() * 100).round(2)}))
attendu = pd.date_range(meteo["date"].min(), meteo["date"].max(), freq="D")
print("jours absents :", len(attendu.difference(meteo["date"])))
print("températures impossibles :", ((meteo["temp_moy_c"] < 0) | (meteo["temp_moy_c"] > 50)).sum())

# %% [markdown]
# ## Exercice 3 ⭐⭐ — Nettoyer la météo
# Écrivez une fonction `nettoyer_meteo(df)` qui :
# 1. remplace les températures impossibles par `NaN` ;
# 2. réindexe sur **tous** les jours (`asfreq("D")`) ;
# 3. interpole la température et l'humidité (méthode `"time"`, au plus 5 jours consécutifs) ;
# 4. ajoute une colonne booléenne `temp_imputee`.
# Combien de valeurs de température restent manquantes après nettoyage ?

# %% tags=["solution"]
def nettoyer_meteo(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().set_index("date").sort_index()
    d.loc[(d["temp_moy_c"] < 0) | (d["temp_moy_c"] > 50), "temp_moy_c"] = np.nan
    d = d.asfreq("D")
    d["temp_imputee"] = d["temp_moy_c"].isna()
    for col in ["temp_moy_c", "humidite_pct"]:
        d[col] = d[col].interpolate(method="time", limit=5)
    return d

propre = nettoyer_meteo(meteo)
print("restant manquant :", propre["temp_moy_c"].isna().sum(), "| imputées :", propre["temp_imputee"].sum())

# %% [markdown]
# ## Exercice 4 ⭐⭐ — Mécanisme de non-réponse
# Dans `enquete_menages.csv`, le revenu manque pour certains ménages.
# 1. Calculez le taux de non-réponse par **région**, par **accès internet** et par **sexe du chef**.
# 2. La non-réponse vous semble-t-elle MCAR ? Argumentez en 3 lignes.

# %% tags=["solution"]
m = pd.read_csv(DATA / "enquete_menages.csv")
m["nr"] = m["revenu_mensuel_fcfa"].isna()
for col in ["region", "acces_internet", "sexe_chef"]:
    print(m.groupby(col)["nr"].mean().mul(100).round(1), "\n")
# Le taux varie fortement avec la région (Dakar, très urbaine, plus élevé) et l'accès internet (proxy du niveau d'éducation
# et du milieu urbain), mais pas avec le sexe du chef. La non-réponse dépend donc de variables observées :
# ce n'est pas MCAR ; l'hypothèse MAR (conditionnellement au milieu/éducation) est plausible, MNAR ne peut être exclu.

# %% [markdown]
# ## Exercice 5 ⭐⭐⭐ — Comparer des imputations
# Sur les ménages dont le revenu est connu, masquez **artificiellement** 20 % des revenus **au hasard (MCAR)**,
# puis comparez l'erreur absolue moyenne (sur les valeurs masquées) de :
# (a) la médiane globale, (b) la médiane par `milieu × education_chef`, (c) `KNNImputer` sur log(revenu), log(dépenses), taille.
# Recommencez avec un masquage **MAR** (35 % chez les urbains, 10 % chez les ruraux). Conclusion ?

# %% tags=["solution"]
from sklearn.impute import KNNImputer

def evaluer(masquer_p, graine=0):
    rng = np.random.default_rng(graine)
    c = m.dropna(subset=["revenu_mensuel_fcfa", "taille_menage"]).copy()
    c = c[c["taille_menage"] != 99]
    verite = c["revenu_mensuel_fcfa"].copy()
    masque = rng.random(len(c)) < masquer_p(c)
    c.loc[masque, "revenu_mensuel_fcfa"] = np.nan
    res = {}
    res["médiane globale"] = c["revenu_mensuel_fcfa"].fillna(c["revenu_mensuel_fcfa"].median())
    res["médiane par groupe"] = c["revenu_mensuel_fcfa"].fillna(
        c.groupby(["milieu", "education_chef"])["revenu_mensuel_fcfa"].transform("median"))
    X = np.column_stack([np.log(c["revenu_mensuel_fcfa"]), np.log(c["depenses_mensuelles_fcfa"]), c["taille_menage"]])
    res["KNN"] = pd.Series(np.exp(KNNImputer(n_neighbors=10).fit_transform(X)[:, 0]), index=c.index)
    return {k: np.abs(v[masque] - verite[masque]).mean().round(0) for k, v in res.items()}

print("MCAR :", evaluer(lambda c: 0.20))
print("MAR  :", evaluer(lambda c: np.where(c["milieu"] == "Urbain", 0.35, 0.10)))
# Les méthodes qui exploitent les autres variables (groupe, KNN avec les dépenses) sont nettement meilleures dans les deux
# cas ; KNN tire profit de la forte corrélation revenu–dépenses.

# %% [markdown]
# ## Exercice 6 ⭐⭐ — Valeurs aberrantes
# Sur la vitesse du vent (`vent_ms`) de la météo nettoyée :
# 1. comptez les valeurs signalées par la règle de Tukey (1,5 × IQR) et par le z-score robuste (|z| > 3,5) ;
# 2. affichez les 5 valeurs les plus élevées : erreurs ou phénomènes réels plausibles ? Que décidez-vous ?

# %% tags=["solution"]
v = propre["vent_ms"].dropna()
q1, q3 = v.quantile([.25, .75]); iqr = q3 - q1
tukey = (v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)
med = v.median(); mad = (v - med).abs().median()
robuste = ((v - med).abs() / (1.4826 * mad)) > 3.5
print("Tukey :", tukey.sum(), "| MAD :", robuste.sum())
print(v.nlargest(5))
# Des vents de 10–11 m/s (≈ 40 km/h) sont tout à fait plausibles à Dakar en saison sèche (alizés, harmattan) :
# ce sont des valeurs réelles extrêmes, à conserver (pas de correction).
