# %% [markdown]
# # Chapitre 8 — Préparer les données pour la modélisation (feature engineering)
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# ## Objectifs
#
# - **Encoder** les variables catégorielles (one-hot, ordinal).
# - **Mettre à l'échelle** les variables numériques (standardisation, min-max, robuste) et **transformer** (log, Box-Cox/Yeo-Johnson).
# - **Créer** des variables utiles (ratios, dates, discrétisation, interactions).
# - Éviter la **fuite de données** grâce aux `Pipeline` et `ColumnTransformer` de scikit-learn.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler,
                                   StandardScaler)

DATA = next(p for p in [Path("data/brutes"), Path("../data/brutes"), Path("../../data/brutes")] if p.exists())
menages = pd.read_csv(DATA / "enquete_menages.csv")
menages.loc[menages["taille_menage"] == 99, "taille_menage"] = np.nan

# %% [markdown]
# **Tâche fil rouge** : prédire si un ménage **utilise le mobile money** (`Oui`/`Non`) à partir de ses caractéristiques.
#
# ## 1. Pourquoi préparer ?
#
# Les algorithmes ne comprennent que des **nombres**, sont souvent sensibles à l'**échelle** des variables
# (distances, régularisation, descente de gradient) et ne tolèrent pas les **NaN** (pour la plupart).

# %% [markdown]
# ## 2. Encoder les variables catégorielles
#
# | Méthode | Pour | Exemple |
# |---|---|---|
# | **One-hot** (indicatrices) | nominales | région → `region_Dakar`, `region_Thiès`, … |
# | **Ordinal** (entiers ordonnés) | ordinales | éducation : Aucun=0 < … < Supérieur=4 |
# | Target / fréquence | très nombreuses modalités | à manier avec précaution (fuite) |

# %%
exemple = menages[["region", "education_chef"]].head(5)
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(transform="pandas")
print(ohe.fit_transform(exemple[["region"]]))

ordre = [["Aucun", "Coranique", "Primaire", "Secondaire", "Supérieur"]]
ordinal = OrdinalEncoder(categories=ordre).set_output(transform="pandas")
print(pd.concat([exemple["education_chef"], ordinal.fit_transform(exemple[["education_chef"]])], axis=1))

# %% [markdown]
# > ⚠️ Ne jamais encoder une variable **nominale** en 1, 2, 3… : le modèle croirait que « Thiès (2) » est entre
# > « Dakar (1) » et « Kaolack (3) ».
# > `handle_unknown="ignore"` évite une erreur si une modalité nouvelle apparaît en production.
#
# ## 3. Mise à l'échelle et transformations

# %%
x = menages[["revenu_mensuel_fcfa"]].dropna()
transfos = {
    "brut": x.values.ravel(),
    "StandardScaler (z-score)": StandardScaler().fit_transform(x).ravel(),
    "MinMaxScaler [0,1]": MinMaxScaler().fit_transform(x).ravel(),
    "RobustScaler (médiane/IQR)": RobustScaler().fit_transform(x).ravel(),
    "log": np.log(x.values.ravel()),
    "Yeo-Johnson": PowerTransformer(method="yeo-johnson").fit_transform(x).ravel(),
}
fig, axes = plt.subplots(2, 3, figsize=(13, 5))
for ax, (nom, v) in zip(axes.ravel(), transfos.items()):
    ax.hist(v, bins=50); ax.set_title(nom, fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# - Les *scalers* **changent l'échelle, pas la forme** de la distribution.
# - `log` / `PowerTransformer` **changent la forme** (réduisent l'asymétrie).
# - `RobustScaler` est peu sensible aux valeurs extrêmes.
# - Les modèles à base d'**arbres** (forêts, boosting) n'ont pas besoin de mise à l'échelle.
#
# ## 4. Créer des variables (feature engineering)
#
# C'est souvent **là** que se joue la performance : traduire la connaissance métier en variables.

# %%
fe = menages.assign(
    revenu_par_personne=lambda d: d["revenu_mensuel_fcfa"] / d["taille_menage"],          # ratio
    taux_depense=lambda d: d["depenses_mensuelles_fcfa"] / d["revenu_mensuel_fcfa"],      # ratio
    log_depenses=lambda d: np.log(d["depenses_mensuelles_fcfa"]),                         # transformation
    tranche_age=lambda d: pd.cut(d["age_chef"], [0, 30, 45, 60, 120],
                                 labels=["<30", "30-44", "45-59", "60+"]),                # discrétisation
    urbain_x_internet=lambda d: ((d["milieu"] == "Urbain") & (d["acces_internet"] == "Oui")).astype(int),  # interaction
    revenu_manquant=lambda d: d["revenu_mensuel_fcfa"].isna().astype(int),                # indicateur de manquant
)
fe[["revenu_par_personne", "taux_depense", "tranche_age", "urbain_x_internet", "revenu_manquant"]].head()

# %% [markdown]
# Pour des **dates** : année, mois, jour de la semaine, jour férié, saison (hivernage), temps écoulé depuis un événement ;
# pour des variables **cycliques** (mois, heure) : encodage $\sin(2\pi m/12)$, $\cos(2\pi m/12)$ pour que décembre soit proche de janvier.
#
# ## 5. La fuite de données (*data leakage*) — l'erreur n°1
#
# **Fuite** = utiliser, pendant l'entraînement, une information qui ne sera pas disponible au moment de la prédiction
# → performance **sur-estimée**, déception en production.
#
# Formes classiques :
# 1. Ajuster un *scaler* / imputeur / encodeur **sur toutes les données** avant de séparer train/test.
# 2. Variable qui **contient la cible** (ex. « a reçu un bonus mobile money » pour prédire « utilise le mobile money »).
# 3. Information **future** dans une série temporelle (ne pas mélanger aléatoirement des données temporelles).
# 4. Doublons présents à la fois dans le train et le test.
#
# Démonstration spectaculaire : sélectionner des variables **avant** la validation croisée sur des données **purement aléatoires**.

# %%
from sklearn.feature_selection import SelectKBest, f_classif

rng = np.random.default_rng(0)
X_bruit = rng.normal(size=(100, 5_000))        # 5000 variables de bruit
y_bruit = rng.integers(0, 2, 100)              # cible aléatoire : AUCUN signal possible

# ❌ Mauvais : sélection sur tout le jeu, puis validation croisée
X_sel = SelectKBest(f_classif, k=20).fit_transform(X_bruit, y_bruit)
mauvais = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_bruit, cv=5).mean()
# ✅ Bon : sélection DANS le pipeline, refaite sur chaque pli d'entraînement
bon = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)),
                      X_bruit, y_bruit, cv=5).mean()
print(f"Exactitude avec fuite : {mauvais:.2f}   |   sans fuite : {bon:.2f}   (hasard = 0.50)")

# %% [markdown]
# ## 6. La solution : `Pipeline` + `ColumnTransformer`
#
# Chaque étape est **ajustée sur l'entraînement seulement** (`fit`), puis **appliquée** (`transform`) au test :
#
# ```text
#               ┌─ numériques  : imputation médiane → log/Yeo-Johnson → standardisation ─┐
#  X (brut) ───►├─ ordinales   : imputation mode    → encodage ordinal                   ├──► modèle
#               └─ nominales   : imputation mode    → one-hot                            ┘
# ```

# %%
cible = (menages["utilise_mobile_money"] == "Oui").astype(int)
X = menages.drop(columns=["id_menage", "utilise_mobile_money"])

num_cols = ["taille_menage", "age_chef", "revenu_mensuel_fcfa", "depenses_mensuelles_fcfa"]
ord_cols = ["education_chef"]
nom_cols = ["region", "milieu", "sexe_chef", "acces_electricite", "acces_internet"]

preparation = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                      ("forme", PowerTransformer(method="yeo-johnson"))]), num_cols),   # PowerTransformer standardise aussi
    ("ord", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("enc", OrdinalEncoder(categories=ordre)),
                      ("std", StandardScaler())]), ord_cols),
    ("nom", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]), nom_cols),
], verbose_feature_names_out=False)

modele = Pipeline([("prep", preparation), ("clf", LogisticRegression(max_iter=1000))])

X_train, X_test, y_train, y_test = train_test_split(X, cible, test_size=0.25, stratify=cible, random_state=42)
modele.fit(X_train, y_train)
print(f"Exactitude test : {modele.score(X_test, y_test):.3f}  (classe majoritaire : {max(y_test.mean(), 1 - y_test.mean()):.3f})")

# %%
modele  # affichage interactif du pipeline dans Jupyter

# %%
noms = modele.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(modele.named_steps["clf"].coef_[0], index=noms).sort_values()
ax = coefs.plot(kind="barh", figsize=(7, 6), color=np.where(coefs > 0, "tab:green", "tab:red"))
ax.set(title="Coefficients (variables standardisées) — effet sur le log-odds", xlabel="coefficient")
plt.tight_layout(); plt.show()

# %% [markdown]
# Avantages du pipeline :
# - **pas de fuite** (tout est appris sur le train) ;
# - **un seul objet** à sauvegarder (`joblib.dump(modele, "modele.joblib")`) et à appliquer à de nouvelles données brutes ;
# - compatible avec la **validation croisée** et la recherche d'hyperparamètres (`GridSearchCV`).

# %%
scores = cross_val_score(modele, X, cible, cv=5, scoring="roc_auc")
print(f"AUC ROC en validation croisée 5 plis : {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# ## À retenir
#
# - One-hot pour le **nominal**, ordinal pour l'**ordinal** ; `handle_unknown="ignore"`.
# - Scalers = échelle ; log/Yeo-Johnson = forme ; arbres = pas besoin.
# - Le feature engineering traduit le **savoir métier** en variables.
# - **Toute étape apprise sur les données va dans le `Pipeline`** → pas de fuite.
#
# **Références** : Kuhn M. & Johnson K. (2019), *Feature Engineering and Selection* (en ligne : <https://www.feat.engineering/>) ;
# Kaufman S. et al. (2012), « Leakage in Data Mining », *ACM TKDD* 6(4) ; guide utilisateur scikit-learn,
# « Common pitfalls » : <https://scikit-learn.org/stable/common_pitfalls.html>.
#
# ➡️ **Chapitre suivant : réduction de dimension, clustering et introduction à la modélisation.**
