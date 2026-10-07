# %% [markdown]
# # Chapitre 7 — Statistique inférentielle
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# ## Objectifs
#
# - Passer de l'**échantillon** à la **population** : estimation, intervalle de confiance, test.
# - Choisir le **test adapté** (t de Student, Mann–Whitney, ANOVA, Kruskal–Wallis, χ²).
# - Interpréter correctement une **p-valeur** et une **taille d'effet**.
# - Utiliser le **bootstrap** et corriger les **tests multiples**.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATA = next(p for p in [Path("data/brutes"), Path("../data/brutes"), Path("../../data/brutes")] if p.exists())
sns.set_theme(style="whitegrid", palette="colorblind")
etu = pd.read_csv(DATA / "resultats_etudiants.csv")
rng = np.random.default_rng(7)
etu.head(3)

# %% [markdown]
# ## 1. Échantillon, population, distribution d'échantillonnage
#
# On observe un **échantillon** ; on veut parler de la **population**. Une statistique (ex. la moyenne $\bar x$)
# varie d'un échantillon à l'autre : sa loi est la **distribution d'échantillonnage**.
#
# **Théorème central limite (TCL)** : pour $n$ assez grand, $\bar X \approx \mathcal N\!\left(\mu, \sigma^2/n\right)$,
# **quelle que soit** la loi des observations (si la variance est finie). L'**erreur standard** est $\mathrm{SE} = \sigma/\sqrt n$.

# %%
population = rng.lognormal(mean=12, sigma=0.7, size=200_000)      # « revenus » très asymétriques
fig, axes = plt.subplots(1, 4, figsize=(15, 3))
axes[0].hist(population, bins=80); axes[0].set_title("Population (asymétrique)")
for ax, n in zip(axes[1:], [2, 10, 100]):
    moyennes = rng.choice(population, size=(5_000, n)).mean(axis=1)
    ax.hist(moyennes, bins=60, density=True)
    ax.set_title(f"Moyennes d'échantillons, n={n}")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Intervalle de confiance (IC)
#
# IC à 95 % pour une moyenne (loi de Student, variance inconnue) :
# $$\bar x \pm t_{0{,}975;\,n-1}\,\frac{s}{\sqrt n}$$
#
# **Interprétation correcte** : si l'on répétait l'enquête un grand nombre de fois, **95 % des intervalles
# ainsi construits** contiendraient la vraie moyenne. (Ce n'est **pas** « 95 % de chance que μ soit dans *cet* intervalle ».)

# %%
x = etu["note_examen"]
moy, se = x.mean(), stats.sem(x)
ic = stats.t.interval(0.95, df=len(x) - 1, loc=moy, scale=se)
print(f"Moyenne = {moy:.2f}, SE = {se:.3f}, IC95% = [{ic[0]:.2f} ; {ic[1]:.2f}]")

# %% [markdown]
# Simulation de la couverture : 100 IC sur des échantillons de taille 30.

# %%
mu_vrai = population.mean()
fig, ax = plt.subplots(figsize=(10, 3))
rate = 0
for i in range(100):
    e = rng.choice(population, 30)
    lo, hi = stats.t.interval(0.95, 29, e.mean(), stats.sem(e))
    contient = lo <= mu_vrai <= hi
    rate += contient
    ax.plot([i, i], [lo, hi], color="tab:blue" if contient else "red")
ax.axhline(mu_vrai, color="black"); ax.set_title(f"{rate}/100 intervalles contiennent la vraie moyenne")
plt.tight_layout(); plt.show()

# %% [markdown]
# Proportion : $\hat p \pm z_{0{,}975}\sqrt{\hat p(1-\hat p)/n}$ (méthode de Wald) — on préfère l'intervalle de
# **Wilson**, plus fiable pour les petites proportions.

# %%
from statsmodels.stats.proportion import proportion_confint

k, n = (etu["admis"] == "Oui").sum(), len(etu)
print("Taux d'admission :", round(k / n, 3), "| IC95% Wilson :", np.round(proportion_confint(k, n, method="wilson"), 3))

# %% [markdown]
# ## 3. Logique d'un test d'hypothèse
#
# 1. **H₀** (hypothèse nulle, « pas d'effet ») et **H₁** (alternative).
# 2. Choisir un seuil **α** (souvent 5 %) *avant* de regarder les données.
# 3. Calculer une **statistique de test** et sa **p-valeur** : probabilité, **si H₀ est vraie**,
#    d'observer un résultat au moins aussi extrême.
# 4. Si $p < α$ : on **rejette H₀** ; sinon on **ne rejette pas** (≠ « H₀ est prouvée »).
#
# |  | H₀ vraie | H₀ fausse |
# |---|---|---|
# | Rejeter H₀ | **Erreur de type I** (α) | ✅ puissance $1-β$ |
# | Ne pas rejeter | ✅ | **Erreur de type II** (β) |
#
# > ⚠️ Une p-valeur **ne mesure pas** l'importance d'un effet. Avec $n$ très grand, un effet minuscule devient
# > « significatif ». **Toujours rapporter une taille d'effet et un IC** (déclaration de l'ASA, Wasserstein & Lazar, 2016).
#
# ## 4. Comparer deux groupes
#
# **Question** : le tutorat améliore-t-il la note d'examen ?

# %%
avec = etu.loc[etu["tutorat"] == "Oui", "note_examen"]
sans = etu.loc[etu["tutorat"] == "Non", "note_examen"]
print(f"avec tutorat : n={len(avec)}, moyenne={avec.mean():.2f} | sans : n={len(sans)}, moyenne={sans.mean():.2f}")

# %% [markdown]
# ### 4.1 Vérifier les conditions
# Le test t suppose des observations **indépendantes** et des moyennes approximativement normales
# (vrai par le TCL si $n$ ≳ 30 par groupe). On utilise par défaut la version de **Welch** (variances inégales).

# %%
print("Shapiro (normalité) p =", round(stats.shapiro(avec).pvalue, 3), "/", round(stats.shapiro(sans).pvalue, 3))
print("Levene (égalité des variances) p =", round(stats.levene(avec, sans).pvalue, 3))

# %% [markdown]
# ### 4.2 Test t de Welch + taille d'effet (d de Cohen)
# $$d = \frac{\bar x_1 - \bar x_2}{s_{\text{poolé}}}\quad(0{,}2 \text{ petit}, 0{,}5 \text{ moyen}, 0{,}8 \text{ grand — repères de Cohen})$$

# %%
res = stats.ttest_ind(avec, sans, equal_var=False)
ic_diff = res.confidence_interval(0.95)
s_pool = np.sqrt(((len(avec) - 1) * avec.var() + (len(sans) - 1) * sans.var()) / (len(avec) + len(sans) - 2))
d = (avec.mean() - sans.mean()) / s_pool
print(f"t = {res.statistic:.2f}, p = {res.pvalue:.2e}")
print(f"différence = {avec.mean() - sans.mean():.2f} points, IC95% = [{ic_diff.low:.2f} ; {ic_diff.high:.2f}]")
print(f"d de Cohen = {d:.2f}")

# %% [markdown]
# **Conclusion rédigée** : les étudiants avec tutorat obtiennent en moyenne ≈ 1 point de plus (IC95 % donné ci-dessus),
# différence statistiquement significative et d'ampleur modérée. ⚠️ Étude **observationnelle** : on ne peut conclure à
# un effet **causal** du tutorat (ceux qui le choisissent sont peut-être plus motivés).
#
# ### 4.3 Alternative non paramétrique : Mann–Whitney (Wilcoxon rang-somme)
# Utile pour des données ordinales, très asymétriques ou de petits échantillons.

# %%
mw = stats.mannwhitneyu(avec, sans, alternative="two-sided")
print(f"U = {mw.statistic:.0f}, p = {mw.pvalue:.2e}")

# %% [markdown]
# ### 4.4 Données appariées
# Mêmes étudiants mesurés deux fois (contrôle continu puis examen) → test t **apparié** (ou Wilcoxon signé).

# %%
ap = stats.ttest_rel(etu["note_examen"], etu["note_controle_continu"])
print(f"gain moyen = {(etu['note_examen'] - etu['note_controle_continu']).mean():.2f}, p = {ap.pvalue:.2e}")

# %% [markdown]
# ## 5. Plus de deux groupes : ANOVA et Kruskal–Wallis
#
# **Question** : la note moyenne diffère-t-elle selon la filière ?

# %%
groupes = [g["note_examen"].values for _, g in etu.groupby("filiere")]
an = stats.f_oneway(*groupes)
kw = stats.kruskal(*groupes)
print(f"ANOVA : F = {an.statistic:.2f}, p = {an.pvalue:.3g} | Kruskal–Wallis : H = {kw.statistic:.2f}, p = {kw.pvalue:.3g}")

# η² (part de variance expliquée par la filière)
grand = etu["note_examen"].mean()
sce_inter = sum(len(g) * (g.mean() - grand) ** 2 for g in groupes)
sce_tot = ((etu["note_examen"] - grand) ** 2).sum()
print(f"η² = {sce_inter / sce_tot:.3f}")

# %% [markdown]
# L'ANOVA dit « au moins une moyenne diffère », pas laquelle → **comparaisons post-hoc** (Tukey HSD) :

# %%
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print(pairwise_tukeyhsd(etu["note_examen"], etu["filiere"], alpha=0.05))

# %%
fig, ax = plt.subplots(figsize=(7, 3))
sns.pointplot(data=etu, x="note_examen", y="filiere", errorbar=("ci", 95), linestyle="none", capsize=0.2, ax=ax)
ax.set(title="Note moyenne par filière (IC 95 %)", xlabel="note d'examen /20", ylabel="")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Deux variables qualitatives : test du χ² d'indépendance
#
# **Question** : l'accès à internet est-il indépendant du milieu (urbain/rural) ?

# %%
menages = pd.read_csv(DATA / "enquete_menages.csv")
tab = pd.crosstab(menages["milieu"], menages["acces_internet"])
chi2, p, ddl, attendus = stats.chi2_contingency(tab)
n = tab.values.sum()
v_cramer = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
print(tab, "\n")
print(f"χ² = {chi2:.1f}, ddl = {ddl}, p = {p:.2e}, V de Cramér = {v_cramer:.2f}")
print("Effectifs attendus tous ≥ 5 :", (attendus >= 5).all())

# %% [markdown]
# Condition : effectifs **attendus** ≥ 5 ; sinon test **exact de Fisher** (`stats.fisher_exact` pour un tableau 2×2).
#
# ## 7. Tester une corrélation

# %%
r, p_r = stats.pearsonr(etu["heures_etude_semaine"], etu["note_examen"])
rho, p_s = stats.spearmanr(etu["heures_etude_semaine"], etu["note_examen"])
print(f"Pearson r = {r:.3f} (p = {p_r:.1e}) | Spearman ρ = {rho:.3f} (p = {p_s:.1e})")

# %% [markdown]
# ## 8. Le bootstrap : un IC pour (presque) n'importe quelle statistique
#
# Idée (Efron, 1979) : ré-échantillonner **avec remise** les données observées pour approcher la distribution
# d'échantillonnage. Très utile pour la **médiane**, un ratio, un écart de médianes…

# %%
rev = menages["revenu_mensuel_fcfa"].dropna()
urb, rur = rev[menages["milieu"] == "Urbain"], rev[menages["milieu"] == "Rural"]

def ecart_medianes(a, b):
    return np.median(a) - np.median(b)

boot = stats.bootstrap((urb.values, rur.values), ecart_medianes, n_resamples=5_000, method="percentile",
                       random_state=0, vectorized=False)
print(f"écart de médianes urbain − rural = {ecart_medianes(urb, rur):,.0f} FCFA")
print(f"IC95% bootstrap = [{boot.confidence_interval.low:,.0f} ; {boot.confidence_interval.high:,.0f}]")

# %% [markdown]
# ## 9. Tests multiples
#
# Avec 20 tests indépendants au seuil 5 %, la probabilité d'au moins un faux positif est $1 - 0{,}95^{20} ≈ 64\%$ !
#
# - **Bonferroni** : seuil α/m (très conservateur) — contrôle le risque d'**au moins une** erreur (FWER).
# - **Benjamini–Hochberg** : contrôle le **taux de fausses découvertes** (FDR), plus puissant ; standard en génomique.

# %%
from statsmodels.stats.multitest import multipletests

# 20 comparaisons « bidons » (aucune vraie différence) + 3 vraies différences
p_vals = [stats.ttest_ind(rng.normal(0, 1, 50), rng.normal(0, 1, 50)).pvalue for _ in range(20)]
p_vals += [stats.ttest_ind(rng.normal(0, 1, 50), rng.normal(0.8, 1, 50)).pvalue for _ in range(3)]
for methode in ["bonferroni", "fdr_bh"]:
    rejet, *_ = multipletests(p_vals, alpha=0.05, method=methode)
    print(f"{methode:<11}: {rejet.sum()} rejets (dont {rejet[:20].sum()} faux positifs)")
print(f"sans correction : {(np.array(p_vals) < 0.05).sum()} rejets (dont {(np.array(p_vals[:20]) < 0.05).sum()} faux positifs)")

# %% [markdown]
# ## 10. Arbre de décision : quel test choisir ?
#
# ```text
# Quelles variables ?
# ├── 1 quanti vs 1 quali à 2 groupes
# │     ├── indépendants : t de Welch      (non param. : Mann–Whitney)
# │     └── appariés     : t apparié       (non param. : Wilcoxon signé)
# ├── 1 quanti vs 1 quali à ≥3 groupes : ANOVA + Tukey   (non param. : Kruskal–Wallis + Dunn)
# ├── 2 quali : χ² d'indépendance        (petits effectifs : Fisher exact)
# └── 2 quanti : corrélation de Pearson  (monotone / extrêmes : Spearman) → régression (chap. 9)
# ```
#
# ## À retenir
#
# - IC et test sont deux faces de la même inférence ; **rapporter les deux + une taille d'effet**.
# - p-valeur = compatibilité des données avec H₀, **pas** la probabilité que H₀ soit vraie.
# - Vérifier les **conditions** (indépendance, normalité approchée, effectifs attendus).
# - Corriger les **tests multiples** ; ne pas « pêcher » la significativité (*p-hacking*).
# - Données observationnelles : **association ≠ causalité**.
#
# **Références** : Wasserstein R. & Lazar N. (2016), « The ASA Statement on p-Values », *The American Statistician* 70(2) ;
# Efron B. & Tibshirani R. (1993), *An Introduction to the Bootstrap* ; Benjamini Y. & Hochberg Y. (1995), *JRSS B* 57(1) ;
# Cohen J. (1988), *Statistical Power Analysis for the Behavioral Sciences*.
#
# ➡️ **Chapitre suivant : préparer les données pour la modélisation (feature engineering).**
