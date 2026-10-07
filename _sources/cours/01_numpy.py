# %% [markdown]
# # Chapitre 1 — Python scientifique et NumPy
#
# **Cours : Analyse et traitement des données** · Auteur : *Issa Gueye*
#
# ## Objectifs
#
# - Comprendre pourquoi **NumPy** est la fondation de tout l'écosystème (pandas, scikit-learn, SciPy…).
# - Créer, indexer et transformer des **tableaux** (`ndarray`).
# - Maîtriser la **vectorisation** et le **broadcasting** pour écrire du code rapide et lisible.
# - Calculer des statistiques descriptives et générer des nombres aléatoires **reproductibles**.

# %%
import time

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Pourquoi NumPy ?
#
# Une liste Python contient des **références vers des objets** dispersés en mémoire.
# Un `ndarray` NumPy contient des **valeurs de même type stockées de façon contiguë** :
# les opérations sont exécutées en C, sans boucle Python.

# %%
n = 1_000_000
liste = list(range(n))
tableau = np.arange(n)

t0 = time.perf_counter()
carres_liste = [x * x for x in liste]
t1 = time.perf_counter()
carres_np = tableau * tableau
t2 = time.perf_counter()

print(f"Boucle Python : {(t1 - t0) * 1000:7.1f} ms")
print(f"NumPy         : {(t2 - t1) * 1000:7.1f} ms  (≈ {(t1 - t0) / (t2 - t1):.0f}× plus rapide)")

# %% [markdown]
# ## 2. Créer des tableaux

# %%
a = np.array([12500, 6500, 700, 2300])          # depuis une liste
z = np.zeros((2, 3))                             # matrice de zéros
u = np.ones(4, dtype=np.int32)                   # type explicite
r = np.arange(0, 1, 0.25)                        # comme range, avec un pas réel
l = np.linspace(0, 1, 5)                         # 5 points régulièrement espacés, bornes incluses
I = np.eye(3)                                    # matrice identité
print(a, a.dtype, a.shape, a.ndim, sep="\n")
print(l)

# %% [markdown]
# | Attribut | Signification |
# |---|---|
# | `shape` | dimensions, ex. `(lignes, colonnes)` |
# | `ndim` | nombre d'axes |
# | `dtype` | type des éléments (`int64`, `float64`, `bool`, …) |
# | `size` | nombre total d'éléments |
#
# ⚠️ Un tableau n'a **qu'un seul dtype**. Mélanger entiers et texte convertit tout en texte :

# %%
print(np.array([1, 2, "3"]).dtype)   # <U21 : chaînes Unicode !

# %% [markdown]
# ## 3. Indexation et découpage

# %%
M = np.arange(1, 13).reshape(3, 4)
print(M)
print("Élément (1, 2)          :", M[1, 2])
print("2e ligne                :", M[1])
print("3e colonne              :", M[:, 2])
print("Sous-matrice            :\n", M[:2, 1:3])
print("Une ligne sur deux      :\n", M[::2])

# %% [markdown]
# ### Indexation booléenne (filtrage) — l'outil le plus utilisé en analyse

# %%
temperatures = np.array([24.1, 27.3, -99.0, 29.8, 31.2, np.nan, 26.4])
valides = (temperatures > -50) & ~np.isnan(temperatures)   # & = ET, | = OU, ~ = NON
print(valides)
print("Températures valides :", temperatures[valides])
print("Moyenne valide        :", temperatures[valides].mean())

# %% [markdown]
# > ⚠️ Utilisez `&`, `|`, `~` (et des parenthèses), **pas** `and`, `or`, `not`, sur des tableaux.
#
# ### Vue ou copie ?
# Un découpage (`M[:2]`) renvoie une **vue** : modifier la vue modifie l'original.
# L'indexation booléenne ou par liste d'indices renvoie une **copie**.

# %%
v = M[0]          # vue
v[0] = 100
print(M[0])       # l'original a changé !
M[0, 0] = 1       # on remet
c = M[[0]].copy() # copie explicite : sûre

# %% [markdown]
# ## 4. Vectorisation et fonctions universelles (ufuncs)
#
# Les opérations s'appliquent **élément par élément** sans boucle :

# %%
prix = np.array([12500, 6500, 700, 2300])
quantites = np.array([3, 2, 10, 4])
chiffre_affaires = prix * quantites
print(chiffre_affaires, chiffre_affaires.sum())
print(np.log(prix).round(2))
print(np.sqrt(quantites))
print(np.where(chiffre_affaires > 10000, "gros", "petit"))   # « si … alors … sinon » vectorisé

# %% [markdown]
# ## 5. Broadcasting
#
# Quand deux tableaux n'ont pas la même forme, NumPy **étire virtuellement** les dimensions de taille 1.
#
# **Règle** : on compare les formes **de droite à gauche** ; deux dimensions sont compatibles si elles
# sont **égales** ou si l'une vaut **1**.
#
# ```text
#   (3, 4)   tableau de données : 3 boutiques × 4 produits
#      (4,)  prix des 4 produits      → étiré en (3, 4)
#   ------
#   (3, 4)   résultat
# ```

# %%
ventes_qte = np.array([[3, 2, 10, 4],
                       [1, 0, 25, 2],
                       [5, 3, 7, 0]])           # 3 boutiques × 4 produits
ca = ventes_qte * prix                          # (3,4) * (4,) -> (3,4)
print(ca)
print("CA par boutique :", ca.sum(axis=1))      # axis=1 : on somme le long des colonnes
print("CA par produit  :", ca.sum(axis=0))      # axis=0 : on somme le long des lignes

# %% [markdown]
# ### Application : standardiser chaque colonne (centrer-réduire)
#
# $$z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

# %%
X = np.random.default_rng(0).normal(loc=[50, 10_000, 3], scale=[10, 2_000, 1], size=(200, 3))
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)   # (200,3) - (3,) : broadcasting
print("moyennes ≈ 0 :", Z.mean(axis=0).round(10))
print("écarts-types = 1 :", Z.std(axis=0, ddof=1))

# %% [markdown]
# > 📝 `ddof=1` donne l'écart-type **corrigé** (division par $n-1$), estimateur sans biais de la variance.
# > Par défaut NumPy utilise `ddof=0`, pandas utilise `ddof=1` : source fréquente d'écarts entre les deux !
#
# ## 6. Statistiques descriptives et valeurs manquantes

# %%
x = np.array([3.0, 7.0, np.nan, 1.0, 9.0])
print(np.mean(x))                       # nan : NaN « contamine » le calcul
print(np.nanmean(x), np.nanmedian(x), np.nanstd(x, ddof=1))
print(np.nanpercentile(x, [25, 50, 75]))

# %% [markdown]
# ## 7. Aléatoire reproductible
#
# Utilisez **toujours** un générateur explicite avec une graine (`seed`) :
# les résultats sont alors **reproductibles** par n'importe qui.

# %%
rng = np.random.default_rng(seed=42)
print(rng.normal(0, 1, 3))
print(rng.integers(1, 7, size=10))                   # 10 lancers de dé
print(rng.choice(["Wave", "Orange Money", "Espèces"], size=5, p=[0.3, 0.2, 0.5]))
echantillon = rng.permutation(10)                      # mélange
print(echantillon)

# %% [markdown]
# ### Illustration : la loi des grands nombres
# La moyenne de lancers de dé converge vers 3,5.

# %%
lancers = rng.integers(1, 7, size=100_000)
moyennes_cumulees = np.cumsum(lancers) / np.arange(1, lancers.size + 1)
for k in [10, 100, 1_000, 10_000, 100_000]:
    print(f"n = {k:>7} : moyenne = {moyennes_cumulees[k - 1]:.4f}")

# %% [markdown]
# ## 8. Un peu d'algèbre linéaire
#
# Utile pour la régression (chapitre 9) et l'ACP.
# Résolvons les **moindres carrés** $\hat\beta = \arg\min \|y - X\beta\|^2$, dont la solution vérifie
# $X^\top X \hat\beta = X^\top y$.

# %%
rng = np.random.default_rng(1)
heures = rng.uniform(0, 20, 100)
note = 6 + 0.4 * heures + rng.normal(0, 1.5, 100)
X = np.column_stack([np.ones_like(heures), heures])   # colonne de 1 pour l'ordonnée à l'origine
beta, *_ = np.linalg.lstsq(X, note, rcond=None)      # plus stable que inv(X.T @ X) @ X.T @ y
print(f"note ≈ {beta[0]:.2f} + {beta[1]:.3f} × heures")

# %% [markdown]
# ## À retenir
#
# | Besoin | Outil |
# |---|---|
# | Calcul rapide sur des nombres | `ndarray` + opérations vectorisées |
# | Filtrer | masque booléen `x[(x > a) & (x < b)]` |
# | Opérations entre formes différentes | broadcasting (comparer les formes de droite à gauche) |
# | Résumer par ligne / colonne | `axis=1` / `axis=0` |
# | Ignorer les NaN | `np.nanmean`, `np.nanmedian`, … |
# | Aléatoire reproductible | `rng = np.random.default_rng(seed)` |
#
# **Références** : Harris C. R. et al. (2020), « Array programming with NumPy », *Nature* 585 ;
# documentation officielle <https://numpy.org/doc/stable/>.
#
# ➡️ **Chapitre suivant : pandas, les fondamentaux.**
