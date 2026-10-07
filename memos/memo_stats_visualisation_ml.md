# Mémo — Statistiques, visualisation et préparation pour la modélisation

*Cours « Analyse et traitement des données » — Issa Gueye*

## Résumés descriptifs
| Variable | Position | Dispersion | Forme |
|---|---|---|---|
| Quantitative symétrique | moyenne | écart-type (`ddof=1`) | — |
| Quantitative asymétrique | **médiane** | IQR (Q3 − Q1), MAD | skewness, kurtosis |
| Qualitative | mode, effectifs, % | nombre de modalités | — |

## Quel graphique ?
| Variables | Graphique |
|---|---|
| 1 quanti | histogramme (`bins="fd"`), KDE, boîte, ECDF |
| 1 quali | barres **triées** |
| quanti × quanti | nuage de points (log si asymétrie), hexbin |
| quanti × quali | boîtes, violons, points + IC |
| quali × quali | barres empilées à 100 %, heatmap de `crosstab` |
| temps | courbe (+ moyenne mobile) |
| multivarié | facettes (`col=`), `pairplot`, heatmap de corrélation, ACP |

Règles : titre = message ; axes et unités ; barres depuis 0 ; palette `colorblind`/`viridis` ; pas de 3D ; trier.

## Quel test ?
| Situation | Paramétrique | Non paramétrique | Taille d'effet |
|---|---|---|---|
| 2 groupes indépendants | `ttest_ind(equal_var=False)` (Welch) | `mannwhitneyu` | d de Cohen |
| 2 mesures appariées | `ttest_rel` | `wilcoxon` | d apparié |
| ≥ 3 groupes | `f_oneway` + Tukey HSD | `kruskal` (+ Dunn) | η² |
| 2 qualitatives | `chi2_contingency` (attendus ≥ 5) | `fisher_exact` (2×2) | V de Cramér |
| 2 quantitatives | `pearsonr` | `spearmanr`, `kendalltau` | r, ρ |
| IC quelconque | — | `stats.bootstrap` | — |
| Plusieurs tests | Bonferroni / Holm (FWER), Benjamini–Hochberg (FDR) : `multipletests` | | |

**p-valeur** : probabilité, si H₀ est vraie, d'observer un résultat au moins aussi extrême.
Elle ne mesure ni l'importance de l'effet, ni la probabilité que H₀ soit vraie → rapporter **IC + taille d'effet**.

Repères de Cohen pour d : 0,2 petit · 0,5 moyen · 0,8 grand. V de Cramér (ddl=1) : 0,1 · 0,3 · 0,5.

## Préparation (scikit-learn)
```python
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("ord", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(categories=[ordre])), ord_cols),
    ("nom", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), nom_cols),
])
modele = make_pipeline(prep, LogisticRegression(max_iter=1000))
cross_val_score(modele, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
```
| Transformation | Effet | Quand |
|---|---|---|
| `StandardScaler` | moyenne 0, écart-type 1 | modèles linéaires, distances, ACP |
| `MinMaxScaler` | [0, 1] | réseaux de neurones, bornes connues |
| `RobustScaler` | médiane / IQR | valeurs extrêmes |
| `np.log`, `PowerTransformer` | réduit l'asymétrie | revenus, montants, comptages |
| One-hot | indicatrices | nominales |
| Ordinal | entiers ordonnés | ordinales |

**Fuite de données** : tout ce qui « apprend » (imputeur, scaler, sélection de variables, encodeur cible) va **dans** le pipeline.
Pas d'information future, pas de variable qui contient la cible.

## ACP & clustering
- Standardiser → `PCA()` → éboulis, Kaiser (λ > 1), % cumulé → cercle des corrélations (`components_.T * sqrt(explained_variance_)`).
- k-means : standardiser ; choisir k (coude, **silhouette**, interprétabilité) ; `n_init=10` ; profiler les segments (médianes).
- CAH Ward : dendrogramme ; DBSCAN pour formes non convexes ; `GaussianMixture` pour appartenances probabilistes.

## Métriques
| Tâche | Métriques |
|---|---|
| Régression | MAE, RMSE, R² ; graphique résidus vs ajustés |
| Classification | matrice de confusion, précision, rappel, F1, AUC ROC ; toujours comparer à une **baseline** (`DummyClassifier`) |
| Clustering | silhouette (interne), ARI (si vérité connue) |
