# Mémo — Checklist de nettoyage et de qualité des données

*Cours « Analyse et traitement des données » — Issa Gueye*

## Avant de commencer
- [ ] La **question métier** et l'unité d'observation sont claires (une ligne = quoi ?).
- [ ] Les données brutes sont **sauvegardées en lecture seule** ; on travaille sur une copie.
- [ ] Le dictionnaire des variables (sens, unité, codes spéciaux) est disponible.

## Audit (diagnostiquer — ne rien corriger encore)
| Dimension | Vérification | Code |
|---|---|---|
| Structure | dimensions, types attendus | `df.shape`, `df.info()` |
| Complétude | manquants par colonne et par ligne | `df.isna().mean()` |
| Unicité | doublons exacts et sur la clé | `df.duplicated()`, `df.id.is_unique` |
| Validité | plages, domaines, formats | `between`, `isin(referentiel)`, regex |
| Cohérence | casse, espaces, synonymes, unités | `value_counts()` |
| Exactitude | valeurs extrêmes plausibles ? | IQR, z robuste (MAD), graphiques |
| Codes spéciaux | 99, -99, 9999, « NSP », « - » | `na_values`, `replace` |

## Corriger (dans cet ordre en général)
1. **Doublons** exacts → `drop_duplicates()` ; doublons de clé → règle métier.
2. **Types** : nombres (`to_numeric(errors="coerce")`), dates (`to_datetime(format=...)`), catégories.
3. **Texte** : `strip`, casse, espaces multiples, accents, dictionnaire de synonymes, vérification contre un **référentiel**.
4. **Valeurs invalides** (règles métier) → `NaN` (puis imputation) ou suppression justifiée.
5. **Manquants** : identifier le mécanisme (MCAR / MAR / MNAR) avant de choisir.
6. **Valeurs aberrantes** : détecter sur la **bonne échelle** (log si asymétrique), décider avec le métier
   (corriger / garder / winsoriser), jamais supprimer en silence.
7. **Variables dérivées** (montant = prix × quantité, indicateurs « imputé »).

## Choisir une stratégie pour les manquants
| Situation | Stratégie |
|---|---|
| < 5 %, MCAR plausible | suppression des lignes acceptable |
| Variable > 50–60 % manquante et peu utile | supprimer la variable |
| MAR lié à des groupes connus | imputation par groupe |
| MAR, variables corrélées disponibles | KNN, régression, `IterativeImputer` (MICE) |
| Série temporelle | `interpolate(method="time", limit=…)`, `ffill` pour des paliers |
| Le fait de manquer est informatif | ajouter un indicateur `x_manquant` |
| Inférence rigoureuse | imputation multiple + règles de Rubin |
| Modélisation | imputeur **dans le Pipeline** (ajusté sur le train) |

## Après
- [ ] Relancer les **règles de validation** : toutes passent.
- [ ] Le **journal** du nettoyage compte ce que chaque étape a changé.
- [ ] Le nettoyage est une **fonction / un script** ré-exécutable de bout en bout.
- [ ] Données propres sauvegardées (**Parquet**) avec la date et la version du script.
