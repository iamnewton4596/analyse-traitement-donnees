# Projet 3 — Segmentation des clients d'un service de mobile money

## Contexte
Un opérateur de mobile money veut adapter ses offres. Il fournit un extrait anonymisé de 2 300 clients
(`data/brutes/clients_mobile_money.csv`, données simulées) avec des indicateurs d'usage mensuels.

## Objectif
Construire une **segmentation** interprétable et actionnable, puis proposer une offre par segment.

## Travail demandé
1. EDA des 6 variables d'usage (distributions, asymétries, corrélations). **Ne pas utiliser** la colonne `segment_reel`
   avant l'étape 5 : elle simule une « vérité » inconnue en situation réelle.
2. Préparation : transformations (log des montants), standardisation — justifiées.
3. **ACP** : nombre d'axes retenus, cercle des corrélations, interprétation des axes.
4. **Clustering** : k-means et CAH (Ward) ; choix de k (coude, silhouette, interprétabilité) ; stabilité (plusieurs graines,
   sous-échantillons).
5. **Évaluation** : comparer votre segmentation à `segment_reel` (indice de Rand ajusté, tableau croisé). Discuter.
6. **Restitution** : un « portrait » de chaque segment (nom, taille, profil médian, offre proposée), 8–10 slides.

## Pour aller plus loin
DBSCAN, mélanges gaussiens (`GaussianMixture`), UMAP / t-SNE pour la visualisation.
