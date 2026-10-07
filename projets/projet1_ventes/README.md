# Projet 1 — Tableau de bord des ventes d'un réseau de boutiques

## Contexte
Un distributeur exploite une quarantaine de boutiques de quartier dans 8 régions du Sénégal. La direction reçoit chaque mois
un export de caisse **brut** (`data/brutes/ventes_boutiques.csv`) issu de logiciels différents, et souhaite un tableau de
bord fiable pour piloter l'activité.

## Questions de la direction
1. Quel est le chiffre d'affaires (CA) par région, par catégorie et par mois ? Quelles tendances et saisonnalités ?
2. Quelles boutiques sont les plus et les moins performantes (CA, nombre de transactions, panier moyen) ?
3. Comment se répartissent les modes de paiement (espèces vs mobile money) selon les régions ? Le mobile money progresse-t-il au fil de l'année ?
4. Les prix d'un même produit diffèrent-ils significativement d'une région à l'autre ?

## Travail demandé
1. **Audit** : produire un rapport de qualité (profil, doublons, formats, valeurs invalides) et des **règles de validation**.
2. **Nettoyage** : écrire une fonction `nettoyer(df) -> (df_propre, journal)` et sauvegarder le résultat en Parquet.
3. **Analyses** répondant aux 4 questions (agrégations, `pivot_table`, `resample`, test de Kruskal–Wallis ou ANOVA pour la question 4).
4. **Visualisation** : 4 à 6 graphiques « prêts pour la direction ». *Bonus* : un tableau de bord HTML interactif (Plotly) ou Streamlit.
5. **Note de synthèse** d'une page : chiffres clés, constats, 3 recommandations.

## Pistes et pièges
- Les dates sont dans 3 formats ; certaines régions sont écrites de nombreuses façons (dont « St-Louis »).
- Certains prix contiennent « FCFA » et des espaces ; certaines quantités sont négatives ou absurdes.
- Attention aux 120 lignes dupliquées (double import).
- Un notebook de démarrage est fourni : [`demarrage.ipynb`](demarrage.ipynb).
