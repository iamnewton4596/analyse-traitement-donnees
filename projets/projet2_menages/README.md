# Projet 2 — Conditions de vie des ménages : inégalités et accès aux services

## Contexte
Une ONG prépare un plaidoyer sur l'inclusion numérique et financière. Elle dispose d'une enquête auprès de 3 000 ménages
(`data/brutes/enquete_menages.csv`, données simulées) et vous demande une analyse statistique rigoureuse.

## Questions
1. Comment le revenu se distribue-t-il ? Quelles différences entre milieux urbain/rural, régions, niveaux d'éducation ?
2. Quels facteurs sont associés à l'accès à internet et à l'usage du mobile money ?
3. La non-réponse sur le revenu biaise-t-elle les résultats ? Comment la traiter ?
4. (Bonus) Peut-on prédire l'usage du mobile money ? Quelles variables comptent le plus ?

## Travail demandé
1. Audit et nettoyage (attention au code **99** de la taille du ménage).
2. Étude de la **non-réponse** : mécanisme probable, comparaison de plusieurs imputations, analyse de sensibilité
   (résultats avec suppression vs imputation).
3. **EDA** complète et **tests** adaptés : intervalles de confiance, Welch / Mann–Whitney, ANOVA / Kruskal–Wallis, χ² + V de Cramér,
   avec **tailles d'effet** et correction des tests multiples.
4. Bonus : pipeline scikit-learn (préparation + régression logistique) évalué par validation croisée.
5. Rapport destiné à l'ONG : messages clés, graphiques, **limites** (données d'enquête, causalité, non-réponse).

## Points d'attention
- Le revenu est très asymétrique : médianes, échelle logarithmique.
- Association ≠ causalité : le milieu urbain est un **facteur de confusion** possible pour beaucoup de relations.
