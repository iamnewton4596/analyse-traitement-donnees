# Jeux de données du cours

> ⚠️ Toutes les données sont **simulées** (`scripts/generer_donnees.py`, graine 2026). Elles imitent des situations
> réalistes au Sénégal mais ne décrivent aucune personne, boutique ou entreprise réelle. Les défauts (manquants,
> doublons, formats incohérents, valeurs aberrantes) sont **volontaires** et servent l'apprentissage.

Regénérer à l'identique : `python scripts/generer_donnees.py`

| Fichier | Lignes | Usage principal |
|---|---|---|
| `brutes/ventes_boutiques.csv` | 5 120 | nettoyage, agrégation, séries temporelles (chap. 3–5, projet 1) |
| `brutes/meteo_dakar.csv` | 1 807 | séries temporelles, interpolation (chap. 4–5, TD 2) |
| `brutes/enquete_menages.csv` | 3 000 | manquants MAR, EDA, tests, préparation (chap. 2, 4, 6–8, projet 2) |
| `brutes/resultats_etudiants.csv` | 600 | inférence, régression, classification (chap. 6, 7, 9) |
| `brutes/clients_mobile_money.csv` | 2 300 | ACP, clustering (chap. 9, projet 3) |
| `brutes/regions_senegal.csv` | 14 | référentiel pour validation et jointures (chap. 4–5) |

`data/propres/` est créé par le chapitre 4 (non versionné).

## Dictionnaire des variables

### ventes_boutiques.csv — une ligne = une transaction de caisse
| Variable | Description | Défauts injectés |
|---|---|---|
| `id_transaction` | identifiant | 120 lignes dupliquées |
| `date` | date de la vente (2025) | 3 formats : `AAAA-MM-JJ`, `JJ/MM/AAAA`, `JJ-MM-AAAA` |
| `region` | région de la boutique | casse, espaces, `St-Louis` pour `Saint-Louis` |
| `boutique` | code boutique `BTQ-XXX-NN` | ~1 % manquant |
| `produit`, `categorie` | article vendu et sa catégorie | — |
| `prix_unitaire` | prix en FCFA | ~12 % stockés en texte « 6 500 FCFA » |
| `quantite` | quantité | ~3 % manquants ; 15 valeurs absurdes (100–999) ; 10 négatives |
| `mode_paiement` | Espèces, Wave, Orange Money, Carte | ~6 % manquants |

### meteo_dakar.csv — une ligne = un jour (2020–2024)
`date`, `temp_moy_c` (°C ; blocs manquants, 4 valeurs −99), `humidite_pct`, `precip_mm` (hivernage juillet–octobre), `vent_ms` (~1 % manquants). 20 jours absents du fichier.

### enquete_menages.csv — une ligne = un ménage
`id_menage`, `region`, `milieu` (Urbain/Rural), `taille_menage` (code **99** = ne sait pas), `age_chef` (~2 % manquants), `sexe_chef`,
`education_chef` (Aucun < Coranique < Primaire < Secondaire < Supérieur), `revenu_mensuel_fcfa` (≈12 % manquants, **MAR** : plus fréquents en milieu
urbain et chez les diplômés du supérieur), `depenses_mensuelles_fcfa`, `acces_electricite`, `acces_internet`, `utilise_mobile_money` (Oui/Non).

### resultats_etudiants.csv — une ligne = un étudiant
`id_etudiant`, `filiere`, `sexe`, `heures_etude_semaine`, `tutorat` (Oui/Non), `note_controle_continu`, `note_examen` (/20), `admis` (note ≥ 10).
Effets simulés : contrôle continu (+), heures (+), tutorat (+≈1 pt), filière ; **aucun** effet du sexe.

### clients_mobile_money.csv — une ligne = un client
`id_client`, `nb_transactions_mois`, `montant_moyen_fcfa`, `part_montant_recu`, `part_paiement_marchand`, `anciennete_mois`, `age`,
`segment_reel` (4 profils simulés : petit_usage, commercant, salarie, diaspora — **à n'utiliser que pour évaluer** une segmentation).

### regions_senegal.csv
Les 14 régions administratives, une zone géographique indicative et une superficie arrondie (ordre de grandeur, usage pédagogique).
