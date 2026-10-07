"""
Génération des jeux de données pédagogiques du cours
« Analyse et traitement des données ».

Auteur : Issa Gueye
Licence : MIT

Toutes les données sont SYNTHÉTIQUES (simulées) mais inspirées de contextes
réalistes au Sénégal. Elles contiennent volontairement des défauts
(valeurs manquantes, doublons, formats incohérents, valeurs aberrantes)
pour l'apprentissage du nettoyage.

Usage :
    python scripts/generer_donnees.py
Les fichiers sont écrits dans data/brutes/.
La graine (SEED) garantit que tout le monde obtient exactement les mêmes fichiers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "data" / "brutes"
SORTIE.mkdir(parents=True, exist_ok=True)

REGIONS = ["Dakar", "Thiès", "Saint-Louis", "Diourbel", "Kaolack", "Ziguinchor", "Louga", "Tambacounda"]
POIDS_REGIONS = np.array([0.30, 0.14, 0.10, 0.13, 0.10, 0.08, 0.08, 0.07])


# ---------------------------------------------------------------------------
# 1. Ventes de boutiques (données « sales » à nettoyer)
# ---------------------------------------------------------------------------
def ventes_boutiques(n=5000):
    produits = {
        "Riz brisé 25kg": ("Alimentaire", 12500),
        "Huile 5L": ("Alimentaire", 6500),
        "Sucre 1kg": ("Alimentaire", 700),
        "Lait en poudre 400g": ("Alimentaire", 2300),
        "Café Touba 250g": ("Alimentaire", 1200),
        "Savon (lot de 4)": ("Hygiène", 1000),
        "Javel 1L": ("Hygiène", 600),
        "Crédit téléphone": ("Services", 1000),
        "Recharge Woyofal": ("Services", 5000),
        "Cahier 200p": ("Fournitures", 500),
        "Stylo (lot de 10)": ("Fournitures", 1500),
        "Pain": ("Alimentaire", 150),
    }
    noms = list(produits)
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    region = rng.choice(REGIONS, size=n, p=POIDS_REGIONS)
    boutique = [f"BTQ-{r[:3].upper()}-{rng.integers(1, 6):02d}" for r in region]
    produit = rng.choice(noms, size=n)
    prix = np.array([produits[p][1] for p in produit], dtype=float)
    # variation régionale du prix (+/- 8 %)
    prix = np.round(prix * rng.normal(1.0, 0.04, n), -1)
    qte = rng.poisson(2, n) + 1
    paiement = rng.choice(["Espèces", "Wave", "Orange Money", "Carte"], size=n, p=[0.45, 0.30, 0.20, 0.05])

    df = pd.DataFrame({
        "id_transaction": [f"T{100000 + i}" for i in range(n)],
        "date": dates,
        "region": region,
        "boutique": boutique,
        "produit": produit,
        "categorie": [produits[p][0] for p in produit],
        "prix_unitaire": prix,
        "quantite": qte.astype(float),
        "mode_paiement": paiement,
    })

    # --- Injection de défauts réalistes -----------------------------------
    # (a) dates dans des formats différents (exports de logiciels différents)
    d = df["date"].dt.strftime("%Y-%m-%d").astype(object)
    idx = rng.choice(n, 700, replace=False)
    d.iloc[idx[:350]] = df["date"].iloc[idx[:350]].dt.strftime("%d/%m/%Y")
    d.iloc[idx[350:]] = df["date"].iloc[idx[350:]].dt.strftime("%d-%m-%Y")
    df["date"] = d
    # (b) casse / espaces incohérents sur la région
    idx = rng.choice(n, 400, replace=False)
    df.loc[idx[:150], "region"] = df.loc[idx[:150], "region"].str.lower()
    df.loc[idx[150:280], "region"] = " " + df.loc[idx[150:280], "region"] + " "
    df.loc[idx[280:], "region"] = df.loc[idx[280:], "region"].str.upper()
    df["region"] = df["region"].replace({"Saint-Louis": "St-Louis"}).where(rng.random(n) < 0.30, df["region"])
    # (c) prix stockés comme texte « 6 500 FCFA » pour une partie des lignes
    prix_txt = df["prix_unitaire"].map(lambda x: f"{x:,.0f}".replace(",", " ")).astype(object)
    masque_txt = rng.random(n) < 0.12
    df["prix_unitaire"] = df["prix_unitaire"].astype(object)
    df.loc[masque_txt, "prix_unitaire"] = prix_txt[masque_txt] + " FCFA"
    # (d) valeurs manquantes
    for col, taux in [("quantite", 0.03), ("mode_paiement", 0.06), ("boutique", 0.01)]:
        df.loc[rng.random(n) < taux, col] = np.nan
    # (e) erreurs de saisie : quantités aberrantes / négatives
    idx = rng.choice(n, 25, replace=False)
    df.loc[idx[:15], "quantite"] = rng.choice([100, 250, 999], 15)
    df.loc[idx[15:], "quantite"] = -rng.integers(1, 4, 10)
    # (f) doublons exacts (double import)
    doublons = df.sample(120, random_state=SEED)
    df = pd.concat([df, doublons], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(SORTIE / "ventes_boutiques.csv", index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------------------
# 2. Météo journalière Dakar (série temporelle avec trous)
# ---------------------------------------------------------------------------
def meteo_dakar():
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    n = len(dates)
    j = dates.dayofyear.values
    # Température : maximum vers septembre-octobre, minimum vers février
    temp = 25.5 + 3.5 * np.sin(2 * np.pi * (j - 160) / 365) + rng.normal(0, 1.0, n)
    # tendance faible au réchauffement
    temp += 0.03 * (dates.year.values - 2020)
    humid = 68 + 12 * np.sin(2 * np.pi * (j - 150) / 365) + rng.normal(0, 4, n)
    # Saison des pluies juillet-octobre
    hivernage = (dates.month >= 7) & (dates.month <= 10)
    proba_pluie = np.where(hivernage, 0.35, 0.01)
    pluie = np.where(rng.random(n) < proba_pluie, rng.gamma(0.8, 14, n), 0.0)
    vent = np.abs(rng.normal(4.5, 1.5, n)) + np.where(dates.month.isin([1, 2, 3, 4]), 1.2, 0)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "temp_moy_c": temp.round(1),
        "humidite_pct": humid.clip(20, 100).round(0),
        "precip_mm": pluie.round(1),
        "vent_ms": vent.round(1),
    })
    # pannes de capteur : blocs de jours manquants + valeurs isolées
    for debut in rng.choice(n - 10, 6, replace=False):
        df.loc[debut:debut + rng.integers(2, 8), ["temp_moy_c", "humidite_pct"]] = np.nan
    df.loc[rng.random(n) < 0.01, "vent_ms"] = np.nan
    # quelques valeurs physiquement impossibles (capteur défaillant)
    df.loc[rng.choice(n, 4, replace=False), "temp_moy_c"] = -99.0
    # jours absents du fichier
    df = df.drop(index=rng.choice(n, 20, replace=False)).reset_index(drop=True)
    df.to_csv(SORTIE / "meteo_dakar.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 3. Enquête ménages (données d'enquête, manquants MAR)
# ---------------------------------------------------------------------------
def enquete_menages(n=3000):
    region = rng.choice(REGIONS, size=n, p=POIDS_REGIONS)
    p_urbain = np.where(region == "Dakar", 0.95, 0.40)
    milieu = np.where(rng.random(n) < p_urbain, "Urbain", "Rural")
    educ_niv = ["Aucun", "Coranique", "Primaire", "Secondaire", "Supérieur"]
    p_educ_urb = [0.18, 0.12, 0.25, 0.30, 0.15]
    p_educ_rur = [0.40, 0.22, 0.22, 0.13, 0.03]
    educ = np.array([rng.choice(educ_niv, p=p_educ_urb if m == "Urbain" else p_educ_rur) for m in milieu])
    taille = np.clip(rng.poisson(np.where(milieu == "Rural", 9, 7)), 1, 30)
    age_chef = np.clip(rng.normal(48, 12, n), 20, 90).round()
    sexe_chef = rng.choice(["H", "F"], n, p=[0.75, 0.25])
    effet_educ = pd.Series(educ).map({"Aucun": 0, "Coranique": 0.05, "Primaire": 0.2, "Secondaire": 0.5, "Supérieur": 1.0}).values
    log_rev = 11.6 + 0.45 * (milieu == "Urbain") + 0.9 * effet_educ + 0.03 * taille + rng.normal(0, 0.55, n)
    revenu = np.round(np.exp(log_rev), -3)  # FCFA / mois
    depenses = np.round(revenu * np.clip(rng.normal(0.82, 0.12, n), 0.3, 1.5), -3)
    electricite = rng.random(n) < np.where(milieu == "Urbain", 0.94, 0.48)
    internet = rng.random(n) < (0.15 + 0.35 * (milieu == "Urbain") + 0.35 * effet_educ)
    mobile_money = rng.random(n) < (0.45 + 0.25 * (milieu == "Urbain") + 0.2 * effet_educ)
    df = pd.DataFrame({
        "id_menage": [f"M{10000 + i}" for i in range(n)],
        "region": region,
        "milieu": milieu,
        "taille_menage": taille,
        "age_chef": age_chef,
        "sexe_chef": sexe_chef,
        "education_chef": educ,
        "revenu_mensuel_fcfa": revenu,
        "depenses_mensuelles_fcfa": depenses,
        "acces_electricite": np.where(electricite, "Oui", "Non"),
        "acces_internet": np.where(internet, "Oui", "Non"),
        "utilise_mobile_money": np.where(mobile_money, "Oui", "Non"),
    })
    # Manquants MAR : les ménages à revenus élevés et urbains refusent plus souvent de répondre
    p_nr = 0.04 + 0.12 * (df["milieu"] == "Urbain") + 0.10 * (df["education_chef"] == "Supérieur")
    df.loc[rng.random(n) < p_nr, "revenu_mensuel_fcfa"] = np.nan
    df.loc[rng.random(n) < 0.02, "age_chef"] = np.nan
    # Code spécial « 99 » utilisé par les enquêteurs pour « ne sait pas »
    df.loc[rng.random(n) < 0.01, "taille_menage"] = 99
    df.to_csv(SORTIE / "enquete_menages.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 4. Résultats d'étudiants (statistiques inférentielles)
# ---------------------------------------------------------------------------
def resultats_etudiants(n=600):
    filiere = rng.choice(["Mathématiques", "Informatique", "Économie", "Biologie"], n)
    sexe = rng.choice(["F", "M"], n)
    heures = np.clip(rng.gamma(4, 2.5, n), 0, 40).round(1)
    tutorat = rng.random(n) < 0.35
    effet_fil = pd.Series(filiere).map({"Mathématiques": -0.5, "Informatique": 0.3, "Économie": 0.6, "Biologie": 0.0}).values
    note_avant = np.clip(rng.normal(10.5, 2.8, n), 0, 20).round(2)
    note = 4.0 + 0.55 * note_avant + 0.12 * heures + 1.1 * tutorat + effet_fil + rng.normal(0, 2.0, n)
    note = np.clip(note, 0, 20).round(2)
    df = pd.DataFrame({
        "id_etudiant": [f"E{2000 + i}" for i in range(n)],
        "filiere": filiere,
        "sexe": sexe,
        "heures_etude_semaine": heures,
        "tutorat": np.where(tutorat, "Oui", "Non"),
        "note_controle_continu": note_avant,
        "note_examen": note,
        "admis": np.where(note >= 10, "Oui", "Non"),
    })
    df.to_csv(SORTIE / "resultats_etudiants.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 5. Clients mobile money (clustering / ACP)
# ---------------------------------------------------------------------------
def clients_mobile_money():
    # 4 segments latents : petits payeurs, commerçants, salariés urbains, diaspora (réception)
    segments = {
        "petit_usage": dict(n=900, nb_tx=(6, 3), montant=(3500, 0.6), recu=(0.3, 0.1), marchand=(0.1, 0.08), anciennete=(10, 6)),
        "commercant": dict(n=500, nb_tx=(85, 20), montant=(18000, 0.5), recu=(0.55, 0.1), marchand=(0.6, 0.12), anciennete=(36, 14)),
        "salarie": dict(n=600, nb_tx=(30, 8), montant=(25000, 0.5), recu=(0.25, 0.08), marchand=(0.35, 0.1), anciennete=(28, 12)),
        "diaspora": dict(n=300, nb_tx=(8, 3), montant=(95000, 0.4), recu=(0.92, 0.05), marchand=(0.05, 0.04), anciennete=(40, 15)),
    }
    lignes = []
    for nom, p in segments.items():
        k = p["n"]
        lignes.append(pd.DataFrame({
            "nb_transactions_mois": np.clip(rng.normal(*p["nb_tx"], k), 1, None).round(),
            "montant_moyen_fcfa": np.round(np.exp(np.log(p["montant"][0]) + rng.normal(0, p["montant"][1], k)), -2),
            "part_montant_recu": np.clip(rng.normal(*p["recu"], k), 0, 1).round(3),
            "part_paiement_marchand": np.clip(rng.normal(*p["marchand"], k), 0, 1).round(3),
            "anciennete_mois": np.clip(rng.normal(*p["anciennete"], k), 1, 120).round(),
            "age": np.clip(rng.normal(34 if nom != "diaspora" else 52, 10, k), 18, 85).round(),
            "segment_reel": nom,  # caché aux étudiants dans le projet ; utile pour l'évaluation
        }))
    df = pd.concat(lignes, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "id_client", [f"C{50000 + i}" for i in range(len(df))])
    df.to_csv(SORTIE / "clients_mobile_money.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 6. Petit référentiel des régions (pour les jointures)
# ---------------------------------------------------------------------------
def referentiel_regions():
    # Chiffres arrondis, à but pédagogique uniquement (ordre de grandeur).
    df = pd.DataFrame({
        "region": REGIONS + ["Fatick", "Kolda", "Matam", "Kaffrine", "Kédougou", "Sédhiou"],
        "zone": ["Ouest", "Ouest", "Nord", "Centre", "Centre", "Sud", "Nord", "Est",
                 "Centre", "Sud", "Nord", "Centre", "Est", "Sud"],
        "superficie_km2": [547, 6601, 19241, 4824, 5357, 7339, 24889, 42706, 7535, 13771, 29445, 11262, 16800, 7341],
    })
    df.to_csv(SORTIE / "regions_senegal.csv", index=False)
    return df


if __name__ == "__main__":
    for f in (ventes_boutiques, meteo_dakar, enquete_menages, resultats_etudiants, clients_mobile_money, referentiel_regions):
        d = f()
        print(f"{f.__name__:<22} -> {d.shape}")
    print(f"Fichiers écrits dans {SORTIE}")
