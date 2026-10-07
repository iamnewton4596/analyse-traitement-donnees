"""Fonctions de nettoyage du jeu « ventes_boutiques » (chapitre 4)."""
import numpy as np
import pandas as pd


def vers_nombre(s: pd.Series) -> pd.Series:
    """Extrait un nombre d'une chaîne du type '6 500 FCFA' ou '6500.0'."""
    nettoye = (s.astype("str")
               .str.replace(r"[^\d,.\-]", "", regex=True)
               .str.replace(",", ".", regex=False))
    return pd.to_numeric(nettoye, errors="coerce")


def vers_date(s: pd.Series, formats=("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")) -> pd.Series:
    """Convertit en essayant successivement plusieurs formats explicites."""
    resultat = pd.Series(pd.NaT, index=s.index, dtype="datetime64[us]")
    for fmt in formats:
        manque = resultat.isna()
        resultat[manque] = pd.to_datetime(s[manque], format=fmt, errors="coerce")
    return resultat


def normaliser_texte(s: pd.Series) -> pd.Series:
    return s.str.strip().str.replace(r"\s+", " ", regex=True).str.title()


def nettoyer_ventes(brut: pd.DataFrame, referentiel_regions: pd.Series):
    """Nettoie le jeu de ventes ; renvoie (DataFrame propre, journal des étapes)."""
    journal = []
    df = brut.copy()
    n = len(df)
    df = df.drop_duplicates()
    journal.append(f"doublons exacts supprimés : {n - len(df)}")
    df["prix_unitaire"] = vers_nombre(df["prix_unitaire"])
    df["date"] = vers_date(df["date"])
    journal.append(f"dates non converties : {df['date'].isna().sum()}")
    df["region"] = normaliser_texte(df["region"]).replace({"St-Louis": "Saint-Louis"})
    inconnues = ~df["region"].isin(referentiel_regions) & df["region"].notna()
    journal.append(f"régions hors référentiel : {inconnues.sum()}")
    mauvais = (df["quantite"] <= 0) | (df["quantite"] > 50)
    df.loc[mauvais, "quantite"] = np.nan
    journal.append(f"quantités invalides -> NaN : {mauvais.sum()}")
    manq = df["quantite"].isna()
    df["quantite"] = df["quantite"].fillna(df.groupby("produit")["quantite"].transform("median"))
    df["quantite_imputee"] = manq
    journal.append(f"quantités imputées (médiane par produit) : {manq.sum()}")
    df["mode_paiement"] = df["mode_paiement"].fillna("Inconnu")
    df["montant"] = df["prix_unitaire"] * df["quantite"]
    df = df.astype({"region": "category", "categorie": "category",
                    "mode_paiement": "category", "quantite": "int64"})
    return df.sort_values("date").reset_index(drop=True), journal
