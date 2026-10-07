"""Fonctions d'audit de qualité des données."""
import pandas as pd


def profil(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau de profilage : une ligne par colonne (type, manquants, modalités, exemple)."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_manquants": df.isna().sum(),
        "pct_manquants": (df.isna().mean() * 100).round(2),
        "n_uniques": df.nunique(),
        "exemple": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    }).sort_values("pct_manquants", ascending=False)


def outliers_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Règle de Tukey : True si hors de [Q1 - k·IQR, Q3 + k·IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def outliers_mad(s: pd.Series, seuil: float = 3.5) -> pd.Series:
    """Score z robuste (médiane / MAD), Iglewicz & Hoaglin (1993)."""
    med = s.median()
    mad = (s - med).abs().median()
    return ((s - med).abs() / (1.4826 * mad)) > seuil
