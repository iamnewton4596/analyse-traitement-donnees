# Mémo pandas (version 3.x)

*Cours « Analyse et traitement des données » — Issa Gueye*

## Lire / écrire
```python
df = pd.read_csv("f.csv", sep=";", decimal=",", thousands=" ", encoding="utf-8",
                 na_values=["NSP", "-"], dtype={"code": str}, parse_dates=["date"], dayfirst=True)
df = pd.read_excel("f.xlsx", sheet_name=None)      # toutes les feuilles -> dict
df = pd.json_normalize(liste_de_dicts, sep="_")     # JSON imbriqué -> table
df.to_parquet("f.parquet", index=False)             # stockage recommandé (types conservés)
```

## Inspecter
| Code | Rôle |
|---|---|
| `df.shape`, `df.info()`, `df.dtypes` | dimensions, types, non-nuls |
| `df.head()`, `df.sample(5)` | aperçu |
| `df.describe()`, `df.describe(include="str")` | résumés num. / texte |
| `df.isna().sum()`, `df.isna().mean()` | manquants |
| `df.duplicated().sum()`, `s.is_unique` | doublons |
| `s.value_counts(normalize=True)`, `s.nunique()` | modalités |

## Sélectionner
```python
df["col"]; df[["a", "b"]]
df.loc[lignes_etiquettes, colonnes]      # fin incluse
df.iloc[0:5, 0:3]                        # positions, fin exclue
df[(df.a > 5) & (df.b == "x")]           # & | ~ et parenthèses
df[df.region.isin(["Dakar", "Thiès"])]
df.query("a > 5 and b == 'x'")
```

## Modifier (pandas 3 : Copy-on-Write)
```python
df.loc[df.x > 5, "col"] = 0              # ✅ toujours .loc
df["col"][df.x > 5] = 0                  # ❌ sans effet en pandas 3
df = df.assign(z=pd.col("x") / pd.col("y"))     # pandas ≥ 3.0
df = df.assign(z=lambda d: d.x / d.y)           # compatible pandas 2
df = df.rename(columns={"ancien": "nouveau"}).drop(columns=["inutile"])
df["c"] = np.where(df.x > 0, "pos", "neg"); np.select([c1, c2], ["a", "b"], default="c")
df["classe"] = pd.cut(df.x, bins=[0, 10, 20, np.inf], labels=["bas", "moyen", "haut"])
df["tercile"] = pd.qcut(df.x, 3, labels=["T1", "T2", "T3"])
```

## Nettoyer
```python
df = df.drop_duplicates(); df.drop_duplicates(subset=["id"], keep="last")
pd.to_numeric(s.str.replace(r"[^\d,.\-]", "", regex=True).str.replace(",", "."), errors="coerce")
pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
s.str.strip().str.lower().str.title(); s.str.replace(r"\s+", " ", regex=True); s.replace({"St-Louis": "Saint-Louis"})
s.fillna(s.median()); s.fillna(df.groupby("g")["x"].transform("median")); s.interpolate(method="time", limit=5)
s.clip(lower=s.quantile(.01), upper=s.quantile(.99))     # winsorisation
df.astype({"region": "category", "n": "Int64"})
```

## Agréger / restructurer
```python
df.groupby("g", observed=True).agg(n=("id", "count"), ca=("montant", "sum"), med=("montant", "median"))
df.groupby("g")["x"].transform("mean")          # même taille que df
df.groupby("g").filter(lambda d: len(d) > 30)   # garder des groupes
pd.pivot_table(df, values="x", index="a", columns="b", aggfunc="sum", margins=True)
pd.crosstab(df.a, df.b, normalize="index")
df.melt(id_vars="id", var_name="variable", value_name="valeur")      # large -> long
long.pivot(index="id", columns="variable", values="valeur")          # long -> large
a.merge(b, on="cle", how="left", validate="many_to_one", indicator=True)
pd.concat([df1, df2], ignore_index=True)
```

## Séries temporelles
```python
s = df.set_index("date")["x"]
s.resample("ME").sum()        # D, W, ME (fin de mois), QE, YE
s.rolling(7, center=True).mean(); s.shift(1); s.diff(); s.pct_change(); s.cumsum()
df.asfreq("D")                # réinsère les dates manquantes
s.dt.year, s.dt.month, s.dt.day_name(), s.dt.quarter
```

## Nouveautés pandas 3.0 à retenir
- Type texte dédié **`str`** par défaut (plus d'`object`).
- **Copy-on-Write** systématique : plus de `SettingWithCopyWarning`, l'affectation en chaîne ne fonctionne plus.
- `pd.col("x")` dans `assign`.
- Dates en résolution **microseconde** par défaut (`datetime64[us]`).
- Fonctionnalités dépréciées supprimées : migrer d'abord vers pandas 2.3 sans avertissement.
