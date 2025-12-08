""" ce fichier sera dédié au feature engineering et permettra d'ajouter des colonnes à partir des données temporelles déjà
incluses dans le dataset"""

#  Moyenne mobile des ventes sur des fenêtres de 28 et 365 jours en partant des jours précédents
def add_rolling_means(df, windows=[28, 365]):
    group_cols = ["item_id","dept_id","cat_id","store_id","state_id"]
    for w in windows:
        df[f"rolling_mean_{w}"] = df.groupby(group_cols)["sales"].shift(1).rolling(w).mean().fillna(0)
    return df


#  Ventes décalées sur des lags de 28 jours et 56 jours
def add_lag_features(df, lags=[28, 56]):
    group_cols = ["item_id","dept_id","cat_id","store_id","state_id"]
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)["sales"].shift(lag).fillna(0)
    return df
