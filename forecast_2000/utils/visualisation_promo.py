import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns

def visualize_prediction_with_discount(model, df, item_id=None, store_id=None, state_id=None,
                         split_date='2016-04-24', history_window=115,
                         price_change_pct=None, sim_start_date=None, sim_end_date=None):
    """
    Visualise les ventes réelles vs prédites avec agrégation flexible et simulation de prix.
    Met à jour automatiquement 'price_ratio' lors de la simulation.
    """
    feature_cols = model.feature_name_
    # --- 0. Sécurité : Gestion des arguments inversés (Compatibilité) ---
    # Si l'utilisateur appelle l'ancienne signature (model, df, item_id, feature_cols)
    # feature_cols recevra l'ID (int/str) et item_id recevra la liste des features.
    # if isinstance(feature_cols, (int, str)) and isinstance(item_id, list):
    #     print("⚠️ Attention : Arguments inversés détectés. Correction automatique...")
    #     feature_cols, item_id = item_id, feature_cols

    split_date = pd.to_datetime(split_date)
    if sim_start_date: sim_start_date = pd.to_datetime(sim_start_date)
    if sim_end_date: sim_end_date = pd.to_datetime(sim_end_date)

    # 1. Construction du masque
    mask = pd.Series(True, index=df.index)
    filters_desc = []

    if item_id is not None:
        mask = mask & (df['item_id'] == item_id)
        filters_desc.append(f"Item: {item_id}")
    if store_id is not None:
        mask = mask & (df['store_id'] == store_id)
        filters_desc.append(f"Magasin: {store_id}")
    if state_id is not None and 'state_id' in df.columns:
        mask = mask & (df['state_id'] == state_id)
        filters_desc.append(f"État: {state_id}")

    if not filters_desc: filters_desc.append("Vision Globale")

    data_subset = df[mask].copy()

    if data_subset.empty:
        print(f"Aucune donnée trouvée pour les filtres : {filters_desc}")
        return

    if not isinstance(data_subset.index, pd.DatetimeIndex):
        try:
            data_subset.index = pd.to_datetime(data_subset.index)
        except Exception:
            print("Erreur: L'index doit être temporel.")
            return

    # 2. Séparation
    test_mask = data_subset.index > split_date
    if not test_mask.any():
        print("Pas de données futures (Test) disponibles.")
        return

    # --- BASELINE ---
    # Vérification que toutes les colonnes features existent
    missing_cols = [c for c in feature_cols if c not in data_subset.columns]
    if missing_cols:
        print(f"Erreur : Colonnes manquantes dans le dataframe : {missing_cols}")
        return

    X_test = data_subset.loc[test_mask, feature_cols].copy()
    try:
        data_subset.loc[test_mask, 'pred_sales'] = model.predict(X_test)
    except Exception as e:
        print(f"Erreur baseline: {e}")
        return

    # --- SIMULATION ---
    is_simulating = price_change_pct is not None and sim_start_date is not None and sim_end_date is not None

    if is_simulating:
        print(f"Simulation : Prix {price_change_pct:+.0%} du {sim_start_date.date()} au {sim_end_date.date()}...")
        X_test_sim = X_test.copy()
        sim_mask = (X_test_sim.index >= sim_start_date) & (X_test_sim.index <= sim_end_date)

        simulation_possible = False

        # 1. Mise à jour du prix brut (si présent)
        if 'sell_price' in X_test_sim.columns:
            X_test_sim.loc[sim_mask, 'sell_price'] *= (1 + price_change_pct)
            simulation_possible = True

        # 2. Mise à jour du price_ratio (si présent)
        if 'price_ratio' in X_test_sim.columns:
            X_test_sim.loc[sim_mask, 'price_ratio'] *= (1 + price_change_pct)
            simulation_possible = True

        if simulation_possible:
            data_subset.loc[test_mask, 'pred_sales_sim'] = model.predict(X_test_sim)
            data_subset.loc[~test_mask, 'pred_sales_sim'] = np.nan
        else:
            print("Attention : Ni 'sell_price' ni 'price_ratio' trouvés dans les features. Simulation ignorée.")
            is_simulating = False

    data_subset.loc[~test_mask, 'pred_sales'] = np.nan

    # 3. Agrégation
    cols_to_agg = ['sales', 'pred_sales']
    if is_simulating: cols_to_agg.append('pred_sales_sim')

    agg_data = data_subset.groupby(level=0)[cols_to_agg].sum()

    start_plot = split_date - pd.Timedelta(days=history_window)
    plot_data = agg_data[agg_data.index > start_plot]

    # 4. Viz
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))

    train_part = plot_data[plot_data.index <= split_date]
    test_part = plot_data[plot_data.index > split_date]

    sns.lineplot(data=train_part, x=train_part.index, y='sales', label='Historique', color='#1f77b4')
    sns.lineplot(data=test_part, x=test_part.index, y='sales', label='Réel', color='green', alpha=0.5)
    sns.lineplot(data=test_part, x=test_part.index, y='pred_sales', label='Prévision Initiale', color='orange', linestyle='--')

    if is_simulating:
        sns.lineplot(data=test_part, x=test_part.index, y='pred_sales_sim', label=f'Scénario ({price_change_pct:+.0%})', color='red', linewidth=2.5)
        plt.axvspan(sim_start_date, sim_end_date, color='red', alpha=0.1)

    plt.axvline(x=split_date, color='black', linestyle=':', alpha=0.5)
    plt.title("Prévisions & Simulation - " + " / ".join(filters_desc), fontsize=16)
    plt.show()
