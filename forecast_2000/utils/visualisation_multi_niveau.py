import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns

def visualize_prediction(model, df, item_id=None, cat_id=None, store_id=None, state_id=None,
                         split_date='2016-04-24', history_window=115):
    """
    Visualise les ventes réelles vs prédites avec agrégation flexible.

    Niveaux d'agrégation possibles :
    1. Produit spécifique : Spécifier `item_id` (et `store_id` optionnel)
    2. Magasin entier : Spécifier `store_id` (laisser `item_id=None`)
    3. État entier : Spécifier `state_id` (laisser les autres None)
    4. Global (National) : Tout laisser à None

    Args:
        model: Le modèle LightGBM entraîné.
        df: Le dataframe complet (indexé par date).
        feature_cols: Liste des colonnes utilisées pour la prédiction (Obligatoire).
        item_id: L'identifiant du produit (Optionnel).
        store_id: L'identifiant du magasin (Optionnel).
        state_id: L'identifiant de l'état (Optionnel).
        split_date: La date de coupure Train/Test.
        history_window: Jours d'historique à afficher.
    """

    feature_cols = model.feature_name_
    # Conversion sécurisée de split_date
    split_date = pd.to_datetime(split_date)

    # 1. Construction du masque de filtrage
    # On commence par un masque True partout
    mask = pd.Series(True, index=df.index)
    filters_desc = []

    if item_id is not None:
        mask = mask & (df['item_id'] == item_id)
        filters_desc.append(f"Item: {item_id}")

    if cat_id is not None:
        mask = mask & (df['cat_id'] == cat_id)
        filters_desc.append(f"Catégorie: {cat_id}")

    if store_id is not None:
        mask = mask & (df['store_id'] == store_id)
        filters_desc.append(f"Magasin: {store_id}")

    if state_id is not None:
        if 'state_id' in df.columns:
            mask = mask & (df['state_id'] == state_id)
            filters_desc.append(f"État: {state_id}")
        else:
            print("Attention: Colonne 'state_id' introuvable pour le filtrage.")

    # Titre par défaut si aucun filtre (Vision Globale)
    if not filters_desc:
        filters_desc.append("Vision Globale (Tous produits/magasins)")

    # Application du filtre sur une copie
    data_subset = df[mask].copy()

    if data_subset.empty:
        print(f"Aucune donnée trouvée pour les filtres : {filters_desc}")
        return

    # S'assurer que l'index est bien au format Datetime
    if not isinstance(data_subset.index, pd.DatetimeIndex):
        try:
            data_subset.index = pd.to_datetime(data_subset.index)
        except Exception:
            print("Erreur: L'index du DataFrame ne peut pas être converti en date.")
            return

    # 2. Séparation Train / Test
    test_mask = data_subset.index > split_date

    if not test_mask.any():
        print("Attention: Pas de données de test (futur) disponibles pour cette sélection.")
        return

    # Prédiction sur la partie Test (uniquement sur les données filtrées pour gagner du temps)
    print(f"Génération des prédictions pour l'agrégation : {' / '.join(filters_desc)}...")
    X_test = data_subset.loc[test_mask, feature_cols]

    try:
        data_subset.loc[test_mask, 'pred_sales'] = model.predict(X_test)
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return

    # Sur l'historique, on met NaN pour ne pas tracer de prédiction
    data_subset.loc[~test_mask, 'pred_sales'] = np.nan

    # 3. Agrégation Temporelle
    # On groupe par Date (index) et on somme les ventes réelles et prédites
    print("Agrégation des données...")
    agg_data = data_subset.groupby(level=0)[['sales', 'pred_sales']].sum()

    # Filtrage fenêtre historique pour l'affichage
    start_plot = split_date - pd.Timedelta(days=history_window)
    plot_data = agg_data[agg_data.index > start_plot]

    # 4. Visualisation Seaborn
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(15, 6))

    train_part = plot_data[plot_data.index <= split_date]
    test_part = plot_data[plot_data.index > split_date]

    # Tracé des courbes
    sns.lineplot(data=train_part, x=train_part.index, y='sales', label='Historique', color='#1f77b4', linewidth=1.5)
    sns.lineplot(data=test_part, x=test_part.index, y='sales', label='Réel', color='green', linewidth=1.5)
    sns.lineplot(data=test_part, x=test_part.index, y='pred_sales', label='Prévision', color='red', linestyle='--', linewidth=2)

    plt.axvline(x=split_date, color='black', linestyle=':', alpha=0.5, label='Début Prévision')

    # Titre dynamique
    title_str = "Prévisions Aggrégées - " + " / ".join(filters_desc)

    plt.title(title_str, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Ventes Totales', fontsize=12)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    plt.show()
