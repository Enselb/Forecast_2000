
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="tqdm")

def features_engineering(df):
    """Crée de nouvelles caractéristiques temporelles pour un DataFrame de ventes.

    Cette fonction enrichit le DataFrame en ajoutant plusieurs types de
    caractéristiques basées sur l'historique des ventes ('sales'), groupées par
    produit ('item_id') et magasin ('store_id').

    Les caractéristiques créées sont :
    - Lags : Valeurs des ventes des jours précédents (J-28, J-35, etc.).
    - Rolling Mean : Moyenne mobile des ventes sur différentes fenêtres de temps.
    - Expanding Mean : Moyenne cumulative des ventes depuis le début de la série.

    Les valeurs NaN générées par ces opérations sont remplacées par 0.

    Args:
        df (pd.DataFrame): DataFrame contenant les données de ventes. Doit inclure
                           les colonnes 'item_id', 'store_id', et 'sales'.

    Returns:
        pd.DataFrame: Le DataFrame d'entrée enrichi avec les nouvelles
                      caractéristiques.
    """

    df = df.drop(columns=['event_name_2', 'event_type_1', 'event_type_2', 'dept_id'])

    # Lags
    lags = [28, 35, 42, 49, 56, 63, 70]
    for lag in tqdm(lags, desc="Création des Lags"):
        col_name = f'lag_{lag}'
        df[col_name] = df.groupby(['item_id', 'store_id'])['sales'].shift(lag)
        df[col_name] = df[col_name].fillna(0)

    # Rolling Window
    # windows = [28]
    # for window in tqdm(windows, desc="Création des Rolling Means"):
    #     col_name = f'rolling_mean_{window}'
    #     df[col_name] = df.groupby(['item_id', 'store_id'])['sales'].transform(lambda s: s.rolling(window).mean())
    #     df[col_name] = df[col_name].fillna(0)

    # Expanding Window
    print("Création de l'Expanding Mean...")
    df['expanding_mean'] = df.groupby(['item_id', 'store_id'])['sales'].transform(lambda s: s.expanding().mean())
    df['expanding_mean'] = df['expanding_mean'].fillna(0)

    # Calcul du prix moyen historique par article
    df['item_avg_price'] = df.groupby('item_id')['sell_price'].transform('mean')

    # Création du Ratio : Prix actuel / Prix moyen
    # Si > 1 : Le produit est plus cher que d'habitude.
    # Si < 1 : Le produit est en "promo".
    df['price_ratio'] = df['sell_price'] / df['item_avg_price']
    df['price_ratio'] = df['price_ratio'].fillna(1.0)

    # Ratio prix vs Catégorie
    df['cat_avg_price'] = df.groupby(['store_id', 'cat_id', 'wm_yr_wk'])['sell_price'].transform('mean')
    df['price_ratio_cat'] = df['sell_price'] / df['cat_avg_price']
    df['price_ratio_cat'] = df['price_ratio_cat'].fillna(1.0)

    # Feature qui regarde si le prix a été changé récemment
    df['price_lag_28'] = df.groupby('item_id')['sell_price'].transform(lambda x: x.shift(28))
    df['price_momentum'] = df['sell_price'] / df['price_lag_28']
    df['price_momentum'] = df['price_momentum'].fillna(1.0)

    print("Feature engineering terminé.")

    return df
