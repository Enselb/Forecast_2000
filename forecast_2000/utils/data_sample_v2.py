import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def get_sample_data():
    """Retourne un dataframe des ventes avec les informations
    prix et évènements calendaires

    """
    # client = storage.Client(project='forecast2000')

    # download fichiers calendar, sales et prices du bucket Google Cloud et
    # sauvegarde dans le répertoire 'data'
    # print("Download des fichiers CSV...")
    # bucket = client.bucket('forecast_2000_raw_data')
    # blob = bucket.blob('calendar.csv')
    # blob.download_to_filename('data/calendar.csv')

    # blob = bucket.blob('sales_train_evaluation.csv')
    # blob.download_to_filename('data/sales_train_evaluation.csv')

    # blob = bucket.blob('sell_prices.csv')
    # blob.download_to_filename('data/sell_prices.csv')

    # # Lecture des csv en DataFrame
    print("Lecture des fichiers CSV...")
    chemin = '~/code/Enselb/Forecast_2000/data'
    data_path = os.path.expanduser(chemin)
    calendar = pd.read_csv(data_path + '/calendar.csv')
    sales = pd.read_csv(data_path + '/sales_train_evaluation.csv')
    prices = pd.read_csv(data_path + '/sell_prices.csv')

    # Remplir les NaN des colonnes d'événements avant le downcasting
    print("Nettoyage des valeurs manquantes pour les événements...")
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in event_cols:
        if col in calendar.columns:
            calendar[col] = calendar[col].fillna('NoEvent')

    # Fonction de réduction de la taille mémoire d'un dataframe
    def downcast(df):
        """
        Réduit l'utilisation de la mémoire du DataFrame en optimisant les types
        numériques (int, float) et en convertissant les chaînes de caractères
        (object) en 'category'.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print(f"Utilisation mémoire initiale : {start_mem:.2f} MB")

        cols = df.dtypes.index.tolist()
        types = df.dtypes.values.tolist()

        # tqdm est utilisé pour suivre la progression de l'itération sur les colonnes
        for i, t in tqdm(enumerate(types), total=len(types), desc="Downcasting"):
            col = cols[i]

            if 'int' in str(t):
                # Traitement des entiers
                if df[col].min() > np.iinfo(np.int8).min and df[col].max() < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif df[col].min() > np.iinfo(np.int16).min and df[col].max() < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif df[col].min() > np.iinfo(np.int32).min and df[col].max() < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            elif 'float' in str(t):
                # Traitement des flottants
                if df[col].min() > np.finfo(np.float16).min and df[col].max() < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

            elif t == object:
                # Traitement des objets
                if col == 'date':
                    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
                else:
                    # Conversion en type 'category' du reste des colonnes
                    df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Utilisation mémoire finale : {end_mem:.2f} MB")
        print(f"Réduction de la mémoire de {(100 * (start_mem - end_mem) / start_mem):.2f}%\n")

        return df

    print("Optimisation des fichiers...")
    # Reduction de la taille des fichiers
    fichiers = [calendar, sales, prices]

    for fichier in fichiers:
        downcast(fichier)

    print("Transposition des ventes...")
    # Transposition des ventes et drop des valeurs manquantes
    df = pd.melt(
        sales,
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='day',
        value_name='sales').dropna()

    # Renommage colonne du calendrier pour le merge
    calendar.rename(columns={'d': 'day'}, inplace=True)

    # Merge des fichiers
    print("Merge des fichiers...")
    df = pd.merge(df, calendar, how='left', on='day')
    df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # drop des colonnes
    df = df.drop(columns=['id', 'day', 'wm_yr_wk'])

    # Echantillonnage du fichier des ventes
    print("Echantillonnage...")
    sub_cat = 'HOBBIES_2'
    year = 2016
    store = 'CA_1'
    df = df[(df['dept_id'] == sub_cat) & (df['store_id'] == store) & (df['year'] == year)]

    print("✅ Execution terminée.")

    return df
