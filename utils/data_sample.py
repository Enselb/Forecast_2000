import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account

def get_sales_sample():
    """Retourne un dataframe des ventes light

    """
    credentials = service_account.Credentials.from_service_account_file(
        '/home/bertrand/code/Enselb/Forecast_2000/forecast2000-ec89eb4db84e.json')

    client = storage.Client(project='forecast2000',
        credentials=credentials)

    # download fichiers calendar, sales et prices du bucket Google Cloud et
    # sauvegarde dans le répertoire 'data'
    bucket = client.bucket('forecast_2000_raw_data')
    blob = bucket.blob('calendar.csv')
    blob.download_to_filename('data/calendar.csv')

    blob = bucket.blob('sales_train_evaluation.csv')
    blob.download_to_filename('data/sales_train_evaluation.csv')

    blob = bucket.blob('sell_prices.csv')
    blob.download_to_filename('data/sell_prices.csv')

    # lecture des csv en DataFrame
    calendar = pd.read_csv('data/calendar.csv')
    sales = pd.read_csv('data/sales_train_evaluation.csv')
    prices = pd.read_csv('data/sell_prices.csv')

    fichiers = [calendar, sales, prices]


    def memory_size_reduction(dataframe):
        """
        Réduit l'utilisation de la mémoire d'un DataFrame en optimisant les types de données.

        Cette fonction parcourt toutes les colonnes du DataFrame qui sont de type
        'int64' ou 'float64'. Pour chacune de ces colonnes, elle tente de convertir
        (downcast) les données vers le plus petit type d'entier non signé ('unsigned')
        possible qui peut contenir toutes les valeurs de la colonne.

        Par exemple, une colonne 'int64' contenant uniquement des valeurs positives
        entre 0 et 200 sera convertie en 'uint8', ce qui réduit considérablement
        l'espace mémoire utilisé.

        Args:
            dataframe (pd.DataFrame): Le DataFrame à optimiser.

        Returns:
            pd.DataFrame: Le DataFrame avec les types de données numériques optimisés.

        """

        for col in dataframe.select_dtypes(include=['int64', 'float64']).columns:
            dataframe[col] = pd.to_numeric(dataframe[col], downcast='unsigned')

        return dataframe

    # Reduction de la taille des fichiers
    fichiers = [calendar, sales, prices]


    for fichier in fichiers:
        fichier = memory_size_reduction(fichier)

    # Echantillonnage du fichier des ventes
    sub_cat = 'HOBBIES_2'
    store = 'CA_1'
    sales_sample = sales[(sales['dept_id'] == sub_cat) & (sales['store_id'] == store)]

    # Transposition des ventes
    # Id colonnes infos
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    valeurs = []
    for colonne in sales_sample.columns:
        if colonne.startswith('d_'):
            valeurs.append(colonne)
    valeurs

    sales_sample_long = sales_sample.melt(id_vars, valeurs)

    # Renommage des colonnes pour le merge
    calendar.rename(columns={'d': 'day'}, inplace=True)
    sales_sample_long.rename(columns={'variable': 'day'}, inplace=True)
    sales_sample_long.rename(columns={'value': 'sales'}, inplace=True)

    # Merge des fichiers
    sales_temp = sales_sample_long.merge(calendar, how='left', on='day')
    sales_light_full = pd.merge(sales_temp, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # 2eme échantillonage sur l'année
    year = 2016
    sales_light_full = sales_light_full[sales_light_full['year'] == year]

    return sales_light_full
