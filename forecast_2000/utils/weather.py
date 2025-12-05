import pandas as pd
import numpy as np
import os


def add_weather_data(df):
    """
    Charge les données météo, les optimise et les fusionne avec le DataFrame d'entrée.
    """
    chemin = '~/code/Enselb/Forecast_2000/data'
    data_path = os.path.expanduser(chemin)
    weather = pd.read_csv(data_path + '/weather_by_state.csv')

    weather['time'] = pd.to_datetime(weather['time'])
    weather.rename(columns={'time': 'date'}, inplace=True)

    # Downcasting pour optimiser la mémoire
    for col in weather.columns:
        dtype = weather[col].dtype

        if 'float' in str(dtype):
            min_val, max_val = weather[col].min(), weather[col].max()
            if min_val > np.finfo(np.float16).min and max_val < np.finfo(np.float16).max:
                weather[col] = weather[col].astype(np.float32) # on se limite à float 32 pour le preprocessing (sparsing)
            elif min_val > np.finfo(np.float32).min and max_val < np.finfo(np.float32).max:
                weather[col] = weather[col].astype(np.float32)

        elif dtype == 'object':
            weather[col] = weather[col].astype('category')

    # Fusionne avec le dataframe principal
    df_merged = pd.merge(df, weather, on=['state_id', 'date'], how='left')
    
    return df_merged
