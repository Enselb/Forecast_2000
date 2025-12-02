import pandas as pd
import numpy as np


def fill_event_nans(dataframe):
    """
    Remplace les valeurs NaN par 'NoEvent' dans les colonnes d'événements spécifiées.

    Args:
        dataframe (pd.DataFrame): DataFrame contenant les colonnes d'événements.

    Returns:
        DataFrame avec les NaN remplacés.
    """
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    for col in event_cols:
        if col in dataframe.columns:
            # Remplacement ciblé uniquement si la colonne contient des NaN
            if dataframe[col].isnull().any():
                dataframe[col] = dataframe[col].fillna('NoEvent')
    return dataframe
