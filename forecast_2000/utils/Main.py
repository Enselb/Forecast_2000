import numpy as np
import pandas as pd
import os
import gc
from pathlib import Path


from forecast_2000.utils.data_size_selector import get_data_size
from forecast_2000.utils.split import split_data
from forecast_2000.utils.val_split import val_split
from forecast_2000.utils.preprocess import preprocess_final
from forecast_2000.utils.model_selector import model_selector
from forecast_2000.utils.Visualisation import visualisation

# Retourne un dataframe des ventes ventes sélectionnées
chemin = '~/code/Enselb/Forecast_2000/data'
data_path = Path(os.path.expanduser(chemin))
# Taille du dataset
print("Sélectionne la taille du dataset :")
print("1 - Sample")
print("2 - from_2014")
print("3 - Full")
choix = input("Entrez le numéro du dataset (1-3) : ")

# Choix du nom du fichier parquet
mapping = {
    "1": "sample.parquet",
    "2": "from_2014.parquet",
    "3": "full.parquet"
}

file_path = data_path / mapping[choix]
print("✅Nom du fichier :", file_path)

# Chargement du dataset
if not file_path.is_file():
    df = get_data_size()
    df.to_parquet(file_path)
    print("✅Fichier créé :", file_path)
else:
    print("✅Chargement de fichier car existant :", file_path)
    df = pd.read_parquet(file_path)

# Train / Test Split Function
X_train_val,X_test,y_train_val,y_test = split_data(df)
print("✅Data splitted1")

# Libération de la mémoire
del df
gc.collect()

# Pour LightGBM et XGBoost, créer un set de validation avec les 28 dernières valeurs du Train
X_train, X_val, y_train, y_val = val_split(X_train_val, y_train_val)
print("✅Data splitted2")

# Pipeline Scikit-learn qui transforme X_train,X_test en X_train_processed et X_test_processed qui seront entraînées dans nos modèles.
X_train_processed, X_val_processed, X_test_processed = preprocess_final(X_train, X_val, X_test)

print("✅X_train_processed :", X_train_processed.shape)
print("✅X_val_processed   :", X_val_processed.shape)
print("✅X_test_processed  :", X_test_processed.shape)

# Sélection du modèle que l'on veut faire tourner
model, y_pred = model_selector(X_train_processed, X_train, X_test, X_test_processed, X_val_processed, X_val, y_train, y_val, y_test)

# Visualisation des résultats du modèle sélectionné
viz = visualisation(y_test, y_pred)
print("✅Visualisation displayed")
