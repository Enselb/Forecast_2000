import numpy as np
import pandas as pd
import os

from forecast_2000.utils.data_size_selector import get_data_size
from forecast_2000.utils.split import split_data
from forecast_2000.utils.val_split import val_split
from forecast_2000.utils.preprocess import preprocess_final
from forecast_2000.utils.model_selector import model_selector
from forecast_2000.utils.Visualisation import visualisation

# Retourne un dataframe des ventes ventes sélectionnées
df = get_data_size()

print("✅Chargement du dataset")

# Train / Test Split Function
X_train, X_test, y_train, y_test = split_data(df)
print("✅Data splitted1")

# Pour LightGBM et XGBoost, créer un set de validation avec les 28 dernières valeurs du Train
X_train, X_val, y_train, y_val = val_split(X_train, y_train)
print("✅Data splitted2")

# Pipeline Scikit-learn qui transforme X_train,X_test en X_train_processed et X_test_processed qui seront entraînées dans nos modèles.
X_train_processed, X_val_processed, X_test_processed = preprocess_final(X_train, X_val, X_test)

print("✅X_train_processed :", X_train_processed.shape)
print("✅X_val_processed   :", X_val_processed.shape)
print("✅X_test_processed  :", X_test_processed.shape)

# Sélection du modèle que l'on veut faire tourner
model = model_selector(X_train_processed, X_train, y_train, X_val_processed, y_val, X_val,X_test, X_test_processed, y_test)
y_pred = model(X_test)

# Visualisation des résultats du modèle sélectionné
viz = os.startfile(visualisation(y_test, y_pred), "open")
print("✅Visualisation displayed")
