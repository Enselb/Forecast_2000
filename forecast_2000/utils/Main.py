import numpy as np
import pandas as pd
import os

from forecast_2000.utils.data_sample import get_sales_sample
from forecast_2000.utils.clean_data import fill_event_nans
from forecast_2000.utils.split import split_data
from forecast_2000.utils.val_split import val_split
from forecast_2000.utils.preprocess import processed_features
from forecast_2000.utils.Visualisation import visualisation
from forecast_2000.utils.model_selector import model_selector

# Retourne un dataframe des ventes light
df = get_sales_sample()

# Remplace les valeurs NaN par 'NoEvent' dans les colonnes d'événements spécifiées.
df = fill_event_nans(df)

print("✅Chargement du dataset")

# Train / Test Split Function
X_train, X_test, y_train, y_test = split_data(df)
print("✅Data splitted1")

# Pour LightGBM et XGBoost, créer un set de validation avec les 28 dernières valeurs du Train
X_train,y_train, X_val, y_val = val_split(X_train, y_train)
print("✅Data splitted2")

# Pipeline Scikit-learn qui transforme X_train,X_test en X_train_processed et X_test_processed qui seront entraînées dans nos modèles.
preprocessor = processed_features(df)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed   = preprocessor.transform(X_val)
X_test_processed  = preprocessor.transform(X_test)

print("✅X_train_processed :", X_train_processed.shape)
print("✅X_val_processed   :", X_val_processed.shape)
print("✅X_test_processed  :", X_test_processed.shape)

# Sélection du modèle que l'on veut faire tourner
model = model_selector(X_train_processed, X_train, y_train, X_val_processed, y_val, X_val,X_test, X_test_processed, y_test)
y_pred = model(X_test)

# Visualisation des résultats du modèle sélectionné
viz = os.startfile(visualisation(y_test, y_pred), "open")
print("✅Visualisation displayed")
