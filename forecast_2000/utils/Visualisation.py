import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecast_2000.utils.split import split_data
from forecast_2000.models.model_XGBoost import evaluate_and_predict_xgboost
def visualisation(model, df, produit, date_de_début, date_de_fin, magasin):

    # On garde uniquement le produit, le magasin et la période choisis
    df_produit = df.loc[
        (df['item_id'] == produit) &
        (df['store_id'] == magasin) &
        (df['date'] >= date_de_début) &
        (df['date'] <= date_de_fin)].copy()

    # Split train / test
    X_train, X_test, y_train, y_test = split_data(df_produit)

    # Prédiction
    model, y_pred = evaluate_and_predict_xgboost(model, X_test, y_test)

    # Conversion en numérique
    y_train = pd.to_numeric(y_train, errors="coerce")
    y_test = pd.to_numeric(y_test, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")

    # Récupération des dates
    dates_train = X_train.index
    dates_test = X_test.index

    window = 8  # on prend comme base 8 jours

    # Pour faciliter le traitement, on transforme en Series
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    y_pred_series = pd.Series(y_pred)

    # Moyenne mobile du train
    y_train_mobile = y_train_series.rolling(window=window).mean()

    # On récupère les derniers points du train pour pouvoir combler le troue dans le test et pred
    last_train = y_train_series.iloc[-(window - 1):]

    # Moyenne mobile du test
    test_concat = pd.concat([last_train, y_test_series])
    y_test_mobile = test_concat.rolling(window=window).mean().iloc[window - 1:]

    # Moyenne mobile prédiction
    pred_concat = pd.concat([last_train, y_pred_series])
    y_pred_mobile = pred_concat.rolling(window=window).mean().iloc[window - 1:]

    # on plot notre graphique
    plt.figure(figsize=(14, 6))

    #Les données brut
    plt.plot(dates_train, y_train, label="Train", color="green", alpha=0.5)
    plt.plot(dates_test, y_test, label="Test", color="blue", alpha=0.5)
    plt.plot(dates_test, y_pred, label="Prédiction", color="red", alpha=0.5)

    # Les moyennes mobiles
    plt.plot(dates_train, y_train_mobile, label="Moyenne mobile du Train", color="black")
    plt.plot(dates_test, y_test_mobile.values, label="Moyenne mobile du Test réel", color="purple")
    plt.plot(dates_test, y_pred_mobile.values, label="Moyenne mobile de la Prédiction", color="orange")

    plt.title("Prédiction vs Réalité")
    plt.xlabel("Date")
    plt.ylabel("Ventes")
    plt.legend()
    plt.grid()
    plt.show()
