from forecast_2000.models.model_RNN import model_LSTM
from forecast_2000.models.model_LightGBM import  model_LightGBM
from forecast_2000.models.model_XGBoost import model_XGB
from forecast_2000.models.auto_arima import auto_arima


def model_selector(X_train_processed, X_train, y_train, X_val_processed, y_val, X_val,X_test, X_test_processed, y_test):
    print("Choisis le modèle à exécuter :")
    print("1 - RNN_LSTM")
    print("2 - LightGBM")
    print("3 - XGBOOST")
    print("4 - AutoArima")

    choix = input("Entrez le numéro du modèle (1-4) : ")

    if choix == "1":
        model = model_LSTM(X_train_processed, y_train, X_val_processed, y_val, X_test_processed, y_test)
        print("✅ Modèle 1 done")

    elif choix == "2":
        model = model_LightGBM(X_train, y_train, X_val, y_val, X_test, y_test)
        print("✅ Modèle 2 done")

    elif choix == "3":
        model = model_XGB(X_train, y_train, X_val, y_val, X_test, y_test)
        print("✅ Modèle 3 done")

    elif choix == "4":
        model = auto_arima(y_train)
        print("✅ Modèle 4 done")

    else:
        print("❌ Modèle non existant")

    return model
