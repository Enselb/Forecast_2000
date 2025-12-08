from forecast_2000.models.model_RNN import model_LSTM
from forecast_2000.models.model_LightGBM import  train_model_LightGBM,evaluate_and_predict
from forecast_2000.models.model_XGBoost import train_xgboost_model,evaluate_and_predict
from forecast_2000.models.auto_arima import auto_arima


def model_selector(X_train_processed, X_train, X_test, X_test_processed, X_val_processed, X_val, y_train, y_val, y_test):
    print("Choisis le modèle à exécuter :")
    print("1 - RNN_LSTM")
    print("2 - LightGBM")
    print("3 - XGBOOST")
    print("4 - AutoArima")

    choix = input("Entrez le numéro du modèle (1-4) : ")

    if choix == "1":
        model = model_LSTM(X_train_processed, X_val_processed, X_test_processed, y_train,  y_val,  y_test)
        print("✅ Modèle 1 done")

    elif choix == "2":
        model = train_model_LightGBM(X_train, X_val, X_test, y_train, y_val, y_test)
        print("✅ Modèle 2 done")

    elif choix == "3":
        model = train_xgboost_model(X_train, X_val, y_train, y_val)
        y_pred = evaluate_and_predict(X_test)
        print("✅ Modèle 3 done")

    elif choix == "4":
        model = auto_arima(y_train)
        print("✅ Modèle 4 done")

    else:
        print("❌ Modèle non existant")

    return model, y_pred
