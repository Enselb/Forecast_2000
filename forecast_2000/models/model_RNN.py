import numpy as np
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import LSTM,Dense,Input
from sklearn.metrics import mean_squared_error

"""
    Build, compile, train and return the model + training history.
"""

def model_LSTM(X_train_processed, X_val_processed,X_test_processed, y_train, y_val, y_test):

    LSTM_model = Sequential()
    LSTM_model.add(Input(shape=(X_train_processed.shape[1], 1)))

    LSTM_model.add(LSTM(units=10, activation='tanh', return_sequences=False))
    LSTM_model.add(Dense(20, activation="relu"))
    LSTM_model.add(Dense(10, activation="relu"))
    LSTM_model.add(Dense(1, activation="linear"))


    LSTM_model.compile(
        loss="mse",
        optimizer="adam",
        metrics=["mae"]
    )

    es = EarlyStopping(
        patience=20,
        restore_best_weights=True
    )


    history = LSTM_model.fit(
        X_train_processed,
        y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val_processed, y_val),
        verbose=1,
        callbacks=[es]
    )

    # Prédictions
    y_pred = LSTM_model.predict(X_test_processed)

    # Performance
    rmse = mean_squared_error(y_test, y_pred)
    print(f"RMSE sur le test set : {rmse:.4f}")

    return LSTM_model, history, y_pred
