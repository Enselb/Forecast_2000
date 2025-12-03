import numpy as np
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import LSTM,Dense,Input

# Initialiser mon modèle et construire le réseau de neurones
def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize CNN
    """
    input_shape = df.shape
    model = Sequential()# j'instentie mon modèle

    model.add(Input(shape = (input_shape[1],1))) # pas sûr de l'INPUT

    model.add(LSTM(units=10,
                   activation='tanh',
                   return_sequences= False
                   ))
    model.add(Dense(20, activation = "relu"))
    model.add(Dense(10, activation = "relue"))
    model.add(Dense(1, activation = "linear"))

    return model
# Compiler le modèle en adaptant les paramètres à une régression
def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """

    model.compile(loss = "mse", optimizer = "metrics", metrics = ["mae"]) # je compile mon modèle avec les paramètres qui correspondent à la fonction d'activation 'linear'
    return model
# entrainer le modèle en intégrant une variable es et tout mettre dans une variable history
def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=,
        patience=,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    es = EarlyStopping( patience = 20, restore_best_weights=True)

    history = model.fit(X_procesed,
                        y_train,
                        validation_split = 0.2,
                        batch_size = 32,
                        verbose = 1,
                        epotch = 100,
                        callbacks = [es])
    return model, history

# Evaluer mon modèle avec mes X/y_test en recupérant la loss et la mae
def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:

    loss,mae = model.evaluate(X_test,y_test)

    return metrics

