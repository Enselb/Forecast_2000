import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from typing import Tuple


def train_model_LightGBM(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
    """
    Entraîne un modèle LightGBM et le retourne.

    Le modèle est entraîné avec un early stopping basé sur un jeu de validation.
    Les colonnes de type 'category' dans X_train sont automatiquement gérées par LightGBM.

    Args:
        X_train: DataFrame des features d'entraînement.
        y_train: Series de la cible d'entraînement.
        X_val: DataFrame des features de validation.
        y_val: Series de la cible de validation.

    Returns:
        Le modèle LGBMRegressor entraîné.
    """
    print("Configuration du modèle LightGBM...")
    lightGBM_model = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.05,
            objective='tweedie',
            tweedie_variance_power=1.1,
            metric='rmse',
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )

    callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    print("Entraînement du modèle LightGBM...")
    lightGBM_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            #eval_metric='rmse',
            #categorical_feature=categorical_cols,
            # Le paramètre 'categorical_feature' est inutile car LightGBM
            # détecte automatiquement les colonnes de type 'category'.
            callbacks=callbacks
        )

    return lightGBM_model

def evaluate_and_predict(model: lgb.LGBMRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """
    Évalue le modèle sur le jeu de test et retourne les prédictions.
    Évalue le modèle sur un jeu de test, affiche le RMSE et retourne les prédictions.
    """
    # Prédictions
    print("Génération des prédictions sur le jeu de test...")
    y_pred = model.predict(X_test)

    # Performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ RMSE sur le jeu de test : {rmse:.4f}")

    return y_pred
