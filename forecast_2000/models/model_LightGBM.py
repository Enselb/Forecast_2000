import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from typing import Tuple
import time
from forecast_2000.models.save_model_local import save_model_joblib


def train_model_LightGBM(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> lgb.LGBMRegressor:
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
    mono_constraints = [0] * len(X_train.columns)
    # Trouver l'index de price_ratio
    if 'price_ratio' in X_train.columns:
        price_idx = X_train.columns.get_loc('price_ratio')
        mono_constraints[price_idx] = -1

    print("Configuration du modèle LightGBM...")
    lightGBM_model = lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.2,
            max_depth=-1,
            objective='tweedie',
            tweedie_variance_power=1.1,
            metric='rmse',
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            monotone_constraints=mono_constraints,
            monotone_constraints_method="basic"
        )

    callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=1)
        ]
    # Enregistrement du temps de départ
    start_time = time.time()

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

    # Enregistrement du temps de fin
    end_time = time.time()

    # Calcul de la durée en secondes
    elapsed_time = end_time - start_time
    # Conversion en heures/minutes si besoin pour M5 (l'entraînement peut être long)
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print("✅ Entrainement terminé")
    print(f"Durée d'entrainement : {minutes} minutes et {seconds} secondes.")

    # Sauvegarde du modèle
    save_model_joblib(lightGBM_model)

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
