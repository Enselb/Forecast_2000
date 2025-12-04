from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Tuple

def train_xgboost_model(X_train, y_train, X_val, y_val) -> XGBRegressor:
    """
    Entraîne un modèle XGBoost et le retourne.

    Note: S'assure que les colonnes catégorielles sont du bon type.
    Cette étape devrait idéalement être faite en amont dans le pipeline de pré-traitement.
    """
    # # Conversion des types pour XGBoost (si pas déjà fait)
    # X = [X_train, X_val]
    # X = [X_train, X_val, X_test]
    # for df in X:
    #     cats = df.select_dtypes(exclude=np.number).columns.tolist()
    #     for col in cats:
    #         # Utiliser .copy() pour éviter les SettingWithCopyWarning
    #         df.loc[:, col] = df[col].astype('category')


    XGB_model = XGBRegressor(
        max_depth=10, n_estimators=1000,
        learning_rate=0.1,
        objective='reg:tweedie',
        tweedie_variance_power=1.1,
        tree_method='hist',
        enable_categorical=True,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )

    print("Entraînement du modèle XGBoost...")
    XGB_model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=1,
    )

    return XGB_model

def evaluate_and_predict(model: XGBRegressor, X_test, y_test) -> Tuple[XGBRegressor, np.ndarray]:
    """
    Évalue le modèle sur le jeu de test et retourne les prédictions.
    """
    # Prédictions
    print("Génération des prédictions sur le jeu de test...")
    y_pred = model.predict(X_test)

    # Performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ RMSE sur le jeu de test : {rmse:.4f}")

    return model, y_pred
