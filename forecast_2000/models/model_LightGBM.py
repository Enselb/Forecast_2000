import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np


def model_LightGBM(X_train, y_train, X_val, y_val, X_test, y_test):
    # Convert to Pandas category
    X = [X_train, X_val, X_test]
    for df in X:
        cats = df.select_dtypes(exclude=np.number).columns.tolist()
        for col in cats:
            df[col] = df[col].astype('category')

    # identification des colonnes catégorielles car LightGBM gère mieux les catégories que le One-Hot Encoding
    categorical_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']

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

    lightGBM_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='rmse',
            categorical_feature=categorical_cols,
            callbacks=callbacks
        )

    # Prédictions
    y_pred = lightGBM_model.predict(X_test)

    # Performance
    rmse = mean_squared_error(y_test, y_pred)
    print(f"RMSE sur le test set : {rmse:.4f}")

    return lightGBM_model, y_pred
