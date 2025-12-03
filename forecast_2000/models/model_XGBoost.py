from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

def model_XGB(X_train, y_train, X_val, y_val, X_test, y_test):
    # Convert to Pandas category
    X = [X_train, X_val, X_test]
    for df in X:
        cats = df.select_dtypes(exclude=np.number).columns.tolist()
        for col in cats:
            df[col] = df[col].astype('category')


    XGB_model = XGBRegressor(
        max_depth=10, n_estimators=500,
        learning_rate=0.1,
        objective='reg:tweedie',
        tweedie_variance_power=1.1,
        tree_method='hist',
        enable_categorical=True,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=5,
        eval_metric='rmse'
    )

    XGB_model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=1,
    )

    # Prédictions
    y_pred = XGB_model.predict(X_test)

    # Performance
    rmse = mean_squared_error(y_test, y_pred)
    print(f"RMSE sur le test set : {rmse:.4f}")

    return XGB_model, y_pred
