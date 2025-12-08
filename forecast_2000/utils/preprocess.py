import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_union
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
'''Scikit-learn pipeline that transforms a cleaned dataset to processed features > X_processed which will
be trained in our models'''
# Fonction preprocess regroupant plusieurs fonctions
def processed_features (df : pd.DataFrame):
    # TIME PIPE
    # Mois
    month_sin = FunctionTransformer(lambda df: np.expand_dims((np.sin(2 * np.pi * df["month"].to_numpy() / 12)),axis=-1))
    month_cos = FunctionTransformer(lambda df: np.expand_dims((np.cos(2 * np.pi * df["month"].to_numpy() / 12)),axis=-1))
    # Jour
    dow_sin = FunctionTransformer(lambda df: np.expand_dims((np.sin(2 * np.pi * df["wday"].to_numpy() / 7)),axis=-1))
    dow_cos = FunctionTransformer(lambda df: np.expand_dims((np.cos(2 * np.pi * df["wday"].to_numpy() / 7)),axis=-1))
    print("✅function transformer done")
    # CATEGORICAL PIPE
    cat_transformer = OneHotEncoder(drop='if_binary',
                                    handle_unknown='ignore',
                                    sparse_output=True)
    print("✅cat transformer done")
    # YEAR_PIPE
    year_transformer = make_pipeline(SimpleImputer(), StandardScaler())
    print("✅year transformer done")
    # NUMERICAL PIPE
    num_transformer = make_pipeline(SimpleImputer(), RobustScaler())
    print("✅num transformer done")
    # Appliquer un column transformer pour paralléliser les séquences
    preproc_int = make_column_transformer(
        (num_transformer, ['snap_CA','snap_TX','snap_WI','sell_price']),
        (cat_transformer, ['item_id','store_id','event_name_1']),
        (year_transformer, ['year']),
        remainder='drop'
    )
    print("✅Preproc int done")
    #  Unir le preproc avec les fonctions créées précédemment
    preproc_prefinal = make_union(month_sin, month_cos, dow_sin, dow_cos)
    preproc_final = make_union(preproc_int,preproc_prefinal)
    return preproc_final
    print("✅Preproc final done")

def preprocess_final(X_train, X_val, X_test):
    preprocessor = processed_features(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    del X_train
    X_val_processed   = preprocessor.transform(X_val)
    del X_val
    X_test_processed  = preprocessor.transform(X_test)


    return pd.DataFrame(X_train_processed),pd.DataFrame(X_val_processed),pd.DataFrame(X_test_processed)
