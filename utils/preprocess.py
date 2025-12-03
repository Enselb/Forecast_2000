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
    month_sin = FunctionTransformer(lambda df: pd.DataFrame(np.sin(2 * np.pi * df["month"] / 12)))
    month_cos = FunctionTransformer(lambda df: pd.DataFrame(np.cos(2 * np.pi * df["month"] / 12)))

    # Jour
    dow_sin = FunctionTransformer(lambda df: pd.DataFrame(np.sin(2 * np.pi * df["wday"] / 7)))
    dow_cos = FunctionTransformer(lambda df: pd.DataFrame(np.cos(2 * np.pi * df["wday"] / 7)))

    # CATEGORICAL PIPE
    cat_transformer = OneHotEncoder(drop='if_binary',
                                    handle_unknown='ignore',
                                    sparse_output=False)
    cat_col = make_column_selector(dtype_include=['object','bool'])

    # YEAR_PIPE
    year_transformer = make_pipeline(SimpleImputer(), StandardScaler())

    # NUMERICAL PIPE
    num_transformer = make_pipeline(SimpleImputer(), RobustScaler())

    # Appliquer un column transformer pour paralléliser les séquences
    preproc_int = make_column_transformer(
        (num_transformer, ['wm_yr_wk','snap_CA','snap_TX','snap_WI','sell_price']),
        (cat_transformer, cat_col),
        (year_transformer, ['year']),
        remainder='passthrough'
    )

    #  Unir le preproc avec les fonctions créées précédemment
    preproc_prefinal = make_union(month_sin, month_cos, dow_sin, dow_cos)
    preproc_final = make_union(preproc_int,preproc_prefinal)
    return preproc_final

def preprocess_final(X_train : pd.DataFrame):

    preprocessor = processed_features(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)

    print("✅ X_processed, with shape", X_train_processed.shape)

    return pd.DataFrame(X_train_processed)
