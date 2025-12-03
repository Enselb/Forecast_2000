import pandas as pd

def val_split(X_train ,y_train) :
    date_fin = X_train.index.max()
    nb_jours = pd.Timedelta(days=28)
    date_seuil = date_fin - nb_jours
    # train final = du début du train au début du val start exclus
    X_train = X_train.loc[:date_seuil]
    y_train= y_train.loc[:date_seuil]
    # Définition du val set
    X_val = X_train.loc[date_seuil:]
    y_val = y_train.loc[date_seuil:]

    return X_train,y_train,X_val, y_val
