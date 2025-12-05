import pandas as pd
"Cette fonctiion split X_train_val et X_train et X_val"
def val_split(X_train_val ,y_train_val) :
    date_fin = X_train_val.index.max()
    nb_jours = pd.Timedelta(days=28)
    date_seuil = date_fin - nb_jours
    date_debut = date_fin - pd.Timedelta(days=27)
    # train final = du début du train au début du val start exclus
    X_train = X_train_val.loc[:date_seuil]
    y_train = y_train_val.loc[:date_seuil]
    # Définition du val set
    X_val = X_train_val.loc[date_debut:]
    y_val = y_train_val.loc[date_debut:]

    return X_train,X_val,y_train,y_val
