def val_split(df) :
    #Définition du nombre de jours pour le val set
    val_horizon = 28
    # Nombre total d'observations dans le train
    n = len(X_train)
    # Position où commence la validation
    val_start = n - val_horizon
    # Définition du val set
    X_val = X_train.iloc[val_start : n]
    y_val = y_train.iloc[val_start : n]
    # train final = du début du train au début du val start exclus
    X_train = X_train.iloc[0 : val_start]
    y_train= y_train.iloc[0 : val_start]

    return X_train,y_train,X_val, y_val
