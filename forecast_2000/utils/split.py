import pandas as pd

## Train / Test Split Function
#Début de  la fonction
def split_data(df) :
    #Convert Date in Datetime + Set New Date into Index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['date'],inplace = True)

    #Définition of the numbers of days for the train limit
    date_fin = df.index.max()
    nb_jours = pd.Timedelta(days=28)
    date_seuil = date_fin - nb_jours

    #Création des Df
    df_train = df.loc[:date_seuil]
    df_test = df.loc[date_seuil:]

    #Définiton X et Y Train
    X_train = df_train.drop(columns=['sales'])
    y_train = df_train['sales']

    #Définition X et Y Test
    X_test= df_test.drop(columns =['sales'])
    y_test =df_test['sales']

    return X_train,X_test,y_train,y_test
