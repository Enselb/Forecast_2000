## Train / Test Split Function
#Début de  la fonction
def split_data(df) :
    #Convert Date in Datetime + Set New Date into Index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['date'],inplace = True)

    #Définition of the numbers of days for the train limit
    date_debut = df.index.min()
    nb_jours = pd.Timedelta(days=28)
    date_seuil = date_debut + nb_jours

    #Création des Df
    df_train = df.loc[:date_seuil]
    df_test = df.loc[date_seuil:]

    #Définiton X et Y Train
    X_train = df_train.drop(columns=['value'])
    y_train = df_train['value']

    #Définition X et Y Test
    X_test= df_test.drop(columns =['value'])
    y_test =df_test['value']

    return X_train,X_test,y_train,y_test
