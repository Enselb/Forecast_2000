import pmdarima as pm

def auto_arima(y_train):

    # Construction et entrainement du modèle auto ARIMA

    smodel = pm.auto_arima(y_train,
                        start_p=0, max_p=3,
                        start_q=0, max_q=0,
                        trend='t',
                        seasonal=False,
                        trace=True)

    return smodel
