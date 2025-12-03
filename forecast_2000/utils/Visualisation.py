import matplotlib.pyplot as plt
import numpy as np
def visualisation(y_train,y_test,y_pred):
    #Construction des axes temporels
    x_train = np.arange(len(y_train)) # nous permet de pouvoir placer le nombre de point que contient y_train sur l'axe du temps en abscisse
    x_test = np.arange(len(y_train), len(y_train) + len(y_test)) #


    window = 500 # on crée une variable window pour
    plt.figure(figsize=(12, 6))
    plt.plot(x_train[-window:],y_train[-window:],label="Train",color="g",alpha=0.7)# courbe train
    plt.plot(x_test,y_test,color="g",linewidth=2) #courbe des valeurs réelle
    plt.plot(x_test,y_pred,label="Prédiction",color="b",linewidth=2,alpha=0.8) # courbe des prédictions

    plt.title("Prédiction vs Réalité")
    plt.xlabel("Jours")
    plt.ylabel("Ventes")
    plt.legend()
    plt.grid()
    plt.savefig()
    plt.show()
