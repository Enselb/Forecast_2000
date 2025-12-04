
from forecast_2000.utils.data_sample_v2 import get_sample_data, downcast
from forecast_2000.utils.data_full import get_full_data, downcast

def get_data_size():
    print("Choisis la taille du dataset :")
    print("1 - Sample")
    print("2 - Full")

    choix = input("Entrez le numéro du dataset (1-2) : ")

    if choix == "1":
        df = get_sample_data()
        df = downcast(df)
        print("✅ Dataset 1 sélectionné")

    elif choix == "2":
        df = get_full_data()
        df = downcast(df)
        print("✅ Dataset 2 sélectionné")

    else:
            print("❌ Taille non existante")

    return df
