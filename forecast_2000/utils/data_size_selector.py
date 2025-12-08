
from forecast_2000.utils.data_sample_v2 import get_sample_data
from forecast_2000.utils.data_from_2014 import get_data_from_2014
from forecast_2000.utils.data_full import get_full_data

def get_data_size():
    print("Choisis la taille du dataset :")
    print("1 - Sample")
    print("2 - from_2014")
    print("3 - Full")

    choix = input("Entrez le numéro du dataset (1-3) : ")

    if choix == "1":
        df = get_sample_data()
        print("✅ Dataset 1 sélectionné")

    elif choix == "2":
        df = get_data_from_2014()
        print("✅ Dataset 2 sélectionné")

    elif choix == "3":
        df = get_full_data()
        print("✅ Dataset 3 sélectionné")

    else:
        print("❌ Taille non existante")

    return df
