import joblib
from datetime import datetime


def save_model_joblib(model):
    """
    Sauvegarde le modèle entraîné sur le disque avec un nom de fichier simplifié :
    [Modèle]_[Date_Heure].joblib
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Ajout des secondes pour unicité

    # Déduction du nom du modèle (ex: LGBM ou XGB)
    model_name = type(model).__name__.replace('Regressor', '').replace('Classifier', '')

    # Construction du nom de fichier simplifié
    model_filename = (f"{model_name}_{timestamp}.joblib")

    # --- SAUVEGARDE DU MODÈLE ---
    joblib.dump(model, model_filename)
    print(f"Modèle sauvegardé sur le disque : {model_filename}")
