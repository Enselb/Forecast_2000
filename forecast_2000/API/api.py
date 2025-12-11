from fastapi import FastAPI
import numpy as np
import pandas as pd
import os
import gc
from forecast_2000.params import *
import joblib
from fastapi.responses import FileResponse
from pathlib import Path
from google.cloud import storage
from forecast_2000.utils.visualisation_promo import visualize_prediction_with_discount


forecast_api = FastAPI()

def load_model() :
    """
    Return a saved model:
    - locally (latest one in alphabetical order)b
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """
    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="LGBM"))
    latest_blob = max(blobs, key=lambda x: x.updated)
    latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH,latest_blob.name)
    latest_blob.download_to_filename(latest_model_path_to_save)

    latest_model = joblib.load(latest_model_path_to_save)

    print(f"✅ Latest model downloaded from cloud storage {latest_blob}")
    return latest_model

forecast_api.state.model = load_model()

def load_X_test() :
    #Load X_test
    file_path = Path(LOCAL_REGISTRY_PATH) / "X_test_20251210.parquet"

    # Chargement du X_train depuis GCS si pas déjà en local et sauvegarde locale
    # EN PARQUET POUR GARDER LES CATEGORIES!!!!
    if not file_path.is_file():
        print("Downloading X_test from cloud storage...")
        X_test = pd.read_csv('gs://forecast_2000_raw_data/X_test_20251210.csv')

        # Les colonnes category ont été perdus pendant la sauvegarde en CSV, on \
        # les reconvertit d'object en category
        for col in X_test.select_dtypes(include=['object']).columns:
            X_test[col] = X_test[col].astype('category')

        X_test.to_parquet(file_path)
        print(f"✅ X_test downloaded and saved locally: {file_path}")
    else:
        print(f"✅ Loading X_test from local file: {file_path}")
        X_test = pd.read_parquet(file_path)

    return X_test

def load_df() :
    #Load X_test
    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="df"))
    latest_blob = max(blobs, key=lambda x: x.updated)
    latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH,latest_blob.name)
    print(f'{latest_model_path_to_save}')
    latest_blob.download_to_filename(latest_model_path_to_save)
    df = pd.read_parquet(latest_model_path_to_save)

    print("✅ DF downloaded from cloud storage")
    return df

X_test=load_X_test()
##if 'rolling_mean_28' in X_test.columns :
    ##X_test= X_test.columns.drop('rolling_mean_28')

# Definition of the end point
@forecast_api.get('/')
def index():
    return {'ok': True}

@forecast_api.get('/predict')
def predict():
    model = forecast_api.state.model
    ##X_test.columns = model.feature_name_
    y_pred = model.predict(X_test.set_index('date',drop=True))

    ##image_path = Path("Plot_Previsions_et_Simulation.png")
    ##if not image_path.is_file():
    ##return {"error": "Image not found on the server"}
    ##return FileResponse(image_path)
    return {"predictions": y_pred.tolist()}

if __name__ == '__main__' :
    predict()
