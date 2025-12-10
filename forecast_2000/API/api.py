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

    print("✅ Latest model downloaded from cloud storage")
    return latest_model

forecast_api.state.model = load_model()

def load_X_test() :
    #Load X_test
    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="X_test"))
    latest_blob = max(blobs, key=lambda x: x.updated)
    latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH,latest_blob.name)
    print(f'{latest_model_path_to_save}')
    latest_blob.download_to_filename(latest_model_path_to_save)

    X_test_preproc = pd.read_csv(latest_model_path_to_save)

    print("✅ X_test preproc downloaded from cloud storage")
    return X_test_preproc

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

##X_test_preproc = load_X_test()

# Definition of the end point
@forecast_api.get('/')
def index():
    return {'ok': True}

@forecast_api.get('/visu_test')
def visualise():
    model = forecast_api.state.model
    df = load_df()
    visualize_prediction_with_discount(model, df, item_id=None, store_id=None, state_id=None,
                         split_date='2016-04-24', history_window=115,
                         price_change_pct=None, sim_start_date=None, sim_end_date=None)
    image_path = Path("/Plot Prévisions & Simulation.png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)
