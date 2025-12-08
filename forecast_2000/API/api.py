from fastapi import FastAPI

forecast_api = FastAPI()

# Definition of the end point
@forecast_api.get('/')
def index():
    return {'ok': True}

@forecast_api.get('/predict')
def predict():
    return {'wait': 64}
