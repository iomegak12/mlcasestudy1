import pickle
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI(title="House Price Prediction Service")


class RealEstateProperty(BaseModel):
    posted_by: int
    under_construction: int
    rera: int
    bhk_numbers: int
    bhk_or_rk: int
    sqft: float
    ready_to_move: int
    resale: int


@app.on_event("startup")
def load_model():
    with open('./models/lgbm_model.pkl', 'rb') as file:
        global model

        model = pickle.load(file)


@app.post("/predict")
def predict(property: RealEstateProperty):
    data_point = np.array(
        [
            [
                property.posted_by,
                property.under_construction,
                property.rera,
                property.bhk_numbers,
                property.bhk_or_rk,
                property.sqft,
                property.ready_to_move,
                property.resale
            ]
        ])

    pred = model.predict(data_point).tolist()
    pred = pred[0]

    print(pred)

    return {
        "prediction": np.round(pred, 2)
    }
