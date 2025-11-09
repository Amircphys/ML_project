import os
from fastapi import FastAPI
import uvicorn
import logging
from typing import Dict, List, Union
from pydantic import conlist, BaseModel
import pandas as pd
import joblib
from dotenv import load_dotenv

from src.features import make_features
from src.models import model_predict

load_dotenv()

app = FastAPI()

logger = logging.Logger(name=__name__)


class PersonParams(BaseModel):
    data: List[conlist(Union[float, str, None], min_length=13, max_length=13)]
    features: List[str]


class HeartDisease(BaseModel):
    ids: List[int]
    probs: List[float]


model = None
transformer = None


def predict_probs(features, model) -> List[float]:
    probs = model_predict(features=features, model=model, predict_proba=True)[:, 1]
    probs = (probs * 100).tolist()
    probs = [round(x, 2) for x in probs]
    return probs


def make_predict(
    data: List[Union[float, str, None]], features: List[str], model, transformer
):

    df = pd.DataFrame(data, columns=features)
    if df.empty:
        return HeartDisease(ids=[], probs=[])
    features = make_features(df, transformer)
    probs = predict_probs(features=features, model=model)
    return HeartDisease(ids=df["id"].tolist(), probs=probs)


@app.on_event("startup")
def load_artefacts():
    global model
    global transformer
    model_path = os.getenv("MODEL_PATH")
    transformer_path = os.getenv("TRANSFORMER_PATH")
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    if model is None:
        logger.warning(f"Model is None object!!!")
    if transformer is None:
        logger.warning(f"Transformer is None object!!!")


@app.get("/health/")
def health_check():
    check_condition = (model is not None) and (transformer is not None)
    if not check_condition:
        return {
            "status": "error",
            "message": "ML artefacts not loaded, service is unhealthy",
        }, 503
    return {"status": "ok", "message": "Service is healthy"}


@app.get("/")
def start_page():
    return {"message": "Hello! This is entripoint for model prediction"}


@app.post("/predict/")
def predict(request: PersonParams):
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
