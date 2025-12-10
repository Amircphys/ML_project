import os
from fastapi import FastAPI
import uvicorn
import logging
from typing import List, Union
from pydantic import conlist, BaseModel
import pandas as pd
import joblib
from dotenv import load_dotenv


load_dotenv()
os.makedirs("./models", exist_ok=True)
app = FastAPI()
logger = logging.Logger(name=__name__)


LOCAL_MODEL_PATH = "./models/model.pkl"
LOCAL_TRANSFORMER_PATH = "./models/transformer.pkl"


class PersonParams(BaseModel):
    data: List[conlist(Union[float, str, None], min_length=13, max_length=13)]
    features: List[str]


class HeartDisease(BaseModel):
    ids: List[int]
    probs: List[float]


model = None
transformer = None


def predict_probs(features, model) -> List[float]:
    if len(features) == 0 or len(features[0]) == 0:
        return []
    probs = model.predict_proba(features)[:, 1]
    probs = (probs * 100).tolist()
    probs = [round(x, 2) for x in probs]
    return probs


def make_features(data: List[Union[float, str, None]], columns: List[str], transformer):
    df = pd.DataFrame(data, columns=columns)
    if df.empty:
        return ([], [])
    features = transformer.transform(df)
    ids = df["id"].tolist()
    return ids, features


def make_predict(
    data: List[Union[float, str, None]], columns: List[str], model, transformer
):
    ids, features = make_features(data, columns, transformer)
    probs = predict_probs(features=features, model=model)
    return HeartDisease(ids=ids, probs=probs)


def load_from_s3(s3_path: str, local_path: str, object_name: str):
    if not os.path.exists(local_path):
        logger.info(f"Start load {object_name} from s3...")
        os.system(f"s3cmd get {s3_path} {local_path}")


@app.on_event("startup")
def load_artefacts():
    global model
    global transformer

    s3_model_path = os.getenv("S3_MODEL_PATH")
    s3_transformer_path = os.getenv("S3_TRANSFORMER_PATH")
    load_from_s3(
        s3_path=s3_model_path, local_path=LOCAL_MODEL_PATH, object_name="model"
    )
    load_from_s3(
        s3_path=s3_transformer_path,
        local_path=LOCAL_TRANSFORMER_PATH,
        object_name="transformer",
    )

    model = joblib.load(LOCAL_MODEL_PATH)
    transformer = joblib.load(LOCAL_TRANSFORMER_PATH)
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
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=8000)
