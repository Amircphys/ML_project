import os
from typing import Union
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from src.entities import TrainingPipelineParams

sklearn_models = Union[LogisticRegression, RandomForestClassifier]


def model_predict(
    features: list,
    model: Union[sklearn_models, CatBoostClassifier],
    predict_proba: bool = False,
) -> list:
    if predict_proba:
        return model.predict_proba(features)
    return model.predict(features)


def evaluate_model(predict: list[float], target: list[float]) -> pd.DataFrame:
    accuracy = accuracy_score(predict, target)
    f1 = f1_score(predict, target)  # .item()
    roc_auc = roc_auc_score(predict, target)  # .item()
    print(f"accuracy: {accuracy}, f1: {f1}, roc_auc: {roc_auc}")
    return pd.DataFrame(
        {
            "accuracy": [round(accuracy, 3)],
            "f1_score": [round(f1, 3)],
            "roc_auc": [round(roc_auc, 3)],
        }
    )


def save_artefacts(
    model: LogisticRegression,
    transformer: ColumnTransformer,
    pipeline_params: TrainingPipelineParams,
):
    joblib.dump(model, pipeline_params.local_model_save_path)
    joblib.dump(transformer, pipeline_params.local_transformer_path)
    os.system(
        f"s3cmd put {pipeline_params.local_model_save_path} {pipeline_params.s3_path_model}"
    )
    os.system(
        f"s3cmd put {pipeline_params.local_transformer_path} {pipeline_params.s3_path_transformer}"
    )
    return model
