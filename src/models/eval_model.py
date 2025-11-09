import os
from typing import Union
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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


def save_model(
    model: LogisticRegression, local_model_save_path: str, s3_model_path: str
):
    with open(local_model_save_path, "wb") as save_file:
        pickle.dump(model, save_file)
    os.system(f"s3cmd put {local_model_save_path} {s3_model_path}")
    return model
