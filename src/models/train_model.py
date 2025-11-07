from dataclasses import asdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from src.entities import TrainingParams


models = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
}


def train_model(features: np.array, target: np.array, training_params: TrainingParams):

    if training_params.model_type == "RandomForest":
        model = RandomForestClassifier(**asdict(training_params.RandomForestArgs))

    elif training_params.model_type == "CatBoost":
        model = CatBoostClassifier(**asdict(training_params.CatBoostArgs))

    model.fit(features, target)
    return model
