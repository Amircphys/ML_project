import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.entities.feature_params import FeatureParams


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder())]
    )
    return categorical_pipeline


def build_numercial_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )
    return numerical_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    categorical_pipeline = build_categorical_pipeline()
    numerical_pipeline = build_numercial_pipeline()
    transformer = ColumnTransformer(
        [
            ("categorical_pipeline", categorical_pipeline, params.categorical_columns),
            ("numeric_pipeline", numerical_pipeline, params.numerical_columns),
        ],
        remainder="drop",
    )
    return transformer


def make_features(df: pd.DataFrame, transformer: ColumnTransformer) -> np.array:
    features = transformer.transform(df)
    return features


def extract_target(df: pd.DataFrame, params: FeatureParams) -> np.array:
    target = df[params.target_column].values
    return target
