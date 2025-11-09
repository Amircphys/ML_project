import sys
import os
import pytest
import warnings
import pandas as pd
import numpy as np
from typing import List
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from online_inference.main import app

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


warnings.filterwarnings("ignore")

FEATURES = [
    "id",
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    "HeartDisease",
]


@pytest.fixture
def client():
    """Фикстура для создания тестового клиента"""
    return TestClient(app)


@pytest.fixture
def sample_request_data():
    """Фикстура с примером валидных данных для запроса"""
    return {
        "data": [[0, 53, "M", "ATA", 120, 175, 0, "Normal", 120, "Y", 1.0, "Flat", 1]],
        "features": FEATURES,
    }


@pytest.fixture(autouse=True)
def mocke_artefacts():
    """Фикстура для мокирования модели и трансформера (запускается для каждого теста)"""
    with patch("online_inference.main.model") as mock_model, patch(
        "online_inference.main.transformer"
    ) as mock_transformer:
        mock_model.predict_proba.return_value = [0.9]
        mock_transformer.transform.return_value = pd.DataFrame([[1, 2, 3]])

        yield mock_model, mock_transformer


class TestPridictEndpoint:
    def test_predict_success(self, client, sample_request_data):
        with patch("online_inference.main.make_features") as mock_make_features, patch(
            "online_inference.main.predict_probs"
        ) as mock_model_predict:
            mock_make_features.return_value = pd.DataFrame([[1, 2, 3, 4, 5]])
            mock_model_predict.return_value = [85.0]
            response = client.post("/predict/", json=sample_request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["ids"] == [0]
            assert data["probs"] == [85.0]
            assert isinstance(data["ids"], List)
            assert isinstance(data["probs"], List)

    def test_predict_with_different_data(self, client):
        data = [
            [44, 40, "M", "ATA", 105, 175, 0, "Normal", 102, "Y", 1.0, "Flat", 1],
            [45, 53, "F", "NAP", 110, 165, 0, "Normal", 101, "Y", 0.0, "Flat", 0],
            [46, 63, "M", "ASY", 100, 185, 0, "Normal", 110, "Y", 1.0, "Flat", 1],
        ]
        with patch("online_inference.main.make_features") as mock_make_features, patch(
            "online_inference.main.predict_probs"
        ) as mock_model_predict:
            mock_make_features.return_value = pd.DataFrame([[1, 2, 3, 4, 5]])
            mock_model_predict.return_value = [85.0, 91.2, 99.0]
            response = client.post(
                "/predict/", json={"data": data, "features": FEATURES}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["ids"] == [44, 45, 46]
            assert data["probs"] == [85.0, 91.2, 99.0]
            assert isinstance(data["ids"], List)
            assert isinstance(data["probs"], List)

    def test_predict_empty_data(self, client):
        """Тест с пустыми данными"""
        empty_data = {"data": [], "features": []}

        response = client.post("/predict/", json=empty_data)
        data = response.json()
        assert data["ids"] == []
        assert data["probs"] == []
        assert response.status_code == 200


class TestHealthEndpoint:
    """Тесты для /health эндпоинта"""

    def test_health_success(self, client):
        """Тест health эндпоинта когда все ок"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "healthy" in data["message"]

    def test_health_model_not_loaded(self, client):
        """Тест health эндпоинта когда модель не загружена"""
        with patch("online_inference.main.model", None), patch(
            "online_inference.main.transformer", None
        ):
            response = client.get("/health")
            data = response.json()
            assert "unhealthy" in data[0]["message"]
