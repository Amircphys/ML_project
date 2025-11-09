import requests
import numpy as np
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("./data/raw/heart.csv")
    df["id"] = range(df.shape[0])
    df = df[
        [
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
    ]
    features = list(df.columns)
    for i in range(0, 50, 5):
        request_data = [
            [x.item() if isinstance(x, np.generic) else x for x in df.iloc[k].tolist()]
            for k in range(i, i + 5)
        ]

        response = requests.post(
            "http://0.0.0.0:8000/predict/",
            json={"data": request_data, "features": features},
        )
        print(response.status_code)
        print(response.json())
