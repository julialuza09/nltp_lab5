import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI(
    title="NTPD ML API",
    description="API do serwowania modelu ML",
    version="1.0.0"
)

# przygotowanie danych i modelu ML
X = np.array([
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 2],
    [5, 2],
    [6, 3],
    [7, 3],
    [8, 4]
])

y = np.array([10, 14, 18, 25, 29, 36, 40, 47])

model = LinearRegression()
model.fit(X, y)

APP_ENV_NAME = os.getenv("APP_ENV_NAME", "local")


class PredictionInput(BaseModel):
    feature1: float = Field(..., description="pierwsza cecha")
    feature2: float = Field(..., description="druga cecha")


class PredictionOutput(BaseModel):
    prediction: float
    environment: str


@app.get("/")
def read_root():
    return {
        "message": "API dziala poprawnie",
        "environment": APP_ENV_NAME
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": APP_ENV_NAME
    }


@app.get("/info")
def info():
    return {
        "model_type": "LinearRegression",
        "n_features": 2,
        "intercept": float(model.intercept_),
        "coefficients": model.coef_.tolist(),
        "environment": APP_ENV_NAME
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    try:
        features = np.array([[data.feature1, data.feature2]])
        prediction = model.predict(features)[0]

        return {
            "prediction": float(prediction),
            "environment": APP_ENV_NAME
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd predykcji: {str(e)}")