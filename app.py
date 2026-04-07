import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
import numpy as np
import redis

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

# konfiguracja Redis z zmiennych środowiskowych
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


class PredictionInput(BaseModel):
    feature1: float = Field(..., description="pierwsza cecha, np. liczba godzin")
    feature2: float = Field(..., description="druga cecha, np. liczba projektów")


class PredictionOutput(BaseModel):
    prediction: float


@app.get("/")
def read_root():
    return {"message": "API dziala poprawnie"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model_type": "LinearRegression",
        "n_features": 2,
        "intercept": float(model.intercept_),
        "coefficients": model.coef_.tolist()
    }


@app.get("/redis-health")
def redis_health():
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        pong = client.ping()
        return {
            "redis_status": "connected",
            "ping": pong
        }
    except Exception as e:
        return {
            "redis_status": "disconnected",
            "error": str(e)
        }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    try:
        if data.feature1 is None or data.feature2 is None:
            raise HTTPException(
                status_code=400,
                detail="Brak wartości: feature1 lub feature2"
            )

        features = np.array([[data.feature1, data.feature2]])
        prediction = model.predict(features)[0]

        return {"prediction": float(prediction)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd predykcji: {str(e)}")