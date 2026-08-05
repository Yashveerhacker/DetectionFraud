from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and scaler on startup
    try:
        artifacts["model"] = joblib.load("fraud_model.pkl")
        artifacts["scaler"] = joblib.load("scaler.pkl")
    except FileNotFoundError:
        raise RuntimeError("Model files not found. Run train.py first.")
    yield
    artifacts.clear()

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

class Transaction(BaseModel):
    features: list[float] = Field(..., min_items=30, max_items=30)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: Transaction):
    if "model" not in artifacts or "scaler" not in artifacts:
        raise HTTPException(status_code=500, detail="Model assets not loaded")

    values = np.array(transaction.features).reshape(1, -1)
    scaled_values = artifacts["scaler"].transform(values)

    prediction = artifacts["model"].predict(scaled_values)[0]
    probabilities = artifacts["model"].predict_proba(scaled_values)[0]

    return {
        "prediction": "Fraud" if prediction == 1 else "Normal",
        "fraud_probability": float(probabilities[1]),
        "is_fraud": bool(prediction == 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)