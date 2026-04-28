from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import os
import json

from src.predict import forecast_state

app = FastAPI()

MODEL_SUMMARY_PATH = "models/summary.json"


class PredictionRequest(BaseModel):
    state: str

    @validator("state")
    def validate_state(cls, v):
        if not v or not v.strip():
            raise ValueError("State cannot be empty")
        return v.strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    try:
        if not os.path.exists(MODEL_SUMMARY_PATH):
            raise HTTPException(status_code=404, detail="Model summary not found")

        with open(MODEL_SUMMARY_PATH, "r") as f:
            return json.load(f)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        state = request.state

        model_path = f"models/{state}.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found for state: {state}")

        forecast = forecast_state(state)

        if not forecast or len(forecast) == 0:
            raise HTTPException(status_code=500, detail="Forecast generation failed")

        return {
            "state": state,
            "forecast": forecast
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")