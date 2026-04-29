from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import os
import json

from src.predict import forecast_state, load_model_file

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_SUMMARY_PATH = "models/summary.json"


class PredictionRequest(BaseModel):
    state: str

    @field_validator("state")
    def validate_state(cls, v):
        if not v or not v.strip():
            raise ValueError("State cannot be empty")
        return v.strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    if not os.path.exists(MODEL_SUMMARY_PATH):
        raise HTTPException(status_code=404, detail="Model summary not found")

    try:
        with open(MODEL_SUMMARY_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        state = req.state.strip().title()

        artifact = load_model_file(state)
        model_type = artifact["model_type"]

        forecast = forecast_state(state)

        return {
            "state": state,
            "model": model_type,
            "forecast": forecast
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="State model not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")