from datetime import datetime

import torch
from fastapi import FastAPI

from model import BurnoutModel
from preprocess import prepare_input
from schemas import EmployeeData, PredictionResponse

app = FastAPI(
    title="Employee Burnout Prediction API",
    description="Predict employee burnout using a trained PyTorch Neural Network.",
    version="1.0.0"
)

# --------------------------------------------------
# Load Model Once
# --------------------------------------------------

model = BurnoutModel()

model.load_state_dict(
    torch.load(
        "employee_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "🚀 Employee Burnout Prediction API is Running",
        "version": "1.0.0",
        "docs": "/docs"
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeData):

    input_tensor = prepare_input(employee)

    with torch.no_grad():

        prediction = model(input_tensor)

    probability = prediction.item()

    prediction_label = (
        "Burnout"
        if probability >= 0.5
        else "No Burnout"
    )

    confidence = (
        probability
        if probability >= 0.5
        else (1 - probability)
    )

    return PredictionResponse(
        success=True,
        prediction=prediction_label,
        probability=round(probability, 4),
        confidence=round(confidence * 100, 2),
        model_version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )