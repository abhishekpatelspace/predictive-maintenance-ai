from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load(
    "saved_models/xgb_model.pkl"
)

class TelemetryInput(BaseModel):
    SoC: float
    SoH: float
    Battery_Voltage: float
    Battery_Current: float
    Battery_Temperature: float
    Motor_Vibration: float
    Motor_RPM: float
    Power_Consumption: float
    Load_Weight: float
    Driving_Speed: float
    Component_Health_Score: float

@app.post("/predict")
def predict(data: TelemetryInput):

    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]

    probability = (
        model.predict_proba(input_df)[0][1]
    )

    return {
        "failure_prediction": int(prediction),
        "failure_probability": float(probability)
    }
