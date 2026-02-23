from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from typing import Optional

from .utils import feature_engineering_pipeline

class TransactionData(BaseModel):
    amount: float
    description: str
    payment_method: str
    status: str
    balance_before: float
    balance_after: float
    sender_id: int
    sender_name: str
    sender_email: str
    sender_phone: str
    sender_dob: str # Assuming date as string for simplicity in API
    sender_gender: str
    sender_country: str
    sender_state: str
    sender_city: str
    sender_upi_id: str
    sender_location_allowed: bool
    sender_bank_holder: str
    sender_bank_name: str
    sender_account_no: str
    sender_ifsc: str
    sender_initial_balance: float
    receiver_id: int
    receiver_name: str
    receiver_upi: str
    receiver_country: str
    receiver_state: str
    receiver_city: str
    receiver_location_allowed: bool
    receiver_bank_holder: str
    receiver_bank_name: str
    receiver_account_no: str
    receiver_ifsc: str
    transaction_device_id: str
    device_platform: str
    device_browser: str
    device_user_agent: str
    device_created_at: str # Assuming datetime as string for simplicity in API
    device_last_used_at: str # Assuming datetime as string for simplicity in API
    device_owner_user_id: int
    transaction_created_at: str # Assuming datetime as string for simplicity in API
    transaction_payment_time: str # Assuming time as string for simplicity in API

class PredictionResult(BaseModel):
    is_fraud: bool
    fraud_probability: float

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "xgboost_fraud_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"XGBoost model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Make sure to train and save the model first.")
    model = None # Or raise an exception to prevent app from starting
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

feature_engineering_pipeline.load_artifacts()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResult)
async def predict_fraud(data: TransactionData):
    if model is None:
        return PredictionResult(is_fraud=False, fraud_probability=0.0) # Or return an error response

    df_raw = pd.DataFrame([data.dict()])

    df_engineered = feature_engineering_pipeline.preprocess_new_data(df_raw)

    fraud_probability = model.predict_proba(df_engineered)[:, 1][0]
    is_fraud = bool(model.predict(df_engineered)[0])

    return PredictionResult(is_fraud=is_fraud, fraud_probability=fraud_probability)
