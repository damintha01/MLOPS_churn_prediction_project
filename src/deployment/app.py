# src/deployment/app.py

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import json

# 1. DEFINE THE INPUT DATA MODEL using Pydantic
# This ensures the incoming JSON has the correct structure and types.
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: str # e.g., 'Month-to-month', 'One year', 'Two year'
    PaymentMethod: str # e.g., 'Electronic check', 'Mailed check', etc.
    # ... add all other features your model needs

# 2. INITIALIZE THE FASTAPI APP
app = FastAPI(title="Churn Prediction API", version="1.0")

# 3. LOAD THE MODEL AT STARTUP
# This is more efficient than loading it on every request.
# We need to set the MLflow tracking server to find the model registry.
# For local testing, you might not need this, but in production it's crucial.
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# The model URI points to the latest production model in the registry.
MODEL_URI = "models:/churn-prediction-model/Production"
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 4. CREATE THE PREDICTION ENDPOINT
@app.post("/predict")
def predict_churn(data: CustomerData):
    if not model:
        return {"error": "Model is not loaded."}

    # Convert the incoming Pydantic object to a pandas DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # IMPORTANT: You must apply the SAME preprocessing steps here
    # that you used during training. For simplicity, we are skipping
    # the full feature engineering pipeline here. In a real project,
    # you would load a saved preprocessing pipeline (e.g., a scikit-learn Pipeline)
    # and apply it to `input_df`.
    # For now, we'll assume the input is already preprocessed.
    # This is a simplification for the guide.

    try:
        # Make a prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        # Return the result as JSON
        return {
            "prediction": int(prediction[0]),
            "churn_probability": float(probability[0])
        }
    except Exception as e:
        return {"error": str(e)}

# 5. ADD A ROOT ENDPOINT FOR HEALTH CHECKS
@app.get("/")
def read_root():
    return {"status": "Churn Prediction API is running"}