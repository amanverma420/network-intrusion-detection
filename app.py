import os
import sys

# Quiet TensorFlow initialization logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf  # Explicitly defined to fix the NameError

app = FastAPI(title="Network Intrusion Detection API")

print("⏳ Initializing pipeline assets...")

# Load all pipeline components globally on startup
encoders = joblib.load("models/encoders.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
cnn_model = tf.keras.models.load_model("models/cnn_model.keras")

class NetworkTrafficInput(BaseModel):
    features: dict

@app.post("/predict")
def predict_intrusion(payload: NetworkTrafficInput):
    try:
        input_df = pd.DataFrame([payload.features])
        input_df = input_df.reindex(columns=feature_columns)
        
        for col, encoder in encoders.items():
            if col in input_df.columns:
                val = input_df.iloc[0][col]
                if val not in encoder.classes_:
                    raise HTTPException(status_code=400, detail=f"Unknown label '{val}' for column '{col}'")
                input_df[col] = encoder.transform([val])[0]
        
        scaled_features = scaler.transform(input_df)
        
        # Random Forest Prediction
        rf_prediction_id = int(rf_model.predict(scaled_features)[0])
        
        # CNN Prediction
        cnn_features = np.expand_dims(scaled_features, axis=-1)
        cnn_probabilities = cnn_model.predict(cnn_features, verbose=0)[0]
        cnn_prediction_id = int(np.argmax(cnn_probabilities))
        
        target_encoder = encoders["connection"]
        rf_label = target_encoder.inverse_transform([rf_prediction_id])[0]
        cnn_label = target_encoder.inverse_transform([cnn_prediction_id])[0]
        
        return {
            "random_forest_prediction": rf_label,
            "cnn_prediction": cnn_label,
            "cnn_confidence": float(np.max(cnn_probabilities))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))