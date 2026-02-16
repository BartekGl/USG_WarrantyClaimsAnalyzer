"""
FastAPI Inference Service
Production-ready API with single/batch predictions and SHAP explanations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import shap
import logging
from datetime import datetime
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="USG Failure Prediction API",
    description="Production-grade API for predicting ultrasound device warranty claims with SHAP explanations",
    version="1.0.0"
)

# Global model objects
MODEL = None
PREPROCESSOR = None
SHAP_EXPLAINER = None
FEATURE_NAMES = None


class DeviceData(BaseModel):
    """Single device data schema"""
    Batch_ID: Optional[str] = None
    Assembly_Temp_C: Optional[float] = None
    Humidity_Percent: Optional[float] = None
    Solder_Temp_C: Optional[float] = None
    Solder_Time_s: Optional[float] = None
    Torque_Nm: Optional[float] = None
    Gap_mm: Optional[float] = None
    Region: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "Batch_ID": "BATCH_001",
                "Assembly_Temp_C": 22.5,
                "Humidity_Percent": 45.0,
                "Solder_Temp_C": 350.0,
                "Solder_Time_s": 3.2,
                "Torque_Nm": 2.5,
                "Gap_mm": 0.15,
                "Region": "EU"
            }
        }


class PredictionRequest(BaseModel):
    """Single prediction request"""
    device_data: Dict[str, Any]
    include_shap: bool = Field(default=True, description="Include SHAP explanation")
    threshold: float = Field(default=0.5, description="Classification threshold")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    devices: List[Dict[str, Any]]
    include_shap: bool = Field(default=False, description="Include SHAP explanations")
    threshold: float = Field(default=0.5, description="Classification threshold")


class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: str
    probability: float
    confidence: float
    threshold: float
    shap_values: Optional[Dict[str, float]] = None
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[Dict[str, Any]]
    total_devices: int
    predicted_failures: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    shap_explainer_loaded: bool
    timestamp: str


def load_artifacts():
    """Load model artifacts on startup"""
    global MODEL, PREPROCESSOR, SHAP_EXPLAINER, FEATURE_NAMES

    models_dir = Path("models")

    try:
        # Load model
        model_path = models_dir / "model.pkl"
        if model_path.exists():
            MODEL = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")

        # Load preprocessor
        preprocessor_path = models_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            PREPROCESSOR = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded successfully")
        else:
            logger.warning(f"Preprocessor file not found at {preprocessor_path}")

        # Load SHAP explainer
        shap_explainer_path = models_dir / "shap_explainer.pkl"
        if shap_explainer_path.exists():
            SHAP_EXPLAINER = joblib.load(shap_explainer_path)
            logger.info("SHAP explainer loaded successfully")
        else:
            logger.warning(f"SHAP explainer file not found at {shap_explainer_path}")

        # Load feature names
        feature_names_path = models_dir / "feature_names.json"
        if feature_names_path.exists():
            import json
            with open(feature_names_path, 'r') as f:
                FEATURE_NAMES = json.load(f)
            logger.info(f"Feature names loaded: {len(FEATURE_NAMES)} features")
        else:
            logger.warning(f"Feature names file not found at {feature_names_path}")

    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    logger.info("Starting USG Failure Prediction API...")
    load_artifacts()
    logger.info("API startup complete")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "USG Failure Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        preprocessor_loaded=PREPROCESSOR is not None,
        shap_explainer_loaded=SHAP_EXPLAINER is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single device prediction endpoint
    Returns prediction with optional SHAP explanation
    """
    start_time = datetime.now()

    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. Please check server health."
        )

    try:
        # Convert to DataFrame
        df = pd.DataFrame([request.device_data])

        # Preprocess
        X_processed = PREPROCESSOR.transform(df)

        # Predict probability
        y_proba = MODEL.predict_proba(X_processed)[0]
        probability_failure = float(y_proba[1])

        # Apply threshold
        prediction = "Yes" if probability_failure >= request.threshold else "No"

        # Calculate confidence
        confidence = max(probability_failure, 1 - probability_failure)

        # SHAP explanation
        shap_values_dict = None
        if request.include_shap and SHAP_EXPLAINER is not None:
            try:
                shap_values = SHAP_EXPLAINER.shap_values(X_processed)

                # Get SHAP values for positive class
                if isinstance(shap_values, list):
                    shap_values_positive = shap_values[1][0]
                else:
                    shap_values_positive = shap_values[0]

                # Create feature -> SHAP value mapping
                feature_names = X_processed.columns.tolist()
                shap_values_dict = {
                    feature: float(value)
                    for feature, value in zip(feature_names, shap_values_positive)
                }

                # Sort by absolute impact
                shap_values_dict = dict(
                    sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
                )

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}")
                shap_values_dict = {"error": "SHAP explanation unavailable"}

        # Log inference time
        inference_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Inference completed in {inference_time*1000:.2f}ms")

        return PredictionResponse(
            prediction=prediction,
            probability=probability_failure,
            confidence=confidence,
            threshold=request.threshold,
            shap_values=shap_values_dict,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint
    Process multiple devices efficiently
    """
    start_time = datetime.now()

    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. Please check server health."
        )

    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.devices)

        # Preprocess
        X_processed = PREPROCESSOR.transform(df)

        # Predict probabilities
        y_proba = MODEL.predict_proba(X_processed)
        probabilities = y_proba[:, 1]

        # Apply threshold
        predictions = ["Yes" if p >= request.threshold else "No" for p in probabilities]

        # Build response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                "device_index": i,
                "prediction": pred,
                "probability": float(prob),
                "confidence": float(max(prob, 1 - prob))
            }

            # Add SHAP if requested
            if request.include_shap and SHAP_EXPLAINER is not None:
                try:
                    shap_values = SHAP_EXPLAINER.shap_values(X_processed.iloc[[i]])

                    if isinstance(shap_values, list):
                        shap_values_positive = shap_values[1][0]
                    else:
                        shap_values_positive = shap_values[0]

                    feature_names = X_processed.columns.tolist()
                    shap_dict = {
                        feature: float(value)
                        for feature, value in zip(feature_names, shap_values_positive)
                    }

                    result["shap_values"] = dict(
                        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                    )
                except Exception as e:
                    logger.warning(f"SHAP failed for device {i}: {str(e)}")

            results.append(result)

        # Count failures
        predicted_failures = sum(1 for p in predictions if p == "Yes")

        # Log batch inference time
        inference_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Batch inference ({len(request.devices)} devices) completed in {inference_time:.2f}s")

        return BatchPredictionResponse(
            predictions=results,
            total_devices=len(request.devices),
            predicted_failures=predicted_failures,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = {
        "model_type": type(MODEL).__name__,
        "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else "unknown",
        "shap_available": SHAP_EXPLAINER is not None,
        "timestamp": datetime.now().isoformat()
    }

    return info


@app.get("/features")
async def get_features():
    """Get list of model features"""
    if FEATURE_NAMES is None:
        raise HTTPException(status_code=503, detail="Feature names not loaded")

    return {
        "features": FEATURE_NAMES,
        "count": len(FEATURE_NAMES),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting USG Failure Prediction API server...")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
