"""
Zar3y — FastAPI Backend (Requirement 5)

REST API for crop disease prediction using the quantized TFLite model.

Endpoints:
- GET  /health          → Health check
- POST /predict         → Accept image, return prediction + Grad-CAM

Run with: uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import io
import base64
import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────
# Add project root to sys.path so we can import src.*
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.inference import TFLitePredictor
from src.settings import DISEASE_INFO, DISPLAY_NAMES

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Zar3y API",
    description="Crop disease detection from leaf photos — powered by MobileNetV3-Small (TFLite INT8)",
    version="1.0.0",
)

# Allow Streamlit (or any frontend) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor — loaded once at startup
predictor: TFLitePredictor | None = None


# ─────────────────────────────────────────────
# Startup: load the TFLite model
# ─────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    """Load the TFLite model into memory on startup."""
    global predictor
    predictor = TFLitePredictor()
    print("TFLite model loaded and ready.")


# ─────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check — returns model status."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
    }


# ─────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded leaf photo.
    Return predicted disease class, confidence, description,
    next-step suggestion, and Grad-CAM overlay as base64.
    Must return in < 500 ms on CPU.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got {file.content_type}",
        )

    t0 = time.perf_counter()

    # Read raw bytes
    image_bytes = await file.read()

    # Run TFLite prediction
    class_name, confidence, probs = predictor.predict(image_bytes)

    # Generate Grad-CAM overlay
    overlay_bgr = predictor.generate_gradcam(image_bytes)
    # Encode overlay to base64 PNG
    _, buffer = cv2.imencode(".png", overlay_bgr)
    grad_cam_base64 = base64.b64encode(buffer).decode("utf-8")

    # Look up disease info
    info = DISEASE_INFO.get(class_name, {})
    description = info.get("description", "No description available.")
    next_step = info.get("next_step", "Consult a local agricultural extension agent.")
    display_name = DISPLAY_NAMES.get(class_name, class_name)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return JSONResponse(
        content={
            "class_name": class_name,
            "display_name": display_name,
            "confidence": round(confidence, 4),
            "description": description,
            "next_step": next_step,
            "grad_cam_base64": grad_cam_base64,
            "inference_ms": round(elapsed_ms, 1),
        }
    )
