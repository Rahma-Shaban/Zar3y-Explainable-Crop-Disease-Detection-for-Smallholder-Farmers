"""
Zar3y — FastAPI Backend (Requirement 5)

REST API for crop disease prediction using the model.
"""

import base64
import time
import cv2
import os
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -----------------------------
# FIX: make project root visible to Python
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from src.inference import TFLitePredictor
from src.settings import DISEASE_INFO, DISPLAY_NAMES


app = FastAPI(
    title="Zar3y API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model once
predictor = TFLitePredictor()


# -------------------------
# Health check
# -------------------------
@app.get("/health")
async def health():
    return {"status": "healthy"}


# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # validate image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    start = time.perf_counter()

    image_bytes = await file.read()

    # inference
    class_name, confidence, probs = predictor.predict(image_bytes)

    overlay = predictor.generate_gradcam(image_bytes)

    _, buffer = cv2.imencode(".png", overlay)
    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    # metadata safety fallback
    info = DISEASE_INFO.get(
        class_name,
        {
            "description": "No description available",
            "next_step": "No recommendation available"
        }
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return JSONResponse(
        content={
            "class_name": class_name,
            "display_name": DISPLAY_NAMES.get(class_name, class_name),
            "confidence": round(confidence, 4),
            "description": info["description"],
            "next_step": info["next_step"],
            "grad_cam_base64": gradcam_base64,
            "inference_ms": round(elapsed_ms, 1),
        }
    )
