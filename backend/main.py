"""
Zar3y — FastAPI Backend (Requirement 5)

This module provides the REST API for crop disease prediction.

Endpoints:
- GET  /health          → Health check
- POST /predict         → Accept image, return prediction

Response schema for /predict:
{
    "class_name": "Tomato___Late_blight",
    "confidence": 0.87,
    "description": "Dark, water-soaked lesions on leaves...",
    "next_step": "Apply copper-based fungicide...",
    "grad_cam_base64": "<base64-encoded overlay image>"
}

Run with: uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

# TODO: Implement FastAPI backend
#
# Key components to implement:
#
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
#
# app = FastAPI(title="Zar3y API", version="1.0.0")
#
# @app.on_event("startup")
# async def load_model():
#     """Load TFLite model on startup."""
#     pass
#
# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     pass
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     """
#     Accept an uploaded leaf photo.
#     Return predicted disease class, confidence, description,
#     next-step suggestion, and Grad-CAM overlay as base64.
#     Must return in < 500ms on CPU.
#     """
#     pass
