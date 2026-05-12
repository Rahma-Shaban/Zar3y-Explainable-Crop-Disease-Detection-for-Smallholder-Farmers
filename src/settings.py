"""
Zar3y — Project-Wide Settings & Constants

All locked hyperparameters, paths, and class definitions live here.
Import this module from any script to ensure consistency.
"""

import os

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Image / batch
# ─────────────────────────────────────────────
IMAGE_SIZE = 224
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
BATCH_SIZE = 32
NUM_CLASSES = 10

# ─────────────────────────────────────────────
# 10 Locked class names (alphabetical — matches
# tf.keras image_dataset_from_directory order)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Corn_(maize)___Common_rust_",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

# ─────────────────────────────────────────────
# Human-readable labels for the Streamlit UI
# ─────────────────────────────────────────────
DISPLAY_NAMES = {
    "Corn_(maize)___Common_rust_": "Corn — Common Rust",
    "Pepper,_bell___Bacterial_spot": "Pepper — Bacterial Spot",
    "Pepper,_bell___healthy": "Pepper — Healthy",
    "Potato___Early_blight": "Potato — Early Blight",
    "Potato___Late_blight": "Potato — Late Blight",
    "Potato___healthy": "Potato — Healthy",
    "Tomato___Early_blight": "Tomato — Early Blight",
    "Tomato___Late_blight": "Tomato — Late Blight",
    "Tomato___Leaf_Mold": "Tomato — Leaf Mold",
    "Tomato___healthy": "Tomato — Healthy",
}

# ─────────────────────────────────────────────
# Static disease info  (plain language — no LLM)
# ─────────────────────────────────────────────
DISEASE_INFO = {
    "Corn_(maize)___Common_rust_": {
        "description": (
            "Common rust appears as small, circular to elongated, "
            "cinnamon-brown pustules on both leaf surfaces. Severe "
            "infections can reduce photosynthesis and yield."
        ),
        "next_step": (
            "Apply a foliar fungicide containing mancozeb or "
            "azoxystrobin when pustules are first noticed. Remove "
            "heavily infected leaves and ensure adequate plant spacing."
        ),
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": (
            "Bacterial spot causes small, dark, water-soaked lesions "
            "on leaves that may coalesce and cause defoliation. Fruit "
            "can develop raised, scab-like spots."
        ),
        "next_step": (
            "Apply copper-based bactericide early. Avoid overhead "
            "irrigation, rotate crops, and use certified disease-free "
            "seed to reduce spread."
        ),
    },
    "Pepper,_bell___healthy": {
        "description": (
            "No disease detected. The leaf appears healthy with "
            "uniform green colour and no visible lesions or spots."
        ),
        "next_step": (
            "No treatment needed. Continue regular watering, "
            "balanced fertilisation, and pest monitoring."
        ),
    },
    "Potato___Early_blight": {
        "description": (
            "Early blight produces dark brown, concentric-ring "
            "(target-shaped) lesions on older leaves. It spreads "
            "upward and can reduce tuber yield significantly."
        ),
        "next_step": (
            "Apply chlorothalonil or mancozeb fungicide preventively. "
            "Remove lower infected leaves, maintain good air "
            "circulation, and avoid overhead irrigation."
        ),
    },
    "Potato___Late_blight": {
        "description": (
            "Late blight causes large, dark, water-soaked lesions "
            "on leaves and stems, often with white mold on the "
            "underside. It spreads rapidly in cool, wet conditions."
        ),
        "next_step": (
            "Apply copper-based or systemic fungicide (e.g., "
            "metalaxyl) immediately. Destroy infected plants, avoid "
            "overhead watering, and harvest tubers promptly."
        ),
    },
    "Potato___healthy": {
        "description": (
            "No disease detected. The leaf appears healthy with "
            "no visible blight, spots, or discoloration."
        ),
        "next_step": (
            "No treatment needed. Maintain regular irrigation, "
            "hilling, and pest scouting schedules."
        ),
    },
    "Tomato___Early_blight": {
        "description": (
            "Early blight shows dark, concentric-ring lesions "
            "(bull's-eye pattern) starting on lower, older leaves. "
            "Severe cases lead to defoliation and reduced fruit quality."
        ),
        "next_step": (
            "Apply mancozeb or chlorothalonil fungicide at first sign. "
            "Prune lower branches for airflow, stake plants, and "
            "practice crop rotation."
        ),
    },
    "Tomato___Late_blight": {
        "description": (
            "Late blight causes large, dark, water-soaked lesions "
            "on leaves and stems. White mold may appear on the "
            "underside of leaves in humid conditions. Spreads fast."
        ),
        "next_step": (
            "Apply copper-based fungicide immediately. Remove and "
            "destroy affected foliage. Avoid wetting leaves; water "
            "at the base. Consider resistant varieties next season."
        ),
    },
    "Tomato___Leaf_Mold": {
        "description": (
            "Leaf mold appears as pale-green to yellow spots on "
            "upper leaf surfaces, with olive-green to brown velvety "
            "mold on the underside. Thrives in high humidity."
        ),
        "next_step": (
            "Improve greenhouse ventilation and reduce humidity. "
            "Apply fungicide (e.g., chlorothalonil). Remove heavily "
            "infected leaves and avoid overhead watering."
        ),
    },
    "Tomato___healthy": {
        "description": (
            "No disease detected. The leaf appears healthy with "
            "vibrant green colour and no visible symptoms."
        ),
        "next_step": (
            "No treatment needed. Continue balanced fertilisation, "
            "regular watering, and periodic scouting for pests."
        ),
    },
}

# ─────────────────────────────────────────────
# Paths  (relative to project root)
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FILTERED_DIR = os.path.join(DATA_DIR, "filtered_data")
SPLIT_DIR = os.path.join(DATA_DIR, "split_data")
TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val")
TEST_DIR = os.path.join(SPLIT_DIR, "test")
FIELD_PHOTOS_DIR = os.path.join(DATA_DIR, "field_photos")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "model_quantized.tflite")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CLASS_NAMES_JSON = os.path.join(PROJECT_ROOT, "class_names.json")

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 10
PHASE2_UNFREEZE_LAYERS = 30

INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-4           # 10× lower

EARLY_STOPPING_PATIENCE = 5

# ─────────────────────────────────────────────
# Split ratios
# ─────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
