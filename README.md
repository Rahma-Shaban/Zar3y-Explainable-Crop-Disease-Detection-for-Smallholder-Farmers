# 🌱 Zar3y — Crop Disease Detection from Phone Photos

> **"Hassan, a smallholder farmer in Beheira, sees yellow spots on his tomato leaves. He takes a photo with his phone."**

Zar3y ("my crop") identifies crop leaf diseases from a single photo — returning a **plain-language diagnosis**, **confidence score**, and **Grad-CAM overlay**.

---

## Team (5 Members)

| Role | Member | Responsibilities |
|------|--------|-----------------|
| Data Engineer | Nourhan | Data prep, augmentation, OOD test set |
| ML Engineer (Train) | shahd | Transfer learning, training pipeline |
| ML Engineer (Eval) | menna | Evaluation, metrics, benchmarks |
| Backend Developer & team leader | Rahma | FastAPI, TFLite inference, quantization |
| Frontend Developer | tasneem | Streamlit demo, Grad-CAM UI, demo recording |

## Dataset

- **Source**: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Subset**: 10 locked classes (Tomato, Potato, Pepper, Corn)
- **Split**: Stratified 70/15/15, **seed=42**

### 10 Locked Classes

| # | Class | Crop |
|---|-------|------|
| 0 | Tomato___healthy | Tomato |
| 1 | Tomato___Early_blight | Tomato |
| 2 | Tomato___Late_blight | Tomato |
| 3 | Tomato___Leaf_Mold | Tomato |
| 4 | Potato___healthy | Potato |
| 5 | Potato___Early_blight | Potato |
| 6 | Potato___Late_blight | Potato |
| 7 | Pepper_bell___healthy | Pepper |
| 8 | Pepper_bell___Bacterial_spot | Pepper |
| 9 | Corn___Common_rust | Corn |

### Per-Class Distribution

| Class | Train | Validation | Test |
|---|---|---|---|
| Corn_(maize)___Common_rust_ | 834 | 179 | 179 |
| Pepper,_bell___Bacterial_spot | 697 | 150 | 150 |
| Pepper,_bell___healthy | 1034 | 222 | 222 |
| Potato___Early_blight | 700 | 150 | 150 |
| Potato___Late_blight | 700 | 150 | 150 |
| Potato___healthy | 106 | 23 | 23 |
| Tomato___Early_blight | 700 | 150 | 150 |
| Tomato___Late_blight | 1336 | 286 | 287 |
| Tomato___Leaf_Mold | 666 | 143 | 143 |
| Tomato___healthy | 1113 | 239 | 239 |

> Note: `Potato___healthy` contains fewer than 500 images. Class weights will be applied during training to handle class imbalance.

## Architecture

```
Streamlit (8501) ──POST /predict──► FastAPI (8000) ──► TFLite INT8 + Grad-CAM
```

## Tech Stack

Python 3.10+ · TensorFlow · MobileNetV3-Small · scikit-learn · FastAPI · Streamlit · TFLite

## Project Structure

```
zar3y-crop-disease/
├── README.md
├── plan.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py                       ← Streamlit demo
├── src/
│   ├── __init__.py
│   ├── settings.py              ← Constants & config
│   ├── data_prep.py             ← Split + augmentation
│   ├── train.py                 ← Transfer learning
│   ├── evaluate.py              ← Metrics + plots
│   ├── quantize_tflite.py       ← INT8 + benchmark
│   ├── grad_cam.py              ← Explainability
│   └── inference.py             ← TFLite runtime
├── backend/
│   ├── __init__.py
│   └── main.py                  ← FastAPI /predict
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── models/                      ← .h5 and .tflite
├── data/
│   ├── plant_village/           ← 10-class subset
│   └── field_photos/            ← OOD test set
├── outputs/
│   ├── grad_cam_examples/
│   └── final_report.md
└── docs/
    └── demo.gif
```

## Setup

```bash
git clone https://github.com/Rahma-Shaban/Zar3y-Explainable-Crop-Disease-Detection-for-Smallholder-Farmers.git
cd Zar3y-Explainable-Crop-Disease-Detection-for-Smallholder-Farmers
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
cp .env.example .env
```

## Training

```bash
python src/data_prep.py          # Prepare data
python src/train.py              # Train model (use Colab for GPU)
python src/evaluate.py           # Evaluate
python src/quantize_tflite.py    # Quantize to INT8
```

## Running the App

```bash
uvicorn backend.main:app --port 8000   # Backend
streamlit run app.py                    # Frontend (localhost:8501)
```

## Evaluation Results

_TODO: Fill after evaluation_

## Quantization Benchmark

_TODO: Fill after quantization_

## OOD Findings

The model was evaluated on real-world field photos (`data/field_photos`) to test generalization beyond the training distribution.

Performance reflects domain shift between clean dataset images and real mobile captures.

- OOD Accuracy: 38.78%
- Total Images: 49
- Correct Predictions: 19

## Limitations

- Trained on lab images; real-field performance may degrade
- Only 4 crops / 10 classes supported
- Static treatment suggestions (not expert-validated)
- English-only UI

## Acceptance Criteria

- [ ] 10 locked classes, 70/15/15 split, seed=42
- [ ] ≥ 90% macro F1 on test split
- [ ] Honest confusion matrix + per-class P/R/F1
- [ ] TFLite INT8 ≥ 3× smaller, < 5% accuracy drop
- [ ] /predict < 500ms on CPU
- [ ] Grad-CAM in Streamlit UI
- [ ] OOD test with ≥ 3 failure cases
- [ ] Total cost = $0
