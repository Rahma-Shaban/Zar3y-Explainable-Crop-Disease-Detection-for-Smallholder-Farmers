# Zar3y — Final Report

> Crop Disease Detection from Phone Photos  
> Graduation Project · Computer Vision + ML Lifecycle + Mobile Deployment

---

## 1. Data

**Source**: PlantVillage dataset (Kaggle) — subset to 10 locked classes covering 4 Egyptian-relevant crops.

**Split**: Stratified 70 / 15 / 15 (seed=42) — reproducible across squads.

| Class | Train | Val | Test |
|---|---|---|---|
| Corn\_(maize)\_\_\_Common\_rust\_ | 834 | 179 | 179 |
| Pepper,\_bell\_\_\_Bacterial\_spot | 697 | 150 | 150 |
| Pepper,\_bell\_\_\_healthy | 1 034 | 222 | 222 |
| Potato\_\_\_Early\_blight | 700 | 150 | 150 |
| Potato\_\_\_Late\_blight | 700 | 150 | 150 |
| Potato\_\_\_healthy | 106 | 23 | 23 |
| Tomato\_\_\_Early\_blight | 700 | 150 | 150 |
| Tomato\_\_\_Late\_blight | 1 336 | 286 | 287 |
| Tomato\_\_\_Leaf\_Mold | 666 | 143 | 143 |
| Tomato\_\_\_healthy | 1 113 | 239 | 239 |

**Class imbalance**: `Potato___healthy` has only 152 images (below the 500-image threshold). Class-weighted loss was applied during training to compensate.

**Augmentation pipeline** (tf.data):
- Random horizontal flip
- Random rotation (±15°)
- Random brightness (±20%)
- Random contrast (±20%)
- Random zoom (±20%)

Augmentation samples saved to `outputs/augmentation_samples.png`.

---

## 2. Model Architecture

**Base model**: MobileNetV3-Small (ImageNet pretrained) — chosen for its small footprint and INT8-quantizability.

**Training strategy** (two-phase transfer learning):

| Phase | Epochs | Learning Rate | Backbone |
|---|---|---|---|
| Phase 1 — Classifier head | 10 | 1×10⁻³ | Frozen |
| Phase 2 — Fine-tune | 10 | 1×10⁻⁴ | Last 30 layers unfrozen |

**Regularisation**: Dropout 0.3 + early stopping (patience 5 on val loss) + ReduceLROnPlateau.

**Callbacks**: EarlyStopping, ModelCheckpoint (save best on val loss), ReduceLROnPlateau (factor 0.2, patience 2).

---

## 3. Evaluation Results

| Metric | Value |
|---|---|
| **Test Accuracy** | 98.23% |
| **Macro F1-Score** | 97.96% |
| **Test Samples** | 1 693 |

The model achieves near-perfect classification on several classes (Corn Common Rust, Pepper Bacterial Spot, Healthy Pepper). Minor confusion exists between visually similar diseases (Tomato Early Blight ↔ Late Blight, Potato Early Blight ↔ Late Blight).

**Artifacts produced**:
- `outputs/confusion_matrix.png`
- `outputs/training_curves.png`
- `outputs/eval_report.json`

---

## 4. Quantization

| Metric | FP32 | INT8 |
|---|---|---|
| **Model size** | 9.14 MB | 1.12 MB |
| **Compression** | — | **8.19×** smaller |

The INT8 TFLite model is suitable for edge / mobile deployment. The quantized model maintains strong classification performance with negligible accuracy delta.

**Artifacts produced**:
- `models/model_quantized.tflite`
- `outputs/quantization_benchmark.json`

---

## 5. OOD Findings

**Out-of-distribution test**: 49 real field photos (sourced from Unsplash / Pexels / personal captures) never seen during training.

| Metric | Value |
|---|---|
| **OOD Accuracy** | 38.78% |
| **Total Images** | 49 |
| **Correct** | 19 |
| **Wrong** | 30 |

**Analysis**: The significant drop from 98% (test set) to 39% (OOD) reflects the domain shift between curated PlantVillage lab images and real-world field captures with variable lighting, backgrounds, and leaf orientations. This is an honest result — we do not hide the gap.

**Failure case patterns** (Grad-CAM analysis):
1. The model attends to background foliage rather than the diseased leaf when the photo contains multiple plants.
2. Overexposed or blurry photos cause the model to fixate on colour artefacts.
3. Some classes with low training data (e.g. Potato healthy with only 152 images) are frequently misclassified.

**Artifacts produced**:
- `outputs/ood_report.json`
- `outputs/grad_cam_examples/` (5 representative overlays)
- `outputs/grad_cam_failure_cases/` (3 annotated failures)

---

## 6. Limitations

1. **Lab-to-field gap**: Trained on clean lab images; real-field accuracy is significantly lower.
2. **Limited scope**: Only 4 crops / 10 classes. Does not cover all diseases found in the Egyptian Delta.
3. **Static suggestions**: Treatment recommendations are hard-coded text, not validated by agronomists.
4. **English-only UI**: No Arabic localisation yet — critical for the target user base.
5. **Single-leaf input**: The model expects a single leaf photo. Multi-leaf or whole-plant images may confuse it.
6. **No temporal tracking**: Cannot monitor disease progression over time.

**Potential improvements**:
- Fine-tune on real Egyptian field photos (data collection campaign)
- Add Arabic language support
- Integrate with agricultural extension agent referral system
- Add multi-crop / multi-disease detection
