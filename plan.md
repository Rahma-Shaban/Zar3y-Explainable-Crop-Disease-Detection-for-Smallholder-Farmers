# 📅 Zar3y — 5-Day Sprint Plan (5 Members)

> Sprint Duration: 5 working days
> Team Size: 5 members

## Team Roles

| ID | Role | Member |
|----|------|--------|
| M1 | Data Engineer | Member 1 |
| M2 | ML Engineer (Training) | Member 2 |
| M3 | ML Engineer (Evaluation) | Member 3 |
| M4 | Backend Developer | Member 4 |
| M5 | Frontend Developer | Member 5 |

---

## Day 1 — Foundation & Data Pipeline

**Theme**: Project setup, environment, data download and preparation.

| Member | Tasks | Deliverables |
|--------|-------|-------------- |
| M1 | Download PlantVillage from Kaggle; filter to 10 locked classes; implement stratified 70/15/15 split (seed=42); compute per-class counts | `src/data_prep.py` (split logic), per-class count table in README |
| M2 | Set up Google Colab notebook for training; implement `src/settings.py` with all constants (classes, image size, paths, hyperparameters) | `notebooks/02_training.ipynb` (skeleton), `src/settings.py` |
| M3 | Set up evaluation framework; create `src/evaluate.py` skeleton with metric function signatures | `src/evaluate.py` (skeleton), `notebooks/03_evaluation.ipynb` (skeleton) |
| M4 | Set up project repo, `.gitignore`, `requirements.txt`, `.env.example`; create FastAPI project skeleton with health-check endpoint | `backend/main.py` (skeleton with `/health`), `requirements.txt` |
| M5 | Set up Streamlit app skeleton with upload/camera widget; design UI layout and page structure | `app.py` (skeleton with upload widget) |

### Day 1 Milestone
- [ ] Repo initialized with full directory structure
- [ ] PlantVillage dataset downloaded and filtered to 10 classes
- [ ] Data split completed with per-class counts documented
- [ ] All skeleton files in place

---

## Day 2 — Augmentation + Training + Backend Wiring

**Theme**: Data augmentation pipeline, begin model training, wire up backend.

| Member | Tasks | Deliverables |
|--------|-------|-------------- |
| M1 | Build `tf.data` augmentation pipeline (flip, rotation ±15°, brightness ±20%, contrast ±20%, zoom); save augmentation samples | `src/data_prep.py` (augmentation), `outputs/augmentation_samples.png` |
| M2 | Implement Phase 1 training: load MobileNetV3-Small, freeze backbone, train classifier head for 10 epochs with early stopping; save checkpoint | `src/train.py` (Phase 1), `models/best_model.h5` (Phase 1) |
| M3 | Begin data exploration notebook; visualize class distribution, sample images per class, augmentation effects | `notebooks/01_data_exploration.ipynb` |
| M4 | Implement `src/inference.py` TFLite runtime helper; wire up `/predict` endpoint to accept image upload and return dummy prediction | `src/inference.py`, `backend/main.py` (working /predict) |
| M5 | Implement Streamlit ↔ FastAPI integration; POST image to backend, display returned prediction | `app.py` (API integration) |

### Day 2 Milestone
- [ ] Augmentation pipeline working with visual samples saved
- [ ] Phase 1 training completed (frozen backbone)
- [ ] FastAPI `/predict` endpoint accepting images
- [ ] Streamlit sending images to backend

---

## Day 3 — Fine-tuning + Evaluation + Grad-CAM

**Theme**: Fine-tune model, run full evaluation, implement explainability.

| Member | Tasks | Deliverables |
|--------|-------|-------------- |
| M1 | Collect 30-50 OOD field photos from Unsplash/Pexels; organize in `data/field_photos/` with labels | `data/field_photos/` (30-50 images) |
| M2 | Implement Phase 2: unfreeze last 30 layers, fine-tune 10 epochs at LR/10 with early stopping; save best checkpoint | `src/train.py` (Phase 2), `models/best_model.h5` (final) |
| M3 | Run full evaluation: overall accuracy, macro F1, per-class P/R/F1, confusion matrix PNG, training curves PNG | `outputs/confusion_matrix.png`, `outputs/training_curves.png`, `outputs/eval_report.json` |
| M4 | Implement `src/quantize_tflite.py`: convert to INT8 with representative dataset (200 images); run size benchmark | `src/quantize_tflite.py` (conversion logic) |
| M5 | Implement `src/grad_cam.py`: Grad-CAM for last conv block; generate side-by-side overlays for 5 test images | `src/grad_cam.py`, `outputs/grad_cam_examples/` |

### Day 3 Milestone
- [ ] Model fully trained (Phase 1 + Phase 2)
- [ ] Full evaluation report with confusion matrix
- [ ] Grad-CAM working and producing overlays
- [ ] OOD test set collected

---

## Day 4 — Quantization + OOD Testing + Integration

**Theme**: Quantize model, benchmark, OOD evaluation, full pipeline integration.

| Member | Tasks | Deliverables |
|--------|-------|-------------- |
| M1 | Run OOD inference on field photos; compute OOD accuracy; identify and annotate 3+ failure cases with Grad-CAM | `outputs/ood_report.json`, 3 annotated failure-case images |
| M2 | Support M4 with quantization accuracy testing; verify accuracy delta < 5% | Quantization accuracy verification |
| M3 | Run latency benchmark: 100 test images on float vs INT8; document size/latency/accuracy table | `outputs/quantization_benchmark.json` |
| M4 | Integrate quantized TFLite model into FastAPI `/predict`; add Grad-CAM base64 overlay to response; verify < 500ms latency | `backend/main.py` (final with TFLite + Grad-CAM) |
| M5 | Display Grad-CAM overlay in Streamlit; add plain-language disease descriptions + next-step suggestions (static dict); polish UI | `app.py` (complete with Grad-CAM display + descriptions) |

### Day 4 Milestone
- [ ] INT8 model quantized and benchmarked (≥ 3× smaller, < 5% accuracy drop)
- [ ] `/predict` returns class + confidence + Grad-CAM in < 500ms
- [ ] OOD evaluation complete with failure cases
- [ ] Full end-to-end pipeline working

---

## Day 5 — Polish, Demo Recording & Documentation

**Theme**: Final testing, documentation, demo recording, submission.

| Member | Tasks | Deliverables |
|--------|-------|-------------- |
| M1 | Finalize `outputs/final_report.md` (data section + OOD findings) | `outputs/final_report.md` (data sections) |
| M2 | Finalize `outputs/final_report.md` (model + training section); verify reproducibility (retrain instructions) | `outputs/final_report.md` (model sections) |
| M3 | Finalize `outputs/final_report.md` (eval + quantization section + limitations); final README update with all results | `outputs/final_report.md` (eval sections), README updated |
| M4 | End-to-end testing; fix bugs; ensure all commands in README work; finalize `.env.example` | Clean, tested codebase |
| M5 | Record 60-90s demo of Hassan scenario; convert to GIF; final UI polish | `docs/demo.gif` |

### Day 5 Milestone
- [ ] `final_report.md` complete (< 4 pages)
- [ ] README has all results tables filled in
- [ ] Demo recording (60-90s) covers Hassan scenario
- [ ] All acceptance criteria verified
- [ ] Repository ready for submission

---

## Acceptance Criteria Checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Dataset: 10 locked classes, 70/15/15 split, seed=42 | ⬜ |
| 2 | Model: ≥ 90% macro F1 on test split | ⬜ |
| 3 | Confusion matrix + per-class P/R/F1 reported honestly | ⬜ |
| 4 | TFLite INT8: ≥ 3× smaller, < 5% accuracy drop | ⬜ |
| 5 | `/predict` latency < 500ms on CPU | ⬜ |
| 6 | Grad-CAM overlays in Streamlit UI | ⬜ |
| 7 | OOD field-photo test with ≥ 3 annotated failures | ⬜ |
| 8 | Total cost = $0 | ⬜ |

---

## Dependencies Between Tasks

```
Day 1: M4(repo) → All members clone
Day 1: M1(data split) → M2(training Day 2)
Day 2: M1(augmentation) → M2(training with augmented data)
Day 2: M2(Phase 1 model) → M3(can begin eval prep)
Day 3: M2(final model) → M3(full eval) → M4(quantization)
Day 3: M2(final model) → M5(Grad-CAM)
Day 4: M4(quantized model) → M3(latency benchmark)
Day 4: M1(OOD photos) → M1(OOD eval with Grad-CAM)
Day 4: M4(API complete) → M5(UI integration)
Day 5: All → Documentation + Demo
```
