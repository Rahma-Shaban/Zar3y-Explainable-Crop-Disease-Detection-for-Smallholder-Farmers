## OOD (Out-of-Distribution) Evaluation

The model was evaluated on a real-world field-photo dataset containing unseen crop leaf images collected from external sources (`data/field_photos/`).

Unlike PlantVillage images (clean background + controlled lighting), these photos include:

- natural lighting variations
- shadows
- cluttered backgrounds
- multiple leaves in one image
- blur and camera noise

### OOD Top-1 Accuracy

| Metric | Value |
|---|---|
| OOD Accuracy | 38.78% |
| Total Images | 49 |
| Correct Predictions | 19 |
| Wrong Predictions | 30 |

### Key Observation

The model performs strongly on the PlantVillage test split, but generalization drops significantly on real field photos due to domain shift between controlled lab images and real agricultural environments.

### Common Failure Cases

- Early Blight vs Late Blight confusion
- Background noise influencing attention maps
- Multiple overlapping leaves causing localization errors
- Low-light mobile photos reducing prediction confidence

### Grad-CAM Explainability

Grad-CAM overlays were generated for:

- representative successful predictions
- difficult OOD failure cases

This helped verify whether the model focused on disease regions or irrelevant background features.

Example outputs are available in:

```text
outputs/grad_cam_examples/
outputs/failure_cases/