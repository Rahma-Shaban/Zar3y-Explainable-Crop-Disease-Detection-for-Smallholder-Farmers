"""
Zar3y — Grad-CAM Explainability (Requirement 4)

This module handles:
1. Implementing Grad-CAM for the last convolutional block of MobileNetV3-Small
2. Generating side-by-side original + Grad-CAM overlay images
3. Saving overlays for 5 representative test images (one per disease)
4. Running inference on OOD field photos (30-50 real photos)
5. Reporting OOD top-1 accuracy + annotating failure cases with Grad-CAM

Deliverables:
- outputs/grad_cam_examples/ (side-by-side overlays)
- data/field_photos/ (OOD test set)
- outputs/ood_report.json
- 3 annotated failure-case images
"""

import os
import glob
import cv2
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# =====================================================
# CONFIG
# =====================================================

SEED = 42
random.seed(SEED)

IMAGE_SIZE = (224,224)
BATCH_SIZE = 32

MODEL_PATH = "models/best_model.keras"

TEST_DIR = "data/split_data/test"

FIELD_PHOTOS_DIR = "data/field_photos"

OUTPUT_DIR = "outputs"

GRAD_CAM_DIR = f"{OUTPUT_DIR}/grad_cam_examples"

FAILURE_DIR = f"{OUTPUT_DIR}/grad_cam_failure_cases"

OOD_REPORT_PATH = f"{OUTPUT_DIR}/ood_report.json"

LAST_CONV_LAYER = "conv_1"

os.makedirs(GRAD_CAM_DIR, exist_ok=True)
os.makedirs(FAILURE_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

model = load_model(MODEL_PATH)

# =====================================================
# LOAD CLASS NAMES
# =====================================================

raw_train_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

class_names = raw_train_ds.class_names

print("Classes:", class_names)

# =====================================================
# PREPROCESS
# =====================================================

def preprocess_img(path):

    img = load_img(path, target_size=IMAGE_SIZE)

    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.cast(img_array, tf.float32)

    img_array = preprocess_input(img_array)

    return np.array(img), img_array

# =====================================================
# PREDICT
# =====================================================

def predict_image(model, img_array, class_names):

    preds = model.predict(img_array, verbose=0)

    pred_index = np.argmax(preds)

    pred_label = class_names[pred_index]

    confidence = float(np.max(preds))

    return pred_label, confidence

# =====================================================
# GRADCAM
# =====================================================

def make_gradcam_heatmap(img_array, model):

    base_model = model.get_layer("MobileNetV3Small")

    last_conv_layer = base_model.get_layer(LAST_CONV_LAYER)

    last_conv_model = tf.keras.Model(
        base_model.input,
        last_conv_layer.output
    )

    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    passed = False

    for layer in base_model.layers:

        if layer.name == LAST_CONV_LAYER:
            passed = True
            continue

        if passed:
            x = layer(x)

    x = model.layers[-3](x)
    x = model.layers[-2](x)
    x = model.layers[-1](x)

    classifier_model = tf.keras.Model(
        classifier_input,
        x
    )

    with tf.GradientTape() as tape:

        conv_output = last_conv_model(img_array)

        tape.watch(conv_output)

        preds = classifier_model(conv_output)

        pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# =====================================================
# OVERLAY
# =====================================================

def overlay_heatmap(img_path, heatmap):

    img = cv2.imread(img_path)

    img = cv2.resize(img, IMAGE_SIZE)

    heatmap = cv2.resize(heatmap, IMAGE_SIZE)

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed = cv2.addWeighted(
        img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return img, superimposed

# =====================================================
# SIDE BY SIDE
# =====================================================

def save_side_by_side(original, overlay, save_path):

    combined = np.hstack([original, overlay])

    cv2.imwrite(save_path, combined)

# =====================================================
# SAVE GRADCAM EXAMPLES
# =====================================================

print("\nGenerating GradCAM examples...")

all_test_images = glob.glob(
    TEST_DIR + "/*/*"
)

sample_images = random.sample(
    all_test_images,
    5
)

for i, img_path in enumerate(sample_images):

    _, img_array = preprocess_img(img_path)

    heatmap = make_gradcam_heatmap(
        img_array,
        model
    )

    original, overlay = overlay_heatmap(
        img_path,
        heatmap
    )

    save_path = f"{GRAD_CAM_DIR}/example_{i}.jpg"

    save_side_by_side(
        original,
        overlay,
        save_path
    )

    print("Saved:", save_path)

# =====================================================
# OOD EVALUATION
# =====================================================

print("\nRunning OOD evaluation...")

all_images = glob.glob(
    FIELD_PHOTOS_DIR + "/*/*"
)

results = []

failures = []

correct = 0
total = 0

for img_path in all_images:

    true_label = img_path.split("/")[-2]

    _, img_array = preprocess_img(img_path)

    pred_label, confidence = predict_image(
        model,
        img_array,
        class_names
    )

    is_correct = pred_label == true_label

    if is_correct:
        correct += 1
    else:

        failures.append({
            "img_path": img_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence
        })

    total += 1

    results.append({
        "image": img_path,
        "true_label": true_label,
        "predicted_label": pred_label,
        "confidence": confidence,
        "correct": bool(is_correct)
    })

accuracy = correct / total

print("\nOOD Accuracy:", round(accuracy * 100,2), "%")

# =====================================================
# SAVE OOD REPORT
# =====================================================

report = {
    "ood_accuracy": float(accuracy),
    "accuracy_percentage": float(accuracy * 100),
    "total_images": int(total),
    "correct_predictions": int(correct),
    "wrong_predictions": int(total - correct)
}

with open(OOD_REPORT_PATH, "w") as f:
    json.dump(report, f, indent=4)

print("Saved:", OOD_REPORT_PATH)

# =====================================================
# FAILURE CASES
# =====================================================

print("\nGenerating failure cases...")

sample_failures = failures[:3]

for i, item in enumerate(sample_failures):

    img_path = item["img_path"]

    _, img_array = preprocess_img(img_path)

    heatmap = make_gradcam_heatmap(
        img_array,
        model
    )

    original, overlay = overlay_heatmap(
        img_path,
        heatmap
    )

    cv2.putText(
        overlay,
        f"TRUE: {item['true_label']}",
        (10,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )

    cv2.putText(
        overlay,
        f"PRED: {item['pred_label']}",
        (10,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )

    combined = np.hstack([original, overlay])

    save_path = f"{FAILURE_DIR}/failure_{i}.jpg"

    cv2.imwrite(save_path, combined)

    print("Saved:", save_path)

print("\nRequirement 4 completed successfully.")
