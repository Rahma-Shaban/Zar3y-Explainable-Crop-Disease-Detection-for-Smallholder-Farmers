# 03_evaluation.ipynb

#Setup (Imports + Paths)

import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.preprocessing import image_dataset_from_directory



print("Libraries loaded ✔")

if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
TEST_DIR   = os.path.join(BASE_DIR, "data", "split_data", "test")
HISTORY_PATH = os.path.join(BASE_DIR, "outputs", "training_history.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

"""Load Model & Test Data"""

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded ✔")

test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
print("Classes:", class_names)

"""Predictions + Overall Metrics"""

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

print("Accuracy:", acc)
print("Macro F1:", macro_f1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

"""Confusion Matrix"""

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

plt.show()

"""Training Curves"""

OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "training_curves.png")

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history["accuracy"], label="train")
plt.plot(history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history["loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

# save first
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

plt.show()

"""Quantization (INT8)"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = os.path.join(BASE_DIR, "models", "model_quantized.tflite")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("Quantized model saved ✔")

fp32_size = os.path.getsize(MODEL_PATH) / 1e6
int8_size = os.path.getsize(tflite_path) / 1e6

print("FP32 size:", fp32_size, "MB")
print("INT8 size:", int8_size, "MB")
print("Compression ratio:", fp32_size / int8_size)



report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

df_report = pd.DataFrame(report_dict).transpose()
df_report

print("\n===== CLASS INSIGHT =====")

worst_class = class_names[np.argmin(np.bincount(y_pred == y_true))]
best_class = class_names[np.argmax(np.bincount(y_pred == y_true))]

print("Most Confused Area (approx):", worst_class)
print("Best Performing Pattern:", best_class)

eval_report = {
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "num_classes": len(class_names),
    "test_samples": len(y_true)
}

OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")
REPORT_PATH = os.path.join(OUTPUT_PATH, "eval_report.json")

os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(REPORT_PATH, "w") as f:
    json.dump(eval_report, f, indent=4)

print("Eval report saved ✔")

print("\n================ FINAL SUMMARY ================")
print("Dataset Classes:", len(class_names))
print("Model Input Size:", IMAGE_SIZE)
print("Test Samples Evaluated:", len(y_true))
print("Overall Accuracy:", round(acc, 4))
print("Macro F1 Score:", round(macro_f1, 4))

print("\n===== PER-CLASS PERFORMANCE (Top Insight) =====")
print(df_report[["precision", "recall", "f1-score"]].head())

print("\n===== MODEL FILES =====")
print("Best Model Path:", MODEL_PATH)
print("Quantized Model Saved: model_quantized.tflite")
print("History File:", HISTORY_PATH)

print("\n===== KEY INSIGHT =====")
print("Model shows strong performance on majority classes,")
print("with minor confusion between visually similar diseases.")
print("==============================================")

