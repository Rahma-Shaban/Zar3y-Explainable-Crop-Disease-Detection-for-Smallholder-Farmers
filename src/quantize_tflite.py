"""
Zar3y — TFLite INT8 Quantization & Benchmark (Requirement 3)

This module handles:
1. Converting best_model.keras to TFLite with INT8 post-training quantization
2. Using a representative dataset (200 images from train set) for calibration
3. Benchmarking:
   a. Model size (MB): float vs INT8
   b. Average inference latency (100 test images): float vs INT8
   c. Accuracy delta on held-out test set
4. Saving benchmark results (including any accuracy drop — do not hide it)

Deliverables:
- models/model_quantized.tflite
- outputs/quantization_benchmark.json
"""

import os
import sys
import json
import time
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ─────────────────────────────────────────────
# Paths (resolve relative to project root)
# ─────────────────────────────────────────────
if "__file__" in dir():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    PROJECT_ROOT = os.getcwd()

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.keras")
TFLITE_PATH = os.path.join(PROJECT_ROOT, "models", "model_quantized.tflite")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "split_data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "split_data", "test")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "outputs", "quantization_benchmark.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CALIBRATION = 200
NUM_LATENCY_RUNS = 100


# ─────────────────────────────────────────────
# 1. Representative dataset generator
# ─────────────────────────────────────────────
def create_representative_dataset(data_dir: str, num_samples: int = NUM_CALIBRATION):
    """Yield float32 images for INT8 calibration."""
    image_paths = glob.glob(os.path.join(data_dir, "*", "*"))
    np.random.seed(42)
    np.random.shuffle(image_paths)
    image_paths = image_paths[:num_samples]

    def generator():
        for path in image_paths:
            img = load_img(path, target_size=IMAGE_SIZE)
            arr = img_to_array(img)
            arr = preprocess_input(arr)
            yield [arr.astype(np.float32)[np.newaxis, ...]]

    return generator


# ─────────────────────────────────────────────
# 2. Convert to TFLite INT8
# ─────────────────────────────────────────────
def convert_to_tflite_int8(model_path: str, representative_gen, output_path: str):
    """Convert Keras model to TFLite with INT8 quantization."""
    model = load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    # Allow fallback to float ops for unsupported int8 ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ TFLite INT8 model saved to {output_path}")
    return tflite_model


# ─────────────────────────────────────────────
# 3. Benchmark model size
# ─────────────────────────────────────────────
def benchmark_model_size(float_path: str, tflite_path: str) -> dict:
    """Compare model sizes in MB."""
    fp32_mb = os.path.getsize(float_path) / 1e6
    int8_mb = os.path.getsize(tflite_path) / 1e6
    return {
        "float32_size_mb": round(fp32_mb, 2),
        "int8_size_mb": round(int8_mb, 2),
        "compression_ratio": round(fp32_mb / int8_mb, 2),
    }


# ─────────────────────────────────────────────
# 4. Benchmark latency
# ─────────────────────────────────────────────
def _tflite_predict(interpreter, input_data):
    """Run a single TFLite inference."""
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    if inp[0]["dtype"] == np.uint8:
        input_data = ((input_data + 1.0) * 127.5).astype(np.uint8)
    else:
        input_data = input_data.astype(np.float32)

    interpreter.set_tensor(inp[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(out[0]["index"])


def benchmark_latency(model_path: str, tflite_path: str, test_dir: str,
                      num_runs: int = NUM_LATENCY_RUNS) -> dict:
    """Measure average inference time for float and TFLite models."""
    # Gather test image paths
    paths = glob.glob(os.path.join(test_dir, "*", "*"))[:num_runs]

    # --- Float model latency ---
    float_model = load_model(model_path)
    float_times = []
    for p in paths:
        img = load_img(p, target_size=IMAGE_SIZE)
        arr = preprocess_input(img_to_array(img)[np.newaxis, ...])
        t0 = time.perf_counter()
        float_model.predict(arr, verbose=0)
        float_times.append(time.perf_counter() - t0)

    # --- TFLite latency ---
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    tflite_times = []
    for p in paths:
        img = load_img(p, target_size=IMAGE_SIZE)
        arr = preprocess_input(img_to_array(img)[np.newaxis, ...])
        t0 = time.perf_counter()
        _tflite_predict(interpreter, arr)
        tflite_times.append(time.perf_counter() - t0)

    return {
        "float32_avg_latency_ms": round(np.mean(float_times) * 1000, 2),
        "int8_avg_latency_ms": round(np.mean(tflite_times) * 1000, 2),
        "speedup": round(np.mean(float_times) / (np.mean(tflite_times) + 1e-9), 2),
        "num_runs": len(paths),
    }


# ─────────────────────────────────────────────
# 5. Benchmark accuracy
# ─────────────────────────────────────────────
def benchmark_accuracy(model_path: str, tflite_path: str, test_dir: str) -> dict:
    """Compare accuracy on the held-out test set."""
    # Float model accuracy
    float_model = load_model(model_path)
    test_ds = image_dataset_from_directory(
        test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )

    y_true, y_pred_float = [], []
    for images, labels in test_ds:
        preds = float_model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_float.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred_float = np.array(y_pred_float)
    float_acc = float(np.mean(y_true == y_pred_float))

    # TFLite accuracy
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    y_pred_tflite = []

    for images, _ in test_ds:
        for img in images:
            arr = preprocess_input(img.numpy()[np.newaxis, ...])
            out = _tflite_predict(interpreter, arr)
            if out.dtype == np.uint8:
                od = interpreter.get_output_details()
                scale, zp = od[0]["quantization"]
                out = (out.astype(np.float32) - zp) * scale
            y_pred_tflite.append(np.argmax(out[0]))

    y_pred_tflite = np.array(y_pred_tflite)
    tflite_acc = float(np.mean(y_true == y_pred_tflite))

    return {
        "float32_accuracy": round(float_acc, 4),
        "int8_accuracy": round(tflite_acc, 4),
        "accuracy_delta": round(tflite_acc - float_acc, 4),
        "test_samples": len(y_true),
    }


# ─────────────────────────────────────────────
# 6. Save benchmark report
# ─────────────────────────────────────────────
def save_benchmark_report(results: dict, output_path: str):
    """Save the full benchmark report to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Benchmark saved to {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Zar3y — TFLite INT8 Quantization & Benchmark")
    print("=" * 60)

    # Step 1: Convert
    print("\n[1/4] Creating representative dataset …")
    rep_gen = create_representative_dataset(TRAIN_DIR)

    print("[2/4] Converting to TFLite INT8 …")
    convert_to_tflite_int8(MODEL_PATH, rep_gen, TFLITE_PATH)

    # Step 2: Size benchmark
    print("[3/4] Benchmarking model size …")
    size_report = benchmark_model_size(MODEL_PATH, TFLITE_PATH)
    print(f"  FP32: {size_report['float32_size_mb']} MB")
    print(f"  INT8: {size_report['int8_size_mb']} MB")
    print(f"  Compression: {size_report['compression_ratio']}×")

    # Step 3: Latency benchmark
    print("[4/4] Benchmarking latency …")
    latency_report = benchmark_latency(MODEL_PATH, TFLITE_PATH, TEST_DIR)
    print(f"  FP32 avg: {latency_report['float32_avg_latency_ms']} ms")
    print(f"  INT8 avg: {latency_report['int8_avg_latency_ms']} ms")
    print(f"  Speedup:  {latency_report['speedup']}×")

    # Step 4: Accuracy benchmark
    print("[5/5] Benchmarking accuracy …")
    acc_report = benchmark_accuracy(MODEL_PATH, TFLITE_PATH, TEST_DIR)
    print(f"  FP32 accuracy: {acc_report['float32_accuracy']}")
    print(f"  INT8 accuracy: {acc_report['int8_accuracy']}")
    print(f"  Delta:         {acc_report['accuracy_delta']}")

    # Save combined report
    full_report = {
        "size": size_report,
        "latency": latency_report,
        "accuracy": acc_report,
    }
    save_benchmark_report(full_report, BENCHMARK_PATH)

    print("\n✅ Quantization & benchmark complete!")
