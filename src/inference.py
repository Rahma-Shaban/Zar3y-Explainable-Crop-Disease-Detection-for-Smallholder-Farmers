"""
Zar3y — TFLite Inference Runtime

Reusable inference helper used by the FastAPI backend.

Responsibilities:
1. Load the quantized TFLite model
2. Preprocess input images (resize, normalize)
3. Run inference and return predicted class + confidence
4. Generate Grad-CAM overlay from the full Keras model
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf

# ─────────────────────────────────────────────
# Resolve paths relative to project root
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_quantized.tflite")
KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.keras")
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "class_names.json")

IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "conv_1"  # last conv layer in MobileNetV3Small


class TFLitePredictor:
    """Wrapper around the quantized TFLite model."""

    def __init__(self, model_path: str = TFLITE_MODEL_PATH):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load class names
        with open(CLASS_NAMES_PATH, "r") as f:
            self.class_names = json.load(f)

        # Lazy-load the full Keras model only when Grad-CAM is requested
        self._keras_model = None

    # ── preprocessing ─────────────────────────
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw bytes → preprocessed float32 numpy array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype(np.float32)
        # Match the MobileNetV3 preprocess_input: scale to [-1, 1]
        img = (img / 127.5) - 1.0
        return np.expand_dims(img, axis=0)

    # ── TFLite predict ────────────────────────
    def predict(self, image_bytes: bytes):
        """Run TFLite inference. Returns (class_name, confidence, probs)."""
        input_data = self.preprocess(image_bytes)

        # Match the expected dtype of the TFLite model
        input_dtype = self.input_details[0]["dtype"]
        if input_dtype == np.uint8:
            # Quantized model: de-normalize back to [0,255]
            input_data = ((input_data + 1.0) * 127.5).astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32)

        self.interpreter.set_tensor(
            self.input_details[0]["index"], input_data
        )
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        # Dequantize if needed
        if output_data.dtype == np.uint8:
            scale, zero_point = self.output_details[0]["quantization"]
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        probs = output_data[0]
        pred_idx = int(np.argmax(probs))
        class_name = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        return class_name, confidence, probs

    # ── Grad-CAM (uses full Keras model) ──────
    def _load_keras_model(self):
        if self._keras_model is None:
            self._keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        return self._keras_model

    def generate_gradcam(self, image_bytes: bytes) -> np.ndarray:
        """Generate a Grad-CAM heatmap overlay as a BGR image (224×224)."""
        model = self._load_keras_model()

        # Decode and resize
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMAGE_SIZE)

        # Preprocess for MobileNetV3
        img_array = img_resized.astype(np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        img_tensor = tf.cast(img_array, tf.float32)

        # Build sub-models
        base_model = model.get_layer("MobileNetV3Small")
        last_conv = base_model.get_layer(LAST_CONV_LAYER)

        conv_model = tf.keras.Model(
            base_model.input, last_conv.output
        )

        # Classifier sub-model (layers after last conv)
        classifier_input = tf.keras.Input(shape=last_conv.output.shape[1:])
        x = classifier_input
        passed = False
        for layer in base_model.layers:
            if layer.name == LAST_CONV_LAYER:
                passed = True
                continue
            if passed:
                x = layer(x)
        # Top layers: GlobalAveragePooling → Dropout → Dense
        x = model.layers[-3](x)
        x = model.layers[-2](x)
        x = model.layers[-1](x)
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            conv_output = conv_model(img_tensor)
            tape.watch(conv_output)
            preds = classifier_model(conv_output)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.math.reduce_max(heatmap) + 1e-8
        heatmap = heatmap.numpy()

        # Overlay on original image
        heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        original_bgr = cv2.resize(img_bgr, IMAGE_SIZE)
        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

        return overlay

    def predict_with_gradcam(self, image_bytes: bytes):
        """Run TFLite prediction + Grad-CAM overlay."""
        class_name, confidence, probs = self.predict(image_bytes)
        overlay = self.generate_gradcam(image_bytes)
        return class_name, confidence, probs, overlay
