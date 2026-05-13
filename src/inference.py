"""
Zar3y — TFLite Inference Runtime

Reusable inference helper used by the FastAPI backend.
"""

import io
import os
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from src.settings import CLASS_NAMES


class TFLitePredictor:
    def __init__(self):

        # -----------------------------
        # FIX: correct model path (GitHub-safe)
        # -----------------------------
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")

        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.img_size = 224

    # -------------------------
    # preprocessing
    # -------------------------
    def preprocess(self, image_bytes):

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.img_size, self.img_size))

        image_np = np.array(image).astype(np.float32)

        # match training pipeline
        image_np = preprocess_input(image_np)

        image_np = np.expand_dims(image_np, axis=0)

        return image_np

    # -------------------------
    # prediction
    # -------------------------
    def predict(self, image_bytes):

        image = self.preprocess(image_bytes)

        predictions = self.model.predict(image, verbose=0)[0]

        class_index = int(np.argmax(predictions))

        class_name = CLASS_NAMES[class_index]
        confidence = float(predictions[class_index])

        return class_name, confidence, predictions.tolist()

    # -------------------------
    # Grad-CAM visualization
    # -------------------------
    def generate_gradcam(self, image_bytes):

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        image_np = np.array(image)

        # Prepare for model
        img_array = self.preprocess(image_bytes)

        # Build Grad-CAM model
        base_model = self.model.get_layer("MobileNetV3Small")
        last_conv_layer_name = "conv_1" # Last conv block
        try:
            last_conv_layer = base_model.get_layer(last_conv_layer_name)
        except ValueError:
            # fallback if name is different
            last_conv_layer = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
        
        last_conv_model = tf.keras.Model(base_model.input, last_conv_layer.output)

        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        passed = False
        for layer in base_model.layers:
            if layer.name == last_conv_layer.name:
                passed = True
                continue
            if passed:
                x = layer(x)

        x = self.model.layers[-3](x)
        x = self.model.layers[-2](x)
        x = self.model.layers[-1](x)
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            conv_output = last_conv_model(img_array)
            tape.watch(conv_output)
            preds = classifier_model(conv_output)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_heat = tf.math.reduce_max(heatmap)
        if max_heat != 0:
            heatmap /= max_heat
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
            0.6,
            heatmap,
            0.4,
            0
        )
        return overlay
