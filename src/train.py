import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# -----------------------------
# Config
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

BASE_PATH = r"D:\Zar3y-Explainable-Crop-Disease-Detection-for-Smallholder-Farmers"

TRAIN_DIR = os.path.join(BASE_PATH, "data", "split_data", "train")
VAL_DIR   = os.path.join(BASE_PATH, "data", "split_data", "val")

MODEL_PATH = os.path.join(BASE_PATH, "models", "best_model.keras")
HISTORY_PATH = os.path.join(BASE_PATH, "outputs", "training_history.json")

os.makedirs(os.path.join(BASE_PATH, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_PATH, "outputs"), exist_ok=True)


# -----------------------------
# Load datasets
# -----------------------------
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

print("Classes:", train_ds.class_names)
NUM_CLASSES = len(train_ds.class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# -----------------------------
# Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomZoom(0.2),
])


# -----------------------------
# Build model
# -----------------------------
def build_model(num_classes):

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)
    x = preprocess_input(x)

    base_model = MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    return model, base_model


# -----------------------------
# Class weights (FIXED)
# -----------------------------
def compute_class_weights(train_ds):

    labels = []

    for _, y in train_ds:
        labels.extend(y.numpy().tolist())

    labels = np.array(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    return dict(enumerate(weights))


# -----------------------------
# Callbacks
# -----------------------------
def get_callbacks():

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2
        )
    ]


# -----------------------------
# Phase 1
# -----------------------------
def train_phase1(model, train_ds, val_ds, epochs, class_weights):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(),
        class_weight=class_weights
    )

    return history


# -----------------------------
# Phase 2
# -----------------------------
def train_phase2(model, base_model, train_ds, val_ds, epochs, class_weights):

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(),
        class_weight=class_weights
    )

    return history


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    model, base_model = build_model(NUM_CLASSES)

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_ds)
    print("Class Weights:", class_weights)

    print("\n🚀 Phase 1 Training...")
    history1 = train_phase1(model, train_ds, val_ds, 10, class_weights)

    print("\n🚀 Phase 2 Fine-tuning...")
    history2 = train_phase2(model, base_model, train_ds, val_ds, 10, class_weights)

    # combine history
combined = {}
for k in history1.history:
    combined[k] = history1.history[k] + history2.history[k]

# convert numpy types → python native types
for k in combined:
    combined[k] = [float(x) for x in combined[k]]

with open(HISTORY_PATH, "w") as f:
    json.dump(combined, f, indent=4)

    print("\n✅ Training completed!")
    print(f"Model saved at: {MODEL_PATH}")