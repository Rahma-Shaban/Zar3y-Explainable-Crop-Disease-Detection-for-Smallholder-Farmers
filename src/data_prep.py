"""
Zar3y — Data Preparation & Augmentation (Requirement 1)

This module handles:
1. Downloading/loading the PlantVillage dataset
2. Filtering to the 10 locked classes
3. Stratified 70/15/15 train/val/test split (seed=42)
4. Computing and reporting per-class counts
5. Building a tf.data augmentation pipeline:
   - Random horizontal flip
   - Random rotation (±15°)
   - Random brightness (±20%)
   - Random contrast (±20%)
   - Random zoom
6. Saving augmentation samples to outputs/augmentation_samples.png

Deliverables:
- Per-class count table (print + update README)
- outputs/augmentation_samples.png
"""
import os
import shutil
import random
import kagglehub
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd


# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)

LOCKED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper,_bell___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Corn_(maize)___Common_rust_"
]


# -----------------------------
# 1. Download Dataset
# -----------------------------
def download_dataset():
    path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print("Dataset path:", path)
    return path


# -----------------------------
# 2. Filter Classes
# -----------------------------
def filter_classes(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for cls in LOCKED_CLASSES:
        src = os.path.join(base_dir, cls)
        dst = os.path.join(output_dir, cls)

        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    print("Filtering done!")


# -----------------------------
# 3. Split Dataset
# -----------------------------
def split_dataset(base_dir, output_dir):

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    def copy_images(images, src, dst):
        os.makedirs(dst, exist_ok=True)
        for img in images:
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    for cls in os.listdir(base_dir):
        class_path = os.path.join(base_dir, cls)
        images = os.listdir(class_path)

        train_imgs, temp = train_test_split(images, test_size=0.3, random_state=SEED)
        val_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=SEED)

        copy_images(train_imgs, class_path, os.path.join(train_dir, cls))
        copy_images(val_imgs, class_path, os.path.join(val_dir, cls))
        copy_images(test_imgs, class_path, os.path.join(test_dir, cls))

    print("Dataset split done!")
    return train_dir, val_dir, test_dir


# -----------------------------
# 4. Count Images
# -----------------------------
def count_images(folder):
    counts = {}
    for cls in os.listdir(folder):
        cls_path = os.path.join(folder, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len(os.listdir(cls_path))
    return counts


# -----------------------------
# 5. Report Distribution
# -----------------------------
def report_distribution(train_counts, val_counts, test_counts):
    data = []
    for cls in sorted(LOCKED_CLASSES):
        data.append({
            'Class': cls,
            'Train': train_counts.get(cls, 0),
            'Validation': val_counts.get(cls, 0),
            'Test': test_counts.get(cls, 0)
        })

    df = pd.DataFrame(data)
    print("\nClass Distribution:")
    print(df)


# -----------------------------
# 6. Augmentation
# -----------------------------
def build_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomZoom(0.2)
    ])


# -----------------------------
# 7. Save Augmentation Samples
# -----------------------------
def save_aug_samples(train_dir, output_path):

    aug = build_augmentation()

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=9
    )

    for images, _ in train_ds.take(1):
        aug_images = aug(images)

        plt.figure(figsize=(10, 10))

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(aug_images[i].numpy().astype("uint8"))
            plt.axis("off")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

    print("Saved:", output_path)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    #  Download dataset
    raw_path = download_dataset()

    
    base_dir = os.path.join(raw_path, "plantvillage dataset", "color")

    #  Local project folders
    filtered_dir = "data/filtered_data"
    split_dir = "data/split_data"
    output_img_path = "outputs/augmentation_samples.png"

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    #  Pipeline
    filter_classes(base_dir, filtered_dir)

    train_dir, val_dir, test_dir = split_dataset(filtered_dir, split_dir)

    train_counts = count_images(train_dir)
    val_counts = count_images(val_dir)
    test_counts = count_images(test_dir)

    report_distribution(train_counts, val_counts, test_counts)

    save_aug_samples(train_dir, output_img_path)