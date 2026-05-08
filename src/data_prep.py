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

# TODO: Implement data preparation pipeline
#
# Key functions to implement:
#
# def download_dataset():
#     """Download PlantVillage from Kaggle if not already present."""
#     pass
#
# def filter_classes(data_dir, locked_classes):
#     """Filter dataset to only the 10 locked classes."""
#     pass
#
# def stratified_split(data_dir, train_ratio, val_ratio, test_ratio, seed):
#     """Create stratified train/val/test split."""
#     pass
#
# def report_class_counts(train_ds, val_ds, test_ds):
#     """Compute and print per-class counts for each split."""
#     pass
#
# def build_augmentation_pipeline():
#     """Build tf.data augmentation pipeline with locked transforms."""
#     pass
#
# def save_augmentation_samples(dataset, output_path):
#     """Save a grid of augmented samples for visual confirmation."""
#     pass
#
# if __name__ == "__main__":
#     # Run full data preparation pipeline
#     pass
