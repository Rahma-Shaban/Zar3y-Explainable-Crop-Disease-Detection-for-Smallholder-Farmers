"""
Zar3y — Transfer-Learning Training (Requirement 2)

This module handles:
1. Loading MobileNetV3-Small pretrained on ImageNet
2. Phase 1: Freeze backbone, train classifier head for 10 epochs
3. Phase 2: Unfreeze last 30 layers, fine-tune for 10 epochs at LR/10
4. Early stopping on val_loss (patience=5)
5. Save best checkpoint to models/best_model.h5
6. Class-weighted loss if any class has < 500 images

Deliverables:
- models/best_model.h5
- Training history for plotting curves
"""

# TODO: Implement training pipeline
#
# Key functions to implement:
#
# def build_model(num_classes, image_size):
#     """Build MobileNetV3-Small with custom classifier head."""
#     pass
#
# def compute_class_weights(train_labels):
#     """Compute class weights for imbalanced classes."""
#     pass
#
# def train_phase1(model, train_ds, val_ds, epochs, class_weights):
#     """Phase 1: frozen backbone training."""
#     pass
#
# def train_phase2(model, train_ds, val_ds, epochs, class_weights):
#     """Phase 2: fine-tune last 30 layers at 10x lower LR."""
#     pass
#
# if __name__ == "__main__":
#     # Run full training pipeline
#     pass
