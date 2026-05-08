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

# TODO: Implement Grad-CAM pipeline
#
# Key functions to implement:
#
# def get_grad_cam_model(model, last_conv_layer_name):
#     """Create a Grad-CAM model targeting the last conv block."""
#     pass
#
# def compute_grad_cam(model, image, class_index):
#     """Compute Grad-CAM heatmap for a given image and class."""
#     pass
#
# def overlay_heatmap(image, heatmap, alpha=0.4):
#     """Overlay Grad-CAM heatmap on the original image."""
#     pass
#
# def save_grad_cam_examples(model, test_images, class_names, output_dir):
#     """Save side-by-side originals + Grad-CAM for 5 representative images."""
#     pass
#
# def run_ood_evaluation(model, field_photos_dir, class_names):
#     """Run inference on OOD field photos and compute accuracy."""
#     pass
#
# def annotate_failure_cases(model, failures, output_dir):
#     """Generate Grad-CAM overlays for failure cases."""
#     pass
#
# if __name__ == "__main__":
#     # Run Grad-CAM generation + OOD evaluation
#     pass
