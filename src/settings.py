"""
Zar3y — Project-Wide Settings & Constants

All locked hyperparameters, paths, and class definitions live here.
Import this module from any script to ensure consistency.
"""

# TODO: Implement the following constants and configurations
#
# - RANDOM_SEED = 42
# - IMAGE_SIZE = 224
# - BATCH_SIZE = 32
# - NUM_CLASSES = 10
#
# - CLASS_NAMES: list of 10 locked class names
#   [Tomato___healthy, Tomato___Early_blight, Tomato___Late_blight,
#    Tomato___Leaf_Mold, Potato___healthy, Potato___Early_blight,
#    Potato___Late_blight, Pepper_bell___healthy,
#    Pepper_bell___Bacterial_spot, Corn___Common_rust]
#
# - DISEASE_INFO: dict mapping class name → {description, next_step}
#   Static text for plain-language explanations shown in the UI.
#
# - Paths: DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIELD_PHOTOS_DIR
#
# - Training hyperparameters:
#   PHASE1_EPOCHS = 10
#   PHASE2_EPOCHS = 10
#   PHASE2_UNFREEZE_LAYERS = 30
#   INITIAL_LR, FINE_TUNE_LR (10× lower)
#   EARLY_STOPPING_PATIENCE = 5
#
# - Split ratios: TRAIN_RATIO=0.70, VAL_RATIO=0.15, TEST_RATIO=0.15
