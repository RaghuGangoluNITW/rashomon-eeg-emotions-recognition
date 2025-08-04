# Configurations and constants.
"""
Configuration file for Rashomon EEG Emotion Recognition.
Edit these values to change default settings across the package.
"""

import torch

# ===============================
# DATA
# ===============================
DATASET_PATH = "data/deap.mat"   # Default dataset path
SAMPLING_RATE = 256              # Hz
NUM_CHANNELS = 32                 # Number of EEG channels

# ===============================
# MODEL
# ===============================
INPUT_DIM = 128                  # Feature vector length after extraction
HIDDEN_DIM = 64
OUTPUT_DIM = 2                   # Binary classification (e.g., high/low emotion)

# ===============================
# TRAINING
# ===============================
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# RASHOMON LOOP
# ===============================
PARAM_GRID = [
    {"hidden_dim": 32, "output_dim": 2},
    {"hidden_dim": 64, "output_dim": 2},
    {"hidden_dim": 128, "output_dim": 2}
]
