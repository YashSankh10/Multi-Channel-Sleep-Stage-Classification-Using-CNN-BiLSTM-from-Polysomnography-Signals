# config.py
#
# Stores all configurations and hyperparameters for the project.

import os

# --- 📂 Data Handling ---
# Assumes the dataset is in a 'data' subdirectory relative to your script's location.
# Example: ./data/sleep-physionet-database-expanded-1.0.0/
DATASET_PATH = "/content/drive/MyDrive/Yash/Sleep-classification/TEST_1/data"

# --- 🧠 Signal Processing ---
# Using 3 channels for enhanced feature extraction
CHANNELS = {
    "eeg": "EEG Fpz-Cz",
    "eog": "EOG horizontal",
    "emg": "EMG submental"
}

# Epoch duration in seconds
EPOCH_DURATION_S = 30

# Resampling frequency in Hz
FS = 100

# AASM label mapping
LABEL_MAP = {
    'Sleep stage W': 0, 'W': 0,
    'Sleep stage 1': 1, 'N1': 1,
    'Sleep stage 2': 2, 'N2': 2,
    'Sleep stage 3': 3, 'N3': 3,
    'Sleep stage 4': 3, # Merge N4 into N3
    'Sleep stage R': 4, 'R': 4,
}
CLASS_NAMES = ['W', 'N1', 'N2', 'N3', 'R']

# --- 🚀 Model & Training ---
N_SPLITS = 5          # Number of folds for subject-wise cross-validation
BATCH_SIZE = 64
EPOCHS = 75           # Max epochs, with EarlyStopping
LEARNING_RATE = 1e-3
MODEL_WEIGHTS_PATH = "sleep_stage_classifier_3ch.weights.h5"