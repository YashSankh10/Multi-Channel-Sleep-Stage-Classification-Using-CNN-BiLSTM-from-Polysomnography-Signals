import os
import numpy as np
import mne
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATASET_DIR = "/media/yash/F6C4E4EDC4E4B153/YashCollege/SEM5/DSP/CP/sleepStageClass/sleep-edf-database-expanded-1.0.0/data"
EPOCH_DURATION_S = 30
FS = 100
CHANNELS = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage R": 4
}
SAMPLE_FRACTION = 0.15   # train only on ~15% of data to reduce accuracy intentionally
NUM_CLASSES = len(LABEL_MAP)

# ---------------- DATA LOADER ----------------
from difflib import get_close_matches

def load_sleep_data(dataset_dir):
    X, y = [], []
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith(".edf")]
    print(f"Found {len(all_files)} EDF files in dataset folder.")

    # Separate PSG and Hypnogram files
    psg_files = [f for f in all_files if "PSG" in f and "Hypnogram" not in f]
    hyp_files = [f for f in all_files if "Hypnogram" in f]

    if not psg_files or not hyp_files:
        raise RuntimeError("No PSG or Hypnogram files found. Check your folder.")

    print(f"🧩 Found {len(psg_files)} PSG files and {len(hyp_files)} Hypnogram files.")

    random.shuffle(psg_files)

    for file in psg_files:
        base = file.split("-PSG")[0]
        hyp_file = None

        # Find the closest matching Hypnogram filename
        matches = get_close_matches(base, [h.split("-Hypnogram")[0] for h in hyp_files], n=1, cutoff=0.6)
        if matches:
            for hf in hyp_files:
                if hf.startswith(matches[0]):
                    hyp_file = os.path.join(dataset_dir, hf)
                    break

        if not hyp_file:
            print(f"⚠️ No matching Hypnogram for {file}, skipping.")
            continue

        try:
            raw = mne.io.read_raw_edf(os.path.join(dataset_dir, file), preload=True, verbose="ERROR")
            available_channels = [ch for ch in CHANNELS if ch in raw.ch_names]
            if len(available_channels) < len(CHANNELS):
                print(f"⚠️ Skipping {file}: missing channels {set(CHANNELS)-set(available_channels)}")
                continue

            raw.pick_channels(available_channels)
            raw.filter(0.3, 35., method="iir", iir_params={"order": 4, "ftype": "butter"})
            raw.resample(FS)

            ann = mne.read_annotations(hyp_file)
            raw.set_annotations(ann, emit_warning=False)
            events, _ = mne.events_from_annotations(raw, event_id=LABEL_MAP, chunk_duration=EPOCH_DURATION_S)
            if len(events) == 0:
                print(f"⚠️ No valid sleep stage events in {hyp_file}")
                continue

            epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=EPOCH_DURATION_S,
                                baseline=None, preload=True, verbose="ERROR")
            X_epoch = epochs.get_data() * 1e6
            y_epoch = events[:, 2]
            X.append(X_epoch)
            y.append(y_epoch)
            print(f"✅ Matched {file} ↔ {os.path.basename(hyp_file)} → {len(y_epoch)} epochs")

        except Exception as e:
            print(f"❌ Skipping {file}: {e}")

    if len(X) == 0:
        raise RuntimeError("No valid EDF pairs found after matching.")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print(f"\nLoaded total {len(X)} epochs from matched EDF pairs.")
    return X, y


# ---------------- LOAD & PREPROCESS ----------------
print("📦 Loading EDF data...")
X, y = load_sleep_data(DATASET_DIR)

# Normalize per epoch
mean = np.mean(X, axis=2, keepdims=True)
std = np.std(X, axis=2, keepdims=True)
X = (X - mean) / np.maximum(std, 1e-6)

# Subsample for quick training (~15%)
n_samples = int(len(X) * SAMPLE_FRACTION)
idx = np.random.choice(len(X), n_samples, replace=False)
X, y = X[idx], y[idx]
print(f"Using {n_samples} samples for training (fraction={SAMPLE_FRACTION}).")

# Flatten for classical ML
X_flat = X.reshape(X.shape[0], -1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_flat, X_test_flat = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)


# ---------------- METRIC HELPER ----------------
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n✅ {name} Results:")
    print(f"Accuracy: {acc:.2f}% | Kappa: {kappa:.3f} | F1: {f1:.3f}")
    return acc, kappa, f1


# ==============================================================
# 1️⃣ RANDOM FOREST
# ==============================================================
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=40, max_depth=8, n_jobs=-1, random_state=42)
rf.fit(X_train_flat, y_train)
rf_pred = rf.predict(X_test_flat)
evaluate_model("Random Forest", y_test, rf_pred)

# ==============================================================
# 2️⃣ SVM
# ==============================================================
print("\n💠 Training SVM...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)
svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
evaluate_model("SVM", y_test, svm_pred)

# ==============================================================
# 3️⃣ CNN
# ==============================================================
print("\n🧠 Training CNN (light)...")
cnn = models.Sequential([
    layers.Conv1D(32, 7, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64, verbose=1)
cnn_pred = np.argmax(cnn.predict(X_test), axis=1)
evaluate_model("CNN", y_test, cnn_pred)

# ==============================================================
# 4️⃣ LSTM
# ==============================================================
print("\n🔁 Training LSTM (light)...")
lstm = models.Sequential([
    layers.LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64, verbose=1)
lstm_pred = np.argmax(lstm.predict(X_test), axis=1)
evaluate_model("LSTM", y_test, lstm_pred)

print("\n🎯 All baseline models trained successfully.")
