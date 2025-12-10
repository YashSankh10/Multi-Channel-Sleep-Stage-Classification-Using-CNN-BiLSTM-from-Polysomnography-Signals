# data_loader.py
#
# Handles loading and preprocessing of multi-channel EDF data.

import os
import numpy as np
import mne
import glob

mne.set_log_level('ERROR')

def load_and_preprocess_data(subject_ids, config):
    """
    Loads, filters, resamples, and epochs data for a given list of subjects.
    
    Args:
        subject_ids (list): List of subject integers (0-19) to process.
        config (module): The configuration module.
    
    Returns:
        tuple: (X, y, groups) containing epoched data, labels, and group identifiers.
    """
    all_epochs_data = []
    all_labels = []
    groups = []
    
    # Get all PSG files from the dataset directory
    psg_files = sorted(glob.glob(os.path.join(config.DATASET_PATH, "*PSG.edf")))
    ann_files = sorted(glob.glob(os.path.join(config.DATASET_PATH, "*Hypnogram.edf")))
    # Filter files based on subject_ids
    subject_psg_files = [psg_files[i] for i in subject_ids]
    subject_ann_files = [ann_files[i] for i in subject_ids]
    
    print(f"\nProcessing subjects: {subject_ids}...")
    for i, (psg_file, ann_file) in enumerate(zip(subject_psg_files, subject_ann_files)):
        raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel='Event marker')
        annots = mne.read_annotations(ann_file)
        raw.set_annotations(annots)
        
        # --- ⚙️ Preprocessing Pipeline ---
        # 1. Select the 3 required channels
        raw.pick_channels(list(config.CHANNELS.values()))
        
        # 2. 4th-order Butterworth bandpass filter
        raw.filter(0.3, 35., method='iir', iir_params={'order': 4, 'ftype': 'butter'})
        
        # 3. Resample
        raw.resample(config.FS, npad='auto')
        
        # 4. Create epochs
        events, _ = mne.events_from_annotations(
            raw, event_id=config.LABEL_MAP, chunk_duration=config.EPOCH_DURATION_S
        )
        epochs = mne.Epochs(
            raw, events, tmin=0., tmax=config.EPOCH_DURATION_S, baseline=None, preload=True
        )
        
        X = epochs.get_data() * 1e6  # to microvolts. Shape: (n_epochs, n_channels, n_samples)
        y = epochs.events[:, 2]
        
        # Per-epoch, per-channel z-score normalization
        for epoch_idx in range(X.shape[0]):
            for chan_idx in range(X.shape[1]):
                mean = np.mean(X[epoch_idx, chan_idx, :])
                std = np.std(X[epoch_idx, chan_idx, :])
                X[epoch_idx, chan_idx, :] = (X[epoch_idx, chan_idx, :] - mean) / std
            
        all_epochs_data.append(X)
        all_labels.append(y)
        groups.extend([subject_ids[i]] * len(y))
        
        print(f"  - Subject {subject_ids[i]}: {len(y)} epochs loaded.")

    X = np.concatenate(all_epochs_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Transpose for Keras: (n_epochs, n_samples, n_channels)
    X = X.transpose(0, 2, 1)
    
    return X, y, np.array(groups)