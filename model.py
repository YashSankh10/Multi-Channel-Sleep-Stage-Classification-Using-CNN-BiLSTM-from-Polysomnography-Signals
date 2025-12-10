# model.py
#
# Defines the multi-channel CNN-BiLSTM model architecture.

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense

def build_multichannel_model(input_shape, num_classes):
    """
    Builds the 1D CNN + BiLSTM model for multi-channel input.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, channels).
        num_classes (int): Number of output classes.
        
    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    
    # --- 1D CNN Feature Extractor ---
    x = Conv1D(filters=64, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=128, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=256, kernel_size=7, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # --- BiLSTM for Temporal Sequence Learning ---
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.5)(x)
    
    # --- Output Layer ---
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model