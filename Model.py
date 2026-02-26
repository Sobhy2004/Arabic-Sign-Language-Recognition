import tensorflow as tf
from tensorflow.keras import layers, models

def get_sign_language_model(input_shape=(22, 126), num_classes=28):
    """
    Optimized Landmark-based Model for Fingerspilling (Alphabets).
    Uses a combination of 1D Conv and GRU to capture spatial and temporal features of finger movements.
    """
    model = models.Sequential(name="Fingerspilling_Model")

    # Input: (Sequence length, Landmark features)
    model.add(layers.Input(shape=input_shape))

    # Extract local spatial relationships between landmarks in each frame
    model.add(layers.Conv1D(128, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(64, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Temporal sequence modeling
    model.add(layers.Bidirectional(layers.GRU(128, return_sequences=True)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Bidirectional(layers.GRU(64, return_sequences=False)))
    model.add(layers.BatchNormalization())

    # Classification head
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    model = get_sign_language_model()
    model.summary()
