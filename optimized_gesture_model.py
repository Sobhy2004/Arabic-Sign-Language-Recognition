import tensorflow as tf
from tensorflow.keras import layers, models

def get_optimized_model(input_shape=(22, 126), num_classes=10):
    """
    Creates an optimized GRU/LSTM model for sign language landmarks.

    Landmark Input: (Sequence Length, 126 Features)
    - Sequence Length: 22 frames
    - 126 Features: 2 hands * 21 landmarks * 3 (x,y,z) coordinates

    Model optimizations:
    - 1D Convolution over landmarks for local feature extraction.
    - Bidirectional GRU (faster and often better for small sequences).
    - Lightweight architecture for mobile/graduation project.
    """
    model = models.Sequential(name="Landmark_Sign_Language_Model")

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # 1D Conv - captures relationships between nearby landmarks (e.g., thumb vs index)
    model.add(layers.Conv1D(64, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    # Temporal processing
    model.add(layers.Bidirectional(layers.GRU(128, return_sequences=False)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    # Dense classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    # Test building the model
    model = get_optimized_model()
    model.summary()
    print(f"\nTrainable Parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
