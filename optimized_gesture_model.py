import tensorflow as tf
from tensorflow.keras import layers, models

def get_optimized_model(input_shape=(22, 64, 48, 1), num_classes=10):
    """
    Creates an optimized CNN-LSTM model for sign language translation.

    Optimizations for Sign Language:
    - Deep CNN with Residual-like structure for better feature extraction from complex hand shapes.
    - Global Average Pooling (GAP) instead of a huge Flatten layer to reduce parameters.
    - ReLU activations and BatchNormalization for training stability.
    - LSTM with Dropout for capturing temporal dynamics of sign motions.
    """
    model = models.Sequential(name="Sign_Language_Model")

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Layer 1
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same')))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Activation('relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    # Layer 2
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same')))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Activation('relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    # Layer 3
    model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same')))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Activation('relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    # Transition to Temporal features
    # Global Average Pooling reduces spatial dimensions from (H,W,C) to (1,1,C)
    # and we flatten that to (C) per frame.
    model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))
    model.add(layers.TimeDistributed(layers.Dropout(0.3)))

    # RNN part - Sequence learning
    model.add(layers.LSTM(128, return_sequences=False, dropout=0.2))

    # Output layer
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    model = get_optimized_model()
    model.summary()
    print(f"\nTrainable Parameters: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
