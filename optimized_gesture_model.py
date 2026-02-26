import tensorflow as tf
from tensorflow.keras import layers, models

def get_optimized_model(input_shape=(22, 64, 48, 1), num_classes=10):
    """
    Creates an optimized CNN-LSTM model for gesture recognition.

    Optimizations:
    - Uses ReLU activation instead of tanh for faster convergence and better gradients.
    - Adds BatchNormalization to stabilize and speed up training.
    - Uses 128 LSTM units (instead of 500) to reduce parameters for low-end devices.
    - Modern Keras API (TensorFlow 2.x).
    """
    model = models.Sequential(name="Optimized_Gesture_Model")

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # CNN part - Feature extraction from each frame
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same')))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Activation('relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same')))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Activation('relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same')))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Activation('relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dropout(0.5)))

    # RNN part - Temporal feature extraction
    model.add(layers.LSTM(128, return_sequences=False))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    model = get_optimized_model()
    model.summary()

    # Verify parameter count
    trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"\nTotal trainable parameters: {trainable_params:,}")
