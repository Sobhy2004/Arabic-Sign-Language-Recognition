import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def get_sign_language_model(input_shape=(22, 126), num_classes=28):
    """
    Accuracy-Optimized Landmark Model for Fingerspilling.
    Changes:
    - Added L2 regularization to prevent overfitting.
    - Used SELU activation for better gradient flow in deep networks.
    - Increased depth of dense layers for better classification power.
    """
    model = models.Sequential(name="Accuracy_Sign_Language_Model")

    model.add(layers.Input(shape=input_shape))

    # Spatial extraction with L2 Regularization
    model.add(layers.Conv1D(128, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(64, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Deep Temporal Sequence modeling
    model.add(layers.Bidirectional(layers.GRU(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))))
    model.add(layers.Dropout(0.4))
    model.add(layers.Bidirectional(layers.GRU(128, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))))
    model.add(layers.BatchNormalization())

    # High-capacity classification head
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    model = get_sign_language_model()
    model.summary()
