import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    """
    Simple Attention layer to weight sequence frames by importance.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = tf.squeeze(tf.tanh(tf.matmul(x, self.W) + self.b), axis=-1)
        at = tf.nn.softmax(et)
        at = tf.expand_dims(at, axis=-1)
        output = x * at
        return tf.reduce_sum(output, axis=1)

    def get_config(self):
        config = super().get_config()
        return config

def get_sign_language_model(input_shape=(22, 126), num_classes=28):
    """
    Fully Optimized Sign Language Model with Attention.
    """
    model = models.Sequential(name="Optimized_Attention_Sign_Model")

    model.add(layers.Input(shape=input_shape))

    # Spatial extraction
    model.add(layers.Conv1D(128, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(64, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Temporal sequence modeling
    model.add(layers.Bidirectional(layers.GRU(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))))
    model.add(layers.Dropout(0.4))

    # Attention Mechanism
    model.add(AttentionLayer())

    # Classification head
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    model = get_sign_language_model()
    model.summary()
