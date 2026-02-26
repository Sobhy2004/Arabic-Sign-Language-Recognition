import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Model import get_sign_language_model
from data_utils import load_data_landmarks, create_dummy_landmarks

def train_arabic(data_path, landmarker_path="hand_landmarker.task", epochs=100, batch_size=32, dummy=False):
    print("--- Training Arabic Fingerspilling Model ---")

    cache_file = "cache/arabic_landmarks.npy"
    if dummy:
        X, y, classes = create_dummy_landmarks(num_classes=28)
    else:
        X, y, classes = load_data_landmarks(data_path, landmarker_path, cache_path=cache_file)
        if X is None: return

    num_classes = len(classes)

    # Save classes for inference
    os.makedirs("models", exist_ok=True)
    with open("models/arabic_classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = get_sign_language_model(input_shape=(22, 126), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

    model.save("models/arabic_fingerspilling.keras")
    print("Arabic model and classes saved.")

if __name__ == "__main__":
    train_arabic("data/arabic_alphabet", dummy=True, epochs=1)
