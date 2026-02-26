import json
import tensorflow as tf
import numpy as np
from data_utils import load_data_landmarks, create_dummy_landmarks

def evaluate_arabic(model_path, classes_path, data_path=None):
    print("--- Evaluating Arabic Fingerspilling Model ---")
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)

    if data_path and os.path.exists(data_path):
        X, y, _ = load_data_landmarks(data_path, cache_path="cache/arabic_test.npy")
    else:
        print("Using dummy test data...")
        X, y, _ = create_dummy_landmarks(num_samples=50, num_classes=len(classes))

    loss, acc = model.evaluate(X, y)
    print(f"Arabic Accuracy: {acc:.2%}")

if __name__ == "__main__":
    import os
    evaluate_arabic("models/arabic_fingerspilling.keras", "models/arabic_classes.json")
