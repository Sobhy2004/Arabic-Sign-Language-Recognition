import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from optimized_gesture_model import get_optimized_model
from data_utils import load_data_landmarks, create_dummy_landmarks

def main():
    parser = argparse.ArgumentParser(description="Train optimized sign language model (Landmarks)")
    parser.add_argument("--data_path", type=str, default="data/signs", help="Path to preprocessed data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dummy", action="store_true", help="Use dummy data for testing")
    args = parser.parse_args()

    if args.dummy:
        print("Using dummy data for testing...")
        X, y, classes = create_dummy_landmarks()
    else:
        if not os.path.exists(args.data_path):
            print(f"Data path {args.data_path} not found.")
            return
        X, y, classes = load_data_landmarks(args.data_path)
        if X is None:
            print("Failed to load data landmarks.")
            return

    num_classes = len(classes)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Landmarks shape is (22, 126)
    model = get_optimized_model(input_shape=(22, 126), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Starting training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=args.epochs,
              batch_size=args.batch_size)

    model.save("optimized_sign_language_model.keras")
    print("Model saved to optimized_sign_language_model.keras")

if __name__ == "__main__":
    main()
