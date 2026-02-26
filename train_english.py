import argparse
import os
from sklearn.model_selection import train_test_split
from optimized_gesture_model import get_optimized_model
from data_utils import load_data, create_dummy_data

def train_english(data_path, epochs=30, batch_size=32, dummy=False):
    print("--- Training English Sign Language Model ---")
    if dummy:
        print("Using dummy English data...")
        X, y, classes = create_dummy_data(num_classes=10)
    else:
        X, y, classes = load_data(data_path)
        if X is None:
            print(f"Error: No data found at {data_path}")
            return

    num_classes = len(classes)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_optimized_model(input_shape=(22, 64, 48, 1), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Starting English training: {len(X_train)} samples, {num_classes} classes.")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    os.makedirs("models", exist_ok=True)
    model.save("models/sign_language_english.keras")
    print("English model saved to models/sign_language_english.keras")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/english")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    train_english(args.data, args.epochs, args.batch_size, args.dummy)
