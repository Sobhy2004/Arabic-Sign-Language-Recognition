import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from optimized_gesture_model import get_optimized_model
import cv2

def load_data(data_path, img_size=(64, 48), sequence_length=22):
    """
    Load data from the directory structure:
    data_path/class_name/sample_dir/frame_img.jpg
    """
    x_data = []
    y_labels = []
    classes = sorted(os.listdir(data_path))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        if not os.path.isdir(cls_path):
            continue

        for sample in os.listdir(cls_path):
            sample_path = os.path.join(cls_path, sample)
            if not os.path.isdir(sample_path):
                continue

            frames = []
            frame_files = sorted(os.listdir(sample_path))
            for f in frame_files:
                img = cv2.imread(os.path.join(sample_path, f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size[1], img_size[0])) # OpenCV uses (width, height)
                frames.append(img / 255.0)
                if len(frames) == sequence_length:
                    break

            # Pad or truncate frames
            if len(frames) == 0:
                continue
            if len(frames) < sequence_length:
                padding = [np.zeros((img_size[0], img_size[1]))] * (sequence_length - len(frames))
                frames.extend(padding)

            x_data.append(np.expand_dims(np.array(frames), -1))
            y_labels.append(class_to_idx[cls])

    return np.array(x_data), tf.keras.utils.to_categorical(y_labels, num_classes=len(classes)), classes

def create_dummy_data(num_samples=100, sequence_length=22, img_size=(64, 48), num_classes=10):
    """Generates synthetic data for testing the training pipeline."""
    X = np.random.rand(num_samples, sequence_length, img_size[0], img_size[1], 1).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    return X, y, [f"Class_{i}" for i in range(num_classes)]

def main():
    parser = argparse.ArgumentParser(description="Train optimized gesture recognition model")
    parser.add_argument("--data_path", type=str, default="Save1", help="Path to preprocessed data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dummy", action="store_true", help="Use dummy data for testing")
    args = parser.parse_args()

    if args.dummy:
        print("Using dummy data for testing...")
        X, y, classes = create_dummy_data()
    else:
        if not os.path.exists(args.data_path):
            print(f"Data path {args.data_path} not found. Please run preprocessing first.")
            return
        X, y, classes = load_data(args.data_path)

    num_classes = len(classes)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_optimized_model(input_shape=(22, 64, 48, 1), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Starting training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=args.epochs,
              batch_size=args.batch_size)

    model.save("optimized_gesture_model.keras")
    print("Model saved to optimized_gesture_model.keras")

if __name__ == "__main__":
    main()
