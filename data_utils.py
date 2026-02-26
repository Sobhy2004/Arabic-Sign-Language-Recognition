import os
import numpy as np
import cv2
import tensorflow as tf
from Hand_tracking import HandTracker

def load_data_landmarks(data_path, landmarker_path="hand_landmarker.task", sequence_length=22, cache_path=None, mode="auto"):
    """
    Loads data and extracts MediaPipe landmarks.

    Modes:
    - "sequence": Assumes data/class/sample_dir/frames...
    - "static": Assumes data/class/image.jpg (individual images)
    - "auto": Automatically detects mode based on subfolder structure.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading landmarks from cache: {cache_path}")
        cached_data = np.load(cache_path, allow_pickle=True).item()
        return cached_data['X'], cached_data['y'], cached_data['classes']

    tracker = HandTracker(landmarker_path)
    x_data = []
    y_labels = []

    if not os.path.exists(data_path):
        return None, None, []

    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        items = os.listdir(cls_path)

        # Detect mode if auto
        current_mode = mode
        if current_mode == "auto":
            has_subdirs = any(os.path.isdir(os.path.join(cls_path, i)) for i in items)
            current_mode = "sequence" if has_subdirs else "static"

        print(f"Processing class: {cls} (Mode: {current_mode})")

        if current_mode == "sequence":
            for sample in items:
                sample_path = os.path.join(cls_path, sample)
                if not os.path.isdir(sample_path): continue

                sequence = []
                frame_files = sorted(os.listdir(sample_path))
                for f in frame_files:
                    if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                    img = cv2.imread(os.path.join(sample_path, f))
                    if img is None: continue
                    lms, _ = tracker.extract_landmarks(img)
                    sequence.append(lms)
                    if len(sequence) == sequence_length: break

                if not sequence: continue
                if len(sequence) < sequence_length:
                    sequence.extend([np.zeros(126)] * (sequence_length - len(sequence)))
                x_data.append(np.array(sequence))
                y_labels.append(class_to_idx[cls])

        else: # static mode
            for f in items:
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                img = cv2.imread(os.path.join(cls_path, f))
                if img is None: continue

                lms, _ = tracker.extract_landmarks(img)
                # Create a "pseudo-sequence" by repeating the same landmark
                # This makes the static data compatible with the GRU model
                sequence = [lms] * sequence_length
                x_data.append(np.array(sequence))
                y_labels.append(class_to_idx[cls])

    tracker.close()

    if not x_data:
        return None, None, classes

    X = np.array(x_data)
    y = tf.keras.utils.to_categorical(y_labels, num_classes=len(classes))

    if cache_path:
        print(f"Saving landmarks to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, {'X': X, 'y': y, 'classes': classes})

    return X, y, classes

def create_dummy_landmarks(num_samples=20, sequence_length=22, num_classes=10):
    X = np.random.rand(num_samples, sequence_length, 126).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    return X, y, [f"Label_{i}" for i in range(num_classes)]
