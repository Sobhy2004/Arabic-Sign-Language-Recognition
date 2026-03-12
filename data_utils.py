import os
import numpy as np
import cv2
import tensorflow as tf
from Hand_tracking import HandTracker

def augment_landmarks(landmarks, noise_level=0.005, scale_range=(0.95, 1.05)):
    """
    Robust landmark augmentation:
    - Jittering (random noise)
    - Scaling (randomly resize the hand)
    - Translation (already handled by normalization, but added small random shift)
    """
    if np.all(landmarks == 0):
        return landmarks

    # Reshape to (21, 3) for easier manipulation
    lms = landmarks.reshape(-1, 3)

    # 1. Random Scaling
    scale = np.random.uniform(*scale_range)
    lms *= scale

    # 2. Random Jittering
    noise = np.random.normal(0, noise_level, lms.shape)
    lms += noise

    return lms.flatten()

def load_data_landmarks(data_path, landmarker_path="hand_landmarker.task", sequence_length=22, cache_path=None, mode="auto", augment=False):
    """
    Loads and extracts hand landmarks with robust augmentation.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading landmarks from cache: {cache_path}")
        cached_data = np.load(cache_path, allow_pickle=True).item()
        X, y, classes = cached_data['X'], cached_data['y'], cached_data['classes']

        if augment:
            print("Applying data augmentation to cached landmarks...")
            X_aug = np.array([np.array([augment_landmarks(f) for f in seq]) for seq in X])
            X = np.concatenate([X, X_aug], axis=0)
            y = np.concatenate([y, y], axis=0)
        return X, y, classes

    tracker = HandTracker(landmarker_path)
    x_data, y_labels = [], []

    if not os.path.exists(data_path):
        return None, None, []

    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        items = os.listdir(cls_path)

        current_mode = mode
        if current_mode == "auto":
            has_subdirs = any(os.path.isdir(os.path.join(cls_path, i)) for i in items)
            current_mode = "sequence" if has_subdirs else "static"

        print(f"Processing {cls} ({current_mode})")

        if current_mode == "sequence":
            for sample in items:
                sample_path = os.path.join(cls_path, sample)
                if not os.path.isdir(sample_path): continue
                sequence = []
                for f in sorted(os.listdir(sample_path)):
                    if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                    img = cv2.imread(os.path.join(sample_path, f))
                    if img is None: continue
                    lms, _ = tracker.extract_landmarks(img)
                    sequence.append(lms)
                    if len(sequence) == sequence_length: break
                if not sequence: continue
                if len(sequence) < sequence_length:
                    sequence.extend([np.zeros(126)] * (sequence_length - len(sequence)))

                seq_np = np.array(sequence)
                x_data.append(seq_np)
                y_labels.append(class_to_idx[cls])
                if augment:
                    x_data.append(np.array([augment_landmarks(f) for f in seq_np]))
                    y_labels.append(class_to_idx[cls])

        else: # static
            for f in items:
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                img = cv2.imread(os.path.join(cls_path, f))
                if img is None: continue
                lms, _ = tracker.extract_landmarks(img)
                if np.all(lms == 0): continue
                x_data.append(np.array([lms] * sequence_length))
                y_labels.append(class_to_idx[cls])
                if augment:
                    x_data.append(np.array([augment_landmarks(lms)] * sequence_length))
                    y_labels.append(class_to_idx[cls])

    tracker.close()
    if not x_data: return None, None, classes

    X, y = np.array(x_data), tf.keras.utils.to_categorical(y_labels, num_classes=len(classes))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, {'X': X, 'y': y, 'classes': classes})
    return X, y, classes

def create_dummy_landmarks(num_samples=20, sequence_length=22, num_classes=10):
    X = np.random.rand(num_samples, sequence_length, 126).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    return X, y, [f"Label_{i}" for i in range(num_classes)]
