import os
import numpy as np
import cv2
import tensorflow as tf
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import core as mp_core

def extract_landmarks_from_image(image, hand_landmarker):
    """
    Extracts 21 landmarks (x, y, z) for both hands using MediaPipe Tasks API.
    Returns a flattened array of 126 values (21 * 3 * 2).
    """
    mp_image = vision.Image.create_from_numpy_array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = hand_landmarker.detect(mp_image)

    landmarks = np.zeros(126)

    if results.hand_landmarks:
        for i, hand_lms in enumerate(results.hand_landmarks):
            if i >= 2: break
            hand_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]).flatten()
            start_idx = i * 63
            landmarks[start_idx : start_idx + 63] = hand_array

    return landmarks, results

def load_data_landmarks(data_path, landmarker_model_path="hand_landmarker.task", sequence_length=22):
    """
    Loads video frames, extracts MediaPipe landmarks, and returns sequences.
    """
    if not os.path.exists(landmarker_model_path):
        print(f"Warning: MediaPipe landmarker model not found at {landmarker_model_path}. Landmarks cannot be extracted from raw images.")
        return None, None, []

    base_options = mp_core.BaseOptions(model_asset_path=landmarker_model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)

    x_data = []
    y_labels = []

    if not os.path.exists(data_path):
        return None, None, []

    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for cls in classes:
            cls_path = os.path.join(data_path, cls)
            for sample in os.listdir(cls_path):
                sample_path = os.path.join(cls_path, sample)
                if not os.path.isdir(sample_path):
                    continue

                sequence = []
                frame_files = sorted(os.listdir(sample_path))
                for f in frame_files:
                    if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    img = cv2.imread(os.path.join(sample_path, f))
                    if img is None:
                        continue

                    landmarks, _ = extract_landmarks_from_image(img, landmarker)
                    sequence.append(landmarks)

                    if len(sequence) == sequence_length:
                        break

                if len(sequence) == 0:
                    continue

                # Pad sequence if necessary
                if len(sequence) < sequence_length:
                    padding = [np.zeros(126)] * (sequence_length - len(sequence))
                    sequence.extend(padding)

                x_data.append(np.array(sequence))
                y_labels.append(class_to_idx[cls])

    if not x_data:
        return None, None, classes

    return np.array(x_data), tf.keras.utils.to_categorical(y_labels, num_classes=len(classes)), classes

def create_dummy_landmarks(num_samples=20, sequence_length=22, num_classes=10):
    """Generates synthetic landmark data (sequence of 126 features)."""
    X = np.random.rand(num_samples, sequence_length, 126).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    return X, y, [f"Class_{i}" for i in range(num_classes)]
