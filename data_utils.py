import os
import numpy as np
import cv2
import tensorflow as tf

def load_data(data_path, img_size=(64, 48), sequence_length=22):
    """
    Load data from the directory structure:
    data_path/class_name/sample_dir/frame_img.jpg
    """
    x_data = []
    y_labels = []
    if not os.path.exists(data_path):
        return None, None, []

    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        for sample in os.listdir(cls_path):
            sample_path = os.path.join(cls_path, sample)
            if not os.path.isdir(sample_path):
                continue

            frames = []
            frame_files = sorted(os.listdir(sample_path))
            for f in frame_files:
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img = cv2.imread(os.path.join(sample_path, f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size[1], img_size[0]))
                frames.append(img / 255.0)
                if len(frames) == sequence_length:
                    break

            if len(frames) == 0:
                continue
            if len(frames) < sequence_length:
                padding = [np.zeros((img_size[0], img_size[1]))] * (sequence_length - len(frames))
                frames.extend(padding)

            x_data.append(np.expand_dims(np.array(frames), -1))
            y_labels.append(class_to_idx[cls])

    if not x_data:
        return None, None, classes

    return np.array(x_data), tf.keras.utils.to_categorical(y_labels, num_classes=len(classes)), classes

def create_dummy_data(num_samples=20, sequence_length=22, img_size=(64, 48), num_classes=10):
    X = np.random.rand(num_samples, sequence_length, img_size[0], img_size[1], 1).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    return X, y, [f"Class_{i}" for i in range(num_classes)]
