import cv2
import numpy as np
import os
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import core as mp_core

class HandTracker:
    def __init__(self, model_path="hand_landmarker.task"):
        """
        Initializes the MediaPipe Hand Landmarker.
        """
        if not os.path.exists(model_path):
            # We don't raise error here for the training phase if we are using cached data,
            # but for extraction or inference, it's needed.
            self.landmarker = None
        else:
            base_options = mp_core.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(self, frame):
        """
        Extracts 21 landmarks for up to 2 hands from a BGR frame.
        Returns a flat array of 126 features (21 landmarks * 3 coordinates * 2 hands).
        """
        if self.landmarker is None:
            return np.zeros(126), None

        # Convert BGR to RGB for MediaPipe
        mp_image = vision.Image.create_from_numpy_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.landmarker.detect(mp_image)

        landmarks = np.zeros(126)
        if results.hand_landmarks:
            for i, hand_lms in enumerate(results.hand_landmarks):
                if i >= 2: break
                # Flatten (x, y, z) for 21 landmarks
                hand_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]).flatten()
                start_idx = i * 63
                landmarks[start_idx : start_idx + 63] = hand_array

        return landmarks, results

    def close(self):
        if self.landmarker:
            self.landmarker.close()
