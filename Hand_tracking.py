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
        Extracts 21 normalized landmarks for up to 2 hands.
        Normalization: Subtracts wrist coordinates from all other points and scales by hand size.
        Returns a flat array of 126 features.
        """
        if self.landmarker is None:
            return np.zeros(126), None

        mp_image = vision.Image.create_from_numpy_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.landmarker.detect(mp_image)

        landmarks = np.zeros(126)
        if results.hand_landmarks:
            for i, hand_lms in enumerate(results.hand_landmarks):
                if i >= 2: break

                # Get raw coordinates
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])

                # 1. Zero-center relative to wrist (landmark 0)
                wrist = coords[0]
                normalized_coords = coords - wrist

                # 2. Scale normalization (invariant to distance from camera)
                # Use max distance from wrist as scaling factor
                max_dist = np.max(np.abs(normalized_coords))
                if max_dist > 0:
                    normalized_coords = normalized_coords / max_dist

                # Flatten and store
                hand_array = normalized_coords.flatten()
                start_idx = i * 63
                landmarks[start_idx : start_idx + 63] = hand_array

        return landmarks, results

    def close(self):
        if self.landmarker:
            self.landmarker.close()
