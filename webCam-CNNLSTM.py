import numpy as np
import cv2
import tensorflow as tf
import sys
import os
import argparse
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import core as mp_core
from data_utils import extract_landmarks_from_image

ARABIC_CLASSES = ['شكراً', 'أهلاً', 'نعم', 'لا', 'مساعدة', 'سلام', 'توقف', 'إلى اللقاء', 'ماذا', 'أنا']
ENGLISH_CLASSES = ['Thank you', 'Hello', 'Yes', 'No', 'Help', 'Peace', 'Stop', 'Goodbye', 'What', 'I']

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Tasks Sign Language Translator")
    parser.add_argument("--lang", choices=['ar', 'en'], required=True, help="Language to translate to")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .keras model file")
    parser.add_argument("--landmarker_model", type=str, default="hand_landmarker.task", help="Path to MediaPipe landmarker .task file")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found.")
        sys.exit(1)

    print(f"Loading {args.lang.upper()} model...")
    model = tf.keras.models.load_model(args.model)
    classes = ARABIC_CLASSES if args.lang == 'ar' else ENGLISH_CLASSES

    if not os.path.exists(args.landmarker_model):
        print(f"MediaPipe hand landmarker model not found at {args.landmarker_model}. Please download it.")
        sys.exit(1)

    base_options = mp_core.BaseOptions(model_asset_path=args.landmarker_model)
    options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_hands=2)

    cap = cv2.VideoCapture(0)
    sequence_buffer = []
    SEQUENCE_LENGTH = 22

    print(f"Starting landmark-based recognition. Press 'q' to quit.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            landmarks, results = extract_landmarks_from_image(frame, landmarker)

            # Sliding window buffer
            sequence_buffer.append(landmarks)
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)

            display_text = "Buffering..."
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence_buffer), 0)
                prediction = model.predict(input_data, verbose=0)
                pred_idx = np.argmax(prediction[0])
                confidence = prediction[0][pred_idx]

                if confidence > 0.7:
                    display_text = f"{classes[pred_idx]} ({confidence:.2f})"
                else:
                    display_text = "Recognizing..."

            cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Sign Language Translator', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
