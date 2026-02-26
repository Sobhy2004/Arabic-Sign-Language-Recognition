import numpy as np
import cv2
import tensorflow as tf
import sys
import os
import argparse

# Global Class mappings for demonstration.
# In production, these should be loaded from a config file corresponding to the trained model.
ARABIC_CLASSES = ['شكراً', 'أهلاً', 'نعم', 'لا', 'مساعدة', 'سلام', 'توقف', 'إلى اللقاء', 'ماذا', 'أنا']
ENGLISH_CLASSES = ['Thank you', 'Hello', 'Yes', 'No', 'Help', 'Peace', 'Stop', 'Goodbye', 'What', 'I']

def diffImg(t0, t1, t2):
    """Computes temporal difference between three frames."""
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

def get_differential_sequence(frames, target_length=22):
    """
    Computes a sequence of differential images from a list of frames.
    """
    diffs = []
    for i in range(1, len(frames) - 1):
        d = diffImg(frames[i-1], frames[i], frames[i+1])
        diffs.append(d)
        if len(diffs) == target_length:
            break

    diffs_np = np.array(diffs)
    diffs_np = np.expand_dims(diffs_np, -1)

    if len(diffs_np) < target_length:
        pad_width = ((0, target_length - len(diffs_np)), (0, 0), (0, 0), (0, 0))
        diffs_np = np.pad(diffs_np, pad_width, mode='constant')

    return diffs_np

def main():
    parser = argparse.ArgumentParser(description="Sign Language Translator (Arabic & English)")
    parser.add_argument("--lang", choices=['ar', 'en'], required=True, help="Language to translate to (ar: Arabic, en: English)")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .keras model file")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found.")
        sys.exit(1)

    print(f"Loading {args.lang.upper()} model from {args.model}...")
    model = tf.keras.models.load_model(args.model)

    classes = ARABIC_CLASSES if args.lang == 'ar' else ENGLISH_CLASSES

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open webcam index 0. Check your hardware connection.")

    frames_buffer = []
    REQUIRED_FRAMES = 24

    print(f"Starting {args.lang.upper()} sign language recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # If webcam not available, we can still process a video file or break
            # In sandbox, it will break here.
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 64))
        frames_buffer.append(resized)

        display_text = "Buffering..."

        if len(frames_buffer) >= REQUIRED_FRAMES:
            diff_seq = get_differential_sequence(frames_buffer, target_length=22)
            input_data = np.expand_dims(diff_seq / 255.0, 0)

            prediction = model.predict(input_data, verbose=0)
            pred_idx = np.argmax(prediction[0])
            confidence = prediction[0][pred_idx]

            if confidence > 0.6:
                display_text = f"{classes[pred_idx]} ({confidence:.2f})"
            else:
                display_text = "Recognizing..."

            frames_buffer = frames_buffer[2:] # Shift buffer for continuous motion

        # Overlay text on the frame
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Translator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
