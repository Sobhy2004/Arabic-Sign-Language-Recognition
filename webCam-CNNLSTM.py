import numpy as np
import cv2
import tensorflow as tf
import sys
import os

# Set of classes used during training.
# In a real scenario, these should be loaded from a config or labels file.
CLASSES = ['Abort', 'Circle', 'Hello', 'No', 'Stop', 'Turn Left', 'Turn Right', 'Turn', 'Warn', 'No_motion']

def diffImg(t0, t1, t2):
    """Computes temporal difference between three frames."""
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

def get_differential_sequence(frames, target_length=22):
    """
    Computes a sequence of differential images from a list of frames.
    If frames = [f0, f1, f2, f3, f4], then diffs = [diff(f0,f1,f2), diff(f1,f2,f3), diff(f2,f3,f4)]
    """
    diffs = []
    for i in range(1, len(frames) - 1):
        d = diffImg(frames[i-1], frames[i], frames[i+1])
        diffs.append(d)
        if len(diffs) == target_length:
            break

    # Convert to numpy array and add channel dimension
    diffs_np = np.array(diffs)
    diffs_np = np.expand_dims(diffs_np, -1) # (seq, 64, 48, 1)

    # Pad if necessary
    if len(diffs_np) < target_length:
        pad_width = ((0, target_length - len(diffs_np)), (0, 0), (0, 0), (0, 0))
        diffs_np = np.pad(diffs_np, pad_width, mode='constant')

    return diffs_np

def main():
    if len(sys.argv) < 2:
        print("Usage: python webCam-CNNLSTM.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        sys.exit(1)

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        # For demonstration purposes in a non-interactive environment, we might skip the loop
        # sys.exit(1)

    frames_buffer = []
    # We need at least 24 frames to get 22 differential images
    # diff(1) = f0,f1,f2
    # diff(22) = f21,f22,f23
    REQUIRED_FRAMES = 24

    print("Starting real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for the model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 64)) # (width, height) to match (64, 48) after resize
        # resized is (64, 48)

        frames_buffer.append(resized)

        display_text = "Buffering..."

        if len(frames_buffer) >= REQUIRED_FRAMES:
            # Prepare input
            diff_seq = get_differential_sequence(frames_buffer, target_length=22)
            # diff_seq is (22, 64, 48, 1)

            # Normalize and add batch dimension
            input_data = np.expand_dims(diff_seq / 255.0, 0)

            # Predict
            prediction = model.predict(input_data, verbose=0)
            pred_idx = np.argmax(prediction[0])
            confidence = prediction[0][pred_idx]

            if confidence > 0.5:
                display_text = f"{CLASSES[pred_idx]} ({confidence:.2f})"
            else:
                display_text = "Uncertain..."

            # Clear buffer or shift? Original code cleared it.
            # Shifting might be better for "real-time", but clearing is simpler.
            frames_buffer = []

        # Overlay text on the original frame
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
