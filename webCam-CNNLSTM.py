import numpy as np
import cv2
import tensorflow as tf
import sys
import os
import argparse
import json
from Hand_tracking import HandTracker
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from gtts import gTTS
import threading

def speak_text(text, lang='ar'):
    """Function to run TTS in a separate thread."""
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("speech.mp3")
        # For Windows/Mac/Linux cross-platform audio playback,
        # we can use 'start' or 'open' command based on OS.
        if sys.platform == "win32":
            os.system("start speech.mp3")
        elif sys.platform == "darwin":
            os.system("open speech.mp3")
        else:
            os.system("mpg123 speech.mp3")
        print(f"TTS ({lang}): {text}")
    except Exception as e:
        print(f"TTS Error: {e}")

class SignLanguageUI:
    def __init__(self, font_path="arial.ttf"):
        self.font_path = font_path if os.path.exists(font_path) else None

    def draw_text(self, frame, text, position, color=(0, 255, 0), font_size=32, is_arabic=False):
        if is_arabic:
            # Reshape Arabic text and handle Bidi (Right-to-Left)
            reshaped_text = arabic_reshaper.reshape(text)
            display_text = get_display(reshaped_text)
        else:
            display_text = text

        if self.font_path:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(self.font_path, font_size)
            draw.text(position, display_text, font=font, fill=color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(frame, display_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            return frame

def main():
    parser = argparse.ArgumentParser(description="Sign Language Fingerspilling to Sentence Translator")
    parser.add_argument("--lang", choices=['ar', 'en'], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)
    parser.add_argument("--landmarker", type=str, default="hand_landmarker.task")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    with open(args.classes, 'r', encoding='utf-8') as f:
        classes = json.load(f)

    tracker = HandTracker(args.landmarker)
    ui = SignLanguageUI()

    cap = cv2.VideoCapture(0)
    sequence_buffer = []
    sentence = []
    last_pred = None
    pred_count = 0
    CONFIRM_FRAMES = 12

    print(f"Starting {args.lang.upper()} Fingerspilling Translator. Press 'q' to quit, 's' to speak, 'c' to clear.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        landmarks, results = tracker.extract_landmarks(frame)
        sequence_buffer.append(landmarks)
        if len(sequence_buffer) > 22:
            sequence_buffer.pop(0)

        # For Arabic, concatenation of fingerspilled characters should be handled by arabic-reshaper later
        current_sentence_text = "".join(sentence)
        status_text = "Buffering..."

        if len(sequence_buffer) == 22:
            input_data = np.expand_dims(np.array(sequence_buffer), 0)
            prediction = model.predict(input_data, verbose=0)
            idx = np.argmax(prediction[0])
            confidence = prediction[0][idx]

            if confidence > 0.85:
                letter = classes[idx]
                status_text = f"Pred: {letter}"

                if letter == last_pred:
                    pred_count += 1
                else:
                    pred_count = 0
                last_pred = letter

                if pred_count == CONFIRM_FRAMES:
                    sentence.append(letter)
                    print(f"Added: {letter}")
                    pred_count = 0
            else:
                status_text = "Recognizing..."
                last_pred = None

        frame = ui.draw_text(frame, status_text, (10, 40), is_arabic=(args.lang == 'ar'))
        frame = ui.draw_text(frame, f"Sentence: {current_sentence_text}", (10, 100), color=(255, 0, 0), is_arabic=(args.lang == 'ar'))

        cv2.imshow('Sign Language Translator', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
        elif key == ord('s'):
            if sentence:
                full_text = "".join(sentence)
                threading.Thread(target=speak_text, args=(full_text, args.lang)).start()

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
