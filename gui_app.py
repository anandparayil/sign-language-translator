import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import pandas as pd
import time

# --- Model & MediaPipe setup ---
model = joblib.load("C:/Users/ANAND/sign_language_translator/models/gesture_classifier_normalized.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()

# --- Normalize Function ---
def normalize_landmarks(landmarks):
    origin_x = landmarks[0].x
    origin_y = landmarks[0].y
    origin_z = landmarks[0].z
    normalized = []
    for lm in landmarks:
        normalized.extend([
            lm.x - origin_x,
            lm.y - origin_y,
            lm.z - origin_z
        ])
    return normalized

# --- GUI App Class ---
class SignTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.video_label = ttk.Label(root)
        self.video_label.pack()

        self.prediction_label = ttk.Label(root, text="Prediction: ", font=("Arial", 16))
        self.prediction_label.pack(pady=10)

        self.sentence = ""
        self.sentence_display = ttk.Label(root, text="Sentence: ", font=("Arial", 14))
        self.sentence_display.pack()

        self.speaking = False

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Clear", command=self.clear_sentence).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="Backspace", command=self.backspace).grid(row=0, column=1, padx=10)
        ttk.Button(button_frame, text="Add Space", command=self.add_space).grid(row=0, column=2, padx=10)  # 👈 Added button
        ttk.Button(button_frame, text="Speak", command=self.speak_sentence).grid(row=0, column=3, padx=10)

        self.cap = cv2.VideoCapture(0)
        self.last_prediction = ""
        self.prediction_count = 0
        self.stable_prediction = ""
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                features = normalize_landmarks(hand_landmarks.landmark)
                if len(features) == 63:
                    features_df = pd.DataFrame([features], columns=model.feature_names_in_)
                    prediction = model.predict(features_df)[0]
                    confidence = max(model.predict_proba(features_df)[0])

                    # Smoothing logic
                    if prediction == self.last_prediction:
                        self.prediction_count += 1
                    else:
                        self.prediction_count = 1
                        self.last_prediction = prediction

                    if self.prediction_count >= 5:
                        if prediction != self.stable_prediction:
                            self.stable_prediction = prediction
                            self.sentence += prediction
                        self.prediction_count = 0

                    self.prediction_label.config(text=f"Prediction: {self.stable_prediction} ({confidence:.2f})")

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert to image and show
        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Update sentence display
        self.sentence_display.config(text=f"Sentence: {self.sentence}")

        self.root.after(10, self.update_video)

    def clear_sentence(self):
        self.sentence = ""

    def backspace(self):
        self.sentence = self.sentence[:-1]

    def add_space(self):  
        self.sentence += " "

    def speak_sentence(self):
        if not self.speaking:
            threading.Thread(target=self._speak, daemon=True).start()

    def _speak(self):
        self.speaking = True  
        engine.say(self.sentence)
        engine.runAndWait()
        time.sleep(0.5) 
        self.speaking = False 

# --- Run the app ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SignTranslatorApp(root)
    root.mainloop()
