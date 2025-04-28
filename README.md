# Real-Time AI-Based Sign Language Translator 🤟🧠

This project implements a real-time sign language translator that detects hand gestures representing A–Z alphabets and converts them into text and speech using AI models and a GUI interface.

## 🚀 Features
- Real-time webcam feed processing
- Hand landmark detection using MediaPipe
- Landmark normalization for robust prediction
- Random Forest Classifier for gesture recognition
- Tkinter-based GUI with sentence construction
- Text-to-Speech (TTS) output

## 🛠️ Tech Stack
- Python
- OpenCV
- MediaPipe
- scikit-learn
- pyttsx3
- Tkinter
- Pillow

## 📂 Project Structure
- `/dataset/` - Training images and extracted landmarks
- `/models/` - Trained Random Forest model
- `/notebooks/` - Jupyter notebooks for data processing and model training
- `/outputs/` - Evaluation plots (Confusion Matrix)
- `gui_app.py` - Main GUI application

## 🧠 Future Enhancements
- Add dynamic gesture recognition
- Expand dataset to include numbers and words
- Build multilingual support (e.g., English ↔ Hindi)
- Deploy on mobile using TensorFlow Lite

## ✨ How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python gui_app.py`

---
