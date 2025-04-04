import numpy as np
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from flask import Flask, request, jsonify, render_template
import json
import os
from typing import List, Dict

# Initialize Flask app with correct template and static folders
app = Flask(__name__, template_folder='templates', static_folder='static')

# Update CORS to allow all origins for now (adjust later if needed)
CORS(app, resources={r"/predict": {"origins": "*"}})

EMOTIONS = ["happy", "sad", "neutral"]
KEY_LANDMARKS = [
    33, 133, 159, 145, 158, 153, 386, 374,  # Eyes
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  # Eyebrows
    61, 291, 78, 306, 13, 14, 17, 84, 181, 91, 314, 402,  # Mouth
    0, 17, 291, 61, 200, 423, 151, 152, 148, 176, 377, 400,  # Face Contour
    1, 4, 5, 195, 197, 2, 98, 327, 358, 412  # Nose & Cheeks
]  # 48 landmarks

class CustomConv1D(Conv1D):
    def __init__(self, *args, **kwargs):
        if 'batch_input_shape' in kwargs:
            del kwargs['batch_input_shape']
        super().__init__(*args, **kwargs)

# Load model with the correct path for Render
model_path = os.path.join(os.path.dirname(__file__), 'Models/emotion_cnn_3class.keras')
try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'Conv1D': CustomConv1D},
        compile=False  # Skip loading the saved optimizer
    )
    # Recompile with legacy Adam optimizer for M1/M2 compatibility
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-6),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def validate_landmarks(landmarks: List[List[float]]) -> bool:
    return len(landmarks) == 468 and all(len(point) == 3 for point in landmarks)

def preprocess_landmarks(landmarks: List[List[float]]) -> np.ndarray:
    landmarks = np.array(landmarks, dtype=np.float32)
    selected = landmarks[KEY_LANDMARKS]  # 468 -> 48
    center = np.mean(selected, axis=0)
    normalized = selected - center
    if len(selected) < 52:  # Pad to (52, 3)
        padding = np.zeros((52 - len(selected), 3), dtype=np.float32)
        normalized = np.vstack((normalized, padding))
    return normalized  # Shape: (52, 3)

# Add a homepage route to serve index.html
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/setup')
def setup():
    return render_template('setup.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request at /predict")
        data = request.get_json()
        print("Payload:", data)

        if not data or 'landmarks' not in data:
            print("Missing 'landmarks' in request.")
            return jsonify({"error": "Missing 'landmarks' in request body"}), 400
        
        landmarks = data['landmarks']

        if not validate_landmarks(landmarks):
            print("Invalid landmark format.")
            return jsonify({"error": "Expected 468 landmarks with 3 coordinates each."}), 400

        processed = preprocess_landmarks(landmarks)
        predictions = model.predict(processed[np.newaxis, ...])[0]

        predictions[2] += 0.1
        emotion_idx = np.argmax(predictions)
        if predictions[2] > 0.3 and predictions[emotion_idx] - predictions[2] < 0.4:
            emotion_idx = 2

        response = {
            "emotion": EMOTIONS[emotion_idx],
            "confidence": float(predictions[emotion_idx])
        }
        print("Prediction result:", response)
        return jsonify(response)
    except Exception as e:
        print("Error during /predict:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable for Render
    port = int(os.getenv("PORT", 8000))
    app.run(host='0.0.0.0', port=port)