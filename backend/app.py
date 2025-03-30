import numpy as np
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from flask import Flask, request, jsonify
import json
from typing import List, Dict

app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": "https://nehabhosale011996.github.io"}})

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

# Load model without compiling it
model = tf.keras.models.load_model(
    'Models/emotion_cnn_3class.keras',
    custom_objects={'Conv1D': CustomConv1D},
    compile=False  # Skip loading the saved optimizer
)

# Recompile with legacy Adam optimizer for M1/M2 compatibility
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-6),  # Use legacy Adam
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

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

@app.route('/predict', methods=['POST'])
def predict():
    landmarks = request.json['landmarks']  # Expects 468 landmarks
    processed = preprocess_landmarks(landmarks)  # Your preprocessing function
    predictions = model.predict(processed[np.newaxis, ...])[0]
    
    # Apply Neutral tweaks
    predictions[2] += 0.1  # Neutral bias
    emotion_idx = np.argmax(predictions)
    if predictions[2] > 0.3 and predictions[emotion_idx] - predictions[2] < 0.4:
        emotion_idx = 2
    
    return jsonify({
        "emotion": EMOTIONS[emotion_idx],
        "confidence": float(predictions[emotion_idx])
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
