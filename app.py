from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Allows all domains (for development)

# If you want to allow only frontend at 127.0.0.1:5500
# CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

# Load trained CNN model
model = tf.keras.models.load_model("models/emoSnap_CNN.h5")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        landmarks = np.array(data["landmarks"]).reshape(1, 468 * 3, 1)  # Reshape for CNN

        # Predict emotion
        prediction = model.predict(landmarks)
        emotions = ["Happy", "Sad", "Angry", "Surprised", "Excited"]
        predicted_emotion = emotions[np.argmax(prediction)]

        return jsonify({"emotion": predicted_emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
