import zipfile
import os
import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import pandas as pd

# Extract ZIP file only if it exists

"""print("Starting ZIP extraction...")
zip_file_path = 'archive.zip'
extract_folder = '.'  # Extract to the current directory

if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"Extracted to {extract_folder}")
else:
    print(f"Warning: Zip file '{zip_file_path}' not found.")

print("ZIP extraction completed.")"""


# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image_path):
    """
    Extracts the 468 facial landmarks using MediaPipe's FaceMesh.
    
    Args:
        image_path (str): Path to the input image.
        
    Returns:
        np.array: Flattened array of 468 facial landmarks, or None if detection fails.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image '{image_path}' not found.")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image '{image_path}'.")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()
    
    return None

def load_data_from_folder(folder_path):
    """
    Loads images from a folder, extracts facial landmarks, and stores them.
    
    Args:
        folder_path (str): Path to the folder containing images.
        
    Returns:
        tuple: (features, labels) where:
               - features is a NumPy array of extracted landmarks.
               - labels is a NumPy array of corresponding labels.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return np.array([]), np.array([])

    features = []
    labels = []

    for subfolder in os.listdir(folder_path):
        emotion_folder = os.path.join(folder_path, subfolder)
        if os.path.isdir(emotion_folder):
            for image_name in os.listdir(emotion_folder):
                image_path = os.path.join(emotion_folder, image_name)
                if image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                    landmarks = extract_landmarks(image_path)
                    if landmarks is not None:
                        features.append(landmarks)
                        labels.append(subfolder)

    return np.array(features), np.array(labels)

# Ensure absolute paths
train_dir = os.path.abspath('/app/extracted/train')
test_dir = os.path.abspath('/app/extracted/train')

print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Load data
train_features, train_labels = load_data_from_folder(train_dir)
test_features, test_labels = load_data_from_folder(test_dir)

print(f"Training data shape: {train_features.shape}")
print(f"Testing data shape: {test_features.shape}")

# Convert extracted features and labels into DataFrame
df_train = pd.DataFrame(train_features)
df_train['label'] = train_labels  # Add labels as the last column

df_test = pd.DataFrame(test_features)
df_test['label'] = test_labels  # Add labels as the last column

# Save to CSV
train_csv_path = "/app/train_features.csv"
test_csv_path = "/app/test_features.csv"

df_train.to_csv(train_csv_path, index=False)
df_test.to_csv(test_csv_path, index=False)