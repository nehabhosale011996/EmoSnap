import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import os

# Create directory if it doesn't exist
if not os.path.exists('/app'):
    os.makedirs('/app')

# Load dataset
train_data = pd.read_csv('/app/train_features.csv')
test_data = pd.read_csv('/app/test_features.csv')

# Handle Missing Values
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Remove Duplicates
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)

# Prepare data (features and labels)
X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)

y_train = to_categorical(label_encoder.transform(y_train), num_classes=len(emotion_labels))
y_test = to_categorical(label_encoder.transform(y_test), num_classes=len(emotion_labels))

# Define CNN Model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape data for CNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Callbacks
checkpoint = ModelCheckpoint('/app/cnn_model.h5', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test),
          callbacks=[checkpoint, reduce_lr, early_stopping])

# Save trained model
model.save('/app/cnn_model_final.h5')

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Emotion to Emoji mapping
emotion_to_emoji = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

# Function to predict emotions for the entire test dataset
def predict_test_data():
    test_features = test_data.drop(columns=['label']).values
    test_features_scaled = scaler.transform(test_features)
    test_features_scaled = test_features_scaled.reshape((test_features_scaled.shape[0], test_features_scaled.shape[1], 1))
    
    model = load_model('/app/cnn_model_final.h5')
    predictions = model.predict(test_features_scaled)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    predicted_emojis = [emotion_to_emoji[label] for label in predicted_labels]
    
    result_df = test_data.copy()
    result_df['Predicted Emotion'] = predicted_labels
    result_df['Predicted Emoji'] = predicted_emojis
    result_df.to_csv('/app/test_predictions.csv', index=False)
    print("Predictions saved to test_predictions.csv")

# Run predictions on test data
predict_test_data()
