import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import kerastuner as kt  # Hyperparameter tuning
import os

# Ensure directory exists
if not os.path.exists('/app'):
    os.makedirs('/app')

# Load dataset
train_data = pd.read_csv('Dataset/train_features.csv')
test_data = pd.read_csv('Dataset/test_features.csv')

# Drop missing values
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Remove duplicate records
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)

# Extract features and labels
X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(emotion_labels))
y_test = to_categorical(y_test, num_classes=len(emotion_labels))

# Reshape data for CNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

### Hyperparameter Tuning with KerasTuner ###
def build_model(hp):
    model = Sequential()
    model.add(Conv1D(filters=hp.Choice('filters', values=[32, 64, 128]), 
                     kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]), 
                     activation='relu', 
                     input_shape=(X_train.shape[1], 1),
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 Regularization
    
    model.add(BatchNormalization())  # Normalization
    model.add(MaxPooling1D(2))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Conv1D(filters=hp.Choice('filters_2', values=[64, 128, 256]), 
                     kernel_size=hp.Choice('kernel_size_2', values=[3, 5]), 
                     activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.3, max_value=0.6, step=0.1)))

    model.add(Flatten())
    model.add(Dense(hp.Choice('dense_units', values=[64, 128, 256]), activation='relu'))
    model.add(Dropout(hp.Float('dropout_3', min_value=0.3, max_value=0.6, step=0.1)))
    
    model.add(Dense(len(emotion_labels), activation='softmax'))  # Output Layer
    
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0005, 0.0001])),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Initialize KerasTuner
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='/app/hp_tuning',
                     project_name='emotion_cnn')

# Run Hyperparameter Tuning
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the best model
model = tuner.hypermodel.build(best_hps)

# Callbacks for training
checkpoint = ModelCheckpoint('Models/cnn_best_model.h5', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, reduce_lr, early_stopping])

# Save final trained model
print("Model saving started.....")
model.save('Models/cnn_model_final.h5')

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
    test_data = pd.read_csv('Dataset/test_features.csv')
    test_features = test_data.drop(columns=['label']).values
    test_features_scaled = scaler.transform(test_features)
    test_features_scaled = test_features_scaled.reshape((test_features_scaled.shape[0], test_features_scaled.shape[1], 1))

    model = load_model('Models/cnn_best_model.h5')
    predictions = model.predict(test_features_scaled)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    predicted_emojis = [emotion_to_emoji[label] for label in predicted_labels]

    result_df = test_data.copy()
    result_df['Predicted Emotion'] = predicted_labels
    result_df['Predicted Emoji'] = predicted_emojis

    
    print("Predictions started......")

    result_df.to_csv("Test Predictions/test_predictions.csv", index=False, encoding="utf-8-sig")
    print("Predictions saved to test_predictions.csv")

# Run predictions
predict_test_data()
