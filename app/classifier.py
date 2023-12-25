from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import librosa





def decode_with_cnn():
    # Load saved model and LabelEncoder
    # loaded_cnn_model = load_model('models/cnn_model.h5')
    loaded_cnn_model = load_model('models/cnn_on_records_model.h5')
    loaded_label_encoder = joblib.load('models/label_encoder.joblib')

    spec_path = 'to_process/input_spec.png'

    # Load and preprocess the input image
    img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))  # Adjust size as needed
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions using the loaded model
    predictions = loaded_cnn_model.predict(img)

    # Decode numerical predictions to class labels
    decoded_predictions = loaded_label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    # Remove the input audio file
    os.remove(spec_path)
    print(decoded_predictions[0], '**********************************')
    return decoded_predictions[0]


def decode_with_tabular():
    input_path = 'to_process/input_audio.wav'

    # Use librosa to extract MFCC features
    audio, sr = librosa.load(input_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Example: Calculate mean of MFCCs
    mfccs_mean = [mfccs.mean(axis=1)]

    features = pd.DataFrame(mfccs_mean, columns=[f'mfcc_{i}' for i in range(len(mfccs_mean[0]))])

    loaded_tabular_model = joblib.load('models/tabular_model.joblib')
    
    prediction = loaded_tabular_model.predict(features)
    
    # Remove the input audio file
    os.remove(input_path)
    return prediction[0]


def decode_with_knn():
    def extract_features(file_path):
        # Load the audio file
        audio, sample_rate = librosa.load(file_path)
        
        # Extracting different types of features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

        # Averaging across time
        mfccs_processed = np.mean(mfccs.T, axis=0)
        chroma_processed = np.mean(chroma.T, axis=0)
        mel_processed = np.mean(mel.T, axis=0)
        contrast_processed = np.mean(contrast.T, axis=0)
        zero_crossing_rate_processed = np.mean(zero_crossing_rate)

        # Concatenating all features
        return np.hstack([mfccs_processed, chroma_processed, mel_processed, contrast_processed, zero_crossing_rate_processed])
    
    input_path = 'to_process/input_audio.wav'

    loaded_knn_model = joblib.load('models/knn_model.joblib')
    loaded_label_encoder = joblib.load('models/label_encoder.joblib')
    loaded_scaler = joblib.load('models/knn_scaler.joblib')

    features = extract_features(input_path)
    features_reshaped = features.reshape(1, -1)
    normalized_features = loaded_scaler.transform(features_reshaped)

    # prediction = loaded_knn_model.predict(normalized_features)
    threshold=0.6
    neighbors = loaded_knn_model.kneighbors(normalized_features, return_distance=False)

    # Count the occurrence of each class in the neighbors
    neighbor_classes = loaded_knn_model._y[neighbors[0]]
    class_counts = np.bincount(neighbor_classes)
    print(class_counts)

    # Find the most common class and its ratio
    most_common_class = np.argmax(class_counts)
    confidence = class_counts[most_common_class] / loaded_knn_model.n_neighbors

    os.remove(input_path)

    # Check if confidence is above the threshold
    if confidence >= threshold:
        # If confident, return the predicted class
        # return loaded_label_encoder.inverse_transform(loaded_knn_model.classes_[most_common_class])[0]
        return loaded_label_encoder.inverse_transform([most_common_class])[0]
    else:
        # If not confident, return the default response
        return 'not_sure'
