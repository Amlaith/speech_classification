from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import librosa
from time import time



def decode_with_cnn():
    # Load saved model and LabelEncoder
    # loaded_cnn_model = load_model('models/cnn_model.h5')
    # loaded_cnn_model = load_model('models/cnn_on_records_model.h5')
    loaded_cnn_model = load_model('models/cnn_phone_model.h5')
    loaded_label_encoder = joblib.load('models/label_encoder.joblib')

    spec_path = 'to_process/input_spec.png'

    # Load and preprocess the input image
    img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (72, 72))
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = loaded_cnn_model.predict(img)

    # Decode numerical predictions to class labels
    # decoded_predictions = loaded_label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    # Remove the input audio file
    # os.remove(spec_path)
    # print(decoded_predictions[0], '**********************************')
    # return decoded_predictions[0]
    return predictions[0]


def decode_with_tabular():
    input_path = 'to_process/input_audio.wav'

    # Use librosa to extract MFCC features
    audio, sr = librosa.load(input_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Example: Calculate mean of MFCCs
    mfccs_mean = [mfccs.mean(axis=1)]

    features = pd.DataFrame(mfccs_mean, columns=[f'mfcc_{i}' for i in range(len(mfccs_mean[0]))])

    loaded_tabular_model = joblib.load('models/tabular_phone_model.joblib')
    
    # prediction = loaded_tabular_model.predict(features)
    probabilities = loaded_tabular_model.predict_proba(features)[0]
    
    # Remove the input audio file
    # os.remove(input_path)
    # return prediction[0]
    return probabilities


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
    
    # start = time()
    
    input_path = 'to_process/input_audio.wav'

    loaded_knn_model = joblib.load('models/knn_phone_model.joblib')
    loaded_label_encoder = joblib.load('models/label_encoder.joblib')
    loaded_scaler = joblib.load('models/knn_phone_scaler.joblib')

    # loading_models = time()

    features = extract_features(input_path)
    features_reshaped = features.reshape(1, -1)
    normalized_features = loaded_scaler.transform(features_reshaped)

    # extracting_features = time()

    # prediction = loaded_knn_model.predict(normalized_features)
    # threshold=0.6
    # neighbors = loaded_knn_model.kneighbors(normalized_features, return_distance=False)

    # # Count the occurrence of each class in the neighbors
    # neighbor_classes = loaded_knn_model._y[neighbors[0]]
    # class_counts = np.bincount(neighbor_classes)
    # print(class_counts)

    # # Find the most common class and its ratio
    # most_common_class = np.argmax(class_counts)
    # confidence = class_counts[most_common_class] / loaded_knn_model.n_neighbors

    # # os.remove(input_path)

    # # print(f'Full time: {time() - start}')
    # # print(f'Loading time: {loading_models - start}')
    # # print(f'extracting time: {extracting_features - loading_models}')

    # # Check if confidence is above the threshold
    # if confidence >= threshold:
    #     # If confident, return the predicted class
    #     return loaded_label_encoder.inverse_transform([most_common_class])[0]
    # else:
    #     # If not confident, return the default response
    #     return 'not_sure'
    probabilities = loaded_knn_model.predict_proba(normalized_features)[0]
    return probabilities


def unified_predict():
    loaded_label_encoder = joblib.load('models/label_encoder.joblib')
    cnn_probs = decode_with_cnn()
    tabular_probs = decode_with_tabular()
    knn_probs = decode_with_knn()
    print(cnn_probs)
    print(tabular_probs)
    print(knn_probs)

    # Average the probabilities from each model
    avg_probs = (cnn_probs + tabular_probs + knn_probs) / 3

    # Determine the class with the highest average probability
    max_prob = np.max(avg_probs)
    max_prob_class = np.argmax(avg_probs)

    # Define a threshold for 'not sure' (can be adjusted based on your preference)
    threshold = 0.5  # Example threshold

    if max_prob < threshold:
        return 'not sure'
    else:
        # Convert class index back to label
        return loaded_label_encoder.inverse_transform([max_prob_class])[0]

# Ensure that 'loaded_label_encoder' is accessible in this scope

