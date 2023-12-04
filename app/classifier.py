from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import librosa





def decode_with_cnn():
    # Load saved model and LabelEncoder
    loaded_cnn_model = load_model('models/cnn_model.h5')
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