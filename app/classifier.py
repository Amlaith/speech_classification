from tensorflow.keras.models import load_model
import cv2
import numpy as np
import joblib
import os


# Load saved model and LabelEncoder
loaded_model = load_model('../notebooks/cnn_model.h5')
loaded_label_encoder = joblib.load('../notebooks/label_encoder.joblib')

spec_path = '../data/specs/to_process/input_spec.png'


def decode_command():
    # Load and preprocess the input image
    img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))  # Adjust size as needed
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions using the loaded model
    predictions = loaded_model.predict(img)

    # Decode numerical predictions to class labels
    decoded_predictions = loaded_label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    # Remove the input audio file
    os.remove(spec_path)
    return decoded_predictions[0]
