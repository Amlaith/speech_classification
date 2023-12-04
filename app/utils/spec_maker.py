import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_spectrogram(input_audio_path, output_spec_path):
    # Load the audio file
    y, sr = librosa.load(input_audio_path, sr=None)
    y, _ = librosa.effects.trim(y)

    # Create a spectrogram
    plt.figure(figsize=(0.72, 0.72))
    spectrogram = librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=512)), ref=np.max),
        y_axis='log',
        x_axis='time'
        )
    plt.axis('off')
    
    # Save the spectrogram
    plt.savefig(output_spec_path, dpi=400, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def process_input_audio(input_audio_path, output_spec_path):
    # Generate the spectrogram
    generate_spectrogram(input_audio_path, output_spec_path)

    # Remove the input audio file
    os.remove(input_audio_path)

def transform_input_to_spec():
    input_audio_path = 'to_process/input_audio.wav'
    output_spec_path = 'to_process/input_spec.png'
    process_input_audio(input_audio_path, output_spec_path)

