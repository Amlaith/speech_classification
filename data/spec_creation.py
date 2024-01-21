import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram(audio_path, save_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
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
    plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def process_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                
                # Create corresponding output subfolder
                subfolder = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, subfolder)
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Create the output spectrogram path
                spectrogram_name = os.path.splitext(file)[0] + '.png'
                output_path = os.path.join(output_subfolder, spectrogram_name)
                
                # Create and save the spectrogram
                create_spectrogram(audio_path, output_path)

if __name__ == "__main__":
    input_folder = "phone_audio"  # audio folder path
    output_folder = "phone_specs"  # spectrograms folder path

    process_folder(input_folder, output_folder)
