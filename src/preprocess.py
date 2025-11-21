import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# --- CONFIGURATION ---
SOURCE_FOLDER = "dataset/train"     # Where your .wav files are
TARGET_FOLDER = "dataset/processed" # Where the images will go

# Image dimensions
# SR=22050, Hop Length=512 -> ~43 frames per second of audio
# Max duration of COLREG signal is roughly 15-20 seconds.
# 20s * 43 = ~860 width. Let's fix width to 860 pixels (approx 20 seconds).
MAX_WIDTH = 860 
N_MELS = 128 # Height of the image (frequency resolution)

def wav_to_spectrogram(file_path):
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=22050)
    
    # 2. Create Mel Spectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
    
    # 3. Convert to Decibels (Log Scale) - makes it look like an image to the eye
    mels_db = librosa.power_to_db(mels, ref=np.max)
    
    # 4. PAD or CROP to ensure strict width (Critical for PyTorch batches!)
    current_width = mels_db.shape[1]
    
    if current_width < MAX_WIDTH:
        # Pad with very low number (silence) on the right side
        padding = MAX_WIDTH - current_width
        mels_db = np.pad(mels_db, ((0, 0), (0, padding)), mode='constant', constant_values=-80)
    else:
        # Crop if too long (unlikely given your generator, but good safety)
        mels_db = mels_db[:, :MAX_WIDTH]
        
    return mels_db

def save_as_image(spectrogram, save_path):
    """
    Saves the matrix as a .png image without axes or white borders
    """
    plt.figure(figsize=(10, 4))
    # Remove axes to save ONLY the data
    plt.axis('off')
    librosa.display.specshow(spectrogram, sr=22050, x_axis='time', y_axis='mel', fmax=8000)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# --- MAIN LOOP ---
if __name__ == "__main__":
    # Find all wav files
    wav_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.wav"))
    print(f"Found {len(wav_files)} files to process...")

    for file_path in wav_files:
        filename = os.path.basename(file_path)
        
        # LOGIC: Extract Label from Filename
        # Your files look like: "01_starboard_turn_001.wav"
        # We want the folder to be: "01_starboard_turn"
        # We split by "_" and remove the last part (the counter number)
        parts = filename.split('_')
        class_name = "_".join(parts[:-1]) # Rejoin everything except the counter
        
        # Create Class Folder
        class_dir = os.path.join(TARGET_FOLDER, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Process
        spec = wav_to_spectrogram(file_path)
        
        # Save as PNG
        # We change extension from .wav to .png
        out_name = filename.replace('.wav', '.png')
        save_path = os.path.join(class_dir, out_name)
        
        save_as_image(spec, save_path)

    print("Processing Complete! Data is ready for PyTorch.")