import librosa
import numpy as np
import os
import glob
import re

# --- CONFIGURATION (MUST MATCH TRAINING SCRIPT) ---
SR = 22050                     # Sample rate
CLIP_DURATION_SEC = 10         # Fixed duration of clips
N_MELS = 128                   # Number of Mel bands (Feature height)
HOP_LENGTH = 512               # Time resolution
# Calculates target width dynamically based on audio length and hop settings
MAX_WIDTH = int(np.ceil(CLIP_DURATION_SEC * (SR / HOP_LENGTH))) 

# Define Input and Output structure
SOURCE_FOLDER = "dataset/train"      # Folder containing ALL .wav files directly
OUTPUT_DIR = "dataset"         # The target directory required by the training script
FEATURES_SUBDIR = "features"           # Subdirectory for NPY files

# Define your class names and map them to integer IDs (MUST MATCH train_colreg_classifier.py)
# NOTE: The keys are the canonical names needed for the training script.
CLASS_MAP = {
    "Alter_Starboard": 0,
    "Alter_Port": 1,
    "Astern_Propulsion": 2,
    "Danger_Signal_Doubt": 3,
    "Overtake_Starboard": 4,
    "Overtake_Port": 5,
    "Agreement_to_Overtake": 6,
    "Blind_Bend_Channel": 7,
    "Not_Under_Command": 8,
    "Noise_Only": 9,
    "Random_Short_Blasts": 10
}

# Helper dictionary to link file shorthand segments (like 'agree_overtake') to the canonical class name.
# This logic will look for the segment inside the filename (e.g., inside '11_agree_overtake_050.wav').
FILE_SEGMENT_TO_CLASS = {
    # File Segments from the provided list (Rule 34 Manoeuvring)
    "starboard_turn": "Alter_Starboard",
    "port_turn": "Alter_Port",
    "astern": "Astern_Propulsion",
    "doubt": "Danger_Signal_Doubt",
    
    # Overtaking Signals
    "overtake_starboard": "Overtake_Starboard",
    "overtake_port": "Overtake_Port",
    "agree_overtake": "Agreement_to_Overtake", 

    # Bend/Warning Signals (Mapping multiple segments to the single 'Blind_Bend_Channel' category)
    "round_starboard": "Blind_Bend_Channel", # Assuming these signals are a single prolonged blast (Rule 34(e))
    "round_port": "Blind_Bend_Channel",      # Same as above
    "making_way": "Blind_Bend_Channel",      # Assuming single prolonged blast (Rule 35(a) or Rule 34(e))

    # Not under command / I am unable to manoeuvre
    "nuc": "Not_Under_Command",

    # Custom/Negative Classes
    "noise_only": "Noise_Only",
    "random_short_blasts": "Random_Short_Blasts",
}


def wav_to_spectrogram(file_path):
    """Loads, pads, computes Mel Spectrogram, and returns the dB-scaled 2D array."""
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=SR)
    
    # Pad or Truncate to the fixed clip duration (CLIP_DURATION_SEC)
    target_samples = int(SR * CLIP_DURATION_SEC)
    if len(y) < target_samples:
        # Pad with silence if shorter than the target clip duration
        y = np.pad(y, (0, target_samples - len(y)), 'constant')
    elif len(y) > target_samples:
        # Truncate if longer than the target clip duration
        y = y[:target_samples]
    
    # 2. Create Mel Spectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    
    # 3. Convert to Decibels (Log Scale) - This is the standard normalization step
    mels_db = librosa.power_to_db(mels, ref=np.max)
    
    # 4. Final Padding/Cropping Check (Ensures all features are exactly the same size: N_MELS x MAX_WIDTH)
    current_width = mels_db.shape[1]
    
    if current_width < MAX_WIDTH:
        padding = MAX_WIDTH - current_width
        mels_db = np.pad(mels_db, ((0, 0), (0, padding)), mode='constant', constant_values=-80)
    elif current_width > MAX_WIDTH:
        mels_db = mels_db[:, :MAX_WIDTH]
        
    return mels_db

def get_class_name_from_filename(filename):
    """
    Infers the canonical class name by checking if any defined segment 
    (from FILE_SEGMENT_TO_CLASS) exists within the filename.
    """
    filename_lower = filename.lower()
    
    for segment, canonical_name in FILE_SEGMENT_TO_CLASS.items():
        # Check if the descriptive segment is present anywhere in the filename
        # This ignores the leading number prefix (e.g., '11_')
        if segment in filename_lower:
            return canonical_name
    return None

# --- MAIN LOOP ---
if __name__ == "__main__":
    
    # Check if the raw audio input directory exists
    if not os.path.exists(SOURCE_FOLDER):
        print(f"ERROR: Raw audio input folder '{SOURCE_FOLDER}' not found.")
        print("Please create this folder and place all your .wav files directly inside it.")
        exit(1)
        
    # Setup output paths
    features_path = os.path.join(OUTPUT_DIR, FEATURES_SUBDIR)
    os.makedirs(features_path, exist_ok=True)
    
    labels = []
    
    # Find all wav files directly in the source folder (flat structure)
    wav_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.wav"))
    wav_files.extend(glob.glob(os.path.join(SOURCE_FOLDER, "*.WAV"))) # Also check for uppercase extension

    print(f"Found {len(wav_files)} files to process...")

    for i, file_path in enumerate(wav_files):
        try:
            filename = os.path.basename(file_path)
            # --- New Logic Here ---
            class_name = get_class_name_from_filename(filename)
            # ----------------------
            
            if class_name is None:
                print(f"  WARNING: Skipping file '{filename}'. Could not map descriptive segment to a COLREG class.")
                continue
                
            class_id = CLASS_MAP[class_name]
            
            # 1. Process WAV to Spectrogram Matrix
            spec_matrix = wav_to_spectrogram(file_path)
            
            # 2. Save as NPY (The raw data array)
            base_filename_no_ext = os.path.splitext(filename)[0]
            # Use the Canonical Class Name for consistency in NPY file naming
            npy_name = f"{class_name}_{base_filename_no_ext}_{i}.npy" 
            save_path = os.path.join(features_path, npy_name)
            
            # This is the crucial step: saving the 2D matrix directly
            np.save(save_path, spec_matrix)
            
            # 3. Record metadata for labels.npy
            labels.append({
                "filepath": save_path,
                "class_name": class_name,
                "class_id": class_id
            })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1} files...")

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    # Final save of the metadata file
    labels_path = os.path.join(OUTPUT_DIR, "labels.npy")
    np.save(labels_path, np.array(labels))
    
    print("\n--- FEATURE GENERATION SUCCESS ---")
    print(f"Total features created: {len(labels)}")
    print(f"Data saved to: {OUTPUT_DIR}/")
    print("You can now run the 'train_colreg_classifier.py' script.")