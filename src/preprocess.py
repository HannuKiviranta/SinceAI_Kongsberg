import librosa
import numpy as np
import os
import glob

# --- CONFIGURATION ---
SR = 22050                     
CLIP_DURATION_SEC = 10         
N_MELS = 128                   
HOP_LENGTH = 512               
MAX_WIDTH = int(np.ceil(CLIP_DURATION_SEC * (SR / HOP_LENGTH))) 

SOURCE_FOLDER = "dataset/train/clean"      
OUTPUT_DIR = "dataset"         
FEATURES_SUBDIR = "features"           

# --- UPDATED CLASS MAP (13 Classes) ---
CLASS_MAP = {
    "Alter_Starboard": 0,
    "Alter_Port": 1,
    "Astern_Propulsion": 2,
    "Danger_Signal_Doubt": 3,
    "Overtake_Starboard": 4,
    "Round_Starboard": 5,
    "Round_Port": 6,
    "Blind_Bend_Making_Way": 7, 
    "Overtake_Port": 8,
    "Agreement_to_Overtake": 9,
    "Not_Under_Command": 10,    
    "Noise_Only": 11,           
    "Random_Short_Blasts": 12   
}

FILE_SEGMENT_TO_CLASS = {
    "starboard_turn": "Alter_Starboard",
    "port_turn": "Alter_Port",
    "astern": "Astern_Propulsion",
    "doubt": "Danger_Signal_Doubt",
    "overtake_starboard": "Overtake_Starboard",
    "overtake_port": "Overtake_Port",
    "agree_overtake": "Agreement_to_Overtake", 
    "round_starboard": "Round_Starboard", 
    "round_port": "Round_Port",      
    
    # MERGE LOGIC: Both map to the same class
    "making_way": "Blind_Bend_Making_Way",
    "blind_bend": "Blind_Bend_Making_Way", 
    
    "nuc": "Not_Under_Command",
    "no_signal": "Noise_Only",
    "random_short": "Random_Short_Blasts",
}

def wav_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    target_samples = int(SR * CLIP_DURATION_SEC)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), 'constant')
    elif len(y) > target_samples:
        y = y[:target_samples]
    
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    
    current_width = mels_db.shape[1]
    if current_width < MAX_WIDTH:
        padding = MAX_WIDTH - current_width
        mels_db = np.pad(mels_db, ((0, 0), (0, padding)), mode='constant', constant_values=-80)
    elif current_width > MAX_WIDTH:
        mels_db = mels_db[:, :MAX_WIDTH]
    return mels_db

def get_class_name_from_filename(filename):
    filename_lower = filename.lower()
    for segment, canonical_name in FILE_SEGMENT_TO_CLASS.items():
        if segment in filename_lower:
            return canonical_name
    return None

if __name__ == "__main__":
    if not os.path.exists(SOURCE_FOLDER):
        print(f"ERROR: Folder '{SOURCE_FOLDER}' not found.")
        exit(1)
        
    features_path = os.path.join(OUTPUT_DIR, FEATURES_SUBDIR)
    os.makedirs(features_path, exist_ok=True)
    
    labels = []
    wav_files = glob.glob(os.path.join(SOURCE_FOLDER, "*.wav"))
    print(f"Found {len(wav_files)} files to process...")

    for i, file_path in enumerate(wav_files):
        try:
            filename = os.path.basename(file_path)
            class_name = get_class_name_from_filename(filename)
            
            if class_name is None:
                print(f"Skipping: {filename}")
                continue
                
            class_id = CLASS_MAP[class_name]
            spec_matrix = wav_to_spectrogram(file_path)
            
            npy_name = f"{class_name}_{os.path.splitext(filename)[0]}_{i}.npy" 
            save_path = os.path.join(features_path, npy_name)
            np.save(save_path, spec_matrix)
            
            labels.append({"filepath": save_path, "class_name": class_name, "class_id": class_id})
            
            if (i + 1) % 100 == 0: print(f"Processed {i+1}...")

        except Exception as e:
            print(f"Error: {e}")

    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), np.array(labels))
    print(f"\nSUCCESS: Processed {len(labels)} files.")