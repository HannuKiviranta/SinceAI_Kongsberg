import librosa
import numpy as np
import soundfile as sf
import os
import random
import glob

# --- CONFIGURATION ---
SR = 22050
OUTPUT_DIR = "dataset/train"
HORNS_DIR = "audio/horns"          
BACKGROUNDS_DIR = "audio/noise/background_noise" 
SAMPLES_PER_CLASS = 50

# --- RANDOMIZATION RANGES ---
RANGE_SHORT = (0.7, 1.1)    
RANGE_LONG = (4.0, 6.0)
RANGE_INTERVAL = (0.8, 1.2) 
RANGE_PAUSE = (2.0, 2.5)    

def load_sound(path, duration=None):
    """Loads audio and TRIMS SILENCE to prevent looping gaps."""
    try:
        y, _ = librosa.load(path, sr=SR, duration=duration)
        
        # --- THE FIX IS HERE ---
        # Trim leading and trailing silence below 20 decibels
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Safety check: if trim removes everything (silent file), use original
        if len(y_trimmed) > 0:
            return y_trimmed
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def get_random_duration(range_tuple):
    return random.uniform(range_tuple[0], range_tuple[1])

def create_blast(horn_raw, duration_sec):
    target_samples = int(duration_sec * SR)
    current_samples = len(horn_raw)
    
    # Fade out edges of the source horn to avoid "clicking" when looping
    # (Optional simple cross-fade smoothing)
    if current_samples > 100:
        horn_raw[:50] = horn_raw[:50] * np.linspace(0, 1, 50)
        horn_raw[-50:] = horn_raw[-50:] * np.linspace(1, 0, 50)

    # Tile (repeat) horn 
    tiled = np.tile(horn_raw, int(np.ceil(target_samples / current_samples)))
    return tiled[:target_samples]

def generate_sample(filename, pattern_def, bg_noise, horn_raw):
    combined_audio = np.array([], dtype=np.float32)
    
    # 1. Random Warmup Silence
    warmup_sec = random.uniform(0.5, 1.5)
    silence_samples = int(warmup_sec * SR)
    
    # Safe background slicing
    if len(bg_noise) > silence_samples:
        max_start = len(bg_noise) - silence_samples
        start_idx = np.random.randint(0, max_start)
        combined_audio = np.concatenate((combined_audio, bg_noise[start_idx : start_idx + silence_samples]))
    else:
        # Fallback if background noise is shorter than requested silence
        tiled_bg = np.tile(bg_noise, int(np.ceil(silence_samples/len(bg_noise))))
        combined_audio = np.concatenate((combined_audio, tiled_bg[:silence_samples]))

    # 2. Build Pattern
    for sound_type, gap_type in pattern_def:
        # A. Blast
        if sound_type == 'short':
            dur = get_random_duration(RANGE_SHORT)
        elif sound_type == 'long':
            dur = get_random_duration(RANGE_LONG)
        
        blast = create_blast(horn_raw, dur)
        combined_audio = np.concatenate((combined_audio, blast))
        
        # B. Gap
        if gap_type == 'none': gap_dur = 0
        elif gap_type == 'interval': gap_dur = get_random_duration(RANGE_INTERVAL)
        elif gap_type == 'pause': gap_dur = get_random_duration(RANGE_PAUSE)
            
        if gap_dur > 0:
            gap_samples = int(gap_dur * SR)
            
            if len(bg_noise) > gap_samples:
                max_start = len(bg_noise) - gap_samples
                start_idx = np.random.randint(0, max_start)
                noise_chunk = bg_noise[start_idx : start_idx + gap_samples]
            else:
                tiled_bg = np.tile(bg_noise, int(np.ceil(gap_samples/len(bg_noise))))
                noise_chunk = tiled_bg[:gap_samples]
                
            combined_audio = np.concatenate((combined_audio, noise_chunk))

    # 3. Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    path = f"{OUTPUT_DIR}/{filename}.wav"
    sf.write(path, combined_audio, SR)

# --- HELPER: LOAD ALL ASSETS ---
def load_asset_library(folder_path, duration=None):
    library = []
    files = glob.glob(os.path.join(folder_path, "*.wav"))
    print(f"Loading {len(files)} files from {folder_path}...")
    
    for f in files:
        y = load_sound(f, duration)
        if y is not None and len(y) > 0:
            library.append(y)
            
    if not library:
        raise ValueError(f"No valid .wav files found in {folder_path}!")
    return library

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    random.seed(42) 
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Libraries
    # Note: Removing duration limit on horn to ensure we get the full blast for trimming
    horns_library = load_asset_library(HORNS_DIR, duration=None) 
    backgrounds_library = load_asset_library(BACKGROUNDS_DIR, duration=60)

    print(f"Starting Generation: {SAMPLES_PER_CLASS} samples per class...")

    scenarios = {
        "01_starboard_turn":      [('short', 'none')],
        "02_port_turn":           [('short', 'interval'), ('short', 'none')],
        "03_astern":              [('short', 'interval'), ('short', 'interval'), ('short', 'none')],
        "04_doubt":               [('short', 'interval')] * 4 + [('short', 'none')],
        "05_round_starboard":     [('short', 'interval'), ('short', 'interval'), ('short', 'interval'), ('short', 'pause'), ('short', 'none')],
        "06_round_port":          [('short', 'interval'), ('short', 'interval'), ('short', 'interval'), ('short', 'pause'), ('short', 'interval'), ('short', 'none')],
        "07_making_way":          [('long', 'none')],
        "08_nuc":                 [('long', 'interval'), ('short', 'interval'), ('short', 'none')],
        "09_overtake_starboard":  [('long', 'interval'), ('long', 'interval'), ('short', 'none')],
        "10_overtake_port":       [('long', 'interval'), ('long', 'interval'), ('short', 'interval'), ('short', 'none')],
        "11_agree_overtake":      [('long', 'interval'), ('short', 'interval'), ('long', 'interval'), ('short', 'none')]
    }

    total_files = 0
    
    for i in range(SAMPLES_PER_CLASS):
        idx = f"{i+1:03d}"
        
        for prefix, pattern in scenarios.items():
            selected_horn = random.choice(horns_library)
            selected_bg = random.choice(backgrounds_library)
            
            generate_sample(f"{prefix}_{idx}", pattern, selected_bg, selected_horn)
            total_files += 1

    print(f"Generation Complete! Created {total_files} labeled audio files.")