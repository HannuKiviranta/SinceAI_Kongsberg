import librosa
import numpy as np
import soundfile as sf
import os
import random
import glob

# --- CONFIGURATION ---
SR = 22050 # NOTE: MUST MATCH THE SR USED IN YOUR RAW ASSETS (22050 in your original script)
OUTPUT_DIR = "dataset/train" # Standardized output folder for the next script to find
HORNS_DIR = "audio/horns"      # Folder containing horn source .wav files

# Define all noise categories and their paths. To add a new type (e.g., 'Alarms'), 
# simply add a new entry here and ensure the folder exists.
NOISE_CATEGORIES = {
    # Continuous Backgrounds (Sea, Engine Rumble, Wind)
    "Backgrounds": "audio/noise/background_noise", 
    
    # Discrete Secondary Events (Mixed ON TOP of a Background)
    "WhiteNoise": "audio/noise/white_noise", 
    "BirdSounds": "audio/noise/bird_sounds", # Placeholder for new assets
    "Alarms": "audio/noise/alarms",          # Placeholder for new assets
    "OtherShips": "audio/horns",             # Using horns as another secondary event
}

SAMPLES_PER_CLASS = 50
SECONDARY_EVENT_PROBABILITY = 0.3 # Probability that a secondary event (bird/alarm/etc.) is mixed in

# --- RANDOMIZATION RANGES ---
RANGE_SHORT = (0.7, 1.1)
RANGE_LONG = (4.5, 6.0)
RANGE_INTERVAL = (0.8, 1.2) 
RANGE_PAUSE = (2.0, 2.5)
RANGE_SNR_SECONDARY = (-10, 5) # SNR range for secondary events (e.g., -10 dB is loud)

# --- UTILITIES ---

def load_sound(path, duration=None):
    """Loads audio and TRIMS SILENCE to prevent looping gaps."""
    try:
        y, _ = librosa.load(path, sr=SR, duration=duration)
        return y.astype(np.float32) # Ensure consistent float type
    except Exception as e:
        # print(f"Error loading {path}: {e}")
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

# --- ASSET LOADING ---
def load_asset_library(folder_paths, duration=None):
    """Loads all assets from a list of folders into one combined library."""
    library = []
    
    # Check if folder_paths is a string (single path) and convert to list if needed
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
        
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Directory not found: {folder_path}. Skipping.")
            continue
            
        files = glob.glob(os.path.join(folder_path, "*.wav"))
        # print(f"Loading {len(files)} files from {folder_path}...")
        
        for f in files:
            y = load_sound(f, duration)
            if y is not None and len(y) > 0:
                library.append(y)
            
    if not library:
        raise ValueError(f"No valid .wav files found in the specified paths!")
    return library

def mix_audio_chunk(main_audio, secondary_audio, snr_db):
    """Mixes a secondary event onto the main audio at a target SNR."""
    
    # 1. Standardize lengths (pad secondary audio if shorter than main_audio)
    if len(secondary_audio) < len(main_audio):
        padding_needed = len(main_audio) - len(secondary_audio)
        secondary_audio = np.pad(secondary_audio, (0, padding_needed), 'constant')
    elif len(secondary_audio) > len(main_audio):
        # Truncate secondary audio if longer than main
        secondary_audio = secondary_audio[:len(main_audio)]

    # 2. Calculate scaling factor based on SNR
    main_power = np.mean(main_audio**2) + 1e-6 # Add epsilon to avoid div by zero
    target_noise_power = main_power / (10**(snr_db / 10))
    current_noise_power = np.mean(secondary_audio**2) + 1e-6
    
    scaling_factor = np.sqrt(target_noise_power / current_noise_power)
    
    # 3. Mix
    mixed = main_audio + (secondary_audio * scaling_factor)
    
    # 4. Re-normalize to prevent clipping
    mixed = mixed / np.max(np.abs(mixed)) * 0.95
    return mixed

# --- MAIN GENERATOR FUNCTION ---

def generate_sample(filename, pattern_def, bg_noise, horn_raw, secondary_events_library):
    
    # 1. Random Warmup Silence and Background Chunk Selection
    warmup_sec = random.uniform(0.5, 1.5)
    silence_samples = int(warmup_sec * SR)
    
    max_start = len(bg_noise) - silence_samples
    start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
    combined_audio = bg_noise[start_idx : start_idx + silence_samples].copy()

    # 2. Build Pattern (Blasts + Gaps)
    for sound_type, gap_type in pattern_def:
        # A. Blast Generation
        if sound_type in ['short', 'long']:
            dur = get_random_duration(RANGE_SHORT) if sound_type == 'short' else get_random_duration(RANGE_LONG)
            blast = create_blast(horn_raw, dur)
            combined_audio = np.concatenate((combined_audio, blast))
        
        # B. Gap Generation
        gap_dur = 0
        if gap_type == 'interval': gap_dur = get_random_duration(RANGE_INTERVAL)
        elif gap_type == 'pause': gap_dur = get_random_duration(RANGE_PAUSE)
            
        if gap_dur > 0:
            gap_samples = int(gap_dur * SR)
            max_start_bg = len(bg_noise) - gap_samples
            start_idx_bg = np.random.randint(0, max_start_bg) if max_start_bg > 0 else 0
            
            noise_chunk = bg_noise[start_idx_bg : start_idx_bg + gap_samples].copy()
            
            if len(noise_chunk) < gap_samples:
                 # Should not happen if assets are long enough, but safety resize
                 noise_chunk = np.resize(noise_chunk, gap_samples)
                 
            combined_audio = np.concatenate((combined_audio, noise_chunk))
            
    # 3. Secondary Event Mixing (Adding the bird/alarm/other ship sound)
    if random.random() < SECONDARY_EVENT_PROBABILITY and secondary_events_library:
        
        # Select a random secondary event (e.g., a short bird call)
        secondary_event = random.choice(secondary_events_library).copy()
        
        
        # --- FIX START: Correctly determine placement and bounds ---
        
        event_len = len(secondary_event)
        combined_len = len(combined_audio)
        
        # Determine the maximum amount of the secondary event that can fit
        max_fit_len = min(event_len, combined_len)
        
        # Choose a random starting point for the event within the combined audio
        # The start point must allow 'max_fit_len' to fit completely if max_fit_len < combined_len
        # If event_len > combined_len, we just place a slice equal to combined_len starting at 0.
        
        # If the secondary event is shorter than the final audio clip:
        if event_len < combined_len:
            max_start_sample = combined_len - event_len
            start_sample = np.random.randint(0, max_start_sample + 1)
            end_sample = start_sample + event_len
            
            event_to_place = secondary_event
            
        # If the secondary event is longer than the final audio clip (or equal):
        else:
            # We only use a slice of the secondary event equal to the combined audio length
            start_sample = 0
            end_sample = combined_len
            event_to_place = secondary_event[:combined_len]
        
        # Create a zero-padded canvas for the secondary event
        secondary_canvas = np.zeros_like(combined_audio)
        
        # Place the event on the canvas
        # The lengths now match exactly: length of secondary_canvas[start:end] == length of event_to_place
        secondary_canvas[start_sample:end_sample] = event_to_place
        
        # --- FIX END ---
        
        # Mix the main audio with the secondary canvas at a random loud SNR
        snr_db = get_random_duration(RANGE_SNR_SECONDARY)
        combined_audio = mix_audio_chunk(combined_audio, secondary_canvas, snr_db)

    # 4. Final Save
    path = f"{OUTPUT_DIR}/{filename}.wav"
    sf.write(path, combined_audio, SR)
    
# --- MAIN EXECUTION ---

if __name__ == "__main__":
    random.seed(42)
    
    # 0. Setup: Ensure output folder and necessary noise folders exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Create dummy folders for new noise types if they don't exist, to avoid errors
    for folder in NOISE_CATEGORIES.values():
        os.makedirs(folder, exist_ok=True)


    # 1. Load all horns and backgrounds
    try:
        horns_library = load_asset_library(HORNS_DIR) 
        
        # Load CONTINUOUS BACKGROUNDS (Must be long clips)
        background_paths = NOISE_CATEGORIES['Backgrounds']
        backgrounds_library = load_asset_library(background_paths, duration=60)
        
        # Load DISCRETE SECONDARY EVENTS (All other noise categories)
        secondary_noise_paths = [v for k, v in NOISE_CATEGORIES.items() if k != 'Backgrounds']
        secondary_events_library = load_asset_library(secondary_noise_paths)
        
    except ValueError as e:
        print(f"\nFATAL ERROR: Asset loading failed. {e}")
        print("Please check that your 'audio/horns' and 'audio/noise/background_noise' folders contain .wav files.")
        exit(1)


    print(f"\nStarting Generation: {SAMPLES_PER_CLASS} samples per class...")
    print(f"Loaded {len(backgrounds_library)} continuous backgrounds and {len(secondary_events_library)} secondary events.")


    # Define the COLREG scenarios map
    scenarios = {
        # Mapping your file prefixes to the pattern definitions:
        "01_starboard_turn":        [('short', 'none')],
        "02_port_turn":             [('short', 'interval'), ('short', 'none')],
        "03_astern":                [('short', 'interval'), ('short', 'interval'), ('short', 'none')],
        "04_doubt":                 [('short', 'interval')] * 4 + [('short', 'none')],
        "05_round_starboard":       [('short', 'interval'), ('short', 'interval'), ('short', 'interval'), ('short', 'pause'), ('short', 'none')],
        "06_round_port":            [('short', 'interval'), ('short', 'interval'), ('short', 'interval'), ('short', 'pause'), ('short', 'interval'), ('short', 'none')],
        "07_making_way":            [('long', 'none')],
        "08_nuc":                   [('long', 'interval'), ('short', 'interval'), ('short', 'none')],
        "09_overtake_starboard":    [('long', 'interval'), ('long', 'interval'), ('short', 'none')],
        "10_overtake_port":         [('long', 'interval'), ('long', 'interval'), ('short', 'interval'), ('short', 'none')],
        "11_agree_overtake":        [('long', 'interval'), ('short', 'interval'), ('long', 'interval'), ('short', 'none')]
    }

    total_files = 0
    
    for i in range(SAMPLES_PER_CLASS):
        idx = f"{i+1:03d}"
        
        for prefix, pattern in scenarios.items():
            # --- THE MAGIC: RANDOM SELECTION ---
            selected_horn = random.choice(horns_library)
            selected_bg = random.choice(backgrounds_library)
            
            generate_sample(
                filename=f"{prefix}_{idx}", 
                pattern_def=pattern, 
                bg_noise=selected_bg, 
                horn_raw=selected_horn, 
                secondary_events_library=secondary_events_library
            )
            total_files += 1

    print(f"\nGeneration Complete! Created {total_files} labeled audio files in {OUTPUT_DIR}/.")

    # --- FINAL CRITICAL WARNING ---
    print("\n\n#####################################################")
    print("##             CRITICAL CONFIGURATION WARNING        ##")
    print("#####################################################")
    print(f"This script used SR={SR}. Your feature extraction script (process_raw_audio_to_npy.py)")
    print("is currently configured for SR=16000.")
    print("\nIf the sample rates do not match, the Mel Spectrograms will be incorrect and the AI model will fail to train.")
    print("\nACTION REQUIRED: Set SR = 22050 in 'process_raw_audio_to_npy.py' to match this generator.")
    print("#####################################################")
