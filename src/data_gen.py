import librosa
import numpy as np
import soundfile as sf
import os
import random
import glob

# --- CONFIGURATION ---
SR = 22050 
OUTPUT_DIR = "dataset/train"

# Source Folders
HORNS_SHORT_DIR = "audio/horns/short"
HORNS_LONG_DIR = "audio/horns/long"

# Define all noise categories
NOISE_CATEGORIES = {
    "Backgrounds": "audio/noise/background_noise", 
    "WhiteNoise": "audio/noise/white_noise", 
    "BirdSounds": "audio/noise/bird_sounds", 
    "Alarms": "audio/noise/alarms",          
    "OtherShips": "audio/horns/short",             
}

SAMPLES_PER_CLASS = 50
SECONDARY_EVENT_PROBABILITY = 0.3 

# --- TIMING CONFIGURATION ---
RANGE_INTERVAL = (0.8, 1.2) 
RANGE_PAUSE = (2.0, 2.5)
RANGE_SNR_SECONDARY = (-10, 5) 

# --- UTILITIES ---

def load_sound(path, duration=None):
    """Loads audio and trims silence to ensure clean concatenation."""
    try:
        y, _ = librosa.load(path, sr=SR, duration=duration)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0:
            return y_trimmed.astype(np.float32)
        return y.astype(np.float32)
    except Exception as e:
        return None

def get_random_duration(range_tuple):
    return random.uniform(range_tuple[0], range_tuple[1])

def process_blast(blast_raw):
    """
    Takes a raw blast and applies a tiny fade in/out 
    to prevent 'clicking' sounds when concatenating.
    """
    # Copy to prevent modifying the original library in memory
    blast = blast_raw.copy()
    
    # Apply tiny 10ms fade to edges
    fade_len = 220 # approx 10ms at 22050Hz
    if len(blast) > fade_len * 2:
        blast[:fade_len] = blast[:fade_len] * np.linspace(0, 1, fade_len)
        blast[-fade_len:] = blast[-fade_len:] * np.linspace(1, 0, fade_len)
        
    return blast

# --- ASSET LOADING ---
def load_asset_library(folder_paths, duration=None):
    library = []
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
        
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            continue
        files = glob.glob(os.path.join(folder_path, "*.wav"))
        for f in files:
            y = load_sound(f, duration)
            if y is not None and len(y) > 0:
                library.append(y)
    return library

def mix_audio_chunk(main_audio, secondary_audio, snr_db):
    if len(secondary_audio) < len(main_audio):
        padding_needed = len(main_audio) - len(secondary_audio)
        secondary_audio = np.pad(secondary_audio, (0, padding_needed), 'constant')
    elif len(secondary_audio) > len(main_audio):
        secondary_audio = secondary_audio[:len(main_audio)]

    main_power = np.mean(main_audio**2) + 1e-6
    target_noise_power = main_power / (10**(snr_db / 10))
    current_noise_power = np.mean(secondary_audio**2) + 1e-6
    
    scaling_factor = np.sqrt(target_noise_power / current_noise_power)
    
    mixed = main_audio + (secondary_audio * scaling_factor)
    mixed = mixed / np.max(np.abs(mixed)) * 0.95 
    return mixed

# --- MAIN GENERATOR FUNCTION ---

def generate_sample(filename, pattern_def, bg_noise, short_horns_list, long_horns_list, secondary_events_library):
    
    # 1. Random Warmup Silence
    warmup_sec = random.uniform(0.5, 1.5)
    silence_samples = int(warmup_sec * SR)
    
    max_start = len(bg_noise) - silence_samples
    start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
    combined_audio = bg_noise[start_idx : start_idx + silence_samples].copy()

    # --- FIX: Select Consistent Horns for this specific sample ---
    # We pick ONE short horn and ONE long horn to use for the entire sequence.
    # This ensures "NUC" (Long, Short, Short) uses the same "voice" for all blasts.
    current_short_horn = random.choice(short_horns_list)
    current_long_horn = random.choice(long_horns_list)

    # 2. Build Pattern
    for sound_type, gap_type in pattern_def:
        
        # --- A. ADD BLAST (Using Pre-selected Horns) ---
        if sound_type == 'short':
            # Use the consistent short horn chosen for this sample
            blast = process_blast(current_short_horn)
        elif sound_type == 'long':
            # Use the consistent long horn chosen for this sample
            blast = process_blast(current_long_horn)
        
        combined_audio = np.concatenate((combined_audio, blast))
        
        # --- B. ADD GAP ---
        gap_dur = 0
        if gap_type == 'interval': gap_dur = get_random_duration(RANGE_INTERVAL)
        elif gap_type == 'pause': gap_dur = get_random_duration(RANGE_PAUSE)
            
        if gap_dur > 0:
            gap_samples = int(gap_dur * SR)
            max_start_bg = len(bg_noise) - gap_samples
            start_idx_bg = np.random.randint(0, max_start_bg) if max_start_bg > 0 else 0
            
            noise_chunk = bg_noise[start_idx_bg : start_idx_bg + gap_samples].copy()
            
            if len(noise_chunk) < gap_samples:
                 noise_chunk = np.resize(noise_chunk, gap_samples)
                 
            combined_audio = np.concatenate((combined_audio, noise_chunk))
            
    # 3. Secondary Event Mixing
    if random.random() < SECONDARY_EVENT_PROBABILITY and secondary_events_library:
        secondary_event = random.choice(secondary_events_library).copy()
        
        event_len = len(secondary_event)
        combined_len = len(combined_audio)
        
        if event_len < combined_len:
            max_start_sample = combined_len - event_len
            start_sample = np.random.randint(0, max_start_sample + 1)
            end_sample = start_sample + event_len
            event_to_place = secondary_event
        else:
            start_sample = 0
            end_sample = combined_len
            event_to_place = secondary_event[:combined_len]
        
        secondary_canvas = np.zeros_like(combined_audio)
        secondary_canvas[start_sample:end_sample] = event_to_place
        
        snr_db = get_random_duration(RANGE_SNR_SECONDARY)
        combined_audio = mix_audio_chunk(combined_audio, secondary_canvas, snr_db)

    # 4. Final Save
    path = f"{OUTPUT_DIR}/{filename}.wav"
    sf.write(path, combined_audio, SR)
    
# --- MAIN EXECUTION ---

if __name__ == "__main__":
    random.seed(42)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for folder in NOISE_CATEGORIES.values():
        os.makedirs(folder, exist_ok=True)

    print("--- Loading Audio Libraries ---")
    
    try:
        short_horns_lib = load_asset_library(HORNS_SHORT_DIR)
        long_horns_lib = load_asset_library(HORNS_LONG_DIR)
        
        bg_path = NOISE_CATEGORIES['Backgrounds']
        backgrounds_lib = load_asset_library(bg_path, duration=60)
        
        secondary_noise_paths = [v for k, v in NOISE_CATEGORIES.items() if k != 'Backgrounds']
        secondary_events_lib = load_asset_library(secondary_noise_paths)
        
    except ValueError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure you have .wav files in audio/horns/short, audio/horns/long, and audio/noise/background_noise")
        exit(1)

    print(f"\nStarting Generation: {SAMPLES_PER_CLASS} samples per class...")
    print(f"Stats: {len(short_horns_lib)} Short Horns, {len(long_horns_lib)} Long Horns")

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
            selected_bg = random.choice(backgrounds_lib)
            
            generate_sample(
                filename=f"{prefix}_{idx}", 
                pattern_def=pattern, 
                bg_noise=selected_bg, 
                short_horns_list=short_horns_lib,
                long_horns_list=long_horns_lib,
                secondary_events_library=secondary_events_lib
            )
            total_files += 1

    print(f"Generation Complete! Created {total_files} labeled audio files in {OUTPUT_DIR}/.")