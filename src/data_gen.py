import librosa
import numpy as np
import soundfile as sf
import os
import random
import glob
import argparse

# --- CONFIGURATION ---
SR = 22050 
OUTPUT_DIR_BASE = "dataset/train"

# Source Folder (Master Horns)
# We will look for any .wav file here to use as a base
HORNS_DIR = "audio/horns" 

# Noise Configuration (Simplified)
NOISE_CATEGORIES = {
    "Backgrounds": "audio/noise/background_noise",
}

SAMPLES_PER_CLASS = 2  # Set to 500 for real training (use 2 for quick testing)
SECONDARY_EVENT_PROBABILITY = 0.65 

# --- TIMING CONFIGURATION ---
RANGE_INTERVAL = (0.7, 1.0)
RANGE_PAUSE    = (3.0, 4.0)
RANGE_SNR_SECONDARY = (-5, 10) 

# --- UTILITIES ---

def load_sound(path, duration=None):
    try:
        y, _ = librosa.load(path, sr=SR, duration=duration)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0: return y_trimmed.astype(np.float32)
        return y.astype(np.float32)
    except Exception as e:
        print(f"   [Load Error] {os.path.basename(path)}: {e}")
        return None

def get_random_duration(range_tuple):
    return random.uniform(range_tuple[0], range_tuple[1])

def process_blast(blast_raw, fade_ms=50):
    """Applies fade in/out to prevent clicking."""
    blast = blast_raw.copy()
    fade_len = int((fade_ms / 1000) * SR)
    if len(blast) > fade_len * 2:
        blast[:fade_len] = blast[:fade_len] * np.linspace(0, 1, fade_len)
        blast[-fade_len:] = blast[-fade_len:] * np.linspace(1, 0, fade_len)
    return blast

def create_short_from_long(long_blast_raw):
    """Creates a ~1.0s Short blast by cutting a slice from the Master (Long) blast."""
    target_sec = random.uniform(0.8, 1.1)
    target_samples = int(target_sec * SR)
    
    total_samples = len(long_blast_raw)
    
    if total_samples <= target_samples:
        # If source is too short, use it as is
        short_blast = long_blast_raw
    else:
        # Cut from the middle to get the most stable tone
        start = (total_samples - target_samples) // 2
        short_blast = long_blast_raw[start : start + target_samples]
        
    return process_blast(short_blast)

def augment_horn_blast(blast, n_steps, rate):
    """Applies pitch/time shift to the raw audio."""
    try:
        blast = librosa.effects.pitch_shift(y=blast, sr=SR, n_steps=n_steps)
        blast = librosa.effects.time_stretch(blast, rate=rate)
    except: pass
    return blast.astype(np.float32)

def load_asset_library(folder_paths, duration=None):
    library = []
    if isinstance(folder_paths, str): folder_paths = [folder_paths]
    
    print(f"DEBUG: Scanning paths: {folder_paths}")
    
    for folder_path in folder_paths:
        # Safety: Ensure we are checking the string path
        if isinstance(folder_path, list): folder_path = folder_path[0]
            
        if not os.path.exists(folder_path):
            print(f"DEBUG: Folder not found inside container: {folder_path}")
            continue
        
        # FIX: Recursive search finds 'master_blast.wav' even if it's in a subfolder
        search_pattern = os.path.join(folder_path, "**", "*.wav")
        files = glob.glob(search_pattern, recursive=True)
        
        print(f"DEBUG: Found {len(files)} .wav files in {folder_path}")
        
        for f in files:
            y = load_sound(f, duration)
            if y is not None and len(y) > 0: library.append(y)
            
    return library

def mix_audio_chunk(main_audio, secondary_audio, snr_db):
    if len(secondary_audio) < len(main_audio):
        padding = len(main_audio) - len(secondary_audio)
        secondary_audio = np.pad(secondary_audio, (0, padding), 'constant')
    elif len(secondary_audio) > len(main_audio):
        secondary_audio = secondary_audio[:len(main_audio)]

    main_rms = np.sqrt(np.mean(main_audio**2)) + 1e-6
    noise_rms = np.sqrt(np.mean(secondary_audio**2)) + 1e-6
    target_noise_rms = main_rms / (10**(snr_db / 20))
    
    scaled_noise = secondary_audio * (target_noise_rms / noise_rms)
    mixed = main_audio + scaled_noise
    
    max_val = np.max(np.abs(mixed))
    if max_val > 0: mixed = mixed / max_val * 0.95
    return mixed

# --- GENERATOR ---

def generate_sample(filename, pattern_def, bg_noise, horns_lib, use_noise=False):
    
    # 1. Warmup
    warmup_samples = int(random.uniform(0.5, 1.5) * SR)
    
    if use_noise and bg_noise is not None:
        max_start = len(bg_noise) - warmup_samples
        start = random.randint(0, max(0, max_start))
        combined_audio = bg_noise[start : start + warmup_samples].copy() * 0.5
    else:
        combined_audio = np.zeros(warmup_samples, dtype=np.float32)

    # --- MASTER SOUND STRATEGY ---
    if not horns_lib: return 
    
    # Pick ONE master file (your master_blast.wav)
    base_source = random.choice(horns_lib)

    # Generate random augmentation parameters ONCE
    aug_steps = random.uniform(-3.0, 3.0) # Pitch shift amount
    aug_rate = random.uniform(0.95, 1.05) # Speed change
    
    # Apply to master source -> This creates the "Ship's Voice"
    base_source_aug = augment_horn_blast(base_source, aug_steps, aug_rate)
    
    # Create both Long and Short blasts from the SAME augmented source
    blast_long = process_blast(base_source_aug)
    blast_short = create_short_from_long(base_source_aug)

    # --- BUILD PATTERN ---
    for sound_type, gap_type in pattern_def:
        if sound_type == 'short':
            combined_audio = np.concatenate((combined_audio, blast_short))
        elif sound_type == 'long':
            combined_audio = np.concatenate((combined_audio, blast_long))
        
        # Add Gap
        if gap_type == 'none': gap_dur = 0
        elif gap_type == 'interval': gap_dur = get_random_duration(RANGE_INTERVAL)
        elif gap_type == 'pause': gap_dur = get_random_duration(RANGE_PAUSE)
            
        if gap_dur > 0:
            gap_samples = int(gap_dur * SR)
            if use_noise and bg_noise is not None:
                max_start = len(bg_noise) - gap_samples
                start = random.randint(0, max(0, max_start))
                noise_chunk = bg_noise[start : start + gap_samples].copy() * 0.5
                combined_audio = np.concatenate((combined_audio, noise_chunk))
            else:
                combined_audio = np.concatenate((combined_audio, np.zeros(gap_samples, dtype=np.float32)))

    path = f"{OUTPUT_DIR}/{filename}.wav"
    sf.write(path, combined_audio, SR)

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['clean', 'noisy'], required=True)
    args = parser.parse_args()
    
    USE_NOISE = (args.mode == 'noisy')
    OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, args.mode)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    try:
        # 1. Load Master Horns (From general folder, recursive search)
        horns_lib = load_asset_library(HORNS_DIR)
        if not horns_lib: 
            raise ValueError(f"No .wav files found in {HORNS_DIR}. Please ensure 'master_blast.wav' is there.")
        
        bg_lib = []
        if USE_NOISE:
            print("🔊 NOISE MODE: ON (Loading backgrounds...)")
            bg_path = NOISE_CATEGORIES.get("Backgrounds")
            
            if bg_path:
                # Check if folder exists
                if not os.path.exists(bg_path):
                    # If not, try to find it recursively or warn user
                    print(f"DEBUG: Background path {bg_path} not found.")
                else:
                    bg_lib = load_asset_library(bg_path, duration=60)
            
            if not bg_lib:
                print("WARNING: No backgrounds found! Reverting to Clean mode.")
                USE_NOISE = False
            else:
                print(f"DEBUG: Loaded {len(bg_lib)} background files.")
        else:
            print("🔇 NOISE MODE: OFF (Generating clean signal)")
            
    except Exception as e:
        print(f"Fatal Error: {e}")
        exit(1)

    print(f"Generating {args.mode.upper()} dataset ({SAMPLES_PER_CLASS} per class)...")
    
    # 12 Classes
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
        "11_agree_overtake":      [('long', 'interval'), ('short', 'interval'), ('long', 'interval'), ('short', 'none')],
        "12_no_signal":           [('none', 'pause'), ('none', 'pause'), ('none', 'pause')]
    }

    total = 0
    for i in range(SAMPLES_PER_CLASS):
        idx = f"{i+1:04d}"
        for prefix, pat in scenarios.items():
            bg = random.choice(bg_lib) if USE_NOISE and bg_lib else None
            generate_sample(f"{prefix}_{idx}", pat, bg, horns_lib, USE_NOISE)
            total += 1
            
    print(f"✅ Done. Generated {total} files.")