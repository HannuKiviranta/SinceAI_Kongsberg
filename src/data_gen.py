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

# Source Folders
HORNS_SHORT_DIR = "audio/horns/short"
HORNS_LONG_DIR = "audio/horns/long"

# Full Noise Categories (Restored)
NOISE_CATEGORIES = {
    "Backgrounds": "audio/noise/background_noise",
    "CalmSea": "audio/noise/calm_sea",
    "Thunderstorm": "audio/noise/thunderstorm",
    "BirdSounds": "audio/noise/bird_sounds",
    "Alarms": "audio/noise/alarms",
    "WhiteNoise": "audio/noise/white_noise"
}

SAMPLES_PER_CLASS = 500
SECONDARY_EVENT_PROBABILITY = 0.65 

# --- TIMING CONFIGURATION ---
RANGE_INTERVAL = (0.7, 1.0)   # Normal gaps
RANGE_PAUSE    = (3.0, 4.0)   # Longer pause for compound signals
RANGE_SNR_SECONDARY = (-5, 10) 

# --- UTILITIES ---

def load_sound(path, duration=None):
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
    blast = blast_raw.copy()
    fade_len = 220 
    if len(blast) > fade_len * 2:
        blast[:fade_len] = blast[:fade_len] * np.linspace(0, 1, fade_len)
        blast[-fade_len:] = blast[-fade_len:] * np.linspace(1, 0, fade_len)
    return blast

def augment_horn_blast(blast):
    n_steps = random.uniform(-1.5, 1.5) 
    try:
        blast = librosa.effects.pitch_shift(y=blast, sr=SR, n_steps=n_steps)
        rate = random.uniform(0.9, 1.1)
        blast = librosa.effects.time_stretch(blast, rate=rate)
    except Exception as e:
        pass 
    return blast.astype(np.float32)

def load_asset_library(folder_paths, duration=None):
    library = []
    if isinstance(folder_paths, str): folder_paths = [folder_paths]
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path): continue
        files = glob.glob(os.path.join(folder_path, "*.wav"))
        for f in files:
            y = load_sound(f, duration)
            if y is not None and len(y) > 0: library.append(y)
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
    max_abs = np.max(np.abs(mixed))
    if max_abs > 0: mixed = mixed / max_abs * 0.95 
    return mixed

# --- MAIN GENERATOR FUNCTION ---

def generate_sample(filename, pattern_def, bg_noise, short_horns_list, long_horns_list, secondary_events_library=None, use_noise=False):
    
    # 1. Warmup (Silence OR Background)
    warmup_sec = random.uniform(0.5, 1.5)
    silence_samples = int(warmup_sec * SR)
    
    if use_noise and bg_noise is not None:
        max_start = len(bg_noise) - silence_samples
        start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        combined_audio = bg_noise[start_idx : start_idx + silence_samples].copy() * 0.5
    else:
        combined_audio = np.zeros(silence_samples, dtype=np.float32)

    # Prepare horns
    base_short = random.choice(short_horns_list)
    base_long = random.choice(long_horns_list)
    current_short_horn = augment_horn_blast(base_short)
    current_long_horn = augment_horn_blast(base_long)

    # 2. Build Pattern
    for sound_type, gap_type in pattern_def:
        
        # A. Add Blast
        if sound_type == 'short':
            blast = process_blast(current_short_horn)
            combined_audio = np.concatenate((combined_audio, blast))
        elif sound_type == 'long':
            blast = process_blast(current_long_horn)
            combined_audio = np.concatenate((combined_audio, blast))
        elif sound_type == 'none':
            pass 
        
        # B. Add Gap (Silence OR Background)
        gap_dur = 0
        if gap_type == 'interval': gap_dur = get_random_duration(RANGE_INTERVAL)
        elif gap_type == 'pause': gap_dur = get_random_duration(RANGE_PAUSE)
            
        if gap_dur > 0:
            gap_samples = int(gap_dur * SR)
            
            if use_noise and bg_noise is not None:
                max_start_bg = len(bg_noise) - gap_samples
                if max_start_bg > 0:
                    start_idx_bg = np.random.randint(0, max_start_bg)
                    noise_chunk = bg_noise[start_idx_bg : start_idx_bg + gap_samples].copy()
                else:
                    tiled = np.tile(bg_noise, int(np.ceil(gap_samples/len(bg_noise))))
                    noise_chunk = tiled[:gap_samples]
                
                combined_audio = np.concatenate((combined_audio, noise_chunk * 0.5))
            else:
                combined_audio = np.concatenate((combined_audio, np.zeros(gap_samples, dtype=np.float32)))
            
    # 3. Secondary Events (RESTORED LOGIC)
    # Only runs if we are in Noisy Mode AND have a library AND pass the probability check
    if use_noise and bg_noise is not None and secondary_events_library and random.random() < SECONDARY_EVENT_PROBABILITY:
        secondary_event = random.choice(secondary_events_library).copy()
        event_len = len(secondary_event)
        combined_len = len(combined_audio)
        
        # Logic to place the event randomly within the clip
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

    # 4. Save
    path = f"{OUTPUT_DIR}/{filename}.wav"
    sf.write(path, combined_audio, SR)
    
# --- MAIN EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['clean', 'noisy'], required=True, help="Generation mode")
    args = parser.parse_args()

    USE_NOISE = (args.mode == 'noisy')
    OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, args.mode) 
    
    random.seed(42)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(f"--- STARTING GENERATION: {args.mode.upper()} ---")
    
    try:
        short_horns_lib = load_asset_library(HORNS_SHORT_DIR)
        long_horns_lib = load_asset_library(HORNS_LONG_DIR)
        if not short_horns_lib or not long_horns_lib: raise ValueError("Horn libraries empty.")
        
        backgrounds_lib = []
        secondary_events_lib = []
        
        if USE_NOISE:
            # Setup noise folders
            for folder in NOISE_CATEGORIES.values():
                if not os.path.exists(folder): 
                    try: os.makedirs(folder, exist_ok=True)
                    except: pass
            
            print("🔊 NOISE MODE: ON (Loading backgrounds...)")
            # Load Continuous Backgrounds
            bg_paths = [NOISE_CATEGORIES[k] for k in ["Backgrounds", "CalmSea", "Thunderstorm"] if k in NOISE_CATEGORIES]
            backgrounds_lib = load_asset_library(bg_paths, duration=60)
            
            # Load Secondary Events (Birds, Alarms, etc.)
            sec_paths = [NOISE_CATEGORIES[k] for k in ["BirdSounds", "Alarms", "WhiteNoise"] if k in NOISE_CATEGORIES]
            secondary_events_lib = load_asset_library(sec_paths)
            
            if not backgrounds_lib:
                print("WARNING: No backgrounds found! Fallback to silence.")
                USE_NOISE = False
        else:
            print("🔇 NOISE MODE: OFF (Generating clean signal)")
        
    except ValueError as e:
        print(f"\nFATAL ERROR: {e}")
        exit(1)

    print(f"\nStarting Generation: {SAMPLES_PER_CLASS} samples per class...")

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
        "12_no_signal":           [('none', 'pause'), ('none', 'pause'), ('none', 'pause'), ('none', 'pause')],
        "13_random_short":        [('short', 'interval')] * 8
    }

    total_files = 0
    for i in range(SAMPLES_PER_CLASS):
        idx = f"{i+1:04d}"
        for prefix, pattern in scenarios.items():
            
            selected_bg = random.choice(backgrounds_lib) if USE_NOISE and backgrounds_lib else None
            
            generate_sample(
                filename=f"{prefix}_{idx}", 
                pattern_def=pattern, 
                bg_noise=selected_bg, 
                short_horns_list=short_horns_lib,
                long_horns_list=long_horns_lib,
                secondary_events_library=secondary_events_lib,
                use_noise=USE_NOISE
            )
            total_files += 1

    print(f"✅ Generation Complete! Created {total_files} {args.mode} files in {OUTPUT_DIR}/.")