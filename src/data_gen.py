import librosa
import numpy as np
import soundfile as sf
import os
import random
import glob
import argparse

# ============================================================
# CONFIGURATION - OPTIMIZED FOR 6.08s HORN
# ============================================================

SR = 22050 
OUTPUT_DIR_BASE = "dataset/train"

# Source Folder (Master Horns)
HORNS_DIR = "audio/horns" 

# Noise Configuration
NOISE_CATEGORIES = {
    "Backgrounds": "audio/noise/background_noise",
}

# ============================================================
# KEY TRAINING PARAMETERS - ADJUST THESE
# ============================================================

SAMPLES_PER_CLASS = 1000  # 🔧 Increased from 2 for real training

# Timing Configuration (in seconds)
RANGE_INTERVAL = (0.8, 1.2)   # 🔧 Gap between consecutive blasts (was 0.7-1.0)
RANGE_PAUSE    = (2.0, 3.0)   # 🔧 Longer pause between groups (was 3.0-4.0)

# Short Blast Configuration
SHORT_BLAST_MIN = 0.8   # 🔧 Minimum short blast duration
SHORT_BLAST_MAX = 1.5   # 🔧 Maximum short blast duration

# Noise Mixing Configuration
SNR_MIN_DB = 3    # 🔧 Minimum Signal-to-Noise Ratio (horn 2x louder than noise)
SNR_MAX_DB = 15   # 🔧 Maximum Signal-to-Noise Ratio (horn 5.6x louder than noise)

# Augmentation ranges
PITCH_SHIFT_RANGE = (-3.0, 3.0)  # Semitones
TIME_STRETCH_RANGE = (0.92, 1.08)  # 🔧 Slightly wider variation

SECONDARY_EVENT_PROBABILITY = 0.65 

# ============================================================
# UTILITIES
# ============================================================

def load_sound(path, duration=None):
    """Load audio file and trim silence."""
    try:
        y, _ = librosa.load(path, sr=SR, duration=duration)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0: 
            return y_trimmed.astype(np.float32)
        return y.astype(np.float32)
    except Exception as e:
        print(f"   [Load Error] {os.path.basename(path)}: {e}")
        return None

def get_random_duration(range_tuple):
    """Get random duration from range tuple."""
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
    """Creates a Short blast by cutting a slice from the Master (Long) blast."""
    # 🔧 Use configurable short blast duration
    target_sec = random.uniform(SHORT_BLAST_MIN, SHORT_BLAST_MAX)
    target_samples = int(target_sec * SR)
    
    total_samples = len(long_blast_raw)
    
    if total_samples <= target_samples:
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
    except: 
        pass
    return blast.astype(np.float32)

def load_asset_library(folder_paths, duration=None):
    """Load all audio files from folder(s)."""
    library = []
    if isinstance(folder_paths, str): 
        folder_paths = [folder_paths]
    
    print(f"DEBUG: Scanning paths: {folder_paths}")
    
    for folder_path in folder_paths:
        if isinstance(folder_path, list): 
            folder_path = folder_path[0]
            
        if not os.path.exists(folder_path):
            print(f"DEBUG: Folder not found inside container: {folder_path}")
            continue
        
        search_pattern = os.path.join(folder_path, "**", "*.wav")
        files = glob.glob(search_pattern, recursive=True)
        
        print(f"DEBUG: Found {len(files)} .wav files in {folder_path}")
        
        for f in files:
            y = load_sound(f, duration)
            if y is not None and len(y) > 0: 
                library.append(y)
            
    return library

def mix_audio_with_snr(signal, noise, snr_db):
    """
    🔧 NEW: Mix signal with noise at specified SNR (Signal-to-Noise Ratio).
    
    SNR = 10 * log10(signal_power / noise_power)
    
    Higher SNR = signal is louder relative to noise
    - SNR 3dB  = signal ~2x louder than noise
    - SNR 10dB = signal ~3.2x louder than noise  
    - SNR 15dB = signal ~5.6x louder than noise
    """
    # Ensure same length
    if len(noise) < len(signal):
        # Loop noise if too short
        repeats = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, repeats)[:len(signal)]
    elif len(noise) > len(signal):
        # Random crop if noise is longer
        max_start = len(noise) - len(signal)
        start = random.randint(0, max_start)
        noise = noise[start : start + len(signal)]
    
    # Calculate power
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:  # Avoid division by zero
        return signal
    
    # Calculate required noise scaling factor
    # SNR = 10 * log10(signal_power / (scale^2 * noise_power))
    # scale = sqrt(signal_power / (noise_power * 10^(SNR/10)))
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    
    # Mix
    mixed = signal + noise * scale
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed / max_val * 0.95
    
    return mixed.astype(np.float32)

# ============================================================
# GENERATOR
# ============================================================

def generate_sample(filename, pattern_def, bg_noise, horns_lib, use_noise=False):
    """Generate a single training sample with the specified pattern."""
    
    # 1. Warmup (random silence/noise at start)
    warmup_sec = random.uniform(0.3, 1.0)
    warmup_samples = int(warmup_sec * SR)
    
    if use_noise and bg_noise is not None:
        # Get random chunk from background noise
        max_start = max(0, len(bg_noise) - warmup_samples)
        start = random.randint(0, max_start) if max_start > 0 else 0
        combined_audio = bg_noise[start : start + warmup_samples].copy()
        # Scale down warmup noise
        combined_audio = combined_audio * 0.3
    else:
        combined_audio = np.zeros(warmup_samples, dtype=np.float32)

    # 2. Select Master Horn
    if not horns_lib: 
        return 
    
    base_source = random.choice(horns_lib)

    # 3. Generate random augmentation (ONCE per sample for consistency)
    aug_steps = random.uniform(*PITCH_SHIFT_RANGE)
    aug_rate = random.uniform(*TIME_STRETCH_RANGE)
    
    # Apply augmentation to create this "ship's voice"
    base_source_aug = augment_horn_blast(base_source, aug_steps, aug_rate)
    
    # Create both Long and Short blasts from the SAME augmented source
    blast_long = process_blast(base_source_aug)
    blast_short = create_short_from_long(base_source_aug)

    # 4. Build Pattern
    for sound_type, gap_type in pattern_def:
        if sound_type == 'short':
            combined_audio = np.concatenate((combined_audio, blast_short))
        elif sound_type == 'long':
            combined_audio = np.concatenate((combined_audio, blast_long))
        elif sound_type == 'none':
            pass  # No sound, just gap
        
        # Add Gap
        if gap_type == 'none': 
            gap_dur = 0
        elif gap_type == 'interval': 
            gap_dur = get_random_duration(RANGE_INTERVAL)
        elif gap_type == 'pause': 
            gap_dur = get_random_duration(RANGE_PAUSE)
        else:
            gap_dur = 0
            
        if gap_dur > 0:
            gap_samples = int(gap_dur * SR)
            if use_noise and bg_noise is not None:
                max_start = max(0, len(bg_noise) - gap_samples)
                start = random.randint(0, max_start) if max_start > 0 else 0
                noise_chunk = bg_noise[start : start + gap_samples].copy() * 0.3
                combined_audio = np.concatenate((combined_audio, noise_chunk))
            else:
                combined_audio = np.concatenate((combined_audio, np.zeros(gap_samples, dtype=np.float32)))

    # 5. Add ending silence/noise
    ending_sec = random.uniform(0.3, 1.0)
    ending_samples = int(ending_sec * SR)
    if use_noise and bg_noise is not None:
        max_start = max(0, len(bg_noise) - ending_samples)
        start = random.randint(0, max_start) if max_start > 0 else 0
        ending = bg_noise[start : start + ending_samples].copy() * 0.3
        combined_audio = np.concatenate((combined_audio, ending))
    else:
        combined_audio = np.concatenate((combined_audio, np.zeros(ending_samples, dtype=np.float32)))

    # 6. 🔧 NEW: Mix with background noise using proper SNR
    if use_noise and bg_noise is not None:
        snr_db = random.uniform(SNR_MIN_DB, SNR_MAX_DB)
        combined_audio = mix_audio_with_snr(combined_audio, bg_noise, snr_db)

    # 7. Save
    path = f"{OUTPUT_DIR}/{filename}.wav"
    sf.write(path, combined_audio, SR)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['clean', 'noisy'], required=True)
    args = parser.parse_args()
    
    USE_NOISE = (args.mode == 'noisy')
    OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, args.mode)
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    try:
        # 1. Load Master Horns
        horns_lib = load_asset_library(HORNS_DIR)
        if not horns_lib: 
            raise ValueError(f"No .wav files found in {HORNS_DIR}.")
        
        print(f"✓ Loaded {len(horns_lib)} horn file(s)")
        
        bg_lib = []
        if USE_NOISE:
            print("🔊 NOISE MODE: ON (Loading backgrounds...)")
            bg_path = NOISE_CATEGORIES.get("Backgrounds")
            
            if bg_path and os.path.exists(bg_path):
                bg_lib = load_asset_library(bg_path, duration=60)
            
            if not bg_lib:
                print("WARNING: No backgrounds found! Reverting to Clean mode.")
                USE_NOISE = False
            else:
                print(f"✓ Loaded {len(bg_lib)} background file(s)")
                print(f"  SNR range: {SNR_MIN_DB}dB to {SNR_MAX_DB}dB")
        else:
            print("🔇 NOISE MODE: OFF (Generating clean signals)")
            
    except Exception as e:
        print(f"Fatal Error: {e}")
        exit(1)

    print(f"\n{'='*50}")
    print(f"Generating {args.mode.upper()} dataset")
    print(f"  Samples per class: {SAMPLES_PER_CLASS}")
    print(f"  Short blast: {SHORT_BLAST_MIN}-{SHORT_BLAST_MAX}s")
    print(f"  Interval gap: {RANGE_INTERVAL[0]}-{RANGE_INTERVAL[1]}s")
    print(f"  Pause gap: {RANGE_PAUSE[0]}-{RANGE_PAUSE[1]}s")
    print(f"{'='*50}\n")
    
    # 12 Classes - COLREG Signal Patterns
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
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{SAMPLES_PER_CLASS} iterations ({total} files)")
            
    print(f"\n✅ Done. Generated {total} files in {OUTPUT_DIR}")