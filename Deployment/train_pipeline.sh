#!/bin/bash
set -e # Exit immediately if any command fails

echo "=========================================================="
echo "🚢  COLREG CLASSIFIER: CURRICULUM TRAINING PIPELINE"
echo "=========================================================="

# 1. Safety Check: Ensure audio files are mounted
if [ -z "$(ls -A /app/audio/horns 2>/dev/null)" ]; then
    echo "❌ ERROR: No horn files found!"
    echo "   Please make sure you mount your local 'audio' folder to '/app/audio'"
    exit 1
fi

# --- PHASE 1: CLEAN DATA ---
echo "Step [1/5]: Generating CLEAN dataset (500 samples/class)..."
python src/data_gen.py --mode clean

echo "Step [2/5]: Preprocessing CLEAN data..."
python src/preprocess.py --source dataset/train/clean --label_file labels_clean.npy

# --- PHASE 2: NOISY DATA ---
# Check if background noise exists in the mounted volume before trying to generate noisy data
if [ -d "/app/audio/noise/background_noise" ] && [ "$(ls -A /app/audio/noise/background_noise)" ]; then
    echo "Step [3/5]: Generating NOISY dataset (500 samples/class)..."
    python src/data_gen.py --mode noisy

    echo "Step [4/5]: Preprocessing NOISY data..."
    python src/preprocess.py --source dataset/train/noisy --label_file labels_noisy.npy
else
    echo "⚠️  WARNING: No background noise found in /app/audio/noise/background_noise"
    echo "   Skipping Noisy Phase generation. Training will be Clean-only."
fi

# --- PHASE 3: TRAINING ---
echo "Step [5/5]: Starting Curriculum Training (Clean -> Noisy)..."
# This script automatically handles loading labels_clean.npy first,
# then fine-tuning on labels_noisy.npy if it exists.
python src/train_colreg_classifier.py

# --- EXPORT ---
echo "Saving final model..."
if [ -f "colreg_classifier_best.pth" ]; then
    cp colreg_classifier_best.pth /app/models/colreg_classifier_best.pth
    echo "✅  SUCCESS! Model saved to 'models/colreg_classifier_best.pth'"
else
    # Fallback: If fine-tuning didn't run, maybe only the clean model exists
    if [ -f "colreg_model_clean.pth" ]; then
        cp colreg_model_clean.pth /app/models/colreg_classifier_best.pth
        echo "✅  SUCCESS! (Clean-only) Model saved to 'models/colreg_classifier_best.pth'"
    else
        echo "❌ ERROR: No model file found after training."
        exit 1
    fi
fi

echo "=========================================================="
echo "   PIPELINE COMPLETE"
echo "=========================================================="
