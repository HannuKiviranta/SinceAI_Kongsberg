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
echo "Step [1/7]: Generating CLEAN dataset (500 samples/class)."
python src/data_gen.py --mode clean

echo "Step [2/7]: Preprocessing CLEAN data."
python src/preprocess.py --source dataset/train/clean --label_file labels_clean.npy

# --- PHASE 2: NOISY DATA ---
if [ -d "/app/audio/noise/background_noise" ] && [ "$(ls -A /app/audio/noise/background_noise)" ]; then
    echo "Step [3/7]: Generating NOISY dataset (500 samples/class)."
    python src/data_gen.py --mode noisy

    echo "Step [4/7]: Preprocessing NOISY data."
    python src/preprocess.py --source dataset/train/noisy --label_file labels_noisy.npy
else
    echo "⚠️  WARNING: No background noise found in /app/audio/noise/background_noise"
    echo "   Skipping Noisy Phase generation. Training will be Clean-only."
fi

# --- PHASE 3: SEAGULL NOISE DATA (optional) ---
if [ -d "/app/audio/noise/seagulls" ] && [ "$(ls -A /app/audio/noise/seagulls)" ]; then
    echo "Step [5/7]: Generating SEAGULL dataset (reduced samples/class)."
    python src/data_gen.py --mode seagulls

    echo "Step [6/7]: Preprocessing SEAGULL data."
    python src/preprocess.py --source dataset/train/seagulls --label_file labels_seagulls.npy
else
    echo "⚠️  WARNING: No seagull noise found in /app/audio/noise/seagulls"
    echo "   Skipping Seagull fine-tuning phase."
fi

# --- PHASE 4: TRAINING ---
echo "Step [7/7]: Starting Curriculum Training (Clean -> Noisy -> Seagulls)..."
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
