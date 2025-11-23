
# 🚢 COLREG Sound Signal Classifier

A robust, containerized Machine Learning system designed to detect and classify maritime sound signals according to **COLREG Rules 34 & 35**.

This project uses a **CNN + GRU** deep learning architecture to identify signals like "Overtaking", "Altering Course", and "Not Under Command" in audio streams. It features a **Curriculum Learning** pipeline that first trains on clean synthetic data and then fine-tunes on noisy data (wind, waves, engine noise) for real-world robustness.

## 🌟 Features

-   **Curriculum Training:** Automated pipeline trains on clean data first, then improves with noisy environments.
    
-   **Synthetic Data Generator:** Creates thousands of labeled samples (`.wav`) mixed with realistic sea/wind noise.
    
-   **Dockerized Workflow:** Zero-dependency setup. One command to generate data, train, and output a model.
    
-   **GPU Acceleration:** Supports NVIDIA CUDA for fast training.
    

## 📁 Repository Structure

```
.
├── Deployment/              # Docker configuration files
│   ├── Dockerfile
│   ├── requirements.txt
│   └── train_pipeline.sh    # Orchestrator script
├── models/                  # Place your pre-trained .pth model here
├── audio/                   # Raw Audio Assets (Input for training)
│   ├── horns/               
│   └── noise/               
├── src/                     # Source Code
├── input_to_predict_COLREG/ # Place your .wav files here to test them
└── README.md

```

## 🚀 Quick Start (Docker)

### 1. Build the Image

Because the Dockerfile is in the `Deployment/` folder, run this specific command from the **root** of the project:

```
docker build -t colreg-classifier -f Deployment/Dockerfile .

```

### 2. Run the Training Pipeline (Optional)

If you **do not** have a model yet, run this to generate data and train one from scratch:

**Linux / Mac:**

```
docker run --rm --gpus all \
  -v "$(pwd)/audio:/app/audio" \
  -v "$(pwd)/models:/app/models" \
  colreg-classifier

```

**Windows (PowerShell):**

```
docker run --rm --gpus all `
  -v "${PWD}/audio:/app/audio" `
  -v "${PWD}/models:/app/models" `
  colreg-classifier

```

## ⚡ Using an Existing / Pre-Trained Model

If you already have a trained model file (e.g., `colreg_classifier_best.pth`), follow these steps to skip training and start predicting immediately.

### Step 1: Place your files

1.  Copy your trained model file into the **`models/`** folder on your computer.
    
    -   _Example:_ `models/colreg_classifier_best.pth`
        
2.  Copy the audio file you want to check into the **`input_to_predict_COLREG/`** folder.
    
    -   _Example:_ `input_to_predict_COLREG/recording.wav`
        

### Step 2: Run Prediction

Run this command to mount your local folders into the container. Docker will read the model from your `models` folder and the audio from your `input` folder.

**Linux / Mac:**

```
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/input_to_predict_COLREG:/app/input" \
  -v "$(pwd)/predictor_logs:/app/predictor_logs" \
  --entrypoint python \
  colreg-classifier \
  src/predictor.py --file /app/input/recording.wav --model /app/models/colreg_classifier_best.pth

```

**Windows (PowerShell):**

```
docker run --rm `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/input_to_predict_COLREG:/app/input" `
  -v "${PWD}/predictor_logs:/app/predictor_logs" `
  --entrypoint python `
  colreg-classifier `
  src/predictor.py --file /app/input/recording.wav --model /app/models/colreg_classifier_best.pth

```

### Example Output

```
========================================
PREDICTION RESULT
========================================
Detected Signal:  OVERTAKE PORT (Two Prolonged, Two Short)
Confidence:       98.45%
----------------------------------------
[Log saved to predictor_logs/prediction_log.txt]

```

## 🛠️ Configuration

You can tweak the system behavior by editing the files in `src/`.

File

Setting

Description

`src/data_gen.py`

`SAMPLES_PER_CLASS`

How many files to generate (Default: 500/phase)

`src/data_gen.py`

`RANGE_SNR_SECONDARY`

How loud the background noise is (in dB)

`src/preprocess.py`

`CLIP_DURATION_SEC`

Length of audio to analyze (Default: 20s)

### Supported Classes

1.  **Alter Starboard** (1 Short)
    
2.  **Alter Port** (2 Short)
    
3.  **Astern Propulsion** (3 Short)
    
4.  **Danger/Doubt** (5 Short)
    
5.  **Round Starboard** (4 Short, 1 Short)
    
6.  **Round Port** (4 Short, 2 Short)
    
7.  **Blind Bend / Making Way** (1 Long)
    
8.  **Overtake Starboard** (2 Long, 1 Short)
    
9.  **Overtake Port** (2 Long, 2 Short)
    
10.  **Agree Overtake** (1 Long, 1 Short, 1 Long, 1 Short)
    
11.  **Not Under Command** (1 Long, 2 Short)
    
12.  **Noise Only** (Background sounds)
    
13.  **Random Short Blasts** (Confusion signal)
    

## 📜 License

This project is open-source. Developed for Maritime Safety AI research.