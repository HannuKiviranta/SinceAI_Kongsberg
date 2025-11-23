
# 🐳 Docker Deployment Guide

This guide provides detailed instructions on how to build, run, and utilize the **COLREG Sound Signal Classifier** using Docker. This approach ensures the application runs consistently on any operating system (Windows, Mac, Linux) without manual dependency installation.

## 📋 Prerequisites

-   **Docker Desktop** installed and running.
    
-   **NVIDIA GPU Drivers** (Optional, for faster training on Windows/Linux).
    
-   **Audio Data:** Your source `.wav` files must be organized in the `audio/` folder as described in the main README.
    

## 1. Build the Docker Image

You must build the image once before using it. This packages Python, PyTorch, Librosa, and all project scripts into a portable container.

**Command (Run from project root):**

```
docker build -t colreg-classifier -f Deployment/Dockerfile .

```

> **Note:** The `-f Deployment/Dockerfile` flag tells Docker where to find the configuration file, while the `.` at the end sets the "build context" to the current folder (so it can see your `src/` and `audio/` folders).

## 2. Train the Model (Full Pipeline)

This single command launches the entire training workflow:

1.  **Generates** synthetic data (Clean & Noisy).
    
2.  **Preprocesses** audio into spectrograms.
    
3.  **Trains** the Neural Network (Curriculum Learning).
    
4.  **Saves** the final model (`colreg_classifier_best.pth`) to your local `models/` folder.
    

### For Linux / Mac (Intel & M1/M2/M3)

```
docker run --rm \
  -v "$(pwd)/audio:/app/audio" \
  -v "$(pwd)/models:/app/models" \
  colreg-classifier

```

### For Windows (PowerShell)

```
docker run --rm `
  -v "${PWD}/audio:/app/audio" `
  -v "${PWD}/models:/app/models" `
  colreg-classifier

```

### For NVIDIA GPU Users (Faster)

Add the `--gpus all` flag to your command:

```
docker run --rm --gpus all ... (rest of command)

```

## 3. Run Prediction (Inference)

Once you have a trained model in your `models/` folder, you can use the container to classify new audio files.

**Scenario:** You have a recording named `recording.wav` inside your `input_to_predict_COLREG/` folder.

### Command (Linux / Mac)

```
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/input_to_predict_COLREG:/app/input" \
  -v "$(pwd)/predictor_logs:/app/predictor_logs" \
  --entrypoint python \
  colreg-classifier \
  src/predictor.py --file /app/input/recording.wav --model /app/models/colreg_classifier_best.pth

```

### Command (Windows PowerShell)

```
docker run --rm `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/input_to_predict_COLREG:/app/input" `
  -v "${PWD}/predictor_logs:/app/predictor_logs" `
  --entrypoint python `
  colreg-classifier `
  src/predictor.py --file /app/input/recording.wav --model /app/models/colreg_classifier_best.pth

```

**Output Example:**

```
========================================
PREDICTION RESULT
========================================
Detected Signal:  OVERTAKE PORT (Two Prolonged, Two Short)
Confidence:       98.45%
----------------------------------------
[Log saved to predictor_logs/prediction_log.txt]

```

## 🛠️ Troubleshooting

-   **"Image not found":** Ensure you ran the `docker build` command in Step 1.
    
-   **"No such file or directory":** Ensure you are running the command from the root `SinceAI_Konsberg` folder.
    
-   **"No audio files found":** Check that your `audio/horns/short` and `audio/horns/long` folders are not empty.