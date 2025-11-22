
# 🚢 COLREG Sound Signal Classifier

A robust Machine Learning system designed to detect and classify maritime sound signals according to the **International Regulations for Preventing Collisions at Sea (COLREGs)**, specifically **Rules 34 & 35**.

This project uses a Deep Learning approach (CNN + GRU) to classify audio signals into categories such as "Altering Course to Starboard", "Overtaking", "Not Under Command", and more. It features a complete pipeline that generates synthetic training data from raw horn samples, preprocesses audio into Mel-Spectrograms, and trains the classifier.

## 🌟 Features

-   **Synthetic Data Generation:** automatically creates thousands of labeled training samples by mixing raw horn blasts with realistic background noise (wind, engine, sea, birds).
    
-   **Deep Learning Model:** Utilizes a Convolutional Neural Network (CNN) for feature extraction and Gated Recurrent Units (GRU) for temporal sequence recognition.
    
-   **Dockerized Workflow:** Entire pipeline (Generation -> Processing -> Training -> Inference) runs with a single command, ensuring reproducibility across any platform.
    
-   **Robust Inference:** capable of detecting signals in noisy environments.
    

## 📁 Project Structure

To use this system, you must organize your local audio assets in the following structure before running the container. The Docker container will mount these folders to read input and save output.

```
.
├── models/                  <-- (Output) The trained model file (.pth) will appear here
├── src/                     <-- Source code (scripts)
├── Dockerfile               <-- Container definition
├── requirements.txt         <-- Python dependencies
├── train_pipeline.sh        <-- Pipeline orchestrator
└── audio/                   <-- (Input) PUT YOUR RAW WAV FILES HERE
    ├── horns/
    │   ├── short/           <-- Place short blast samples here (~1s)
    │   └── long/            <-- Place long blast samples here (~4-6s)
    └── noise/
        ├── background_noise/<-- Long clips: Wind, Engine, Sea noise
        ├── bird_sounds/     <-- (Optional) Seagulls, etc.
        ├── alarms/          <-- (Optional) Deck alarms
        ├── white_noise/     <-- (Optional) Electronic static
        ├── calm_sea/        <-- (Optional) Water lapping
        └── thunderstorm/    <-- (Optional) Heavy weather

```

## 🚀 Quick Start (Docker)

### 1. Build the Image

First, build the Docker image. This installs all dependencies (PyTorch, Librosa, etc.).

```
docker build -t colreg-classifier .

```

### 2. Run Training Pipeline

Run the container to start the full workflow:

1.  **Generate** synthetic dataset from your `audio/` folder.
    
2.  **Preprocess** audio into Spectrograms.
    
3.  **Train** the model.
    
4.  **Save** the result to your local `models/` folder.
    

**Linux / Mac:**

```
docker run --rm \
  -v "$(pwd)/audio:/app/audio" \
  -v "$(pwd)/models:/app/models" \
  colreg-classifier

```

**Windows (PowerShell):**

```
docker run --rm `
  -v "${PWD}/audio:/app/audio" `
  -v "${PWD}/models:/app/models" `
  colreg-classifier

```

Once finished, you will see `colreg_classifier_best.pth` inside your `models/` folder.

## 🔎 How to Use (Inference)

You can use the same Docker image to classify a new audio file.

Assume you have a file named **`test_signal.wav`** in your current directory.

**Linux / Mac:**

```
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/test_signal.wav:/app/input.wav" \
  --entrypoint python \
  colreg-classifier \
  src/predictor.py --file /app/input.wav --model /app/models/colreg_classifier_best.pth

```

**Windows (PowerShell):**

```
docker run --rm `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/test_signal.wav:/app/input.wav" `
  --entrypoint python `
  colreg-classifier `
  src/predictor.py --file /app/input.wav --model /app/models/colreg_classifier_best.pth

```

### Example Output

```
========================================
PREDICTION RESULT
========================================
Detected Signal:  OVERTAKE PORT (Two Prolonged, Two Short)
Confidence:       98.45%
----------------------------------------

```

## 🛠️ Configuration

You can adjust training parameters in `src/data_gen.py` and `src/train_colreg_classifier.py`:

-   `SAMPLES_PER_CLASS`: Number of synthetic files to generate per class (Default: 50).
    
-   `SR` (Sample Rate): Default is 22050Hz.
    
-   `SECONDARY_EVENT_PROBABILITY`: Chance of adding birds/alarms to the background.
    

## 📋 Supported Classes

1.  **Alter Starboard** (1 Short)
    
2.  **Alter Port** (2 Short)
    
3.  **Astern Propulsion** (3 Short)
    
4.  **Danger/Doubt** (5 Short)
    
5.  **Round Starboard** (4 Short, 1 Short)
    
6.  **Round Port** (4 Short, 2 Short)
    
7.  **Making Way** (1 Long)
    
8.  **Not Under Command** (1 Long, 2 Short)
    
9.  **Overtake Starboard** (2 Long, 1 Short)
    
10.  **Overtake Port** (2 Long, 2 Short)
    
11.  **Agree Overtake** (1 Long, 1 Short, 1 Long, 1 Short)
    

## 📜 License

This project is open-source. Please attribute the contributors if used in production or academic work.