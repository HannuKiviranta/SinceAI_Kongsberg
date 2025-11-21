🚢 COLREG Sound Signal Classifier (Autonomous Vessel Stack)This repository contains the Machine Learning pipeline, model architecture, and deployment environment (via Docker) for classifying maritime COLREG maneuvering and warning signals from raw audio/Mel Spectrogram features.The solution addresses the challenge of recognizing sequences of short (S) and long (L) blasts (e.g., S-S-S for "Astern Propulsion") under noisy maritime conditions.📁 Repository Structure.
├── colreg_features/
│   ├── features/
│   │   ├── Alter_Starboard_001.npy  # Your 2D Mel Spectrogram Feature Files
│   │   └── ...
│   └── labels.npy                  # Metadata for all features (Class ID, Path)
├── src/
│   ├── model.py                    # (Optional: If you pull the NN module out of train.py)
│   ├── train_colreg_classifier.py  # Primary training script (Python)
│   └── predict.py                  # **REQUIRED:** Inference script for deployment (to be written)
├── deployment/
│   ├── Dockerfile                  # Docker build instructions for the containerized pipeline
│   └── requirements.txt            # Python dependencies (PyTorch, Librosa, NumPy)
├── docs/
│   └── one_pager_summary.pdf       # Submission: Project summary, findings, limitations
├── test_samples/
│   ├── example_sss_horn.wav        # Submission: Example audio files for demo
│   └── example_noise_only.wav
├── .gitignore                      # Defines files/folders to ignore (e.g., trained model, large data)
└── README.md
🚀 Quick Start (Dockerized Training)This pipeline is designed for a single-command training run using Docker, fulfilling the deployment requirement.PrerequisitesDocker must be installed and running.Your feature data (colreg_features/) must be ready, following the structure above.The Dockerfile, requirements.txt, and train_colreg_classifier.py must be in their respective locations.1. Build the Training ImageNavigate to the project root directory and build the Docker image:docker build -t colreg-trainer ./deployment
2. Run the TrainingExecute the training script. This command mounts your local colreg_features directory into the container, allowing the script to read your data and save the final model artifact back to your machine.# Ensure you are in the project root directory:
docker run --rm \
  -v "$(pwd)/colreg_features:/app/colreg_features" \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd):/app" \
  colreg-trainer python src/train_colreg_classifier.py
3. Deployment (Inference)After training, the best model weights will be saved as colreg_classifier_best.pth in your root directory. The next step is to finalize the predict.py script and run it via Docker to classify new audio files.
