🚢 COLREG Sound Signal Classifier (Autonomous Vessel Stack)This repository contains the Machine Learning pipeline, model architecture, and deployment environment (via Docker) for classifying maritime COLREG maneuvering and warning signals from pre-computed Mel Spectrogram features.The solution addresses the critical challenge of recognizing standardized sequences of short (S) and long (L) blasts (e.g., S-S-S for "Astern Propulsion") under realistic, noisy maritime conditions, providing input for automated navigation systems.⚙️ Core Technology & ArchitectureThe classification is performed using a specialized Deep Learning model designed for sequential audio features:Feature Extraction: Mel Spectrograms (2D time-frequency representations) are used as input, capturing both the horn's frequency content and the precise timing of the blasts.Model: A Convolutional Neural Network (CNN)  filters noise and extracts spectral features, feeding its output into a Gated Recurrent Unit (GRU) (a type of Recurrent Neural Network).Sequence Modeling: The GRU learns the temporal patterns (the sequence of S and L blasts) which define the COLREG signal, making it robust to variations in horn timbre and environmental noise.  
  
📁 Repository Structure.  
├── colreg_features/  
│   ├── features/  
│   │   ├── Alter_Starboard_001.npy  # Your 2D Mel Spectrogram Feature Files  
│   │   └── ...  
│   └── labels.npy                  # Metadata for all features (Class ID, Path)  
├── src/  
│   ├── model.py                    # (Optional: Contains the ColregClassifier class)  
│   ├── train_colreg_classifier.py  # Primary script: loads features, trains model, saves weights  
│   └── predict.py                  # **REQUIRED:** Inference script for real-time classification  
├── deployment/  
│   ├── Dockerfile                  # Defines the container environment and dependencies  
│   └── requirements.txt            # Python dependencies (PyTorch, Librosa, NumPy)  
├── docs/  
│   └── one_pager_summary.pdf       # Submission: Project summary and findings  
├── test_samples/  
│   ├── example_sss_horn.wav        # Submission: Example raw audio files for testing  
│   └── example_noise_only.wav  
├── .gitignore                      # Ensures large artifacts (data, model files) are ignored  
└── README.md  
🛠️ PrerequisitesTo build and run the training pipeline, you need:Docker: Must be installed and running on your system.Data: Your pre-computed Mel Spectrogram features (as NumPy .npy files) must be placed in the colreg_features/features/ subdirectory, and the corresponding labels.npy metadata file must be present in the colreg_features/ root.🚀 Quick Start (Dockerized Training)The training process is fully containerized, ensuring a reproducible environment and simplifying deployment.  
1. Build the Training ImageNavigate to the project root directory and execute the following command to build the Docker image using the configuration defined in deployment/Dockerfile:docker build -t colreg-trainer ./deployment  
2. Run the TrainingExecute the training script within the container. The -v (volume) flags are critical: they map your local data and source code directories into the container, allowing the trained model to be saved back to your host machine.# Ensure you are in the project root directory:  
docker run --rm \  
  -v "$(pwd)/colreg_features:/app/colreg_features" \  
  -v "$(pwd)/src:/app/src" \  
  -v "$(pwd):/app" \  
  colreg-trainer python src/train_colreg_classifier.py  
Output: The command will output the loss and validation accuracy for each epoch, and save the final best model weights as colreg_classifier_best.pth in your project root directory.3. Deployment (Inference)Once the model is trained, the next step is to create the predict.py script. This script will load the colreg_classifier_best.pth file and be wrapped in a final, smaller Docker image for real-time edge deployment. You will need to define a similar Docker run command to classify new input .wav files.
