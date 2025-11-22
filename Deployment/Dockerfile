# Use a slim Python image for a smaller footprint
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install necessary system dependencies
# These are crucial for libraries like soundfile and librosa
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libblas3 \
    liblapack3 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements and install Python dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy all source code (Generator, Preprocessor, and Trainer)
# We copy them into the correct folder structure (/app/src/)
RUN mkdir -p src/
COPY src/train_colreg_classifier.py src/
COPY src/data_gen.py src/       # <-- Data Generator
COPY src/preprocess.py src/     # <-- Feature Extractor

# 4. Create the required data mount point
# This directory MUST match the mount point you use in the 'docker run' command.
# It aligns with your host directory structure (dataset/ features, train, labels.npy)
RUN mkdir -p dataset

# Set the default command to run the training script
CMD ["python", "src/train_colreg_classifier.py"]
