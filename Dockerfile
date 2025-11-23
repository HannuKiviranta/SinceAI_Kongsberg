# USE NVIDIA BASE IMAGE (Includes CUDA 12.1 and PyTorch pre-installed)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system libraries (libsndfile is still needed for Librosa)
# We use DEBIAN_FRONTEND=noninteractive to prevent timezone prompts
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libsndfile1 \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Note: We remove 'torch' and 'torchaudio' from requirements.txt below
# because the base image already has the optimized GPU versions!
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY predictor.py .
COPY train_pipeline.sh .

# Windows Fix & Permissions
RUN dos2unix train_pipeline.sh && chmod +x train_pipeline.sh

# Directories
RUN mkdir -p /app/audio /app/models /app/dataset

ENTRYPOINT ["./train_pipeline.sh"]