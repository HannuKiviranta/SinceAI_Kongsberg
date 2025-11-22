# Use a lightweight version of Python 3.9
FROM python:3.9-slim

# Set environment variables to keep Python output clean and unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system libraries required for audio processing (libsndfile is mandatory for Librosa)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache for dependencies
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts into the container
COPY src/ src/
COPY predictor.py .
COPY train_pipeline.sh .

# Make the pipeline script executable
RUN chmod +x train_pipeline.sh

# Create directories that will be used for mounting external data volumes
RUN mkdir -p /app/audio /app/models /app/dataset

# Set the default command to run the training pipeline when the container starts
ENTRYPOINT ["./train_pipeline.sh"]