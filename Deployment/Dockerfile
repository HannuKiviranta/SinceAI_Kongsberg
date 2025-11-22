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

# 3. Copy all source code (Generator and Trainer)
# We assume source files are in the local 'src/' directory
COPY src/train_colreg_classifier.py src/
# If you run the generator in the container:
COPY src/generate_colreg_audio.py src/

# 4. Create the required data mount points
# These directories will be volume-mounted from the host machine
RUN mkdir -p colreg_features
# If you decide to generate audio inside Docker:
# RUN mkdir -p raw_audio_input 

# Set the default command to guide the user (but this will be overridden by the run command)
CMD ["python", "src/train_colreg_classifier.py"]
