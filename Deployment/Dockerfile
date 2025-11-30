# 1. Base Image (NVIDIA PyTorch)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install System Dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libsndfile1 \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Copy and Install Python Dependencies
COPY Deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Source Code and Web Interface
COPY src/ src/
COPY web/ web/

# 6. Copy Scripts
COPY Deployment/train_pipeline.sh .
COPY Deployment/start_web.sh .

# 7. Fix Permissions (Windows line endings)
RUN dos2unix train_pipeline.sh && chmod +x train_pipeline.sh && \
    dos2unix start_web.sh && chmod +x start_web.sh

# 8. Create Required Directories
RUN mkdir -p /app/audio /app/models /app/dataset /app/web/temp_uploads

# 9. Expose Ports
EXPOSE 5000 8000

# 10. Default Command (Training Pipeline)
CMD ["./train_pipeline.sh"]