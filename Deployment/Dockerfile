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

# 4. Install Python Dependencies
COPY Deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Source Code
COPY src/ src/
COPY web/ web/

# 6. Copy Pipeline Script
COPY Deployment/train_pipeline.sh .

# 7. Fix Permissions (Windows line endings)
RUN dos2unix train_pipeline.sh && chmod +x train_pipeline.sh

# 8. Create Required Directories
RUN mkdir -p /app/audio /app/models /app/dataset /app/web/temp_uploads

# 9. Expose Ports for Web Interface
EXPOSE 5000 8000

# 10. Default command - can be overridden
ENTRYPOINT ["./train_pipeline.sh"]
