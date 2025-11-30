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
# We copy from Deployment/ folder
COPY Deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Source Code
COPY src/ src/

# 6. Copy Web Application
# Copy the whole web folder (html, js, css)
COPY web/ web/
# Copy the server script to root for easy execution
COPY web/api_server.py .

# 7. Copy Pipeline Script
COPY Deployment/train_pipeline.sh .

# 8. Fix Permissions
RUN dos2unix train_pipeline.sh && chmod +x train_pipeline.sh

# 9. Create Data Directories
RUN mkdir -p /app/audio /app/models /app/dataset /app/predictor_logs /app/temp_uploads

# 10. Run Default Command
ENTRYPOINT ["./train_pipeline.sh"]