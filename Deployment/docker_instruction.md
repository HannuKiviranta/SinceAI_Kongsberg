# 🐳 Docker Deploy (Build • Train • Serve • Predict)

Run the COLREG Sound Signal Classifier entirely via Docker.

## Prerequisites

Required directory structure:
```
audio/        # horn wav sources
models/       # trained models saved here
dataset/      # optional synthetic data
src/
web/
Deployment/
```

---

## Build Image

From project root:

```bash
docker build -t colreg-classifier -f Deployment/Dockerfile .
```

---

## Train Model

Generates synthetic data, trains CNN+GRU, saves best model.

### Linux/macOS

```bash
docker run --rm \
  -v "$(pwd)/audio:/app/audio" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/dataset:/app/dataset" \
  colreg-classifier
```

### Windows PowerShell

```powershell
docker run --rm `
  -v "${PWD}/audio:/app/audio" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/dataset:/app/dataset" `
  colreg-classifier
```

### Enable GPU

Add the following flag to enable GPU support:

```bash
--gpus all
```

Model will be saved to:
```
models/colreg_classifier_best.pth
```

---

## Run API + Web UI

### Linux/macOS

```bash
docker run --rm \
  -p 5000:5000 \
  -p 8000:8000 \
  -v "$(pwd)/audio:/app/audio" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/dataset:/app/dataset" \
  colreg-classifier \
  ./start_web.sh
```

### Windows PowerShell

```powershell
docker run --rm `
  -p 5000:5000 `
  -p 8000:8000 `
  -v "${PWD}/audio:/app/audio" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/dataset:/app/dataset" `
  colreg-classifier `
  ./start_web.sh
```

### Access the Services

- **API** → http://localhost:5000
- **UI** → http://localhost:8000

---

## Predict Single File

Required files:
- `models/colreg_classifier_best.pth`
- `input_to_predict_COLREG/recording.wav`

```bash
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/input_to_predict_COLREG:/app/input" \
  -v "$(pwd)/predictor_logs:/app/predictor_logs" \
  --entrypoint python \
  colreg-classifier \
  src/predictor.py \
    --file /app/input/recording.wav \
    --model /app/models/colreg_classifier_best.pth
```

---

## Troubleshooting

### Image not found

Rebuild the image:
```bash
docker build -t colreg-classifier -f Deployment/Dockerfile .
```

### No horn files

Ensure `audio/horns/*.wav` files are present in the audio directory.

### UI "Cannot connect to API"

Make sure port 5000 is properly mapped:
```bash
-p 5000:5000
```

### Port in use

Use alternative ports:
```bash
-p 5500:5000 -p 8500:8000
```

### Check GPU

```bash
docker ps
docker exec -it <container-id> nvidia-smi
```

---