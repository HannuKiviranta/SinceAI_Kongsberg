try_1 Branch

- 1. Build Docker Image

docker build -t colreg-classifier -f Deployment/Dockerfile .

- 2. Starts APO and Frontedn (web UI)

docker run --rm `
  -p 5000:5000 `
  -p 8000:8000 `
  -v "${PWD}/audio:/app/audio" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/dataset:/app/dataset" `
  colreg-classifier `
  ./start_web.sh


UI → http://localhost:8000

API (if you want to test) → http://localhost:5000


- 3. Full pipeline

docker run --rm --gpus all `
  -v "${PWD}/audio:/app/audio" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/dataset:/app/dataset" `
  colreg-classifier
