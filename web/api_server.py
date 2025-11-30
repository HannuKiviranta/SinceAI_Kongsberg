from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import librosa
import os
import sys
import time
import subprocess
from werkzeug.utils import secure_filename

# --- PATH SETUP ---
# Add 'src' to the system path so we can import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from your existing scripts
from src.predictor import ColregClassifier, preprocess_audio, COLREG_CLASSES, N_MELS

app = Flask(__name__)
CORS(app)  # Enable CORS

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav'}
# FIX: Point to the correct location in your structure
MODEL_PATH = "models/colreg_classifier_best.pth" 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  WARNING: Model not found at {MODEL_PATH}")
            return False
        
        # Initialize Architecture
        model = ColregClassifier(num_classes=len(COLREG_CLASSES), input_height=N_MELS)
        
        # Load Weights
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- ROUTES ---

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check if the system is ready"""
    return jsonify({
        'status': 'ready' if model is not None else 'model_not_loaded',
        'device': str(device),
        'model_path': MODEL_PATH,
        'classes_count': len(COLREG_CLASSES)
    })

@app.route('/api/classify', methods=['POST'])
def classify_audio():
    """Classify uploaded audio file"""
    start_time = time.time()
    
    if model is None:
        # Try loading it again in case it was added later
        if not load_model():
            return jsonify({'error': 'Model not loaded. Train or upload a model first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .wav allowed'}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    
    try:
        file.save(filepath)
        
        # Check Audio Properties
        y, sr = librosa.load(filepath, sr=22050)
        duration = len(y) / sr
        
        # Preprocess using your existing logic
        input_tensor = preprocess_audio(filepath)
        
        if input_tensor is None:
            raise ValueError("Preprocessing returned None")
        
        # Inference
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, 1)
            
            confidence = score.item() * 100
            idx = predicted_idx.item()
            
            # Handle Unknowns (Threshold Check)
            if confidence < 65.0:
                predicted_class = "UNKNOWN / UNRECOGNIZED"
                status = "IGNORED"
            else:
                predicted_class = COLREG_CLASSES[idx]
                status = "ACCEPTED"

        # Prepare clean probability dictionary
        probs_dict = {
            COLREG_CLASSES[i]: round(probabilities[0][i].item() * 100, 2) 
            for i in range(len(COLREG_CLASSES))
        }

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'status': status,
            'audio_duration': round(duration, 2),
            'processing_time': round(time.time() - start_time, 3),
            'probabilities': probs_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/train/generate', methods=['POST'])
def generate_training_data():
    """Trigger data generation"""
    try:
        # We allow passing 'clean' or 'noisy' via query param ?mode=clean
        mode = request.args.get('mode', 'clean')
        
        cmd = ['python', 'src/data_gen.py', '--mode', mode]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        return jsonify({
            'success': result.returncode == 0,
            'command': " ".join(cmd),
            'output': result.stdout,
            'error': result.stderr
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/full_pipeline', methods=['POST'])
def run_pipeline():
    """Triggers the full training pipeline"""
    try:
        # We run the bash script directly
        result = subprocess.run(['./train_pipeline.sh'], capture_output=True, text=True)
        
        # Reload model if successful
        if result.returncode == 0:
            load_model()
            
        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("🚢 COLREG API SERVER STARTING")
    print("="*50)
    
    # Initial Model Load
    load_model()
    
    # Run on 0.0.0.0 to be accessible outside Docker
    app.run(host='0.0.0.0', port=5000)