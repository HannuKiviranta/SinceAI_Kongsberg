from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import librosa
import os
import sys
import time
from werkzeug.utils import secure_filename

# Add parent directory to path to import predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model architecture from predictor
from predictor import ColregClassifier, preprocess_audio, COLREG_CLASSES, N_MELS, MODEL_PATH

app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"WARNING: Model not found at {MODEL_PATH}")
            return False
        
        model = ColregClassifier(num_classes=len(COLREG_CLASSES), input_height=N_MELS)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check if the system is ready"""
    return jsonify({
        'status': 'ready' if model is not None else 'not_ready',
        'model_loaded': model is not None,
        'device': str(device),
        'classes': len(COLREG_CLASSES)
    })

@app.route('/api/classify', methods=['POST'])
def classify_audio():
    """Classify uploaded audio file"""
    start_time = time.time()
    
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .wav files allowed'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get audio duration
        y, sr = librosa.load(filepath, sr=22050)
        duration = len(y) / sr
        
        # Preprocess audio
        input_tensor = preprocess_audio(filepath)
        
        if input_tensor is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to process audio file'}), 400
        
        # Perform inference
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = COLREG_CLASSES[predicted_idx.item()]
            confidence = score.item() * 100
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up temporary file
        os.remove(filepath)
        
        # Return results
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'processing_time': round(processing_time, 3),
            'audio_duration': round(duration, 2),
            'all_probabilities': {
                COLREG_CLASSES[i]: round(probabilities[0][i].item() * 100, 2) 
                for i in range(len(COLREG_CLASSES))
            }
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/generate', methods=['POST'])
def generate_training_data():
    """Trigger data generation script"""
    try:
        import subprocess
        result = subprocess.run(
            ['python', 'src/data_gen.py'],
            capture_output=True,
            text=True,
            timeout=300
        )
        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/preprocess', methods=['POST'])
def preprocess_data():
    """Trigger preprocessing script"""
    try:
        import subprocess
        result = subprocess.run(
            ['python', 'src/preprocess.py'],
            capture_output=True,
            text=True,
            timeout=300
        )
        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/train', methods=['POST'])
def train_model_api():
    """Trigger training script"""
    try:
        import subprocess
        result = subprocess.run(
            ['python', 'src/train_colreg_classifier.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Reload model after training
        if result.returncode == 0:
            load_model()
        
        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("COLREG SOUND SIGNAL CLASSIFIER - API SERVER")
    print("="*60)
    
    # Try to load model
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠️  WARNING: Model not loaded. Train the model first!")
        print("    You can still use the training endpoints.\n")
    
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)