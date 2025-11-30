import matplotlib
matplotlib.use('Agg') # Critical for running in Docker without a monitor
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
import librosa
import librosa.display
import os
import sys
import time
import subprocess
from werkzeug.utils import secure_filename

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Robust Import Strategy
try:
    from src.predictor import ColregClassifier, preprocess_audio, COLREG_CLASSES, N_MELS, SR, HOP_LENGTH
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
    from predictor import ColregClassifier, preprocess_audio, COLREG_CLASSES, N_MELS, SR, HOP_LENGTH

# Setup Web Folder
WEB_FOLDER = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(WEB_FOLDER) != 'web':
    WEB_FOLDER = 'web'

app = Flask(__name__, static_folder=WEB_FOLDER)
CORS(app)

# --- CONFIGURATION ---
UPLOAD_FOLDER = '/app/temp_uploads' if os.path.exists('/app') else 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav', 'webm', 'ogg'} # Webm/Ogg needed for browser mic
MODEL_PATH = "models/colreg_classifier_best.pth" 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  WARNING: Model not found at {MODEL_PATH}")
            return False
        
        model = ColregClassifier(num_classes=len(COLREG_CLASSES), input_height=N_MELS)
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

def generate_spectrogram_image(y, sr):
    """Generates a Base64 PNG string of the Mel-Spectrogram for the UI"""
    # Generate Mel Spectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mels_db = librosa.power_to_db(mels, ref=np.max)

    # Plotting
    plt.figure(figsize=(10, 3))
    plt.style.use('dark_background') 
    # Remove axes for a cleaner look
    plt.axis('off')
    librosa.display.specshow(mels_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='magma')
    plt.tight_layout(pad=0)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    
    # Encode
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# --- ROUTES ---

@app.route('/')
def index():
    return send_from_directory(WEB_FOLDER, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(WEB_FOLDER, path)

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'ready' if model is not None else 'model_not_loaded',
        'device': str(device),
        'model_path': MODEL_PATH
    })

# --- 1. PREVIEW ENDPOINT (Visualizes Audio) ---
@app.route('/api/preview', methods=['POST'])
def preview_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(f"{int(time.time())}_{file.filename}")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(filepath)
        # Load audio
        y, sr = librosa.load(filepath, sr=22050)
        
        # Generate Visual
        spectrogram_image = generate_spectrogram_image(y, sr)
        
        return jsonify({
            'success': True,
            'spectrogram_image': spectrogram_image,
            'duration': len(y)/sr,
            'temp_filename': filename # Send this back to UI so it can request classification next
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 2. CLASSIFY ENDPOINT (Analyze Existing File) ---
@app.route('/api/classify', methods=['POST'])
def classify_audio():
    if model is None:
        if not load_model():
            return jsonify({'error': 'Model not loaded'}), 500
    
    # Get filename from the previous 'preview' step
    filename = request.form.get('filename')
    filepath = None

    if filename:
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': 'File expired or not found.'}), 400
    elif 'file' in request.files:
        # Direct upload fallback
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
    else:
        return jsonify({'error': 'No filename or file provided'}), 400
    
    start_time = time.time()
    
    try:
        input_tensor = preprocess_audio(filepath)
        
        if input_tensor is None:
            raise ValueError("Preprocessing returned None")
        
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, 1)
            
            confidence = score.item() * 100
            idx = predicted_idx.item()
            
            # Unknown Logic
            if confidence < 65.0:
                predicted_class = "UNKNOWN / UNRECOGNIZED"
                status = "IGNORED"
            else:
                predicted_class = COLREG_CLASSES[idx]
                status = "ACCEPTED"

        # Formatting probabilities for UI
        probs_dict = {
            COLREG_CLASSES[i]: round(probabilities[0][i].item() * 100, 2) 
            for i in range(len(COLREG_CLASSES))
        }

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'status': status,
            'processing_time': round(time.time() - start_time, 3),
            'probabilities': probs_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("🚢 COLREG API SERVER STARTING")
    print("="*50)
    load_model()
    app.run(host='0.0.0.0', port=5000)