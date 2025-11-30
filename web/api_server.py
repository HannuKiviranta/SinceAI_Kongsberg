from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import librosa
import os
import sys
import time
import subprocess
from werkzeug.utils import secure_filename
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Must be BEFORE pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'web/temp_uploads'
ALLOWED_EXTENSIONS = {'wav'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model Configuration (Match your actual project)
SR = 22050
HOP_LENGTH = 512
N_MELS = 128
CLIP_DURATION_SEC = 20
MAX_WIDTH = int(np.ceil(CLIP_DURATION_SEC * (SR / HOP_LENGTH)))

# Updated to match your actual 12 classes
COLREG_CLASSES = [
    "Alter Starboard",           # 0
    "Alter Port",                # 1
    "Astern Propulsion",         # 2
    "Danger Signal (Doubt)",     # 3
    "Overtake Starboard",        # 4
    "Round Starboard",           # 5
    "Round Port",                # 6
    "Blind Bend / Making Way",   # 7
    "Overtake Port",             # 8
    "Agreement to Overtake",     # 9
    "Not Under Command",         # 10
    "Noise Only"                 # 11
]

MODEL_PATH = "models/colreg_classifier_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# --- Model Architecture (Must match train_colreg_classifier.py) ---
class ColregClassifier(nn.Module):
    def __init__(self, num_classes, input_height):
        super(ColregClassifier, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
            nn.Dropout(0.4)
        )
        
        self.output_channels = 64
        self.reduced_height = input_height // 2 // 2 // 4
        gru_input_size = self.output_channels * self.reduced_height
        
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        cnn_out = self.cnn(x)
        B, C, H, T = cnn_out.size()
        gru_input = cnn_out.permute(0, 3, 1, 2).contiguous().view(B, T, C * H)
        gru_out, _ = self.gru(gru_input)
        last_forward = gru_out[:, -1, :128]
        last_backward = gru_out[:, 0, 128:]
        gru_output_combined = torch.cat((last_forward, last_backward), dim=1)
        return self.classifier(gru_output_combined)

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Model not found at {MODEL_PATH}")
            return False
        
        model = ColregClassifier(num_classes=len(COLREG_CLASSES), input_height=N_MELS)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ Model loaded from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        target_samples = int(SR * CLIP_DURATION_SEC)
        
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), 'constant')
        elif len(y) > target_samples:
            y = y[:target_samples]
        
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mels_db = librosa.power_to_db(mels, ref=np.max)
        
        current_width = mels_db.shape[1]
        if current_width < MAX_WIDTH:
            padding = MAX_WIDTH - current_width
            mels_db = np.pad(mels_db, ((0, 0), (0, padding)), mode='constant', constant_values=-80)
        elif current_width > MAX_WIDTH:
            mels_db = mels_db[:, :MAX_WIDTH]
        
        tensor = torch.tensor(mels_db).float().unsqueeze(0).unsqueeze(0)
        
        # Return both tensor and spectrogram
        return tensor, mels_db
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None, None

# --- API ENDPOINTS ---

@app.route('/api/status', methods=['GET'])
def get_status():
    """System status check"""
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        'status': 'ready' if model is not None else 'not_ready',
        'model_loaded': model is not None,
        'model_exists': model_exists,
        'device': str(device),
        'classes': len(COLREG_CLASSES),
        'model_path': MODEL_PATH
    })

@app.route('/api/classify', methods=['POST'])
@app.route('/api/classify', methods=['POST'])
def classify_audio():
    """Classify uploaded audio"""
    start_time = time.time()
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .wav files allowed'}), 400
    
    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get duration
        y, sr = librosa.load(filepath, sr=SR)
        duration = len(y) / sr
        
        # Preprocess - NOW RETURNS BOTH
        input_tensor, mels_db = preprocess_audio(filepath)
        if input_tensor is None:
            return jsonify({'error': 'Failed to process audio'}), 400
        
        # Inference
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = COLREG_CLASSES[predicted_idx.item()]
            confidence = score.item() * 100
        
        processing_time = time.time() - start_time
        
        # Create spectrogram visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(mels_db, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Time Frames', color='white')
        ax.set_ylabel('Mel Frequency Bands', color='white')
        ax.set_title('Mel Spectrogram', color='#f5c041', fontsize=14, weight='bold')
        
        # Style to match UI
        fig.patch.set_facecolor('#010711')
        ax.set_facecolor('#010711')
        ax.tick_params(colors='#7dd0ff')
        for spine in ax.spines.values():
            spine.set_color('#7dd0ff')
        
        # Colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Amplitude (dB)', color='white')
        cbar.ax.yaxis.set_tick_params(color='#7dd0ff')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#7dd0ff')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#010711', edgecolor='none', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'processing_time': round(processing_time, 3),
            'audio_duration': round(duration, 2),
            'spectrogram': f'data:image/png;base64,{image_base64}',
            'all_probabilities': {
                COLREG_CLASSES[i]: round(probabilities[0][i].item() * 100, 2)
                for i in range(len(COLREG_CLASSES))
            }
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/train/generate_clean', methods=['POST'])
def generate_clean_data():
    """Generate clean training data"""
    try:
        result = subprocess.run(
            ['python', 'src/data_gen.py', '--mode', 'clean'],
            capture_output=True,
            text=True,
            timeout=600
        )
        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/generate_noisy', methods=['POST'])
def generate_noisy_data():
    """Generate noisy training data"""
    try:
        result = subprocess.run(
            ['python', 'src/data_gen.py', '--mode', 'noisy'],
            capture_output=True,
            text=True,
            timeout=600
        )
        return jsonify({
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/full_pipeline', methods=['POST'])
def run_full_pipeline():
    """Run complete training pipeline (Clean + Noisy + Train)"""
    try:
        # Step 1: Generate Clean
        print("Step 1/5: Generating clean data...")
        result1 = subprocess.run(
            ['python', 'src/data_gen.py', '--mode', 'clean'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result1.returncode != 0:
            return jsonify({
                'success': False,
                'error': 'Clean data generation failed',
                'output': result1.stderr
            }), 500
        
        # Step 2: Preprocess Clean
        print("Step 2/5: Preprocessing clean data...")
        result2 = subprocess.run(
            ['python', 'src/preprocess.py', '--source', 'dataset/train/clean', '--label_file', 'labels_clean.npy'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result2.returncode != 0:
            return jsonify({
                'success': False,
                'error': 'Clean preprocessing failed',
                'output': result2.stderr
            }), 500
        
        # Step 3: Generate Noisy
        print("Step 3/5: Generating noisy data...")
        result3 = subprocess.run(
            ['python', 'src/data_gen.py', '--mode', 'noisy'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result3.returncode != 0:
            return jsonify({
                'success': False,
                'error': 'Noisy data generation failed',
                'output': result3.stderr
            }), 500
        
        # Step 4: Preprocess Noisy
        print("Step 4/5: Preprocessing noisy data...")
        result4 = subprocess.run(
            ['python', 'src/preprocess.py', '--source', 'dataset/train/noisy', '--label_file', 'labels_noisy.npy'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result4.returncode != 0:
            return jsonify({
                'success': False,
                'error': 'Noisy preprocessing failed',
                'output': result4.stderr
            }), 500
        
        # Step 5: Train Model
        print("Step 5/5: Training model (this may take 10-30 minutes)...")
        result5 = subprocess.run(
            ['python', 'src/train_colreg_classifier.py'],
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result5.returncode == 0:
            # Reload model
            load_model()
            
            return jsonify({
                'success': True,
                'output': result5.stdout,
                'message': 'Full pipeline completed successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Training failed',
                'output': result5.stderr
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/train', methods=['POST'])
def train_model_endpoint():
    """Train model only (assumes data is already generated)"""
    try:
        result = subprocess.run(
            ['python', 'src/train_colreg_classifier.py'],
            capture_output=True,
            text=True,
            timeout=3600
        )
        
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
    print("="*70)
    print("🚢  COLREG SOUND SIGNAL CLASSIFIER - API SERVER")
    print("="*70)
    print(f"Device: {device}")
    print(f"Classes: {len(COLREG_CLASSES)}")
    print(f"Model Path: {MODEL_PATH}")
    
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠️  WARNING: Model not loaded!")
        print("    Train the model using the TRAINING tab in the web interface.")
        print("    Or run: python src/train_colreg_classifier.py\n")
    
    print("\n🌐 Starting Flask API server...")
    print("   API: http://localhost:5000")
    print("   Web UI: http://localhost:8000 (run separately)")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)