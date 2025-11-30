import torch
import torch.nn as nn
import numpy as np
import librosa
import argparse
import os
import sys
import datetime
import glob

# --- 1. CONFIGURATION ---
N_MELS = 128
SR = 22050
HOP_LENGTH = 512
CLIP_DURATION_SEC = 20 

# Calculate width dynamically
MAX_WIDTH = int(np.ceil(CLIP_DURATION_SEC * (SR / HOP_LENGTH)))

MODEL_PATH = "models/colreg_classifier_best.pth"

# Log Configuration
LOG_DIR = "predictor_logs"
LOG_FILE = "prediction_log.txt" 

# Confidence Threshold (Below this % = Unknown)
CONFIDENCE_THRESHOLD = 65.0 

# Class mapping (12 Classes)
COLREG_CLASSES = [
    "Alter Starboard",          # 0
    "Alter Port",               # 1
    "Astern Propulsion",        # 2
    "Danger Signal (Doubt)",    # 3
    "Overtake Starboard",       # 4
    "Round Starboard",          # 5
    "Round Port",               # 6
    "Blind Bend / Making Way",  # 7 
    "Overtake Port",            # 8
    "Agreement to Overtake",    # 9
    "Not Under Command",        # 10
    "Noise Only"                # 11
]

# --- 2. MODEL ARCHITECTURE ---
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
            nn.Dropout(0.3)
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

# --- 3. UTILS ---
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
        return tensor
    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        return None

def log_prediction(audio_file, predicted_class, confidence):
    if not os.path.exists(LOG_DIR):
        try: os.makedirs(LOG_DIR)
        except OSError: pass

    log_path = os.path.join(LOG_DIR, LOG_FILE)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] File: {os.path.basename(audio_file)} | Prediction: {predicted_class} | Confidence: {confidence:.2f}%\n"
    
    try:
        with open(log_path, "a") as f: f.write(log_entry)
    except Exception: pass

def predict_single_file(model, device, file_path):
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (Not found)")
        return

    input_tensor = preprocess_audio(file_path)
    if input_tensor is None: return

    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        score, predicted_idx = torch.max(probabilities, 1)
        
        confidence = score.item() * 100
        
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "UNKNOWN / UNRECOGNIZED SIGNAL"
            status = "IGNORED (Low Confidence)"
        else:
            predicted_class = COLREG_CLASSES[predicted_idx.item()]
            status = "ACCEPTED"

    log_prediction(file_path, predicted_class, confidence)

    print("-" * 50)
    print(f"File:       {os.path.basename(file_path)}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Status:     {status}")

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to a single .wav file")
    parser.add_argument("--dir", type=str, help="Path to a folder containing .wav files")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to .pth model")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    try:
        model = ColregClassifier(num_classes=len(COLREG_CLASSES), input_height=N_MELS)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        model.eval() 
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Check if user provided a directory or a single file
    if args.dir:
        if not os.path.exists(args.dir):
            print(f"ERROR: Directory {args.dir} not found.")
            sys.exit(1)
            
        files = glob.glob(os.path.join(args.dir, "*.wav"))
        print(f"Found {len(files)} wav files in {args.dir}")
        
        for f in files:
            predict_single_file(model, device, f)
            
    elif args.file:
        predict_single_file(model, device, args.file)
    
    else:
        # Default behavior: Look in the standard input folder if no args provided
        default_input_dir = "/app/input" # Docker path
        # Check if running locally or in docker to decide default
        if not os.path.exists(default_input_dir) and os.path.exists("input_to_predict_COLREG"):
             default_input_dir = "input_to_predict_COLREG"
             
        if os.path.exists(default_input_dir):
            print(f"No arguments provided. Scanning default folder: {default_input_dir}")
            files = glob.glob(os.path.join(default_input_dir, "*.wav"))
            for f in files:
                predict_single_file(model, device, f)
        else:
            print("Usage: python predictor.py --file <path> OR --dir <path>")

    print("\n" + "="*50)
    print(f"Done. Logs saved to {os.path.join(LOG_DIR, LOG_FILE)}")
    print("="*50 + "\n")