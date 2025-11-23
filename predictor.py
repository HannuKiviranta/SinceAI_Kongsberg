import torch
import torch.nn as nn
import numpy as np
import librosa
import argparse
import os
import sys
import datetime

# --- 1. CONFIGURATION ---
N_MELS = 128
SR = 22050
HOP_LENGTH = 512
CLIP_DURATION_SEC = 20 

# Calculate width dynamically
MAX_WIDTH = int(np.ceil(CLIP_DURATION_SEC * (SR / HOP_LENGTH)))

MODEL_PATH = "colreg_classifier/colreg_classifier_best.pth"

# Log Configuration
LOG_DIR = "predictor_logs"
LOG_FILE = "prediction_log.txt" 

# Class mapping (13 Classes)
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
    "Noise Only",               # 11
    "Random Short Blasts"       # 12
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
        print(f"Error processing audio file: {e}")
        return None

def log_prediction(audio_file, predicted_class, confidence):
    """Appends the prediction result to a log file in a specific folder."""
    
    # Ensure log directory exists
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            print(f"Warning: Could not create log directory {LOG_DIR}: {e}")
            return

    # Full path to the log file
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] File: {os.path.basename(audio_file)} | Prediction: {predicted_class} | Confidence: {confidence:.2f}%\n"
    
    try:
        with open(log_path, "a") as f:
            f.write(log_entry)
        print(f"   [Log saved to {log_path}]")
    except Exception as e:
        print(f"Warning: Could not save log: {e}")

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COLREG Sound Signal Classifier")
    parser.add_argument("--file", type=str, required=True, help="Path to the .wav file to test")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to .pth model file")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = ColregClassifier(num_classes=len(COLREG_CLASSES), input_height=N_MELS)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        model.eval() 
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    if not os.path.exists(args.file):
        print("ERROR: Audio file not found.")
        sys.exit(1)

    input_tensor = preprocess_audio(args.file)
    
    if input_tensor is None:
        sys.exit(1)

    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        score, predicted_idx = torch.max(probabilities, 1)
        predicted_class = COLREG_CLASSES[predicted_idx.item()]
        confidence = score.item() * 100

    # Log result to Text File in the new folder
    log_prediction(args.file, predicted_class, confidence)

    print("\n" + "="*40)
    print(f"PREDICTION RESULT")
    print("="*40)
    print(f"Detected Signal:  {predicted_class.upper()}")
    print(f"Confidence:       {confidence:.2f}%")
    print("-" * 40)
    
    if confidence < 70.0:
        print("WARNING: Confidence is low. Result may be unreliable.")
    
    print("="*40 + "\n")