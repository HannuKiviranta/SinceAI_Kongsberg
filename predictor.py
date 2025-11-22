import torch
import torch.nn as nn
import numpy as np
import librosa
import argparse
import os
import sys

# --- 1. CONFIGURATION ---
# These MUST match the training settings exactly
N_MELS = 128
MAX_WIDTH = 860  # Approx 20 seconds of audio at sr=22050, hop=512
MODEL_PATH = "colreg_classifier/colreg_classifier_best.pth"

# Class mapping (Must be in the exact same order as training)
COLREG_CLASSES = [
    "Alter Starboard",          # 0
    "Alter Port",               # 1
    "Astern Propulsion",        # 2
    "Danger Signal (Doubt)",    # 3
    "Overtake Starboard",       # 4
    "Overtake Port",            # 5
    "Agreement to Overtake",    # 6
    "Blind Bend / Narrow Channel", # 7
    "Noise Only",               # 8
    "Random Short Blasts"       # 9
]

# --- 2. MODEL ARCHITECTURE ---
# This class must be identical to the one used in training
class ColregClassifier(nn.Module):
    def __init__(self, num_classes, input_height):
        super(ColregClassifier, self).__init__()
        
        # 1. Convolutional Block
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
        
        # Calculate hidden sizes
        self.output_channels = 64
        self.reduced_height = input_height // 2 // 2 // 4 
        gru_input_size = self.output_channels * self.reduced_height 

        # 2. Recurrent Block
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        B, C, H, T = cnn_out.size()
        
        # Prepare for GRU
        gru_input = cnn_out.permute(0, 3, 1, 2).contiguous().view(B, T, C * H)
        gru_out, _ = self.gru(gru_input)
        
        # Get last state
        last_forward = gru_out[:, -1, :128]
        last_backward = gru_out[:, 0, 128:]
        gru_output_combined = torch.cat((last_forward, last_backward), dim=1)
        
        return self.classifier(gru_output_combined)

# --- 3. PREPROCESSING UTILS ---
def preprocess_audio(file_path):
    """Loads wav, converts to mel spectrogram, pads to correct size, returns Tensor."""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=22050)
        
        # Convert to Mel Spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=8000)
        mels_db = librosa.power_to_db(mels, ref=np.max)
        
        # Pad or Crop to MAX_WIDTH
        current_width = mels_db.shape[1]
        if current_width < MAX_WIDTH:
            # Pad with silence (-80dB)
            padding = MAX_WIDTH - current_width
            mels_db = np.pad(mels_db, ((0, 0), (0, padding)), mode='constant', constant_values=-80)
        else:
            # Crop
            mels_db = mels_db[:, :MAX_WIDTH]
            
        # Create Tensor: (Batch=1, Channel=1, Height, Width)
        tensor = torch.tensor(mels_db).float().unsqueeze(0).unsqueeze(0)
        return tensor
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description="COLREG Sound Signal Classifier")
    parser.add_argument("--file", type=str, required=True, help="Path to the .wav file to test")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to .pth model file")
    args = parser.parse_args()

    # 1. Check Model Path
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        print("Please train the model or adjust the path.")
        sys.exit(1)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model} on {device}...")
    
    try:
        model = ColregClassifier(num_classes=len(COLREG_CLASSES), input_height=N_MELS)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        model.eval() # Important: Set to evaluation mode (disables dropout)
    except Exception as e:
        print(f"Failed to load model architecture: {e}")
        sys.exit(1)

    # 3. Preprocess Audio
    print(f"Processing audio file: {args.file}")
    if not os.path.exists(args.file):
        print("ERROR: Audio file not found.")
        sys.exit(1)

    input_tensor = preprocess_audio(args.file)
    
    if input_tensor is None:
        sys.exit(1)

    # 4. Inference
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad(): # No need to calculate gradients for inference
        outputs = model(input_tensor)
        
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the highest probability class
        score, predicted_idx = torch.max(probabilities, 1)
        predicted_class = COLREG_CLASSES[predicted_idx.item()]
        confidence = score.item() * 100

    # 5. Output Results
    print("\n" + "="*40)
    print(f"PREDICTION RESULT")
    print("="*40)
    print(f"Detected Signal:  {predicted_class.upper()}")
    print(f"Confidence:       {confidence:.2f}%")
    print("-" * 40)
    
    # Optional: Warning for low confidence
    if confidence < 70.0:
        print("WARNING: Confidence is low. Result may be unreliable.")
        print("Possible causes: Background noise too high, or unknown signal.")
    
    print("="*40 + "\n")