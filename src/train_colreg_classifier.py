import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random

# --- 1. CONFIGURATION AND COLREG DEFINITIONS ---

# ML Feature Constants
N_MELS = 128                   # Number of Mel bands (Feature height) - MUST MATCH YOUR SAVED FEATURES
# HOP_LENGTH = 512             # (Not needed for training, but kept for reference)

# COLREG signal patterns (Used only for mapping class IDs to names)
COLREG_CLASSES = {
    "Alter Starboard":          ["S"],
    "Alter Port":               ["S", "S"],
    "Astern Propulsion":        ["S", "S", "S"],
    "Danger Signal (Doubt)":    ["S", "S", "S", "S", "S"],
    "Overtake Starboard":       ["L", "L", "S"],
    "Overtake Port":            ["L", "L", "S", "S"],
    "Agreement to Overtake":    ["L", "S", "L", "S"],
    "Blind Bend / Narrow Channel": ["L"],
    # Negative examples
    "Noise Only":               ["SILENCE"],
    "Random Short Blasts":      ["S"] * 8, 
}
CLASS_NAMES = list(COLREG_CLASSES.keys())
N_CLASSES = len(CLASS_NAMES)
print(f"Defined {N_CLASSES} classification classes.")

# --- 2. FEATURE GENERATION UTILITIES (Removed as requested) ---
# All previous functions for synthesizing audio and computing features (generate_sine_wave, 
# create_horn_blast, mix_signal_with_noise, generate_colreg_dataset_features) have been removed.
# The script now relies solely on your existing data files.

# --- 3. PYTORCH DATASET AND MODEL DEFINITION ---

class ColregDataset(Dataset):
    """PyTorch Dataset for loading pre-computed 2D Mel Spectrogram features from .npy files."""
    def __init__(self, labels_list):
        self.labels = labels_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_data = self.labels[idx]
        feature_path = label_data["filepath"]
        
        # 1. Load Pre-computed 2D Feature Array (the "2D image")
        # Feature array shape: (N_MELS, Time_Frames)
        # Note: This loads the dB-scaled Mel Spectrogram array.
        try:
            mel_spectrogram_db = np.load(feature_path)
        except FileNotFoundError:
            print(f"Error: Feature file not found at {feature_path}. Check your labels.npy and data structure.")
            raise
        
        # 2. Convert to PyTorch Tensor and add channel dimension
        # Input shape required: (C, H, W) -> (1, N_MELS, Time_Frames)
        features = torch.tensor(mel_spectrogram_db).float().unsqueeze(0)
        
        # 3. Get Class Label
        label = torch.tensor(label_data["class_id"], dtype=torch.long)
        
        return features, label

class ColregClassifier(nn.Module):
    """
    CNN-GRU Model Architecture for Sequence Classification:
    1. CNN (Spectral Feature Extraction): Learns horn timbre/pitch from frequency bands.
    2. GRU (Temporal Pattern Recognition): Learns the sequence (S, S, S, L-S-L-S) over time.
    """
    def __init__(self, num_classes, input_height):
        super(ColregClassifier, self).__init__()
        
        # 1. Convolutional Block (Time distributed feature extraction)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)), # Drastically reduce frequency dim
            
            nn.Dropout(0.3)
        )
        
        # Calculate the size of the feature vector after CNN reduction
        self.output_channels = 64
        self.reduced_height = input_height // 2 // 2 // 4 # N_MELS / 16 (128/16 = 8)
        gru_input_size = self.output_channels * self.reduced_height 

        # 2. Recurrent Block (Sequence Modeling)
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 3. Output Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64), # 128 hidden * 2 directions (bidirectional)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, N_MELS, Time_Frames)
        cnn_out = self.cnn(x)
        
        # Prepare for GRU: Flatten (Channels * Height) dimension and swap Time dimension
        # cnn_out shape: (B, C, H, T) -> (B, T, C*H)
        B, C, H, T = cnn_out.size()
        gru_input = cnn_out.permute(0, 3, 1, 2).contiguous().view(B, T, C * H)
        
        # Pass through GRU
        gru_out, _ = self.gru(gru_input)
        
        # Use the output from the last time step for classification
        last_forward = gru_out[:, -1, :128]
        last_backward = gru_out[:, 0, 128:]
        gru_output_combined = torch.cat((last_forward, last_backward), dim=1)
        
        # Final classification layer
        logits = self.classifier(gru_output_combined)
        return logits


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, model_path="colreg_model.pth"):
    """Main function to train and evaluate the PyTorch model."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n--- Starting Training on {device} ---")
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")
        
        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} with improved accuracy: {best_accuracy:.2f}%")

    print("\n--- Training Complete ---")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Final model saved to {model_path}")


if __name__ == '__main__':
    
    # 1. Setup paths
    DATA_DIR = "dataset/" # Directory containing labels.npy and the 'features' folder
    MODEL_FILE = "colreg_classifier_best.pth"
    
    # 2. Data Loading Check
    labels_file = os.path.join(DATA_DIR, "labels.npy")
    if not os.path.exists(labels_file):
        print("ERROR: Data metadata file not found!")
        print(f"Please ensure your data is located in '{DATA_DIR}' and that '{labels_file}' exists.")
        print("The 'labels.npy' file must contain the list of dictionaries linking to your Mel Spectrogram .npy feature files.")
        exit(1)
    
    print("Loading existing 2D feature metadata from labels.npy.")
    all_labels = np.load(labels_file, allow_pickle=True)

    # 3. Split Data
    random.shuffle(all_labels)
    split_idx = int(0.8 * len(all_labels))
    train_labels = all_labels[:split_idx]
    val_labels = all_labels[split_idx:]

    # 4. Create DataLoaders
    train_dataset = ColregDataset(train_labels)
    val_dataset = ColregDataset(val_labels)
    
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Total samples: {len(all_labels)}. Training on {len(train_labels)}, Validating on {len(val_labels)}.")

    # 5. Initialize Model and Start Training
    model = ColregClassifier(num_classes=N_CLASSES, input_height=N_MELS)
    train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=100, 
        learning_rate=0.001,
        model_path=MODEL_FILE
    )

    print("\n--- Next Steps ---")
    print(f"Use the saved model file ({MODEL_FILE}) in a separate inference script (predict.py) inside your Docker container for deployment.")