import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import random

# --- CONFIGURATION ---
N_MELS = 128                   
N_CLASSES = 13 

# --- DATASET DEFINITION ---
class ColregDataset(Dataset):
    def __init__(self, labels_list):
        self.labels = labels_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_data = self.labels[idx]
        feature_path = label_data["filepath"]
        
        try:
            mel_spectrogram_db = np.load(feature_path)
        except FileNotFoundError:
            print(f"Error: Feature file not found at {feature_path}")
            raise
        
        # Convert to Tensor (1 Channel, Height, Width)
        features = torch.tensor(mel_spectrogram_db).float().unsqueeze(0)
        label = torch.tensor(label_data["class_id"], dtype=torch.long)
        
        return features, label

# --- MODEL ARCHITECTURE (CNN + GRU) ---
class ColregClassifier(nn.Module):
    def __init__(self, num_classes, input_height):
        super(ColregClassifier, self).__init__()
        
        # 1. CNN Block (Extracts Frequency Features)
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
        
        # Calculate flatten size for GRU input
        self.output_channels = 64
        self.reduced_height = input_height // 2 // 2 // 4 
        gru_input_size = self.output_channels * self.reduced_height 

        # 2. GRU Block (Extracts Time Sequence Features)
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 3. Classifier Block
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64), # Bidirectional = 2 * hidden_size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # CNN Forward
        cnn_out = self.cnn(x)
        
        # Prepare for GRU: (Batch, Channels, Freq, Time) -> (Batch, Time, Features)
        B, C, H, T = cnn_out.size()
        gru_input = cnn_out.permute(0, 3, 1, 2).contiguous().view(B, T, C * H)
        
        # GRU Forward
        gru_out, _ = self.gru(gru_input)
        
        # Take last hidden state (Forward + Backward)
        last_forward = gru_out[:, -1, :128]
        last_backward = gru_out[:, 0, 128:]
        gru_output_combined = torch.cat((last_forward, last_backward), dim=1)
        
        return self.classifier(gru_output_combined)

# --- TRAINING ENGINE ---
def run_training_phase(phase_name, labels_file, model, num_epochs, lr, save_path, load_path=None):
    """
    Runs a complete training session (Phase).
    Can optionally load weights from a previous phase (Fine-Tuning).
    """
    print(f"\n" + "="*50)
    print(f"🚀 STARTING PHASE: {phase_name.upper()}")
    print(f"="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 1. Load Previous Weights (if requested)
    if load_path:
        if os.path.exists(load_path):
            print(f"🔄 Loading weights from previous phase: {load_path}")
            try:
                model.load_state_dict(torch.load(load_path, map_location=device))
                print("✅ Weights loaded successfully. Fine-tuning...")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load weights ({e}). Starting fresh.")
        else:
            print(f"⚠️ Previous model {load_path} not found. Starting fresh.")
    
    # 2. Load Data Labels
    if not os.path.exists(labels_file):
        print(f"❌ Skipping Phase '{phase_name}': Metadata file '{labels_file}' not found.")
        return

    print(f"📂 Loading data from: {labels_file}")
    all_labels = np.load(labels_file, allow_pickle=True)
    random.shuffle(all_labels)
    
    # Split 80/20
    split_idx = int(0.8 * len(all_labels))
    train_labels = all_labels[:split_idx]
    val_labels = all_labels[split_idx:]
    
    print(f"   Training Samples: {len(train_labels)} | Validation Samples: {len(val_labels)}")
    
    train_loader = DataLoader(ColregDataset(train_labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(ColregDataset(val_labels), batch_size=32, shuffle=False)
    
    # 3. Setup Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_accuracy = 0.0
    
    # 4. Training Loop
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
        
        # Validation
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
        scheduler.step(val_accuracy)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{phase_name}] Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_accuracy:.2f}% (LR: {current_lr:.6f})")
        
        # Save Best
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
    
    print(f"✅ Phase '{phase_name}' Complete. Best Accuracy: {best_accuracy:.2f}%")
    print(f"💾 Model saved to: {save_path}")

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    # Define Paths
    DATA_DIR = "dataset/" 
    LABELS_CLEAN = os.path.join(DATA_DIR, "labels_clean.npy")
    LABELS_NOISY = os.path.join(DATA_DIR, "labels_noisy.npy")
    
    # Intermediate model (after clean training)
    MODEL_CLEAN = "colreg_model_clean.pth"
    # Final model (after noisy training)
    MODEL_FINAL = "colreg_classifier_best.pth"
    
    # Initialize Architecture
    model = ColregClassifier(num_classes=N_CLASSES, input_height=N_MELS)

    # --- PHASE 1: CLEAN TRAINING (Curriculum Start) ---
    # Higher Learning Rate (0.001), More Epochs (30)
    run_training_phase(
        phase_name="Clean",
        labels_file=LABELS_CLEAN,
        model=model,
        num_epochs=35, 
        lr=0.001,
        save_path=MODEL_CLEAN,
        load_path=None # Start from scratch
    )

    # --- PHASE 2: NOISY FINE-TUNING (Curriculum Advanced) ---
    # Lower Learning Rate (0.0001) to preserve knowledge, fewer epochs (20)
    # Only runs if labels_noisy.npy exists
    run_training_phase(
        phase_name="Noisy",
        labels_file=LABELS_NOISY,
        model=model,
        num_epochs=20, 
        lr=0.0001, 
        save_path=MODEL_FINAL,
        load_path=MODEL_CLEAN # Load weights from Phase 1
    )
    
    # Fallback: If Phase 2 didn't run (no noisy data), ensure we still have a "best" model file
    if not os.path.exists(MODEL_FINAL) and os.path.exists(MODEL_CLEAN):
        print("\n⚠️ No noisy training performed. Using Clean model as Final model.")
        import shutil
        shutil.copy(MODEL_CLEAN, MODEL_FINAL)
    
    print(f"\n🎉 Full Curriculum Training Workflow Complete!")