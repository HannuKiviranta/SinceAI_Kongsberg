import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import random
import shutil

# ============================================================
# CONFIGURATION - OPTIMIZED
# ============================================================

N_MELS = 128                   
N_CLASSES = 12  # Must match the classes defined in preprocess.py

# ============================================================
# TRAINING HYPERPARAMETERS - ADJUST THESE
# ============================================================

# Phase 1: Clean Training
CLEAN_EPOCHS = 2          # 
CLEAN_LR = 0.001           # Learning rate for clean phase
CLEAN_BATCH_SIZE = 32

# Phase 2: Noisy Fine-tuning
NOISY_EPOCHS = 2          # 
NOISY_LR = 0.0001          # Lower LR to not break clean learning
NOISY_BATCH_SIZE = 32

# Phase 3: Seagull Fine-tuning (optional)
SEAGULL_EPOCHS = 2        # Smaller, just fine-tune a bit
SEAGULL_LR = 0.00005      # Even lower LR
SEAGULL_BATCH_SIZE = 32


# Regularization
DROPOUT_CNN = 0.4          # Dropout after CNN layers
DROPOUT_FC = 0.5           # Dropout before final classifier

# Learning Rate Scheduler
LR_PATIENCE = 5            # Epochs to wait before reducing LR
LR_FACTOR = 0.5            # Factor to reduce LR by

# Early Stopping (optional - prevents overfitting)
EARLY_STOP_PATIENCE = 10   # Stop if no improvement for N epochs

# ============================================================
# DATASET DEFINITION
# ============================================================

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
        
        features = torch.tensor(mel_spectrogram_db).float().unsqueeze(0)
        label = torch.tensor(label_data["class_id"], dtype=torch.long)
        
        return features, label

# ============================================================
# MODEL ARCHITECTURE (CNN + GRU)
# ============================================================

class ColregClassifier(nn.Module):
    def __init__(self, num_classes, input_height, dropout_cnn=DROPOUT_CNN, dropout_fc=DROPOUT_FC):
        super(ColregClassifier, self).__init__()
        
        # 1. CNN Block (Extracts Frequency Patterns)
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
            nn.Dropout(dropout_cnn)
        )
        
        # Calculate input size for the GRU
        self.output_channels = 64
        self.reduced_height = input_height // 2 // 2 // 4 
        gru_input_size = self.output_channels * self.reduced_height 

        # 2. GRU Block (Extracts Time Patterns)
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 3. Classifier Block
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
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

# ============================================================
# TRAINING ENGINE
# ============================================================

def run_training_phase(phase_name, labels_file, model, num_epochs, lr, batch_size, 
                       save_path, load_path=None, early_stop_patience=EARLY_STOP_PATIENCE):
    """
    Runs a training loop with early stopping and learning rate scheduling.
    """
    print(f"\n" + "="*60)
    print(f"🚀 STARTING PHASE: {phase_name.upper()}")
    print(f"="*60)
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {lr}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Early Stop Patience: {early_stop_patience}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n--- Starting Training on {device} ---")
    
    # 1. Load Previous Weights (if applicable)
    if load_path:
        if os.path.exists(load_path):
            print(f"🔄 Loading weights from previous phase: {load_path}")
            try:
                model.load_state_dict(torch.load(load_path, map_location=device))
                print("✅ Weights loaded. Starting Fine-Tuning.")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load weights ({e}). Starting fresh.")
        else:
            print(f"⚠️ Previous model {load_path} not found. Starting fresh.")
    
    # 2. Load Data Labels
    if not os.path.exists(labels_file):
        print(f"⚠️ Skipping Phase '{phase_name}': Metadata file '{labels_file}' not found.")
        return False

    print(f"📂 Loading dataset: {labels_file}")
    all_labels = np.load(labels_file, allow_pickle=True).tolist()
    random.shuffle(all_labels)
    
    # Split 80/20
    split_idx = int(0.8 * len(all_labels))
    train_labels = all_labels[:split_idx]
    val_labels = all_labels[split_idx:]
    
    print(f"   Training Samples: {len(train_labels)} | Validation Samples: {len(val_labels)}")
    
    train_loader = DataLoader(ColregDataset(train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ColregDataset(val_labels), batch_size=batch_size, shuffle=False)
    
    # 3. Setup Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True)
    
    best_accuracy = 0.0
    epochs_without_improvement = 0
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_loss = val_loss / len(val_loader.dataset)
        
        scheduler.step(val_accuracy)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Progress output
        print(f"[{phase_name}] Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Save Best Model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            epochs_without_improvement = 0
            print(f"   💾 New best! Saved to {save_path}")
        else:
            epochs_without_improvement += 1
        
        # Early Stopping
        if early_stop_patience and epochs_without_improvement >= early_stop_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            print(f"   No improvement for {early_stop_patience} epochs")
            break
    
    print(f"\n✅ Phase '{phase_name}' Complete!")
    print(f"   Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"   Model saved to: {save_path}")
    return True

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    
    print("="*60)
    print("🚢 COLREG CLASSIFIER - CURRICULUM TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Clean Phase: {CLEAN_EPOCHS} epochs @ LR={CLEAN_LR}")
    print(f"  Noisy Phase: {NOISY_EPOCHS} epochs @ LR={NOISY_LR}")
    print(f"  Seagull Phase: {SEAGULL_EPOCHS} epochs @ LR={SEAGULL_LR}")
    print(f"  CNN Dropout: {DROPOUT_CNN}")
    print(f"  FC Dropout: {DROPOUT_FC}")
    print(f"  Early Stop Patience: {EARLY_STOP_PATIENCE}")
    
    # Define Paths
    DATA_DIR = "dataset/" 
    LABELS_CLEAN = os.path.join(DATA_DIR, "labels_clean.npy")
    LABELS_NOISY = os.path.join(DATA_DIR, "labels_noisy.npy")
    LABELS_SEAGULLS = os.path.join(DATA_DIR, "labels_seagulls.npy")
    
    MODEL_CLEAN = "colreg_model_clean.pth"
    MODEL_FINAL = "colreg_classifier_best.pth"
    
    # Initialize Model
    model = ColregClassifier(num_classes=N_CLASSES, input_height=N_MELS)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # --- PHASE 1: CLEAN TRAINING ---
    success_clean = run_training_phase(
        phase_name="Clean",
        labels_file=LABELS_CLEAN,
        model=model,
        num_epochs=CLEAN_EPOCHS,
        lr=CLEAN_LR,
        batch_size=CLEAN_BATCH_SIZE,
        save_path=MODEL_CLEAN,
        load_path=None
    )

    # --- PHASE 2: NOISY FINE-TUNING ---
    success_noisy = run_training_phase(
        phase_name="Noisy",
        labels_file=LABELS_NOISY,
        model=model,
        num_epochs=NOISY_EPOCHS,
        lr=NOISY_LR,
        batch_size=NOISY_BATCH_SIZE,
        save_path=MODEL_FINAL,
        load_path=MODEL_CLEAN
    )
    
      # --- PHASE 3: SEAGULL FINE-TUNING (optional) ---
    success_seagulls = run_training_phase(
        phase_name="Seagulls",
        labels_file=LABELS_SEAGULLS,
        model=model,
        num_epochs=SEAGULL_EPOCHS,
        lr=SEAGULL_LR,
        batch_size=SEAGULL_BATCH_SIZE,
        save_path=MODEL_FINAL,      # keep saving to the same final model
        load_path=MODEL_FINAL       # start from the already fine-tuned noisy model
    )

    # Fallback Logic
    if not success_noisy:
        if success_clean:
            print("\n⚠️ No noisy training performed. Using Clean model as Final.")
            if os.path.exists(MODEL_CLEAN):
                shutil.copy(MODEL_CLEAN, MODEL_FINAL)
        else:
            print("\n❌ Critical Error: Both training phases failed.")
            exit(1)
    
    print(f"\n{'='*60}")
    print(f"🎉 FULL TRAINING WORKFLOW COMPLETE!")
    print(f"{'='*60}")