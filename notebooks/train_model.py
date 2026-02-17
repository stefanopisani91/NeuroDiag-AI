
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import json

# Configuration
X_TRAIN_FILE = "X_train_global.npy"
X_TEST_FILE = "X_test_global.npy"
Y_TRAIN_FILE = "y_train_global.npy"
Y_TEST_FILE = "y_test_global.npy"
TARGET_NAMES_FILE = "target_names.json"

MODEL_FILE = "psycho_global_model.pth"
HISTORY_PLOT = "global_training_history.png"

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
PATIENCE = 5

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class PsychoGlobalModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(PsychoGlobalModel, self).__init__()
        # Input Layer -> Dense 128
        self.layer1 = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Dense 128 -> Dense 64
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Dense 64 -> Dense 32
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Output Layer
        self.output = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs > threshold).float()
    
    # Accuracy (Exact match for multilabel? Or subset? Standard is often subset or hamming)
    # Using simple element-wise accuracy for reporting as requested (common proxy)
    accuracy = (y_pred == y_true).float().mean().item()
    
    # Precision & Recall (Global/Micro)
    tp = (y_pred * y_true).sum().item()
    fp = (y_pred * (1 - y_true)).sum().item()
    fn = ((1 - y_pred) * y_true).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return accuracy, precision, recall

def train_model():
    print("Loading data...")
    try:
        X_train = np.load(X_TRAIN_FILE)
        X_test = np.load(X_TEST_FILE)
        y_train = np.load(Y_TRAIN_FILE)
        y_test = np.load(Y_TEST_FILE)
        
        with open(TARGET_NAMES_FILE, 'r') as f:
            target_names = json.load(f)
            
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Convert to Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Validation Split (Manual 20% from train)
    val_size = int(0.2 * len(X_train_tensor))
    train_size = len(X_train_tensor) - val_size
    
    full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]
    
    print(f"Input Features: {input_shape}")
    print(f"Target Classes: {num_classes}")
    print(f"Device: {device}")
    
    # Initialize Model
    model = PsychoGlobalModel(input_shape, num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Deep Training...")
    
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            acc, _, _ = calculate_metrics(targets, outputs)
            running_acc += acc * inputs.size(0)
            
        epoch_loss = running_loss / train_size
        epoch_acc = running_acc / train_size
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                acc, _, _ = calculate_metrics(targets, outputs)
                val_running_acc += acc * inputs.size(0)
                
        val_loss = val_running_loss / val_size
        val_acc = val_running_acc / val_size
        
        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(epoch_acc)
        history['val_accuracy'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Save Model
    print(f"Saving model to {MODEL_FILE}...")
    torch.save(model.state_dict(), MODEL_FILE)
    
    # Plot History
    print(f"Saving history plot to {HISTORY_PLOT}...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Val')
    plt.title('Global Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Global Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT)
    
    # Final Evaluation
    print("\nEvaluating on Test Set...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)
        test_loss = loss.item()
        test_acc, test_prec, test_recall = calculate_metrics(y_test_tensor, outputs)
    
    print("-" * 30)
    print("FINAL GLOBAL TEST RESULTS")
    print("-" * 30)
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    train_model()
