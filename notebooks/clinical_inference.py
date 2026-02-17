
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import json

# Files
MODEL_FILE = "psycho_global_model.pth"
SCALER_FILE = "scaler_global.pkl"
FEATURE_NAMES_FILE = "feature_names_global.json"
TARGET_NAMES_FILE = "target_names.json"

# --- 1. Re-define Model Class (Must match training structure) ---
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

# --- 2. Load Artifacts ---
print("Initializing Global Diagnostic System...")
try:
    scaler = joblib.load(SCALER_FILE)
    with open(FEATURE_NAMES_FILE, 'r') as f:
        feature_names = json.load(f)
    with open(TARGET_NAMES_FILE, 'r') as f:
        target_names = json.load(f)
    print("Artifacts loaded.")
except FileNotFoundError as e:
    print(f"Critical Error: Missing file {e}")
    exit(1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_dim = len(feature_names)
num_classes = len(target_names)

model = PsychoGlobalModel(input_dim, num_classes).to(device)

try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print(f"Model loaded (Input: {input_dim}, Output: {num_classes}).")
except FileNotFoundError:
    print(f"Error: {MODEL_FILE} not found.")
    exit(1)

# List of numerical columns for scaling (Must match training script)
NUMERICAL_COLS = [
    "eta", "scolarita_anni", "HAM_D", "GAD_7", 
    "PANSS_total", "UPDRS_III", "MMSE"
]

# --- 3. Prediction Function ---
def predict_patient(patient_dict):
    """
    Diagnoses a patient from a sparse dictionary.
    """
    # 1. Create base dictionary with defaults
    # Most features are binary 0, except some defaults if needed (like gender)
    full_data = {feat: 0 for feat in feature_names}
    
    # 2. Update with input data
    for k, v in patient_dict.items():
        if k in full_data:
            full_data[k] = v
        else:
            print(f"Warning: Unknown feature '{k}' ignored.")
            
    # 3. Handle specific formatting (Sesso string -> int)
    if isinstance(full_data.get("sesso"), str):
         full_data["sesso"] = 1 if full_data["sesso"].upper() == "F" else 0
         
    # 4. Scaling
    # Create DF for scaler to work correctly with names
    # Construct a single-row DF with correct columns (numerical only)
    # Scaler was fitted on ['eta', 'scolarita_anni', ...] (subset)
    
    numerical_data = {col: [full_data[col]] for col in NUMERICAL_COLS}
    numerical_df = pd.DataFrame(numerical_data)
    
    # Transform
    scaled_values = scaler.transform(numerical_df)[0]
    
    # Update full_data with scaled values
    for i, col in enumerate(NUMERICAL_COLS):
        full_data[col] = scaled_values[i]
        
    # 5. Create Input Tensor (Strict Order)
    input_vector = [full_data[feat] for feat in feature_names]
    input_tensor = torch.tensor([input_vector], dtype=torch.float32).to(device)
    
    # 6. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = logits.cpu().numpy()[0]
        
    # 7. Format Output
    results = {name.replace("TARGET_", ""): round(float(p) * 100, 2) 
               for name, p in zip(target_names, probs)}
    
    # Sort
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_results

# --- 4. Test Case: "Dr. House Scenario" ---
if __name__ == "__main__":
    # Case: Young woman, visual issues + sensory + fatigue -> likely MS
    # Also Depressed (Reactive?)
    test_case = {
        'eta': 32, 'sesso': 1, # F
        'neurite_ottica': 1,
        'parestesie_formicolii': 1,
        'fatica_astenia': 1,
        'umore_depresso': 1,
        'insonnia': 1,
        'HAM_D': 14,
        'MMSE': 30,
        # Default others implicitly 0
        'scolarita_anni': 16 # Assume some education
    }
    
    print("\n" + "="*50)
    print("CONSULTO CLINICO SPECIALE: CASO 'DR. HOUSE'")
    print("="*50)
    print("Sintomatologia:")
    for k, v in test_case.items():
        print(f"  - {k}: {v}")
        
    print("-" * 50)
    print("Analisi Differenziale AI:")
    
    diagnosis = predict_patient(test_case)
    
    for condition, prob in diagnosis.items():
        if prob > 1.0: # Show only relevant > 1%
            bar = "█" * int(prob // 2)
            print(f"{condition:<15} : {prob:>6.2f}% {bar}")
            
    print("-" * 50)
    print("Interpretazione:")
    top_disease = list(diagnosis.keys())[0]
    top_prob = list(diagnosis.values())[0]
    
    if top_prob > 50:
        print(f"Diagnosi Primaria: {top_disease}")
        # Comorbidities?
        others = [k for k,v in diagnosis.items() if v > 40 and k != top_disease]
        if others:
            print(f"Comorbidità Rilevate: {', '.join(others)}")
    else:
        print("Quadro clinico complesso/incerto.")
