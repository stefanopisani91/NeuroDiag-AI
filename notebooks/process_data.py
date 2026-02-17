
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

# Configuration
INPUT_FILE = "patients_global.csv"
OUTPUT_SCALER = "scaler_global.pkl"
TARGET_NAMES_FILE = "target_names.json"

# Output files
X_TRAIN_FILE = "X_train_global.npy"
X_TEST_FILE = "X_test_global.npy"
Y_TRAIN_FILE = "y_train_global.npy"
Y_TEST_FILE = "y_test_global.npy"

SEED = 42

def process_data_global():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # ---------------------------------------------------------
    # 1. Identify Columns
    # ---------------------------------------------------------
    
    # Targets
    target_cols = [c for c in df.columns if c.startswith("TARGET_")]
    print(f"Found {len(target_cols)} targets: {target_cols}")
    
    # Save target names mapping
    with open(TARGET_NAMES_FILE, "w") as f:
        json.dump(target_cols, f)
    print(f"Saved target names to {TARGET_NAMES_FILE}")
    
    # Numerical features to scale
    numerical_cols = [
        "eta", 
        "scolarita_anni", 
        "HAM_D", 
        "GAD_7", 
        "PANSS_total", 
        "UPDRS_III", 
        "MMSE"
    ]
    
    # Categorical
    if "sesso" in df.columns:
        print("Encoding 'sesso'...")
        df["sesso"] = df["sesso"].map({"M": 0, "F": 1})
        
    # Drop non-feature columns
    drop_cols = ["id", "active_profiles", "familiarita_disturbi"] + target_cols
    # "familiarita_disturbi" is technically a feature (binary 0/1), usually kept. 
    # Logic: Keep binary symptoms AND binary history.
    # The prompt explicitly listed "sesso" and numericals for scaling.
    # Binary symptoms were "leave as is".
    # "familiarita_disturbi" is binary (0/1), so should be kept as feature.
    # Correcting drop list to KEEP "familiarita_disturbi"
    drop_cols = ["id", "active_profiles"] + target_cols
    
    # Create Feature Matrix X
    X = df.drop(columns=drop_cols).copy()
    
    # Ensure floats
    for col in numerical_cols:
        X[col] = X[col].astype(float)
        
    feature_names = list(X.columns)
    print(f"Total features: {len(feature_names)}")

    # ---------------------------------------------------------
    # 2. Scaling (Fit on Train)
    # ---------------------------------------------------------
    
    # Extract Y
    y = df[target_cols].values
    
    # Split
    print("Splitting data (80% Train, 20% Test)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    
    scaler = MinMaxScaler()
    
    # Copy to avoid SettingWithCopy
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    print(f"Scaling numerical columns: {numerical_cols}...")
    scaler.fit(X_train[numerical_cols])
    
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Save Scaler
    print(f"Saving scaler to {OUTPUT_SCALER}...")
    joblib.dump(scaler, OUTPUT_SCALER)
    
    # ---------------------------------------------------------
    # 3. Export
    # ---------------------------------------------------------
    
    # Convert to float32 numpy
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print("Saving .npy files...")
    np.save(X_TRAIN_FILE, X_train_np)
    np.save(X_TEST_FILE, X_test_np)
    np.save(Y_TRAIN_FILE, y_train)
    np.save(Y_TEST_FILE, y_test)
    
    # Also save feature names for inference later!
    # (Crucial for mapping input dict to array index)
    with open("feature_names_global.json", "w") as f:
        json.dump(feature_names, f)
    
    print("\nProcessing Complete.")
    print("-" * 30)
    print(f"X_train shape: {X_train_np.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test_np.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print("-" * 30)

if __name__ == "__main__":
    process_data_global()
