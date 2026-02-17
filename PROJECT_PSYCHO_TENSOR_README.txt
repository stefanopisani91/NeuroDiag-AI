PROJECT PSYCHO-TENSOR DOCUMENTATION
=====================================
Last Updated: 2026-02-11

1. PROJECT OVERVIEW
-------------------
"Psycho-Tensor" is an AI-powered Clinical Decision Support System (CDSS) designed to assist clinicians in diagnosing psychiatric and neurological disorders. It uses a neural network to analyze patient symptoms, demographics, and psychometric scale scores (HAM-D, PANSS, etc.) to evaluate the probability of various conditions (e.g., Depression, Schizophrenia, Parkinson's, ALS).

The project is currently in version 4 (`app_wizard_v4.py`), featuring a professional "OsteoEasy-style" UI.

2. CORE ARCHITECTURE
--------------------
The system follows a standard Machine Learning pipeline:
Data Generation -> Data Processing -> Model Training -> Inference -> User Interface

A. DATA GENERATION
   - Script: `generate_patients_global.py`
   - Purpose: Generates a synthetic dataset of 30,000+ patients based on probabilistic clinical profiles (e.g., "DEPRESSION" profile has high prob of "insomnia" + "depressed mood").
   - Output: `patients_global.csv`

B. DATA PROCESSING
   - Script: `process_data_global.py`
   - Purpose: Cleans data, encodes categoricals (Sex M/F -> 0/1), scales numerical inputs (Age, Test Scores) using MinMax scaling, and splits into Train/Test sets.
   - Outputs: 
     - `scaler_global.pkl` (Saved Scaler for inference)
     - `feature_names_global.json` (Ordered list of input features)
     - `target_names.json` (List of possible diagnoses)
     - `*.npy` files (Binary training data)

C. MODEL TRAINING
   - Script: `train_model_global.py`
   - Model: `PsychoGlobalModel` (PyTorch Feed-Forward Neural Network).
   - Architecture: Input -> Dense(128) -> Dense(64) -> Dense(32) -> Output(Sigmoid).
   - Output: `psycho_global_model.pth` (Saved weights).

D. INFERENCE ENGINE
   - Module: `clinical_inference_global.py`
   - Purpose: Loads the saved artifacts (Model, Scaler, JSONs) and provides the `predict_patient()` function used by the app and tests.

E. USER INTERFACE (Streamlit)
   - Main App: `app_wizard_v4.py`
   - Features:
     - "OsteoEasy" Professional UI Style (Teal/Clean).
     - Clinical Wizard: Guided anamnesis for symptoms.
     - Psychometrics: Sliders for HAM-D, GAD-7, PANSS, etc.
     - Database: Local SQLite (`clinical_sessions.db`) to save patient visits.
     - Admin Area: Password protected ("admin") to export data.

3. FILE STRUCTURE EXPLANATION
-----------------------------

--- ACTIVE FILES (DO NOT DELETE) ---
[APP]
- `app_wizard_v4.py`: The CURRENT production application. Run with `streamlit run app_wizard_v4.py`.
- `clinical_sessions.db`: SQLite database storing patient history.

[ML PIPELINE - GLOBAL VERSION]
- `generate_patients_global.py`: Generates the master dataset.
- `process_data_global.py`: Prepares data for training.
- `train_model_global.py`: Trains the Neural Network.
- `clinical_inference_global.py`: Backend module for predictions.

[TESTING]
- `test_clinical_case.py`: Standalone script to verify model logic with a specific clinical case (e.g., "Case 17M Anorexia").

[ARTIFACTS (Generated)]
- `psycho_global_model.pth`: Active Model weights.
- `scaler_global.pkl`: Active Scaler.
- `feature_names_global.json`: Active Feature mapping.
- `target_names.json`: Active Target class names.
- `patients_global.csv`: Raw synthetic dataset.
- `*_global.npy`: Training tensors.

--- LEGACY / OBSOLETE FILES (CANDIDATES FOR CLEANUP) ---
- `app_wizard_v3.py`, `app_wizard_v2.py`, `app_wizard.py`: Old UI versions.
- `app.py`, `app_guided.py`: Very old prototypes.
- `process_data.py`, `train_model.py`: Old ML pipeline (non-global).
- `generate_patients_v2.py`: Old generator.
- `clinical_inference.py`: Old inference module.
- `patients_dataset_v2.csv`: Old dataset.
- `psycho_tensor_model.pth`: Old model weights.
- `scaler.pkl`: Old scaler.
- `X_*.npy`, `y_*.npy`: Old training data (non-global).

4. HOW TO RUN
-------------
1. Training (if needed):
   `python generate_patients_global.py`
   `python process_data_global.py`
   `python train_model_global.py`

2. Running the App:
   `streamlit run app_wizard_v4.py`

3. Testing Inference:
   `python test_clinical_case.py`
