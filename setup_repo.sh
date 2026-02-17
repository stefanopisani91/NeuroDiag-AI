#!/bin/bash

# Create directories
mkdir -p app data models notebooks

# Move App files
mv app_wizard_v5.py app/app.py
mv NVD.png app/
mv NVD_cropped.png app/

# Move Data files
if [ -f clinical_sessions.db ]; then
    rm clinical_sessions.db # Remove potentially containing real data, recreate empty based on schema or let app recreate
fi
# Recreate empty db is safer or just let app do it. App does init_db() on load.
# mv clinical_sessions.db data/ # If we wanted to keep it
mv patient_schema_template.json data/
mv patients_global.csv data/
mv session_recovery.json data/

# Move Models files
mv psycho_global_model.pth models/
mv scaler_global.pkl models/
mv feature_names_global.json models/
mv target_names.json models/

# Move Notebooks/Scripts
mv generate_patients_global.py notebooks/generate_patients.py
mv train_model_global.py notebooks/train_model.py
mv process_data_global.py notebooks/process_data.py
mv clinical_inference_global.py notebooks/clinical_inference.py
mv global_training_history.png notebooks/
mv X_test_global.npy notebooks/
mv X_train_global.npy notebooks/
mv y_test_global.npy notebooks/
mv y_train_global.npy notebooks/
mv test_clinical_case.py notebooks/

# Cleanup old versions if desired, or move to backup folder
mkdir -p backup
mv app_wizard_v4.py backup/

echo "Project structure organized!"
ls -R
