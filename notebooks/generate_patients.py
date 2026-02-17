
import numpy as np
import pandas as pd
import uuid

# Set seed
np.random.seed(42)

NUM_PATIENTS = 30000

# ---------------------------------------------------------
# 1. Define Clinical Profiles & Symptoms
# ---------------------------------------------------------

# --- A. Symptom List (Expanded) ---
SYMPTOMS_PSYCH = [
    # Depression/Anxiety
    "insonnia", "anedonia", "umore_depresso", "ansia_eccessiva",
    "attacchi_di_panico", "agitazione_psicomotoria", "isolamento_sociale",
    "ridotto_bisogno_sonno", "euforia_grandiosita", "apatia",
    # Psychosis
    "allucinazioni_uditive", "allucinazioni_visive", "deliri",
    "disorganizzazione_pensiero",
    # Obsessive/Anxious (New)
    "ossessioni", "compulsioni_rituali", "flashback", 
    "evitamento_trauma", "ipervigilanza",
    # Impulsive/Personality (New)
    "disattenzione_cronica", "iperattivita_impulsivita", 
    "instabilita_affettiva", "autolesionismo", "paura_abbandono",
    # Eating/Addiction (New)
    "restrizione_alimentare_severa", "abbuffate", 
    "vomito_autoindotto", "craving_sostanze"
]

SYMPTOMS_NEURO = [
    # Motor/Degenerative
    "tremore_a_riposo", "bradicinesia", "rigidita_muscolare", 
    "instabilita_posturale", "fascicolazioni", "atrofia_muscolare",
    "deficit_forza_progressivo",
    # Cognitive
    "perdita_memoria_breve_termine", "disorientamento_spazio_temporale",
    "difficolta_linguaggio_afasia", "aprassia",
    # Sensory/Cord (New)
    "parestesie_formicolii", "neurite_ottica", "segno_lhermitte",
    "fatica_astenia",
    # Epilepsy/Headache (New)
    "convulsioni", "assenze_incantamento", "aura_visiva",
    "cefalea_pulsante_severa", "fotofobia_fonofobia"
]

ALL_SYMPTOMS = SYMPTOMS_PSYCH + SYMPTOMS_NEURO

# --- B. Specific "Must Have" / "Severe" Lists for Logic ---
SEVERE_PSYCHOTIC = ["allucinazioni_uditive", "allucinazioni_visive", "deliri"]
SEVERE_MOTOR = ["tremore_a_riposo", "bradicinesia", "rigidita_muscolare", "instabilita_posturale", "fascicolazioni", "atrofia_muscolare"]
SEVERE_COGNITIVE = ["difficolta_linguaggio_afasia", "aprassia", "disorientamento_spazio_temporale"]

# --- C. Profiles Definition ---
BASE_PROB = 0.03  # Low noise

def create_profile(high_prob_symptoms, prob=0.85):
    profile = {s: BASE_PROB for s in ALL_SYMPTOMS}
    for s in high_prob_symptoms:
        if s in profile:
            profile[s] = prob
    return profile

PROFILES = {
    # --- Psychiatry ---
    "DEPRESSION": create_profile([
        "umore_depresso", "anedonia", "insonnia", "isolamento_sociale", 
        "apatia", "fatica_astenia", "perdita_memoria_breve_termine"
    ], 0.90),
    
    "ANXIETY": create_profile([
        "ansia_eccessiva", "attacchi_di_panico", "agitazione_psicomotoria",
        "insonnia", "ipervigilanza"
    ], 0.85),
    
    "BIPOLAR": create_profile([
        "euforia_grandiosita", "ridotto_bisogno_sonno", "agitazione_psicomotoria",
        "compulsioni_rituali", "umore_depresso", "instabilita_affettiva"
    ], 0.85),
    
    "SCHIZOPHRENIA": create_profile([
        "deliri", "allucinazioni_uditive", "allucinazioni_visive",
        "disorganizzazione_pensiero", "isolamento_sociale", "apatia", "anedonia"
    ], 0.90),
    
    "OCD": create_profile([
        "ossessioni", "compulsioni_rituali", "ansia_eccessiva", "insonnia"
    ], 0.90),
    
    "PTSD": create_profile([
        "flashback", "evitamento_trauma", "ipervigilanza", "insonnia",
        "ansia_eccessiva", "isolamento_sociale", "instabilita_affettiva"
    ], 0.90),
    
    "ADHD": create_profile([
        "disattenzione_cronica", "iperattivita_impulsivita", 
        "instabilita_affettiva", "agitazione_psicomotoria"
    ], 0.85),
    
    "BPD": create_profile([
        "instabilita_affettiva", "paura_abbandono", "autolesionismo",
        "iperattivita_impulsivita", "umore_depresso", "ansia_eccessiva"
    ], 0.90),
    
    "DCA": create_profile([
        "restrizione_alimentare_severa", "abbuffate", "vomito_autoindotto",
        "ansia_eccessiva", "ossessioni" # Body image obsessions
    ], 0.90), # Specific logic will split Anorexia/Bulimia
    
    # --- Neurology ---
    "PARKINSON": create_profile([
        "tremore_a_riposo", "bradicinesia", "rigidita_muscolare",
        "instabilita_posturale", "apatia", "insonnia", "ansia_eccessiva",
        "fatica_astenia"
    ], 0.90),
    
    "ALZHEIMER": create_profile([
        "perdita_memoria_breve_termine", "disorientamento_spazio_temporale",
        "difficolta_linguaggio_afasia", "aprassia", "apatia", "isolamento_sociale",
        "agitazione_psicomotoria"
    ], 0.90),
    
    "EPILEPSY": create_profile([
        "convulsioni", "assenze_incantamento", "aura_visiva",
        "perdita_memoria_breve_termine" # post-ictal
    ], 0.85),
    
    "MIGRAINE": create_profile([
        "cefalea_pulsante_severa", "fotofobia_fonofobia", "aura_visiva",
        "insonnia"
    ], 0.90),
    
    "MS": create_profile([ # Multiple Sclerosis
        "parestesie_formicolii", "neurite_ottica", "segno_lhermitte",
        "fatica_astenia", "instabilita_posturale", "deficit_forza_progressivo",
        "insonnia", "umore_depresso"
    ], 0.85),
    
    "ALS": create_profile([ # SLA
        "deficit_forza_progressivo", "fascicolazioni", "atrofia_muscolare",
        "crampi", "fatica_astenia" # Crampi not in list, keep existing
    ], 0.95), # Very specific
    
    "HEALTHY": {s: 0.01 for s in ALL_SYMPTOMS} # Extremely low base
}

# ---------------------------------------------------------
# 2. Generation Logic with HARD CONSTRAINTS
# ---------------------------------------------------------

data = []
profile_keys = list(PROFILES.keys())

# Adjusted weights for a varied dataset (Healthy ~10%, common things ~10%, rare things ~3-5%)
# Total 16 profiles + Healthy
weights = {
    "DEPRESSION": 0.12, "ANXIETY": 0.12, "HEALTHY": 0.10,
    "BIPOLAR": 0.05, "SCHIZOPHRENIA": 0.04, "OCD": 0.04, "PTSD": 0.04, "ADHD": 0.05, "BPD": 0.04, "DCA": 0.04,
    "PARKINSON": 0.05, "ALZHEIMER": 0.05, "EPILEPSY": 0.04, "MIGRAINE": 0.08, "MS": 0.04, "ALS": 0.02
}
# Normalize weights
prob_list = [weights[k] for k in profile_keys]
prob_list = np.array(prob_list) / np.sum(prob_list)

def noise(l, h): return np.random.randint(l, h+1)

for _ in range(NUM_PATIENTS):
    # --- 1. Select Profile ---
    is_comorbid = np.random.rand() < 0.20 # 20% Comorbidity
    
    main_profile = np.random.choice(profile_keys, p=prob_list)
    active_profiles = [main_profile]
    
    # Comorbidity Logic
    if is_comorbid and main_profile != "HEALTHY":
        # Realistic Pairs mapping
        compatible = []
        if main_profile == "ALS": compatible = ["DEPRESSION"] # Only Dep
        elif main_profile == "MS": compatible = ["DEPRESSION", "ANXIETY"]
        elif main_profile == "PARKINSON": compatible = ["DEPRESSION", "ALZHEIMER"] # Demenza
        elif main_profile == "BPD": compatible = ["DEPRESSION", "ANXIETY", "PTSD", "DCA"]
        elif main_profile == "DEPRESSION": compatible = ["ANXIETY", "OCD", "PTSD", "BPD", "PARKINSON", "MS", "ALS", "MIGRAINE"]
        elif main_profile == "ADHD": compatible = ["ANXIETY", "DEPRESSION", "BIPOLAR"] # frequent overlap
        else:
            # Generic valid list (excluding incompatible)
            compatible = [k for k in profile_keys if k not in ["HEALTHY", "ALS", "MS", "SCHIZOPHRENIA", "ALZHEIMER", main_profile]]
        
        if compatible:
            secondary = np.random.choice(compatible)
            # HARD EXCLUSION: SLA + Parkinson
            if (main_profile == "ALS" and secondary == "PARKINSON") or (main_profile == "PARKINSON" and secondary == "ALS"):
                pass # Skip
            else:
                active_profiles.append(secondary)

    # --- 2. Generate Symptoms (Base + Profile) ---
    patient_symptoms = {}
    
    # Combine probabilities
    active_probs = {s: 0.01 for s in ALL_SYMPTOMS} # Base noise 1%
    for p_name in active_profiles:
        p_probs = PROFILES[p_name]
        for s in ALL_SYMPTOMS:
            active_probs[s] = max(active_probs[s], p_probs.get(s, 0.01))
            
    # Generate
    for s in ALL_SYMPTOMS:
        patient_symptoms[s] = 1 if np.random.rand() < active_probs[s] else 0

    # --- 3. HARD CLINICAL CONSTRAINTS (Enforcement) ---
    
    p_set = set(active_profiles) # quick lookup
    
    # A. SLA (ALS)
    if "ALS" in p_set:
        # MUST HAVE
        patient_symptoms["deficit_forza_progressivo"] = 1
        if patient_symptoms["fascicolazioni"] == 0 and patient_symptoms["atrofia_muscolare"] == 0:
            # Force one
            force_sym = np.random.choice(["fascicolazioni", "atrofia_muscolare"])
            patient_symptoms[force_sym] = 1
        # MUST NOT HAVE
        patient_symptoms["parestesie_formicolii"] = 0
        
    # B. Sclerosi Multipla (MS)
    if "MS" in p_set:
        # MUST HAVE 2 of list
        ms_signs = ["parestesie_formicolii", "neurite_ottica", "instabilita_posturale", "fatica_astenia", "segno_lhermitte"]
        current_signs = sum([patient_symptoms[s] for s in ms_signs])
        if current_signs < 2:
            # Add missing
            needed = 2 - current_signs
            available = [s for s in ms_signs if patient_symptoms[s] == 0]
            if len(available) >= needed:
                to_add = np.random.choice(available, needed, replace=False)
                for s in to_add: patient_symptoms[s] = 1
    
    # C. Epilessia
    if "EPILEPSY" in p_set:
        if patient_symptoms["convulsioni"] == 0 and patient_symptoms["assenze_incantamento"] == 0:
             patient_symptoms["convulsioni"] = 1 # Default strict
             
    # D. Emicrania (Migraine)
    if "MIGRAINE" in p_set:
        patient_symptoms["cefalea_pulsante_severa"] = 1
        if patient_symptoms["fotofobia_fonofobia"] == 0 and patient_symptoms["aura_visiva"] == 0:
            patient_symptoms["fotofobia_fonofobia"] = 1
            
    # E. OCD
    if "OCD" in p_set:
        if patient_symptoms["ossessioni"] == 0 and patient_symptoms["compulsioni_rituali"] == 0:
             patient_symptoms["ossessioni"] = 1
        # Incompatible Schizo
        if "SCHIZOPHRENIA" not in p_set:
             # Remove bizarre delusions if present by noise
             patient_symptoms["deliri"] = 0
             
    # F. ADHD
    if "ADHD" in p_set:
         if patient_symptoms["disattenzione_cronica"] == 0 and patient_symptoms["iperattivita_impulsivita"] == 0:
             patient_symptoms["disattenzione_cronica"] = 1
             
    # G. BPD
    if "BPD" in p_set:
        patient_symptoms["instabilita_affettiva"] = 1
        if patient_symptoms["paura_abbandono"] == 0 and patient_symptoms["autolesionismo"] == 0:
             patient_symptoms["paura_abbandono"] = 1
             
    # H. DCA (Eating Disorders)
    if "DCA" in p_set:
         # Constraint: Anorexia (restriction) vs Bulimia (binge)
         # Can't have strict restriction + constant bingeing (simplify logic)
         if patient_symptoms["restrizione_alimentare_severa"] == 1 and patient_symptoms["abbuffate"] == 1:
             # Flip a coin for subtype
             if np.random.rand() < 0.5:
                 patient_symptoms["abbuffate"] = 0 # Anorexia
             else:
                 patient_symptoms["restrizione_alimentare_severa"] = 0 # Bulimia
    
    # I. HEALTHY IS HEALTHY
    if "HEALTHY" in p_set and len(p_set) == 1:
        # Strict zero for severe
        for s in SEVERE_PSYCHOTIC + SEVERE_MOTOR + SEVERE_COGNITIVE + ["convulsioni", "deficit_forza_progressivo", "neurite_ottica", "autolesionismo"]:
             patient_symptoms[s] = 0

    # --- 4. Age & Demographics Logic ---
    age = noise(18, 65)
    gender = np.random.choice(["M", "F"])
    
    if "ALZHEIMER" in p_set: age = noise(60, 95)
    elif "PARKINSON" in p_set: age = noise(50, 85) # Rare < 40 constraint
    elif "ALS" in p_set: age = noise(40, 80) # > 40
    elif "MS" in p_set: 
        age = noise(20, 50)
        gender = np.random.choice(["F", "M"], p=[0.7, 0.3]) # F > M
        
    # --- 5. Generate Scales (Correlated) ---
    # HAM-D
    ham_d = noise(0, 4)
    if patient_symptoms["umore_depresso"]: ham_d += noise(5, 10)
    if patient_symptoms["anedonia"]: ham_d += noise(4, 7)
    if patient_symptoms["autolesionismo"]: ham_d += noise(5, 10) # Severe dep
    if "HEALTHY" in p_set: ham_d = min(7, ham_d)
    
    # GAD-7
    gad_7 = noise(0, 3)
    if patient_symptoms["ansia_eccessiva"]: gad_7 += noise(5, 10)
    if patient_symptoms["ipervigilanza"]: gad_7 += noise(3, 6)
    if "HEALTHY" in p_set: gad_7 = min(5, gad_7)
    
    # PANSS
    panss = noise(30, 35)
    if any(patient_symptoms[s] for s in SEVERE_PSYCHOTIC): panss += noise(20, 50)
    if "HEALTHY" in p_set: panss = 30
    
    # UPDRS-III
    updrs = 0
    if "PARKINSON" in p_set: 
        updrs = noise(15, 30)
        if patient_symptoms["tremore_a_riposo"]: updrs += noise(10, 20)
        if patient_symptoms["rigidita_muscolare"]: updrs += noise(10, 20)
    
    # MMSE
    mmse = noise(27, 30)
    if "ALZHEIMER" in p_set: mmse = noise(10, 24)
    if patient_symptoms["disorientamento_spazio_temporale"]: mmse -= noise(3, 6)
    if "HEALTHY" in p_set: mmse = max(27, mmse)

    # --- 6. Targets ---
    targets = {f"TARGET_{k}": 1 if k in p_set else 0 for k in profile_keys if k != "HEALTHY"}
    # Handle mutually exclusive / hierarchy
    if targets.get("TARGET_BIPOLAR"): targets["TARGET_DEPRESSION"] = 0 # Hierarchy
    # BPD often co-diagnosed, keep both.
    
    # Map back to requested target names if slightly different
    # Request: MDD, GAD, BIPOLAR, SCHIZO, PARKINSON, ALZHEIMER
    # New: OCD, PTSD, ADHD, DCA, BPD, EPILEPSY, MIGRAINE, MS, ALS
    # My dict keys are profile names: DEPRESSION -> TARGET_DEPRESSION.
    # I should align with requested TARGET_MDD
    
    final_targets = {
        "TARGET_MDD": targets.get("TARGET_DEPRESSION", 0),
        "TARGET_GAD": targets.get("TARGET_ANXIETY", 0),
        "TARGET_BIPOLAR": targets.get("TARGET_BIPOLAR", 0),
        "TARGET_SCHIZO": targets.get("TARGET_SCHIZOPHRENIA", 0),
        "TARGET_PARKINSON": targets.get("TARGET_PARKINSON", 0),
        "TARGET_ALZHEIMER": targets.get("TARGET_ALZHEIMER", 0),
        "TARGET_OCD": targets.get("TARGET_OCD", 0),
        "TARGET_PTSD": targets.get("TARGET_PTSD", 0),
        "TARGET_ADHD": targets.get("TARGET_ADHD", 0),
        "TARGET_DCA": targets.get("TARGET_DCA", 0),
        "TARGET_BPD": targets.get("TARGET_BPD", 0),
        "TARGET_EPILEPSY": targets.get("TARGET_EPILEPSY", 0),
        "TARGET_MIGRAINE": targets.get("TARGET_MIGRAINE", 0),
        "TARGET_MS": targets.get("TARGET_MS", 0),
        "TARGET_ALS": targets.get("TARGET_ALS", 0)
    }

    # Record
    record = {
        "id": str(uuid.uuid4()),
        "active_profiles": "+".join(sorted(list(p_set))),
        "eta": int(age),
        "sesso": gender,
        # ...symptoms...
        **patient_symptoms,
        # ...scales...
        "HAM_D": int(ham_d), "GAD_7": int(gad_7), "PANSS_total": int(panss),
        "UPDRS_III": int(updrs), "MMSE": int(mmse),
        "scolarita_anni": noise(5, 20),
        "familiarita_disturbi": np.random.choice([0, 1]),
        # ...targets...
        **final_targets
    }
    data.append(record)

# Export
df = pd.DataFrame(data)
cols_start = ["id", "active_profiles", "eta", "sesso", "scolarita_anni", "familiarita_disturbi"]
cols_scales = ["HAM_D", "GAD_7", "PANSS_total", "UPDRS_III", "MMSE"]
cols_targets = sorted(list(final_targets.keys()))
cols_symptoms = ALL_SYMPTOMS

df = df[cols_start + cols_symptoms + cols_scales + cols_targets]

print(f"Generated {NUM_PATIENTS} patients.")
print(df["active_profiles"].value_counts().head(20))
df.to_csv("patients_global.csv", index=False)
print("Saved patients_global.csv")
