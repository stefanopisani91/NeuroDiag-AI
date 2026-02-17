#!/usr/bin/env python3
"""
test_clinical_case.py
━━━━━━━━━━━━━━━━━━━━━
Standalone Clinical Case Test: "Case 17M — Anorexia Nervosa with Suicidal Risk"

Loads the global diagnostic backend and injects a realistic clinical profile
derived from published literature, then prints the model's differential diagnosis.

Usage:
    python test_clinical_case.py
"""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Import Backend ---
# clinical_inference_global loads artifacts (model, scaler, feature names)
# at module level. predict_patient() is the main inference function.
from clinical_inference_global import predict_patient, feature_names


def build_case_17m_anorexia_suicide():
    """
    Constructs the clinical profile for:
    Case 17M — Anorexia Nervosa with Suicidal Risk
    
    Source: Composite from published case reports on adolescent 
    eating disorders with comorbid suicidality.
    
    Profile:
        - 17-year-old male
        - Anorexia Nervosa (restricting + purging subtype)
        - Active suicidal ideation with prior attempt
        - Severe depressive episode (HAM-D estimated 28)
        - Generalized anxiety (GAD-7 estimated 16)
        - Intact cognition (MMSE 30)
        - BMI 16.1 (Note: BMI is not a model feature)
    """
    
    # Start with all features zeroed
    input_data = {feat: 0 for feat in feature_names}
    
    # ── Anagrafica ──────────────────────────────────────────────
    input_data["eta"] = 17
    input_data["sesso"] = 0                # 0 = Maschio
    input_data["scolarita_anni"] = 10      # Stimato (2° superiore)
    # BMI = 16.1 → non è una feature del modello, ma informa il quadro clinico
    
    # ── Sintomi DCA (Eating Disorder) ──────────────────────────
    # Chiavi che potrebbero non esistere vengono gestite con safe_set
    symptom_keys = {
        "abbuffate": 1,
        "vomito_autoindotto": 1,           # condotte_eliminazione → mapped
        "condotte_eliminazione": 1,        # alias, verrà ignorato se assente
        "restrizione_alimentare_severa": 1, # alias con _severa
        "restrizione_alimentare": 1,       # fallback senza _severa
        "disturbo_immagine_corporea": 1,   # potrebbe non esistere
    }
    
    # ── Sintomi Psichiatrici Comorbidi ─────────────────────────
    psychiatric_keys = {
        "autolesionismo": 1,
        "ideazione_suicidaria": 1,         # potrebbe non esistere
        "tentativo_suicidio": 1,           # potrebbe non esistere
        "instabilita_affettiva": 1,
        "iperattivita_impulsivita": 1,     # impulsività → mapped
        "impulsivita": 1,                  # fallback
        "ansia_eccessiva": 1,
        "umore_depresso": 1,               # coerente con HAM-D 28
        "insonnia": 1,                     # comune in DCA+MDD
        "anedonia": 1,                     # correlato a depressione severa
        "segni_fisici_malnutrizione": 1,   # potrebbe non esistere
    }
    
    # ── Scale Cliniche (Stimate dal testo) ─────────────────────
    scale_keys = {
        "HAM_D": 28,           # Depressione grave / rischio suicidario
        "GAD_7": 16,           # Ansia severa
        "PANSS_total": 45,     # Disregolazione affettiva, non psicosi franca
        "MMSE": 30,            # Cognizione intatta
        "UPDRS_III": 0,        # Nessun segno parkinsoniano
    }
    
    # ── Safe Injection ─────────────────────────────────────────
    # Only set keys that actually exist in the model's feature_names
    applied = []
    skipped = []
    
    all_overrides = {}
    all_overrides.update(symptom_keys)
    all_overrides.update(psychiatric_keys)
    all_overrides.update(scale_keys)
    
    for key, value in all_overrides.items():
        if key in feature_names:
            input_data[key] = value
            applied.append(key)
        else:
            skipped.append(key)
    
    return input_data, applied, skipped


def print_header():
    """Print a clean clinical header."""
    print()
    print("━" * 60)
    print("  CONSULTO CLINICO — CASO DALLA LETTERATURA")
    print("━" * 60)
    print("  Paziente:    Maschio, 17 anni")
    print("  BMI:         16.1 (sottopeso severo)")
    print("  Quadro:      Anorexia Nervosa (restrict/purge)")
    print("                + Ideazione suicidaria + Tentativo")
    print("                + Depressione maggiore severa")
    print("  Fonte:       Case report composito (letteratura)")
    print("━" * 60)


def print_results(diagnosis, applied, skipped):
    """Print results in a clean, readable format."""
    
    # Feature injection report
    print(f"\n  Feature iniettate: {len(applied)}")
    for feat in sorted(applied):
        print(f"    ✓ {feat}")
    
    if skipped:
        print(f"\n  ⚠  Feature ignorate (non nel modello): {len(skipped)}")
        for feat in sorted(skipped):
            print(f"    ✗ {feat}")
    
    # Differential diagnosis
    print("\n" + "─" * 60)
    print("  DIAGNOSI DIFFERENZIALE AI")
    print("─" * 60)
    
    # Top 3
    top_items = list(diagnosis.items())[:3]
    print("\n  🏥  Top 3 Diagnosi:")
    for i, (condition, prob) in enumerate(top_items, 1):
        bar_len = int(prob // 2)
        bar = "█" * bar_len
        medal = ["🥇", "🥈", "🥉"][i-1]
        print(f"    {medal}  {condition:<20s}  {prob:>6.2f}%  {bar}")
    
    # All diagnoses > 1%
    print("\n  📋  Quadro Completo (prob > 1%):")
    for condition, prob in diagnosis.items():
        if prob > 1.0:
            bar_len = int(prob // 2)
            bar = "▓" * bar_len
            print(f"      {condition:<20s}  {prob:>6.2f}%  {bar}")
    
    # Clinical interpretation
    print("\n" + "─" * 60)
    print("  INTERPRETAZIONE")
    print("─" * 60)
    
    top_disease = top_items[0][0]
    top_prob = top_items[0][1]
    
    if top_prob > 50:
        print(f"  ▶ Diagnosi primaria:  {top_disease} ({top_prob:.1f}%)")
        comorbid = [k for k, v in diagnosis.items() if v > 30 and k != top_disease]
        if comorbid:
            print(f"  ▶ Comorbidità:        {', '.join(comorbid)}")
    elif top_prob > 30:
        print(f"  ▶ Diagnosi probabile: {top_disease} ({top_prob:.1f}%)")
        print(f"  ▶ Quadro complesso — considerare comorbidità multiple")
        comorbid = [k for k, v in diagnosis.items() if v > 20 and k != top_disease]
        if comorbid:
            print(f"  ▶ DD da considerare:  {', '.join(comorbid)}")
    else:
        print(f"  ▶ Quadro incerto — nessuna diagnosi supera il 30%")
        print(f"  ▶ Top candidate:      {top_disease} ({top_prob:.1f}%)")
        print(f"  ▶ Approfondimento clinico necessario")
    
    # DCA-specific check
    dca_prob = diagnosis.get("DCA", 0)
    mdd_prob = diagnosis.get("MDD", 0)
    bpd_prob = diagnosis.get("BPD", 0)
    
    print()
    if dca_prob > 10:
        print(f"  💡 Nota: DCA rilevato al {dca_prob:.1f}% — coerente con il profilo clinico")
    else:
        print(f"  ⚠  Nota: DCA a solo {dca_prob:.1f}% — il modello potrebbe non catturare")
        print(f"          pienamente i disturbi alimentari con questo set di feature")
    
    if mdd_prob > 20:
        print(f"  💡 Nota: MDD al {mdd_prob:.1f}% — coerente con HAM-D 28 (depressione grave)")
    
    if bpd_prob > 15:
        print(f"  💡 Nota: BPD al {bpd_prob:.1f}% — instabilità affettiva + autolesionismo")
    
    print("\n" + "━" * 60)
    print("  Fine consulto")
    print("━" * 60)
    print()


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build the clinical case
    input_data, applied, skipped = build_case_17m_anorexia_suicide()
    
    # Print header
    print_header()
    
    # Run inference
    diagnosis = predict_patient(input_data)
    
    # Print formatted results
    print_results(diagnosis, applied, skipped)
