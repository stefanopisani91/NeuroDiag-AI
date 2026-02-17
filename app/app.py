
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import json
import time
import sqlite3
import random
from datetime import datetime
import pathlib
import os

# --- Configuration (Must be first) ---
st.set_page_config(
    page_title="Gestione Clinica - Psycho-Tensor",
    page_icon="⚕️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- UI Theme Toggle ---
USE_MODERN_UI = True

# --- 1. STYLE INJECTION (CSS) ---
if not USE_MODERN_UI:
    st.markdown("""
<style>
    /* GLOBAL RESET & FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif !important;
        background-color: #F1F3F5 !important; /* Light Gray Background like OsteoEasy */
        color: #2C3E50 !important;
    }

    /* HIDE STREAMLIT HEADER (transparent, no space) */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stApp {margin-top: -30px;}
    footer {display: none;}

    /* Native Sidebar Toggle - Styled as Teal Pill */
    [data-testid="collapsedControl"] {
        position: fixed !important;
        top: 80px !important;
        left: 12px !important;
        z-index: 999999 !important;
        background-color: #009688 !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        box-shadow: 0 2px 10px rgba(0, 150, 136, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: #00796B !important;
        box-shadow: 0 4px 14px rgba(0, 150, 136, 0.55) !important;
        transform: scale(1.08) !important;
    }
    [data-testid="collapsedControl"] button {
        color: #FFFFFF !important;
        background: transparent !important;
        border: none !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] span {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Sidebar slide animation */
    [data-testid="stSidebar"] {
        transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1),
                    opacity 0.3s ease !important;
    }

    /* SIDEBAR STYLING - Professional Dark/Teal or Clean White */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #D1D5DB;
    }
    
    /* INPUT FIELDS - Pill/Rounded Rect style */
    div.row-widget.stTextInput > div > div > input {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 8px; /* Slightly rounded */
        color: #374151;
        padding: 10px 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div.row-widget.stTextInput > div > div > input:focus {
        border-color: #009688;
        box-shadow: 0 0 0 2px rgba(0, 150, 136, 0.2);
    }
    
    /* HEADERS */
    h1, h2, h3, h4, .main-question {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }
    
    .main-question {
        font-size: 28px !important; /* Balanced size */
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
        line-height: 1.3;
        color: #111827 !important;
    }
    
    .technical-label {
        font-size: 14px;
        font-weight: 600;
        color: #9CA3AF; /* Gray 400 */
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* SUBTITLES - Rounded Teal Border Box */
    .section-label {
        color: #009688; /* TEAL PRIMARY */
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
        margin-bottom: 15px;
        border: 1.5px solid #009688;
        border-radius: 8px;
        padding: 10px 20px;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* CARDS/CONTAINERS */
    .card-container {
        background-color: #FFFFFF;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 1px solid #F3F4F6;
    }

    /* BUTTONS - OsteoEasy Style */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        height: 55px; /* Consistent height */
    }

    /* PROGRESS BAR */
    .stProgress > div > div > div > div {
        background-color: #009688; /* TEAL */
    }

    /* SLIDER styling */
    div[data-testid="stSlider"] div[role="slider"] { background-color: #009688 !important; }
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] { color: #009688 !important; }
    div[data-testid="stSlider"] [class*="StyledThumb"] { background-color: #009688 !important; }

    /* Custom Header Container */
    .main-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        background-color: #FFFFFF;
        border-bottom: 1px solid #E0E0E0;
        margin-bottom: 30px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* RADIO BUTTONS */
    div[role="radiogroup"] > label > div:first-child {
        background-color: #009688 !important;
        border-color: #009688 !important;
    }
    
    /* CHECKBOX */
    div[data-testid="stCheckbox"] label span[role="checkbox"][aria-checked="true"] {
        background-color: #009688 !important;
        border-color: #009688 !important;
    }
    
    /* EXPANDER DESIGN */
    div[data-testid="stExpander"] {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* TOASTS */
    div[data-testid="stToast"] {
        background-color: #FFFFFF !important;
        color: #333 !important;
        border-left: 5px solid #009688 !important;
    }

</style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #F8FAFC !important;
        color: #334155 !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    .stApp { margin-top: -30px; }
    footer { display: none; }

    /* Sidebar Toggle Pill */
    [data-testid="collapsedControl"] {
        position: fixed !important;
        top: 80px !important;
        left: 12px !important;
        z-index: 999999 !important;
        background-color: #0E766E !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        box-shadow: 0 2px 10px rgba(14, 118, 110, 0.35) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: #0A5D57 !important;
        box-shadow: 0 4px 14px rgba(14, 118, 110, 0.5) !important;
        transform: scale(1.08) !important;
    }
    [data-testid="collapsedControl"] button {
        color: #FFFFFF !important;
        background: transparent !important;
        border: none !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] span {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1),
                    opacity 0.3s ease !important;
    }

    /* Input Fields */
    div.row-widget.stTextInput > div > div > input {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        color: #334155;
        padding: 10px 15px;
        box-shadow: none;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div.row-widget.stTextInput > div > div > input:focus {
        border-color: #0E766E;
        box-shadow: 0 0 0 3px rgba(14, 118, 110, 0.12);
    }

    /* Headers */
    h1, h2, h3, h4, .main-question {
        color: #1E293B !important;
        font-weight: 700 !important;
    }
    .main-question {
        font-size: 26px !important;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
        line-height: 1.4;
        color: #1E293B !important;
    }
    .technical-label {
        font-size: 13px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Section Labels — Clean bottom border only */
    .section-label {
        color: #0E766E;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-align: center;
        margin-bottom: 15px;
        border: none;
        border-bottom: 2px solid #0E766E;
        border-radius: 0;
        padding: 0 0 8px 0;
        width: 100%;
        box-sizing: border-box;
    }

    /* Cards */
    .card-container {
        background-color: #FFFFFF;
        padding: 32px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
    }

    /* Buttons — Ghost default */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
        font-size: 14px !important;
        border: 1px solid #CBD5E1 !important;
        background-color: transparent !important;
        color: #334155 !important;
    }
    div.stButton > button:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
    }

    /* Primary buttons — Solid Teal */
    button[kind="primary"][data-testid="stBaseButton-primary"] {
        background-color: #0E766E !important;
        color: white !important;
        border: none !important;
        font-size: 15px !important;
        height: 48px !important;
    }
    button[kind="primary"][data-testid="stBaseButton-primary"]:hover {
        background-color: #0A5D57 !important;
        box-shadow: 0 4px 12px rgba(14, 118, 110, 0.25) !important;
    }

    /* Sidebar buttons — Minimal ghost */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #64748B !important;
        border: 1px solid #E2E8F0 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        height: auto !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #F8FAFC !important;
        color: #0E766E !important;
        border-color: #0E766E !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div { background-color: #0E766E; }

    /* Slider */
    div[data-testid="stSlider"] div[role="slider"] { background-color: #0E766E !important; }
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] { color: #0E766E !important; }
    div[data-testid="stSlider"] [class*="StyledThumb"] { background-color: #0E766E !important; }

    /* Header Container */
    .main-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        background-color: #FFFFFF;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 30px;
    }

    /* Radio Buttons */
    div[role="radiogroup"] > label > div:first-child {
        background-color: #0E766E !important;
        border-color: #0E766E !important;
    }

    /* Checkbox */
    div[data-testid="stCheckbox"] label span[role="checkbox"][aria-checked="true"] {
        background-color: #0E766E !important;
        border-color: #0E766E !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        background-color: #FFFFFF !important;
    }

    /* Toasts */
    div[data-testid="stToast"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border-left: 4px solid #0E766E !important;
    }
</style>
    """, unsafe_allow_html=True)

# --- 2. Database Management ---
# Robust Path Handling
BASE_DIR = pathlib.Path(__file__).parent.resolve()
ROOT_DIR = BASE_DIR.parent

DB_FILE = str(ROOT_DIR / "data" / "clinical_sessions.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            nome TEXT,
            cognome TEXT,
            eta INTEGER,
            sesso INTEGER,
            features_json TEXT,
            diagnosis_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def default_json_converter(o):
    if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(o)
    elif isinstance(o, (np.float_, np.float16, np.float32, 
        np.float64)):
        return float(o)
    elif isinstance(o, (np.ndarray,)): 
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def save_patient_session(nome, cognome, eta, sesso, features, diagnosis):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO patients (timestamp, nome, cognome, eta, sesso, features_json, diagnosis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now(), nome, cognome, eta, sesso, json.dumps(features, default=default_json_converter), json.dumps(diagnosis, default=default_json_converter)))
    conn.commit()
    conn.close()

def load_recent_patients(limit=10):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(f"SELECT timestamp, nome, cognome, eta, diagnosis_json FROM patients ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def get_full_db_csv():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()
    return df.to_csv(index=False).encode('utf-8')

def reset_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM patients")
    conn.commit()
    conn.close()

# Initialize DB on load
init_db()


# --- Clinical Pseudonym Generator (Color-Animal, standard in RCTs) ---
_PSEUDO_COLORS = [
    "Amber", "Azure", "Bronze", "Cedar", "Cobalt", "Coral", "Crimson", "Copper",
    "Dune", "Ember", "Flint", "Frost", "Gold", "Granite", "Hazel", "Indigo",
    "Iron", "Ivory", "Jade", "Jasper", "Lapis", "Linen", "Maple", "Marble",
    "Moss", "Navy", "Nickel", "Umber", "Violet"
]
_PSEUDO_ANIMALS = [
    "Badger", "Bear", "Bison", "Condor", "Crane", "Deer", "Dolphin", "Eagle",
    "Elk", "Falcon", "Finch", "Fox", "Gazelle", "Griffin", "Hare", "Hawk",
    "Heron", "Ibex", "Jaguar", "Jay", "Kestrel", "Lark", "Leopard", "Lynx",
    "Mantis", "Otter", "Owl", "Panther", "Puma", "Quail", "Wolf", "Zebra"
]

def generate_pseudonym():
    """Generate a unique Color-Animal pseudonym not already in the DB."""
    conn = sqlite3.connect(DB_FILE)
    existing = set()
    try:
        df = pd.read_sql_query("SELECT nome, cognome FROM patients", conn)
        existing = set(zip(df['nome'], df['cognome']))
    except Exception:
        pass
    finally:
        conn.close()
    
    for _ in range(100):
        color = random.choice(_PSEUDO_COLORS)
        animal = random.choice(_PSEUDO_ANIMALS)
        if (color, animal) not in existing:
            return color, animal
    return random.choice(_PSEUDO_COLORS), f"{random.choice(_PSEUDO_ANIMALS)}{random.randint(10,99)}"


# --- 3. Model & Resources ---
# --- 3. Model & Resources ---
MODEL_FILE = str(ROOT_DIR / "models" / "psycho_global_model.pth")
SCALER_FILE = str(ROOT_DIR / "models" / "scaler_global.pkl")
FEATURE_NAMES_FILE = str(ROOT_DIR / "models" / "feature_names_global.json")
TARGET_NAMES_FILE = str(ROOT_DIR / "models" / "target_names.json")

class PsychoGlobalModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(PsychoGlobalModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_shape, 128), nn.ReLU(), nn.Dropout(0.3))
        self.layer2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.layer3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2))
        self.output = nn.Sequential(nn.Linear(32, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load(SCALER_FILE)
        with open(FEATURE_NAMES_FILE, 'r') as f: feature_names = json.load(f)
        with open(TARGET_NAMES_FILE, 'r') as f: target_names = json.load(f)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        input_dim = len(feature_names)
        num_classes = len(target_names)
        model = PsychoGlobalModel(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()
        return model, scaler, feature_names, target_names, device
    except Exception as e:
        return None, None, None, None, None

model, scaler, feature_names, target_names, device = load_resources()

if not model:
    st.error("System Error: Resources not found.")
    st.stop()

# --- SESSION RECOVERY (JSON) ---
RECOVERY_FILE = str(ROOT_DIR / "data" / "session_recovery.json")

def save_recovery_state():
    try:
        state = {
            "step": st.session_state.get("step"),
            "patient_data": st.session_state.get("patient_data"),
            "patient_meta": st.session_state.get("patient_meta"),
            "current_macro_index": st.session_state.get("current_macro_index", 0),
            "current_micro_index": st.session_state.get("current_micro_index", 0),
            "depth_level": st.session_state.get("depth_level", 0),
            "saved_eta": st.session_state.get("saved_eta"),
            "saved_scolarita": st.session_state.get("saved_scolarita"),
            "saved_sesso_index": st.session_state.get("saved_sesso_index"),
            "input_nome": st.session_state.get("input_nome"),
            "input_cognome": st.session_state.get("input_cognome"),
            "timestamp": str(datetime.now())
        }
        with open(RECOVERY_FILE, "w") as f:
            json.dump(state, f, default=default_json_converter)
        st.toast("Stato Sessione Salvato su File")
    except Exception as e:
        st.error(f"Errore Salvataggio: {e}")

def load_recovery_state():
    try:
        with open(RECOVERY_FILE, "r") as f:
            state = json.load(f)
        if "step" in state: st.session_state.step = state["step"]
        if "patient_data" in state: st.session_state.patient_data = state["patient_data"]
        if "patient_meta" in state: st.session_state.patient_meta = state["patient_meta"]
        if "current_macro_index" in state: st.session_state.current_macro_index = state["current_macro_index"]
        if "current_micro_index" in state: st.session_state.current_micro_index = state["current_micro_index"]
        if "depth_level" in state: st.session_state.depth_level = state["depth_level"]
        if "saved_eta" in state: st.session_state.saved_eta = state["saved_eta"]
        if "saved_scolarita" in state: st.session_state.saved_scolarita = state["saved_scolarita"]
        if "saved_sesso_index" in state: st.session_state.saved_sesso_index = state["saved_sesso_index"]
        if "input_nome" in state: st.session_state.input_nome = state["input_nome"]
        if "input_cognome" in state: st.session_state.input_cognome = state["input_cognome"]
        st.toast("Sessione Ripristinata!")
        time.sleep(0.5)
        st.rerun()
    except FileNotFoundError:
        st.error("Nessun file di salvataggio trovato.")
    except Exception as e:
        st.error(f"Errore Ripristino: {e}")

# --- 4. Sidebar: ADMIN GATE ---
with st.sidebar:
    # --- BRANDING ---
    if not USE_MODERN_UI:
        st.markdown("""
            <div style="text-align: left; margin-bottom: 25px; margin-top: 10px;">
                <div style="font-size: 26px; font-weight: 800; color: #009688; letter-spacing: -0.5px;">Clinical<span style="color:#2C3E50">Support</span></div>
                <div style="font-size: 13px; font-weight: 500; color: #9CA3AF; margin-top:4px;">Software Clinico Ospedaliero</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:10px; margin-bottom: 30px;">
            <div style="color:#009688; font-weight:600; font-size:14px; padding:8px 12px; background:#E0F2F1; border-radius:6px;">Nuova Visita</div>
            <div style="color:#6B7280; font-weight:500; font-size:14px; padding:8px 12px;">Pazienti</div>
            <div style="color:#6B7280; font-weight:500; font-size:14px; padding:8px 12px;">Agenda</div>
            <div style="color:#6B7280; font-weight:500; font-size:14px; padding:8px 12px;">Impostazioni</div>
        </div>
        <hr style="border:0; border-top:1px solid #E0E0E0; margin:20px 0;">
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align: left; margin-bottom: 25px; margin-top: 10px;">
                <div style="font-size: 26px; font-weight: 800; color: #0E766E; letter-spacing: -0.5px;">Clinical<span style="color:#1E293B">Support</span></div>
                <div style="font-size: 12px; font-weight: 500; color: #94A3B8; margin-top:4px; letter-spacing: 0.5px;">Software Clinico Ospedaliero</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:6px; margin-bottom: 30px;">
            <div style="color:#0E766E; font-weight:600; font-size:14px; padding:8px 12px; background:#F0FDFA; border-radius:6px; border-left: 3px solid #0E766E;">Nuova Visita</div>
            <div style="color:#64748B; font-weight:500; font-size:14px; padding:8px 12px;">Pazienti</div>
            <div style="color:#64748B; font-weight:500; font-size:14px; padding:8px 12px;">Agenda</div>
            <div style="color:#64748B; font-weight:500; font-size:14px; padding:8px 12px;">Impostazioni</div>
        </div>
        <hr style="border:0; border-top:1px solid #E2E8F0; margin:20px 0;">
        """, unsafe_allow_html=True)

    # === MANUAL SESSION MANAGEMENT ===
    st.markdown('<p style="font-size:12px; font-weight:600; color:#9CA3AF; margin-top:10px; margin-bottom:5px;">GESTIONE SESSIONE</p>', unsafe_allow_html=True)
    c_save, c_load = st.columns(2)
    with c_save:
        if st.button("SALVA STATO", use_container_width=True, key="btn_salva_stato"):
            save_recovery_state()
    with c_load:
        if st.button("RIPRISTINA", use_container_width=True, key="btn_ripristina"):
            load_recovery_state()
            
    st.markdown("---")

    # --- ADMIN AREA ---
    with st.expander("AREA CLINICA RISERVATA"):
        admin_password = st.text_input("Credenziali Medico", type="password", key="admin_pass", label_visibility="collapsed", placeholder="Sblocco")
        
        if admin_password == "admin":
            st.success("Sessione Autenticata")
            st.markdown("---")
            st.caption("DASHBOARD REPARTO")
            
            # Load stats
            try:
                conn = sqlite3.connect(DB_FILE)
                count_res = pd.read_sql_query("SELECT COUNT(*) as count FROM patients", conn)
                count = count_res['count'][0] if not count_res.empty else 0
                conn.close()
                st.metric("Cartelle Attive", count)
                
                st.markdown("##### Pazienti Recenti")
                recent_df = load_recent_patients(10)
                if not recent_df.empty:
                    st.dataframe(recent_df[['timestamp', 'nome', 'cognome']], hide_index=True, use_container_width=True)
                else:
                    st.caption("Nessuna cartella trovata.")
                    
                st.markdown("---")
                # Download
                csv_data = get_full_db_csv()
                st.download_button(
                    label="ESPORTA INTERO DB (CSV)",
                    data=csv_data,
                    file_name="clinical_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("---")
                # Reset Zone
                if st.button("FORMATTA DATABASE", use_container_width=True, type="primary"):
                    reset_database()
                    st.warning("Database formattato.")
                    time.sleep(1)
                    st.rerun()
                        
            except Exception as e:
                st.error(f"Errore: {e}")
        elif admin_password:
            st.error("Credenziali non valide.")


# --- 5. Main Content Area ---

# Header Bar
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; }
    div[data-testid="stImage"] { margin-top: -20px; margin-bottom: -20px; }
    img { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1]) 
with c2:
    st.image(str(BASE_DIR / "NVD.png"), use_container_width=True)


# --- 6. WIZARD FLOW (Onion Logic Restored) ---
WIZARD_FLOW = [
    {
        "id": "macro_psych",
        "title": "Sfera Psichica (Umore & Ansia)",
        "micro_questions": [
            {"feat": ["umore_depresso"], "label": "Umore Deflesso", "levels": ["Si sente spesso triste, senza speranza o piange facilmente?", "Le capita di sentirsi giù di corda per gran parte della giornata, come se avesse una nuvola nera sopra la testa?", "A volte sente che le energie emotive sono così basse che anche le piccole cose sembrano pesanti?"]},
            {"feat": ["anedonia"], "label": "Perdita di Piacere", "levels": ["Ha perso interesse o piacere nelle attività che prima gradiva?", "Le cose che prima la divertivano (hobby, amici) ora le sembrano 'insipide' o indifferenti?", "Sente una specie di 'piattezza', come se non riuscisse più a godersi i momenti belli?"]},
            {"feat": ["ansia_eccessiva"], "label": "Ansia Generalizzata", "levels": ["Si sente costantemente preoccupato, teso o incapace di rilassarsi?", "La sua mente tende a correre sempre avanti, immaginando problemi futuri o scenari negativi?", "Sente spesso un nodo allo stomaco o una tensione muscolare che non va via?"]},
             {"feat": ["attacchi_di_panico", "agitazione_psicomotoria"], "label": "Panico & Agitazione", "levels": ["Ha episodi improvvisi di terrore intenso con palpitazioni o sudorazione?", "Le è mai capitato di sentirsi improvvisamente sopraffatto dalla paura, come se stesse per perdere il controllo?", "Ha mai momenti in cui il cuore batte forte senza motivo apparente, facendole mancare il respiro?"]},
            {"feat": ["insonnia"], "label": "Disturbi del Sonno", "levels": ["Ha difficoltà ad addormentarsi o si sveglia troppo presto?", "Il suo sonno è disturbato, interrotto o non riposante?", "Si rigira nel letto a lungo con pensieri che non la lasciano tranquilla?"]},
            {"feat": ["isolamento_sociale"], "label": "Ritiro Sociale", "levels": ["Tende a evitare attivamente amici o situazioni sociali?", "Preferisce restare a casa piuttosto che uscire, sentendo che gli altri la 'stancano'?", "Si sente più al sicuro quando è solo e non deve interagire con nessuno?"]},
            {"feat": ["fatica_astenia"], "label": "Astenia Psichica", "levels": ["Soffre di una stanchezza fisica o mentale che non passa col riposo?", "Si sente spesso 'svuotato' o privo di forze anche senza aver fatto sforzi particolari?", "Ogni piccola attività le sembra richiedere uno sforzo immenso?"]}
        ]
    },
    {
        "id": "macro_trauma",
        "title": "Pensieri Intrusivi & Trauma",
        "micro_questions": [
            {"feat": ["ossessioni"], "label": "Ideazione Ossessiva", "levels": ["Ha pensieri intrusivi, immagini o idee fisse che le causano ansia?", "Le capita di avere preoccupazioni che sembrano eccessive ma che le tornano sempre in mente contro la sua volontà?", "A volte sente che la sua mente si 'incanta' su certi pensieri fastidiosi che non riesce a scacciare?"]},
            {"feat": ["compulsioni_rituali"], "label": "Compulsioni", "levels": ["Si sente costretto a compiere certe azioni (lavarsi, controllare, contare) per calmare l'ansia?", "Ha delle piccole regole o abitudini rigide che deve seguire affinché 'tutto vada bene'?", "Sente un disagio crescente se non esegue certi gesti in un modo specifico?"]},
            {"feat": ["flashback"], "label": "Flashback Traumatici", "levels": ["Le capita di rivivere improvvisamente ricordi traumatici come se stessero accadendo ora?", "A volte si sente trasportato nel passato da un odore, un suono o un'emozione improvvisa?", "Ha mai momenti in cui un vecchio ricordo brutto le sembra vivido e reale?"]},
            {"feat": ["evitamento_trauma"], "label": "Evitamento Agorafobico", "levels": ["Evita attivamente luoghi, persone o pensieri che le ricordano un evento doloroso?", "Cerca di girare alla larga da situazioni che potrebbero risvegliare certi ricordi?", "Tende a bloccare certi pensieri o a evitare conversazioni su temi specifici?"]},
            {"feat": ["ipervigilanza"], "label": "Ipervigilanza", "levels": ["Si sente costantemente in allerta, come se fosse in pericolo?", "Fa molta attenzione a chi le sta intorno quando è fuori casa, controllando le uscite?", "Si sente sempre 'sull'attenti', faticando a rilassarsi completamente anche al sicuro?"]}
        ]
    },
     {
        "id": "macro_psychosis",
        "title": "Percezione & Realtà",
        "micro_questions": [
            {"feat": ["allucinazioni_uditive"], "label": "Allucinazioni Uditive", "levels": ["Sente voci, sussurri o suoni che altri non sentono?", "Le capita, nel silenzio, di percepire rumori che sembrano reali ma non hanno una fonte esterna?", "A volte ha l'impressione che la sua mente le faccia sentire suoni che solo lei percepisce?"]},
            {"feat": ["allucinazioni_visive"], "label": "Allucinazioni Visive", "levels": ["Vede cose, ombre o figure che gli altri non vedono?", "Le capita di vedere qualcosa con la coda dell'occhio, ma poi non c'è nulla?", "A volte le sue percezioni visive le sembrano un po' alterate, come se vedesse le cose deformate?"]},
            {"feat": ["deliri"], "label": "Ideazione Delirante", "levels": ["È convinto di essere perseguitato, spiato o vittima di complotti?", "Ha mai la sensazione sgradevole che le persone parlino male di lei o tramino alle sue spalle?", "Si sente a volte come se fosse al centro di un'attenzione ostile o speciale non giustificata?"]},
            {"feat": ["euforia_grandiosita"], "label": "Grandiosità", "levels": ["Si è sentito invincibile, con poteri speciali o in missione per conto di forze superiori?", "Ha avuto periodi di energia eccessiva in cui sentiva di poter fare qualsiasi cosa senza dormire?", "Le è capitato di avere idee o progetti grandiosi che gli altri trovavano irrealistici?"]},
            {"feat": ["disorganizzazione_pensiero"], "label": "Disorganizzazione", "levels": ["Fa fatica a seguire un filo logico nei pensieri o gli altri dicono che i suoi discorsi sono confusi?", "Le capita di saltare da un argomento all'altro senza un nesso apparente?", "A volte si sente mentalmente confuso, come se i pensieri fossero disordinati o 'nebbiosi'?"]}
        ]
    },
    {
        "id": "macro_eating",
        "title": "Comportamento Alimentare",
        "micro_questions": [
            {"feat": ["restrizione_alimentare_severa"], "label": "Restrizione Calorica", "levels": ["Ha ridotto drasticamente l'assunzione di cibo per paura di ingrassare?", "Le capita di saltare i pasti o mangiare molto poco perché si vede troppo pesante?", "Fa molta attenzione a limitare ciò che mangia per un forte timore di prendere peso?"]},
            {"feat": ["abbuffate"], "label": "Abbuffate", "levels": ["Ha episodi in cui mangia grandi quantità di cibo perdendo il controllo?", "Le capita di mangiare molto rapidamente o di nascosto fino a sentirsi spiacevolmente pieno?", "A volte sente che quando inizia a mangiare qualcosa che le piace non riesce più a fermarsi?"]},
            {"feat": ["vomito_autoindotto"], "label": "Condotte di Eliminazione", "levels": ["Si induce il vomito o usa lassativi per controllare il peso dopo aver mangiato?", "Ha mai cercato di 'rimediare' a un pasto abbondante eliminandolo fisicamente?", "Le è mai capitato di voler espellere il cibo ingerito per alleviare il senso di colpa?"]},
            {"feat": ["autolesionismo"], "label": "Autolesionismo", "levels": ["Si è mai procurato ferite, tagli o bruciature intenzionalmente?", "Quando sta molto male, le è mai capitato di farsi del male fisico per scaricare la tensione?", "Ha mai compiuto gesti impulsivi contro il proprio corpo in momenti di forte stress?"]},
            {"feat": ["craving_sostanze"], "label": "Craving da Sostanze", "levels": ["Sente un desiderio irrefrenabile di assumere alcol, droghe o farmaci?", "Ha momenti in cui il pensiero di bere o usare sostanze diventa dominante?", "Sente a volte il bisogno di 'staccare la spina' usando sostanze che sa potrebbero farle male?"]}
        ]
    },
    {
        "id": "macro_behavior",
        "title": "Temperamento",
        "micro_questions": [
            {"feat": ["instabilita_affettiva"], "label": "Labilità Emotiva", "levels": ["Il suo umore cambia molto rapidamente nell'arco della giornata?", "Le capita di passare da una grande allegria a una profonda tristezza o rabbia in pochi minuti?", "Si sente emotivamente 'sulle montagne russe' senza un motivo apparente?"]},
            {"feat": ["paura_abbandono"], "label": "Angoscia Abbandonica", "levels": ["Ha una paura intensa di essere abbandonato o lasciato solo?", "Si sente eccessivamente ansioso quando una persona cara non risponde o si allontana?", "Ha bisogno di continue rassicurazioni sul fatto che le persone le vogliano bene?"]},
            {"feat": ["iperattivita_impulsivita"], "label": "Impulsività", "levels": ["Agisce spesso d'impulso senza riflettere sulle conseguenze?", "Le capita di interrompere gli altri o di non riuscire ad aspettare il proprio turno?", "Ha difficoltà a 'mettere il freno' alle sue reazioni o ai suoi desideri immediati?"]},
             {"feat": ["apatia"], "label": "Apatia", "levels": ["Ha perso la motivazione a fare qualsiasi cosa, anche curare la sua igiene personale?", "Si sente indifferente a ciò che le accade intorno, come se non le importasse più di nulla?", "Tende a trascorrere le giornate senza fare nulla, per pura inerzia?"]},
             {"feat": ["ridotto_bisogno_sonno"], "label": "Iperattivazione", "levels": ["Dorme pochissimo (2-3 ore) sentendosi comunque pieno di energie?", "Le capita di passare la notte sveglio lavorando o facendo cose senza sentirsi stanco il giorno dopo?", "Ha periodi in cui il sonno le sembra 'tempo perso' e preferisce restare attivo?"]}
        ]
    },
    {
        "id": "macro_neuro",
        "title": "Neurologia Generale",
        "micro_questions": [
             {"feat": ["tremore_a_riposo"], "label": "Tremore", "levels": ["Ha notato un tremore alle mani o ad altre parti del corpo quando è rilassato?", "Le capita che le mani tremino leggermente quando non sta facendo nulla, magari appoggiate sulle gambe?", "Sente a volte delle vibrazioni involontarie nei muscoli anche quando è completamente fermo?"]},
            {"feat": ["bradicinesia"], "label": "Rallentamento Motorio", "levels": ["Si sente più lento o impacciato nei movimenti quotidiani?", "Ha l'impressione che i suoi movimenti siano diventati 'macchinosi' o richiedano più sforzo del solito?", "Nota che gesti semplici come vestirsi o camminare le richiedono più tempo di prima?"]},
            {"feat": ["rigidita_muscolare"], "label": "Rigidità", "levels": ["Sente i muscoli rigidi o 'legati', come se facesse fatica a sciogliersi?", "Avverte una resistenza nei movimenti, come se le articolazioni fossero arrugginite?", "Le capita di sentire il corpo 'duro' o poco flessibile, specialmente al risveglio o dopo essere stato fermo?"]},
            {"feat": ["instabilita_posturale"], "label": "Equilibrio", "levels": ["Ha problemi di equilibrio o si sente instabile quando cammina?", "Le è capitato di inciampare spesso o di sentirsi insicuro sui piedi senza un motivo chiaro?", "Sente a volte di sbandare o di non essere ben saldo a terra, temendo di cadere?"]},
             {"feat": ["fascicolazioni"], "label": "Fascicolazioni", "levels": ["Nota dei piccoli guizzi o contrazioni muscolari sotto la pelle?", "Le capitano quei 'saltelli' involontari dei muscoli (come alla palpebra o al braccio) che non riesce a fermare?", "Sente a volte come dei piccoli vermicelli che si muovono sotto pelle in varie parti del corpo?"]},
             {"feat": ["convulsioni", "assenze_incantamento"], "label": "Episodi Critici", "levels": ["Ha mai avuto crisi convulsive, svenimenti o momenti di 'assenza'?", "Le è mai capitato di 'spegnersi' improvvisamente e risvegliarsi confuso, o di incantarsi perdendo il filo per secondi?", "I familiari le hanno mai detto che ha avuto scosse o che fissava il vuoto senza rispondere?"]},
             {"feat": ["atrofia_muscolare"], "label": "Perdita Massa Muscolare", "levels": ["Ha notato che i suoi muscoli si sono assottigliati o ridotti di volume?", "Vede le sue braccia o gambe più magre del solito, come se i muscoli si stessero consumando?", "Sente che le sue forme fisiche sono cambiate, con una perdita di tono muscolare visibile?"]},
             {"feat": ["deficit_forza_progressivo"], "label": "Debolezza Progressiva", "levels": ["Sente che la forza nelle braccia o nelle gambe sta diminuendo nel tempo?", "Fa più fatica di un mese fa a sollevare oggetti, salire le scale o alzarsi dalla sedia?", "Nota che la sua capacità fisica si sta riducendo giorno dopo giorno, rendendola più debole?"]}
        ]
    },
    {
        "id": "macro_sensory",
        "title": "Sintomi Sensoriali & Altro",
        "micro_questions": [
            {"feat": ["neurite_ottica"], "label": "Visione (Neurite)", "levels": ["Ha avuto cali improvvisi della vista, appannamenti o dolore muovendo un occhio?", "Le è capitato di vedere 'nebbia' o colori sbiaditi con un occhio solo, forse con dolore?", "Ha mai avuto episodi in cui la vista è peggiorata rapidamente in pochi giorni?"]},
            {"feat": ["segno_lhermitte"], "label": "Scossa Elettrica (Lhermitte)", "levels": ["Avverte mai una sensazione di scossa elettrica lungo la schiena piegando il collo?", "Le capita, chinando la testa in avanti, di sentire un brivido elettrico che scende lungo la colonna?", "Sente come una vibrazione fastidiosa o dolorosa nella schiena quando flette il collo?"]},
            {"feat": ["aura_visiva"], "label": "Aura Visiva", "levels": ["Vede luci lampeggianti, linee a zig-zag o macchie cieche prima di stare male?", "Le capita di avere disturbi visivi transitori come scintille o tunnel visivo?", "Ha mai episodi in cui la vista si altera con luci o forme strane per qualche minuto?"]},
            {"feat": ["cefalea_pulsante_severa"], "label": "Cefalea Pulsante", "levels": ["Soffre di mal di testa molto forti, pulsanti, che le impediscono di fare cose?", "Il dolore alla testa è martellante e peggiora muovendosi o con la luce?", "Ha attacchi di emicrania che la costringono a stare al buio e in silenzio?"]},
            {"feat": ["fotofobia_fonofobia"], "label": "Ipersensibilità Luci/Suoni", "levels": ["La luce forte o i rumori intensi le danno molto fastidio o dolore?", "Durante il mal di testa, preferisce stare al buio perché la luce le fa male agli occhi?", "I suoni normali le sembrano a volte insopportabilmente forti o fastidiosi?"]}
        ]
    },
    {
        "id": "macro_cog",
        "title": "Funzioni Cognitive",
        "micro_questions": [
            {"feat": ["perdita_memoria_breve_termine"], "label": "Memoria Recente", "levels": ["Dimentica spesso cose accadute di recente o ripete le stesse domande?", "Le capita di entrare in una stanza e non ricordare perché ci è andato, o di perdere il filo del discorso?", "Sente che i piccoli dettagli di ogni giorno le sfuggono più facilmente di un tempo?"]},
            {"feat": ["disattenzione_cronica"], "label": "Attenzione", "levels": ["Fa fatica a mantenere la concentrazione su un compito o una lettura?", "Si distrae facilmente durante una conversazione o un'attività, perdendo il punto?", "Sente che la sua mente vaga spesso altrove, rendendo difficile finire ciò che ha iniziato?"]},
            {"feat": ["difficolta_linguaggio_afasia"], "label": "Linguaggio", "levels": ["Ha difficoltà a trovare le parole giuste mentre parla?", "Le capita di avere la parola 'sulla punta della lingua' ma di non riuscire a dirla?", "Sente che il suo vocabolario si è impoverito o che a volte usa parole sbagliate senza volerlo?"]},
             {"feat": ["aprassia"], "label": "Manualità", "levels": ["Ha difficoltà a compiere gesti complessi o usare oggetti comuni (es. chiavi, posate)?", "Si sente a volte impacciato nell'eseguire azioni motorie che prima faceva in automatico?", "Le capita di non sapere bene come maneggiare un oggetto che ha sempre usato?"]},
             {"feat": ["disorientamento_spazio_temporale"], "label": "Orientamento", "levels": ["Le capita di confondersi su che giorno è o dove si trova?", "Si è mai sentito smarrito in luoghi che conosce bene o ha faticato a ritrovare la strada di casa?", "A volte perde il senso del tempo o ha dubbi su quale momento della giornata sia?"]},
             {"feat": ["parestesie_formicolii"], "label": "Sensibilità", "levels": ["Avverte formicolii, intorpidimenti o sensazioni strane sulla pelle?", "Le capita di sentire parti del corpo 'addormentate' o come se ci fossero spilli?", "Nota alterazioni della sensibilità (caldo/freddo/tatto) in alcune zone del corpo?"]}
        ]
    }
]

# --- 7. State Management ---
if 'step' not in st.session_state: st.session_state.step = 'intro_anagrafica'
if 'current_macro_index' not in st.session_state: st.session_state.current_macro_index = 0
if 'current_micro_index' not in st.session_state: st.session_state.current_micro_index = 0
if 'depth_level' not in st.session_state: st.session_state.depth_level = 0
if 'patient_data' not in st.session_state: st.session_state.patient_data = {feat: 0 for feat in feature_names}

def record_answer(features_to_set, value):
    for f in features_to_set:
        st.session_state.patient_data[f] = value

def reset_depth():
    st.session_state.depth_level = 0

def next_question():
    st.session_state.current_micro_index += 1
    reset_depth()


# --- 8. Custom Button CSS Injection per Step (Full Width) ---
st.markdown("""
<style>
/* BUTTON STYLES FOR WIZARD - OSTEOEASY INSPIRED */

/* Container tweaks */
div[data-testid="column"], div[data-testid="stColumn"] {
    padding: 0 5px;
}

/* NO Button (Outline Gray) - Left */
div[data-testid="column"]:nth-of-type(1) button {
    background-color: #FFFFFF !important; 
    color: #6B7280 !important;
    border: 1px solid #D1D5DB !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
div[data-testid="column"]:nth-of-type(1) button:hover {
    border-color: #9CA3AF !important;
    background-color: #F9FAFB !important;
    color: #374151 !important;
}

/* MAYBE Button (Soft Blue/Gray) - Center */
div[data-testid="column"]:nth-of-type(2) button {
    background-color: #EFF6FF !important; 
    color: #3B82F6 !important;
    border: 1px solid #BFDBFE !important;
    box-shadow: none !important;
}
div[data-testid="column"]:nth-of-type(2) button:hover {
    background-color: #DBEAFE !important;
}

/* YES Button (Primary Teal) - Right */
div[data-testid="column"]:nth-of-type(3) button {
    background-color: #009688 !important; 
    color: #FFFFFF !important;
    border: 1px solid #009688 !important;
    box-shadow: 0 2px 4px rgba(0, 150, 136, 0.2) !important;
}
div[data-testid="column"]:nth-of-type(3) button:hover {
    background-color: #00796B !important;
    border-color: #00796B !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 6px rgba(0, 150, 136, 0.25) !important;
}
</style>
""", unsafe_allow_html=True)


# --- 9. UI Rendering ---

if st.session_state.step == 'intro_anagrafica':
    
    # --- Pre-fill pseudonym ---
    if st.session_state.get("_generate_pseudo"):
        color, animal = generate_pseudonym()
        st.session_state["input_nome"] = color
        st.session_state["input_cognome"] = animal
        del st.session_state["_generate_pseudo"]
    
    # CARD CONTAINER
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">IDENTIFICAZIONE PAZIENTE</p>', unsafe_allow_html=True)
    
    # --- Identification Fields ---
    c1, c2, c3 = st.columns([2, 2, 1.5])
    
    # Init keys if missing (since we removed value= default)
    if "input_nome" not in st.session_state: st.session_state.input_nome = ""
    if "input_cognome" not in st.session_state: st.session_state.input_cognome = ""

    with c1:
        st.markdown('<label style="font-size:12px; font-weight:600; color:#4B5563; margin-bottom:4px; display:block;">NOME</label>', unsafe_allow_html=True)
        nome_input = st.text_input("Nome", key="input_nome", label_visibility="collapsed")
    with c2:
        st.markdown('<label style="font-size:12px; font-weight:600; color:#4B5563; margin-bottom:4px; display:block;">COGNOME</label>', unsafe_allow_html=True)
        cognome_input = st.text_input("Cognome", key="input_cognome", label_visibility="collapsed")
    with c3:
        st.markdown('<div style="height: 25px;"></div>', unsafe_allow_html=True) 
        if st.button("Genera ID Anonimo", use_container_width=True, key="btn_genera_id"):
            st.session_state["_generate_pseudo"] = True
            st.rerun()

    # Session state is auto-updated by widgets via 'key' parameter

    st.markdown('</div>', unsafe_allow_html=True) # End Card
    
    st.markdown('<hr style="border:0; border-top:1px solid #E5E7EB; margin: 15px 0;">', unsafe_allow_html=True)
    
    # CARD CONTAINER 2
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">DATI DEMOGRAFICI</p>', unsafe_allow_html=True) # Changed label

    # --- Age & Schooling ---
    c_age, c_school = st.columns([1, 3])
    
    with c_age:
        st.markdown('<label style="font-size:12px; font-weight:600; color:#4B5563; margin-bottom:4px; display:block;">ETÀ</label>', unsafe_allow_html=True)
        eta_input = st.text_input("Età", key="input_eta", value=st.session_state.get("saved_eta", ""), placeholder="18", max_chars=2, label_visibility="collapsed")
    
    with c_school:
        st.markdown('<label style="font-size:12px; font-weight:600; color:#4B5563; margin-bottom:4px; display:block;">LIVELLO ISTRUZIONE (ANNI)</label>', unsafe_allow_html=True)
        scolarita = st.slider("Anni di Scolarità", 0, 25, st.session_state.get("saved_scolarita", 13), label_visibility="collapsed")
        
    eta = 18 # Default
    if eta_input and eta_input.strip() and eta_input.strip().isdigit():
        eta = int(eta_input.strip())

    st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
    
    # --- Gender only (Familiarità removed here) ---
    c_gender_container = st.columns([1])[0]
    with c_gender_container:
         st.markdown('<label style="font-size:12px; font-weight:600; color:#4B5563; margin-bottom:4px; display:block;">SESSO BIOLOGICO</label>', unsafe_allow_html=True)
         sesso_label = st.radio("Sesso", ["Maschio", "Femmina"], index=st.session_state.get("saved_sesso_index", 0), horizontal=True, label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True) # End Card 2
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # --- CTA Buttons Styling ---
    if not USE_MODERN_UI:
        st.markdown("""
        <style>
        /* Genera ID Anonimo — Light Brown */
        button[kind="secondary"][data-testid="stBaseButton-secondary"] {
            background-color: #A1887F !important;
            color: white !important;
            font-size: 16px !important;
            height: 50px !important;
            border: none !important;
        }
        button[kind="secondary"][data-testid="stBaseButton-secondary"]:hover {
            background-color: #8D6E63 !important;
        }

        /* Sidebar buttons — Canary Yellow */
        [data-testid="stSidebar"] button[kind="secondary"][data-testid="stBaseButton-secondary"] {
            background-color: #FFF176 !important;
            color: #5D4037 !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            height: auto !important;
            border: 1px solid #FDD835 !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"][data-testid="stBaseButton-secondary"]:hover {
            background-color: #FFEE58 !important;
        }

        /* Prosegui al Triage — Orange */
        button[kind="primary"][data-testid="stBaseButton-primary"] {
            background-color: #EF6C00 !important;
            color: white !important;
            font-size: 16px !important;
            height: 50px !important;
            border: none !important;
        }
        button[kind="primary"][data-testid="stBaseButton-primary"]:hover {
            background-color: #E65100 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    # Modern mode: global CSS already handles button styling
    
    col_center = st.columns([1, 1, 1])[1]
    with col_center:
        if st.button("Prosegui al Triage", use_container_width=True, type="primary"):
            if not nome_input or not cognome_input:
                st.error("Dati identificativi obbligatori.")
            elif not eta_input:
                st.error("Età obbligatoria.")
            else:
                # Save state
                st.session_state.patient_data["eta"] = eta
                st.session_state.patient_data["scolarita_anni"] = scolarita
                st.session_state.patient_data["sesso"] = 1 if sesso_label == "Femmina" else 0
                
                st.session_state["saved_eta"] = eta_input
                st.session_state["saved_scolarita"] = scolarita
                st.session_state["saved_sesso_index"] = 0 if sesso_label == "Maschio" else 1
                st.session_state.patient_meta = {"nome": nome_input, "cognome": cognome_input} 

                st.session_state.step = 'intro_familiarita'
                st.rerun()


elif st.session_state.step == 'intro_familiarita':
    
    # Header for Familiarità
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #111827; font-weight: 700; font-family: 'Montserrat', sans-serif;">Anamnesi Familiare</h2>
        <p style="color: #6B7280; font-size: 14px;">Valutazione del rischio genetico e ambientale</p>
    </div>
    """, unsafe_allow_html=True)

    # Main Question Card
    # Using the same style as Wizard for consistency
    st.markdown(f"""
    <div style="
        background-color: white; 
        padding: 40px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); 
        margin-bottom: 20px;
        text-align: center;
    ">
        <p class="main-question">
            Il paziente ha parenti di primo grado con patologie psichiatriche e/o neurologiche diagnosticate?
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Big Buttons (NO / SI) - Manually styled for this step as columns logic is strict
    c_empty_l, c_no, c_si, c_empty_r = st.columns([0.5, 1, 1, 0.5])
    
    with c_no:
        if st.button("NO", use_container_width=True): 
            st.session_state.patient_data["familiarita_disturbi"] = 0
            st.session_state.step = 'wizard'
            st.rerun()

    with c_si:
        if st.button("SÌ", use_container_width=True, type="primary"): 
            st.session_state.patient_data["familiarita_disturbi"] = 1
            st.session_state.step = 'wizard'
            st.rerun()
            
    # Fix the button styling for this specific layout if needed, but type="primary" helps.

elif st.session_state.step == 'wizard':
    
    macro_idx = st.session_state.current_macro_index
    if macro_idx >= len(WIZARD_FLOW):
        st.session_state.step = 'scales'
        st.rerun()
        
    current_macro = WIZARD_FLOW[macro_idx]
    micro_idx = st.session_state.current_micro_index
    
    if micro_idx >= len(current_macro["micro_questions"]):
        st.session_state.current_macro_index += 1
        st.session_state.current_micro_index = 0
        reset_depth()
        st.rerun()
        
    question_data = current_macro["micro_questions"][micro_idx]
    depth = st.session_state.depth_level
    if depth >= len(question_data["levels"]): depth = len(question_data["levels"]) - 1
    current_text = question_data["levels"][depth]
    current_label = question_data["label"]
    
    # Progress
    total_steps = sum(len(m["micro_questions"]) for m in WIZARD_FLOW)
    past_steps = sum(len(m["micro_questions"]) for i, m in enumerate(WIZARD_FLOW) if i < macro_idx)
    current_step_abs = past_steps + micro_idx
    
    # UI Layout -> CARD
    st.markdown('<div class="card-container" style="padding: 40px;">', unsafe_allow_html=True)
    
    # Header Info
    st.markdown(f'<p class="section-label" style="text-align:left; color:#9CA3AF; border:none; padding:0; margin-bottom:5px;">SEZIONE {macro_idx + 1} / {len(WIZARD_FLOW)} - {current_macro["title"]}</p>', unsafe_allow_html=True)
    st.progress((current_step_abs + 1) / total_steps)
    st.markdown(f'<div style="height:20px"></div>', unsafe_allow_html=True)
    
    # QUESTION BLOCK
    st.markdown(f'<p class="technical-label">Sintomo: {current_label}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="main-question">{current_text}</p>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="height:40px"></div>', unsafe_allow_html=True)
    
    # --- ONION LOGIC BUTTONS ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # ASSENTE -> Next
        if st.button("ASSENTE", use_container_width=True):
            record_answer(question_data["feat"], 0)
            next_question()
            st.rerun()
            
    with c2:
        # DUBBIO -> Logic
        if st.button("DUBBIO / APPROFONDIRE", use_container_width=True):
            if depth < 2:
                # Onion Logic: Go deeper
                st.session_state.depth_level += 1
                st.rerun()
            else:
                # Max depth reached -> Record 0 (Prudence)
                st.toast("Limite approfondimento raggiunto. Registrato come assente.", icon="ℹ️")
                record_answer(question_data["feat"], 0)
                next_question()
                st.rerun()
                
    with c3:
        # PRESENTE -> Next
        if st.button("PRESENTE", use_container_width=True):
            record_answer(question_data["feat"], 1)
            next_question()
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.step == 'scales':

    # --- Calculator Reference Links ---
    CALC_LINKS = {
        "HAM_D":  ("MDCalc", "https://www.mdcalc.com/calc/10043/hamilton-depression-rating-scale-hamd"),
        "GAD_7":  ("MDCalc", "https://www.mdcalc.com/calc/1727/gad-7-general-anxiety-disorder-7"),
        "PANSS":  ("MDCalc", "https://www.mdcalc.com/calc/10414/positive-negative-syndrome-scale-panss-schizophrenia"),
        "UPDRS":  ("NeuroToolkit", "https://neurotoolkit.com/updrs/"),
        "MMSE":   ("AITASIT", "https://www.aitasit.org/calcs/Minimentstatexamnt/index.html"),
    }
    _link_color = "#0E766E" if USE_MODERN_UI else "#009688"

    def _calc_badge(key):
        """Return an HTML badge linking to the external calculator."""
        source, url = CALC_LINKS[key]
        return (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'style="display:inline-block; font-size:12px; font-weight:600; '
            f'color:{_link_color}; text-decoration:none; padding:3px 8px; '
            f'border:1px solid {_link_color}; border-radius:6px; '
            f'margin-bottom:6px; transition:all .2s ease;">'
            f'🔗 Calcola su {source} ↗</a>'
        )

    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">VALUTAZIONE PSICOMETRICA</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#6B7280; margin-bottom:20px;">Inserimento punteggi scale cliniche standardizzate.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_calc_badge("HAM_D"), unsafe_allow_html=True)
        st.session_state.patient_data["HAM_D"] = st.slider("HAM-D (Hamilton Depression Rating Scale)", 0, 50, 0)
        st.markdown(_calc_badge("GAD_7"), unsafe_allow_html=True)
        st.session_state.patient_data["GAD_7"] = st.slider("GAD-7 (General Anxiety Disorder)", 0, 21, 0)
        st.markdown(_calc_badge("PANSS"), unsafe_allow_html=True)
        st.session_state.patient_data["PANSS_total"] = st.slider("PANSS (Positive/Negative Syndrome Scale)", 30, 150, 30)
    with c2:
        st.markdown(_calc_badge("MMSE"), unsafe_allow_html=True)
        st.session_state.patient_data["MMSE"] = st.slider("MMSE (Mini-Mental State Exam)", 0, 30, 30)
        st.markdown(_calc_badge("UPDRS"), unsafe_allow_html=True)
        st.session_state.patient_data["UPDRS_III"] = st.slider("UPDRS-III (Motor Examination)", 0, 100, 0)

    st.markdown('</div>', unsafe_allow_html=True)

    # Scales CTA
    if not USE_MODERN_UI:
        st.markdown("""
        <style>
        div.stButton > button[kind="secondary"] {
            background-color: #EF6C00 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
    # Modern mode: global CSS handles button styling
    
    col_center = st.columns([1, 1, 1])[1]
    with col_center:
        if st.button("ELABORA DIAGNOSI", use_container_width=True, type="primary"):
            st.session_state.step = 'result'
            st.rerun()

elif st.session_state.step == 'result':
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">RISULTATO ELABORAZIONE</p>', unsafe_allow_html=True)
    
    with st.spinner("Analisi inferenziale in corso..."):
        num_cols = ["eta", "scolarita_anni", "HAM_D", "GAD_7", "PANSS_total", "UPDRS_III", "MMSE"]
        try:
            p_data = st.session_state.patient_data.copy()
            meta = st.session_state.patient_meta
            
            # --- Inference ---
            input_df = pd.DataFrame([{col: p_data[col] for col in num_cols}])
            scaled_vals = scaler.transform(input_df)[0]
            for i, col in enumerate(num_cols): p_data[col] = scaled_vals[i]
            
            input_vec = [p_data[feat] for feat in feature_names]
            input_tensor = torch.tensor([input_vec], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                probs = model(input_tensor).cpu().numpy()[0]
                
            results = sorted(zip(target_names, probs), key=lambda x: x[1], reverse=True)
            results = [(str(name), float(prob)) for name, prob in results]
            
            # --- Save to DB (Silent/Background) ---
            raw_data = st.session_state.patient_data.copy()
            
            if 'data_saved' not in st.session_state:
                save_patient_session(
                    meta["nome"], 
                    meta["cognome"], 
                    raw_data["eta"], 
                    raw_data["sesso"], 
                    raw_data, 
                    results
                )
                st.session_state.data_saved = True
                st.toast("Dati clinicizzati correttamente.", icon="✅")
            
            # RESULTS TABLE
            st.markdown(f"<h3 style='text-align:center;'>Paziente: {meta['nome']} {meta['cognome']}</h3>", unsafe_allow_html=True)
            st.markdown("<hr style='margin: 15px 0; border-top:1px solid #eee;'>", unsafe_allow_html=True)
            
            top_disease, top_prob = results[0]
            top_name = top_disease.replace('TARGET_', '')
            
            st.markdown(f"""
            <div style="text-align:center; padding: 20px; background-color: #E0F2F1; border-radius: 8px; border: 1px solid #B2DFDB; margin-bottom: 20px;">
                <div style="font-size: 14px; font-weight: 600; color: #00796B; text-transform: uppercase;">Ipotesi Diagnostica Principale</div>
                <div style="font-size: 32px; font-weight: 800; color: #004D40; margin: 5px 0;">{top_name}</div>
                <div style="font-size: 18px; color: #00695C;">Probabilità stimata: <strong>{(top_prob * 100):.2f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Diagnosi Differenziale")
            for name, p in results[1:4]:
                dis = name.replace("TARGET_", "")
                val = p * 100
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px;">
                    <span>{dis}</span>
                    <span style="font-weight:600;">{val:.1f}%</span>
                </div>
                <div style="background-color: #E0E0E0; border-radius: 4px; height: 8px; width: 100%;">
                    <div style="background-color: #757575; height: 8px; border-radius: 4px; width: {val}%;"></div>
                </div>
                <div style="height: 10px;"></div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Errore: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("NUOVA VALUTAZIONE", use_container_width=True):
        st.session_state.clear()
        st.rerun()
