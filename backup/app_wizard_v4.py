
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

# --- Configuration (Must be first) ---
st.set_page_config(
    page_title="Gestione Clinica - Psycho-Tensor",
    page_icon="PT",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 1. OSTEOSAY STYLE INJECTION (CSS) ---
st.markdown("""
<style>
    /* GLOBAL RESET & FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif !important;
        background-color: #F1F3F5 !important; /* Light Gray Background like OsteoEasy */
        color: #2C3E50 !important;
    }

    /* HIDE STREAMLIT ELEMENTS */
    header {visibility: hidden;}
    .stApp {margin-top: -30px;}
    footer {display: none;}

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
        font-size: 32px !important;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
        line-height: 1.3;
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
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 1px solid #F3F4F6;
    }

    /* BUTTONS - OsteoEasy Style */
    /* Primary Action Buttons (Orange CTA) */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    /* Specific override for "CTA" buttons vs "Option" buttons will be done via key/classes or inline */

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
    .header-logo {
        font-size: 20px;
        font-weight: 800;
        color: #009688;
        letter-spacing: -0.5px;
    }
    .header-status {
        font-size: 13px;
        color: #6B7280;
        font-weight: 500;
        background-color: #F3F4F6;
        padding: 6px 12px;
        border-radius: 20px;
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

# --- 2. Database Management ---
DB_FILE = "clinical_sessions.db"

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

def save_patient_session(nome, cognome, eta, sesso, features, diagnosis):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO patients (timestamp, nome, cognome, eta, sesso, features_json, diagnosis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now(), nome, cognome, eta, sesso, json.dumps(features), json.dumps(diagnosis)))
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
    "Moss", "Navy", "Nickel", "Obsidian", "Olive", "Onyx", "Opal", "Pearl",
    "Pine", "Quartz", "Raven", "Ruby", "Rust", "Sage", "Sand", "Scarlet",
    "Shadow", "Sienna", "Silver", "Slate", "Steel", "Stone", "Teal", "Timber",
    "Umber", "Violet"
]
_PSEUDO_ANIMALS = [
    "Badger", "Bear", "Bison", "Condor", "Crane", "Deer", "Dolphin", "Eagle",
    "Elk", "Falcon", "Finch", "Fox", "Gazelle", "Griffin", "Hare", "Hawk",
    "Heron", "Ibex", "Jaguar", "Jay", "Kestrel", "Lark", "Leopard", "Lynx",
    "Mantis", "Marten", "Merlin", "Moose", "Newt", "Orca", "Osprey", "Otter",
    "Owl", "Panther", "Puma", "Quail", "Raven", "Robin", "Seal", "Sparrow",
    "Stork", "Swan", "Swift", "Tiger", "Viper", "Wren", "Wolf", "Wombat",
    "Yak", "Zebra"
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
    
    for _ in range(500):  # Safety limit
        color = random.choice(_PSEUDO_COLORS)
        animal = random.choice(_PSEUDO_ANIMALS)
        if (color, animal) not in existing:
            return color, animal
    # Fallback: add numeric suffix
    return random.choice(_PSEUDO_COLORS), f"{random.choice(_PSEUDO_ANIMALS)}{random.randint(10,99)}"


# --- 3. Model & Resources ---
MODEL_FILE = "psycho_global_model.pth"
SCALER_FILE = "scaler_global.pkl"
FEATURE_NAMES_FILE = "feature_names_global.json"
TARGET_NAMES_FILE = "target_names.json"

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

# --- 4. Sidebar: ADMIN GATE ---
with st.sidebar:
    # --- BRANDING ---
    st.markdown("""
        <div style="text-align: left; margin-bottom: 25px; margin-top: 10px;">
            <div style="font-size: 26px; font-weight: 800; color: #009688; letter-spacing: -0.5px;">Clinical<span style="color:#2C3E50">Support</span></div>
            <div style="font-size: 13px; font-weight: 500; color: #9CA3AF; margin-top:4px;">Software Clinico Ospedaliero</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation mock to look like OsteoEasy
    st.markdown("""
    <div style="display:flex; flex-direction:column; gap:10px; margin-bottom: 30px;">
        <div style="color:#009688; font-weight:600; font-size:14px; padding:8px 12px; background:#E0F2F1; border-radius:6px;">Nuova Visita</div>
        <div style="color:#6B7280; font-weight:500; font-size:14px; padding:8px 12px;">Pazienti</div>
        <div style="color:#6B7280; font-weight:500; font-size:14px; padding:8px 12px;">Agenda</div>
        <div style="color:#6B7280; font-weight:500; font-size:14px; padding:8px 12px;">Impostazioni</div>
    </div>
    <hr style="border:0; border-top:1px solid #E0E0E0; margin:20px 0;">
    """, unsafe_allow_html=True)

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
                    # Double check logic usually here, kept simple for v4
                    reset_database()
                    st.warning("Database formattato.")
                    time.sleep(1)
                    st.rerun()
                        
            except Exception as e:
                st.error(f"Errore: {e}")
        elif admin_password:
            st.error("Credenziali non valide.")


# --- 5. Main Content Area ---

# Header Bar (OsteoEasy style: simple, clean top bar)
st.markdown("""
<style>
    /* COMPACT HEADER CSS */
    .block-container {
        padding-top: 1rem !important; /* Reduce default top padding */
        padding-bottom: 0rem !important;
    }
    div[data-testid="stImage"] {
        margin-top: -20px; /* Pull image up */
        margin-bottom: -20px; /* Pull content below up */
    }
    img {
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1]) 
with c2:
    st.image("NVD.png", use_container_width=True)

st.markdown("""
<div class="main-header-container" style="display:none;"></div>
""", unsafe_allow_html=True)


# --- 6. WIZARD FLOW (Data) ---
WIZARD_FLOW = [
    {"id": "macro_psych", "title": "Sfera Psichica", "question": "Variazioni nel tono dell'umore o nei livelli di ansia rilevati.", "micro_questions": [
        {"feat": ["umore_depresso"], "levels": ["Umore deflesso", "Sentimenti di disperazione", "Bassa energia emotiva"]},
        {"feat": ["anedonia"], "levels": ["Perdita di interesse", "Anedonia", "Appiattimento affettivo"]},
        {"feat": ["ansia_eccessiva"], "levels": ["Preoccupazione costante", "Rimuginio", "Tensione somatica"]},
        {"feat": ["attacchi_di_panico", "agitazione_psicomotoria"], "levels": ["Attacchi di panico", "Paura di perdere il controllo", "Palpitazioni sine causa"]},
        {"feat": ["insonnia"], "levels": ["Insonnia iniziale/terminale", "Sonno disturbato", "Rimuginio notturno"]},
        {"feat": ["isolamento_sociale"], "levels": ["Ritiro sociale", "Evitamento", "Esaurimento sociale"]},
        {"feat": ["fatica_astenia"], "levels": ["Astenia", "Affaticabilità", "Mancanza di forze"]}
    ]},
    {"id": "macro_trauma", "title": "Pensieri Intrusivi", "question": "Presenza di ideazione ossessiva o riattivazione traumatica.", "micro_questions": [
        {"feat": ["ossessioni"], "levels": ["Pensieri intrusivi", "Preoccupazioni eccessive", "Ideazione fissa"]},
        {"feat": ["compulsioni_rituali"], "levels": ["Rituali (washing/checking)", "Rigidità comportamentale", "Ripetitività"]},
        {"feat": ["flashback"], "levels": ["Flashback traumatici", "Riattivazione sensoriale", "Vividità mnestica"]},
        {"feat": ["evitamento_trauma"], "levels": ["Evitamento fobico", "Evitamento trigger", "Soppressione del pensiero"]},
        {"feat": ["ipervigilanza"], "levels": ["Ipervigilanza", "Scanning ambientale", "Stato di allerta"]}
    ]},
    {"id": "macro_psychosis", "title": "Percezione", "question": "Alterazioni della percezione sensoriale o del contenuto del pensiero.", "micro_questions": [
        {"feat": ["allucinazioni_uditive"], "levels": ["Allucinazioni uditive", "Percezioni senza oggetto", "Voci dialoganti"]},
        {"feat": ["allucinazioni_visive"], "levels": ["Allucinazioni visive", "Distorsioni percettive", "Visioni periferiche"]},
        {"feat": ["deliri"], "levels": ["Ideazione persecutoria", "Riferimento", "Interpretazione delirante"]},
        {"feat": ["euforia_grandiosita"], "levels": ["Ideazione grandiosa", "Iperattività finalizzata", "Progettualità irrealistica"]},
        {"feat": ["disorganizzazione_pensiero"], "levels": ["Deragliamento", "Tangenzialità", "Confusione mentale"]}
    ]},
    {"id": "macro_eating", "title": "Comportamento Alimentare", "question": "Condotte alimentari disfunzionali o disregolazione degli impulsi.", "micro_questions": [
        {"feat": ["restrizione_alimentare_severa"], "levels": ["Restrizione calorica", "Digiuno", "Controllo del peso"]},
        {"feat": ["abbuffate"], "levels": ["Abbuffate oggettive", "Perdita di controllo", "Fame compulsiva"]},
        {"feat": ["vomito_autoindotto"], "levels": ["Condotte di eliminazione", "Uso di lassativi", "Compenso post-prandiale"]},
        {"feat": ["autolesionismo"], "levels": ["Autolesionismo", "Tagli/Bruciature", "Gesti impulsivi auto-diretti"]},
        {"feat": ["craving_sostanze"], "levels": ["Craving da sostanze", "Uso di alcol/farmaci", "Comportamento di ricerca"]}
    ]},
    {"id": "macro_behavior", "title": "Temperamento", "question": "Disregolazione emotiva e tratti di personalità.", "micro_questions": [
        {"feat": ["instabilita_affettiva"], "levels": ["Labilità emotiva", "Disregolazione affettiva", "Oscillazioni dell'umore"]},
        {"feat": ["paura_abbandono"], "levels": ["Angoscia abbandonica", "Dipendenza affettiva", "Ricerca di rassicurazione"]},
        {"feat": ["iperattivita_impulsivita"], "levels": ["Impulsività", "Acting out", "Scarso controllo inibitorio"]},
        {"feat": ["apatia"], "levels": ["Avolizione", "Indifferenza", "Inerzia motoria"]},
        {"feat": ["ridotto_bisogno_sonno"], "levels": ["Ipoesigenza di sonno", "Iperattivazione notturna", "Energia soggettiva aumentata"]}
    ]},
    {"id": "macro_neuro", "title": "Neurologia Generale", "question": "Segni motori e alterazioni del tono muscolare.", "micro_questions": [
        {"feat": ["tremore_a_riposo"], "levels": ["Tremore a riposo", "Tremore posturale", "Microvibrazioni involontarie"]},
        {"feat": ["bradicinesia"], "levels": ["Bradicinesia", "Rallentamento motorio", "Impaccio motorio"]},
        {"feat": ["rigidita_muscolare"], "levels": ["Rigidità muscolare", "Ipertono plastico", "Resistenza al movimento passivo"]},
        {"feat": ["instabilita_posturale"], "levels": ["Instabilità posturale", "Disturbo dell'equilibrio", "Cadute ricorrenti"]},
        {"feat": ["fascicolazioni"], "levels": ["Fascicolazioni", "Guizzi muscolari involontari", "Contrazioni sottocutanee"]},
        {"feat": ["convulsioni"], "levels": ["Crisi convulsive", "Perdita di coscienza", "Episodi critici"]}
    ]},
    {"id": "macro_cog", "title": "Funzioni Cognitive", "question": "Deficit mnesici, attentivi e del linguaggio.", "micro_questions": [
        {"feat": ["perdita_memoria_breve_termine"], "levels": ["Deficit memoria a breve termine", "Amnesia anterograda", "Riduzione span mnesico"]},
        {"feat": ["disattenzione_cronica"], "levels": ["Disattenzione cronica", "Deficit attentivo", "Ridotta concentrazione"]},
        {"feat": ["difficolta_linguaggio_afasia"], "levels": ["Afasia/Anomia", "Difficoltà di accesso lessicale", "Parafasie"]},
        {"feat": ["aprassia"], "levels": ["Aprassia ideomotoria", "Deficit prassico", "Impaccio gestuale"]},
        {"feat": ["disorientamento_spazio_temporale"], "levels": ["Disorientamento S/T", "Confusione topografica", "Perdita dei riferimenti temporali"]},
        {"feat": ["parestesie_formicolii"], "levels": ["Parestesie", "Formicolii/Intorpidimento", "Sensibilità alterata"]}
    ]}
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

/* NO Button (Outline Gray) */
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

/* MAYBE Button (Soft Blue/Gray) */
div[data-testid="column"]:nth-of-type(2) button {
    background-color: #EFF6FF !important; 
    color: #3B82F6 !important;
    border: 1px solid #BFDBFE !important;
    box-shadow: none !important;
}
div[data-testid="column"]:nth-of-type(2) button:hover {
    background-color: #DBEAFE !important;
}

/* YES Button (Primary Teal) */
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
    
    # --- Pre-fill pseudonym BEFORE widgets (Streamlit requires this order) ---
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
    
    with c1:
        st.markdown('<label style="font-size:12px; font-weight:600; color:#4B5563; margin-bottom:4px; display:block;">NOME</label>', unsafe_allow_html=True)
        nome_input = st.text_input("Nome", key="input_nome", value=st.session_state.get("input_nome", ""), label_visibility="collapsed")
    with c2:
        st.markdown('<label style="font-size:12px; font-weight:600; color:#4B5563; margin-bottom:4px; display:block;">COGNOME</label>', unsafe_allow_html=True)
        cognome_input = st.text_input("Cognome", key="input_cognome", value=st.session_state.get("input_cognome", ""), label_visibility="collapsed")
    with c3:
        st.markdown('<div style="height: 25px;"></div>', unsafe_allow_html=True) 
        if st.button("Genera ID Anonimo", use_container_width=True):
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
    if eta_input.strip() and eta_input.strip().isdigit():
        eta = int(eta_input.strip())

    st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
    
    # --- Gender only (Familiarità removed here) ---
    c_gender_container = st.columns([1])[0]
    with c_gender_container:
         st.markdown('<label style="font-size:12px; font-weight:600; color:#4B5563; margin-bottom:4px; display:block;">SESSO BIOLOGICO</label>', unsafe_allow_html=True)
         sesso_label = st.radio("Sesso", ["Maschio", "Femmina"], index=st.session_state.get("saved_sesso_index", 0), horizontal=True, label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True) # End Card 2
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # --- CTA Button (Orange) ---
    col_center = st.columns([1, 1, 1])[1]
    with col_center:
        if st.button("Prosegui al Triage ➡️", use_container_width=True, type="primary"):
            if not nome_input or not cognome_input:
                st.error("Dati identificativi obbligatori.")
            elif not eta_input:
                st.error("Età obbligatoria.")
            else:
                # Save state
                st.session_state.patient_data["eta"] = eta
                st.session_state.patient_data["scolarita_anni"] = scolarita
                st.session_state.patient_data["sesso"] = 1 if sesso_label == "Femmina" else 0
                
                # Verify logic: save inputs for "back" navigation if implemented later
                st.session_state["saved_eta"] = eta_input
                st.session_state["saved_scolarita"] = scolarita
                st.session_state["saved_sesso_index"] = 0 if sesso_label == "Maschio" else 1
                st.session_state.patient_meta = {"nome": nome_input, "cognome": cognome_input} # Save meta here

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
        <p style="font-size: 20px; font-weight: 600; color: #1F2937; margin-bottom: 30px;">
            Il paziente ha parenti di primo grado con patologie psichiatriche e/o neurologiche diagnosticate?
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Big Buttons (NO / SI)
    c_no, c_si = st.columns([1, 1])
    
    with c_no:
        if st.button("NO", use_container_width=True): # Will use "nth-of-type(1)" style (Gray) if managed by column index, but let's be safe.
            # Actually our CSS styles buttons by nth-of-type in columns. 
            # If we have 2 columns, col 1 is type 1 (Gray style), col 2 is type 2 (Blue style).
            # Wait, the user requested: NO (Gray), SI (Teal).
            # Our current CSS: col 1 -> Gray, col 2 -> Blue, col 3 -> Teal.
            # We need to ensure SI gets the Teal style.
            # Workaround: Use 3 columns hidden or adjust CSS. 
            # Or better: Rely on Streamlit's type="primary" for SI to force some styling, but our CSS overrides it likely.
            # Let's trust the columns. 
            # To get Teal on the right button (which is col 2 here), we might need to adjust CSS or use a 3-column trick: [1, 0.1, 1] empty middle?
            # No, that's hacky.
            
            st.session_state.patient_data["familiarita_disturbi"] = 0
            st.session_state.step = 'wizard'
            st.rerun()

    with c_si:
        # To force Teal color, I might need inline style or a trick.
        # But wait, looking at CSS lines 508: "div[data-testid="column"]:nth-of-type(3) button" is Teal.
        # Here we only have 2 columns.
        # If I want the second button to be Teal, I should update the CSS or layout.
        # I'll stick to logical flow now. If style is "Soft Blue" for Yes, it's acceptable, but user asked for Teal.
        # Let's restart the columns to ensure specific targeting? No.
        
        if st.button("SÌ", use_container_width=True, type="primary"): 
            st.session_state.patient_data["familiarita_disturbi"] = 1
            st.session_state.step = 'wizard'
            st.rerun()

    # CSS Patch for this step to make the 2nd column button Teal (Simulating a YES action)
    st.markdown("""
    <style>
    /* Force 2nd column button to be Teal in this specific step */
    div[data-testid="column"]:nth-of-type(2) button {
        background-color: #009688 !important; 
        color: #FFFFFF !important;
        border: 1px solid #009688 !important;
    }
    div[data-testid="column"]:nth-of-type(2) button:hover {
        background-color: #00796B !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
    
    # Progress
    total_steps = sum(len(m["micro_questions"]) for m in WIZARD_FLOW)
    past_steps = sum(len(m["micro_questions"]) for i, m in enumerate(WIZARD_FLOW) if i < macro_idx)
    current_step_abs = past_steps + micro_idx
    
    # UI Layout -> CARD
    st.markdown('<div class="card-container" style="padding: 40px;">', unsafe_allow_html=True)
    
    st.markdown(f'<p class="section-label" style="text-align:left; color:#9CA3AF;">SEZIONE {macro_idx + 1} / {len(WIZARD_FLOW)}</p>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="margin:0 0 10px 0; font-size:24px;">{current_macro["title"]}</h3>', unsafe_allow_html=True)
    st.progress((current_step_abs + 1) / total_steps)
    
    st.markdown(f'<div style="height:30px"></div>', unsafe_allow_html=True)
    
    st.markdown(f'<p style="font-size:22px; font-weight:600; color:#374151; line-height:1.4;">{current_text}</p>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="height:40px"></div>', unsafe_allow_html=True)
    
    # Buttons
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("ASSENTE", use_container_width=True):
            record_answer(question_data["feat"], 0)
            next_question()
            st.rerun()
            
    with c2:
        if st.button("DUBBIO / APPROFONDIRE", use_container_width=True):
            if depth < 2:
                st.session_state.depth_level += 1
                st.rerun()
            else:
                st.toast("Registrato come assente.")
                record_answer(question_data["feat"], 0)
                next_question()
                st.rerun()
                
    with c3:
        if st.button("PRESENTE", use_container_width=True):
            record_answer(question_data["feat"], 1)
            next_question()
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.step == 'scales':
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">VALUTAZIONE PSICOMETRICA</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#6B7280; margin-bottom:20px;">Inserimento punteggi scale cliniche standardizzate.</p>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.patient_data["HAM_D"] = st.slider("HAM-D (Hamilton Depression Rating Scale)", 0, 50, 0)
        st.session_state.patient_data["GAD_7"] = st.slider("GAD-7 (General Anxiety Disorder)", 0, 21, 0)
        st.session_state.patient_data["PANSS_total"] = st.slider("PANSS (Positive/Negative Syndrome Scale)", 30, 150, 30)
    with c2:
        st.session_state.patient_data["MMSE"] = st.slider("MMSE (Mini-Mental State Exam)", 0, 30, 30)
        st.session_state.patient_data["UPDRS_III"] = st.slider("UPDRS-III (Motor Examination)", 0, 100, 0)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Orange CTA
    st.markdown("""
    <style>
    div.stButton > button[kind="secondary"] {
        background-color: #EF6C00 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_center = st.columns([1, 1, 1])[1]
    with col_center:
        if st.button("ELABORA DIAGNOSI", use_container_width=True):
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
                st.toast("Dati clinicizzati correttamente.")
            
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
