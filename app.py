import io
import time
import requests
import base64
from datetime import datetime
from PIL import Image

import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
BACKEND_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{BACKEND_URL}/predict"

st.set_page_config(
    page_title="🌾 Zar3y Intelligence Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'recent_uploads' not in st.session_state:
    st.session_state.recent_uploads = []

# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=5)
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def get_severity_details(class_name):
    cls_lower = class_name.lower()
    if "healthy" in cls_lower:
        return "Healthy", "#3fb950"  # Green
    elif "early" in cls_lower or "spot" in cls_lower or "mold" in cls_lower or "blight" in cls_lower or "curl" in cls_lower or "rust" in cls_lower:
        return "Warning", "#e3b341"  # Amber
    else:
        return "Critical", "#f85149" # Red

def extract_crop_name(class_name):
    return class_name.split("___")[0].replace("_", " ")

def get_products_for_disease(class_name):
    cls_lower = class_name.lower()
    if "healthy" in cls_lower:
        return [
            {"name": "Organic Compost", "desc": "Maintain soil health", "icon": "🌱"},
            {"name": "Balanced NPK Fertilizer", "desc": "Routine nutrient feed", "icon": "🌾"}
        ]
    elif "blight" in cls_lower or "rust" in cls_lower or "spot" in cls_lower or "mold" in cls_lower:
        return [
            {"name": "Copper Fungicide", "desc": "Broad-spectrum fungal control", "icon": "🛡️"},
            {"name": "Neem Oil Extract", "desc": "Organic antifungal/pest deterrent", "icon": "💧"},
            {"name": "Sulfur Dust", "desc": "Prevents rust and powdery mildew", "icon": "🧪"}
        ]
    elif "curl" in cls_lower or "virus" in cls_lower or "mosaic" in cls_lower:
        return [
            {"name": "Insecticidal Soap", "desc": "Control disease-carrying aphids", "icon": "🐞"},
            {"name": "Reflective Mulch", "desc": "Repel whiteflies", "icon": "✨"}
        ]
    else:
        return [
            {"name": "Broad-Spectrum Treatment", "desc": "General disease control", "icon": "🛡️"}
        ]

is_online = check_backend()

# ==========================================
# GLOBAL STYLING (CSS)
# ==========================================
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
        background-color: #0a0e14;
        color: #e6edf3;
    }

    /* Main Container & Grid Dot Background */
    .stApp {
        background-color: #0a0e14;
        background-image: radial-gradient(#21262d 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* Remove Default Padding */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0e14; 
    }
    ::-webkit-scrollbar-thumb {
        background: #00c9a7; 
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3fb950; 
    }

    /* Header Section */
    .header-container {
        background-color: #161b22;
        border-left: 4px solid #00c9a7;
        border-bottom: 1px solid #21262d;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border-radius: 0 0 8px 8px;
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-left .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #3fb950, #00c9a7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-left .app-tagline {
        color: #8b949e;
        margin: 0;
        font-size: 0.95rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Scanning Line Animation */
    .scanning-line {
        position: absolute;
        bottom: 0;
        left: -100%;
        width: 50%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00c9a7, transparent);
        animation: scan 4s linear infinite;
    }
    @keyframes scan {
        0% { left: -50%; }
        100% { left: 100%; }
    }

    /* Status Badge */
    .status-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #0d1117;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #21262d;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    .status-online {
        background-color: #3fb950;
        box-shadow: 0 0 8px #3fb950;
        animation: pulse 2s infinite;
    }
    .status-offline {
        background-color: #f85149;
        box-shadow: 0 0 8px #f85149;
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* HTML Metric Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.2rem;
        display: flex;
        flex-direction: column;
        gap: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value {
        color: #e6edf3;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
    }
    .metric-data {
        color: #58a6ff;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-badge {
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        width: fit-content;
    }
    
    /* Progress Bar */
    .progress-track {
        background: #21262d;
        height: 4px;
        border-radius: 2px;
        width: 100%;
        margin-top: 4px;
        overflow: hidden;
    }
    .progress-fill {
        background: linear-gradient(90deg, #00c9a7, #3fb950);
        height: 100%;
    }

    /* Explainability Viewer */
    .viewer-title {
        display: inline-block;
        font-size: 1.4rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 1rem;
        position: relative;
    }
    .viewer-title::after {
        content: '';
        position: absolute;
        width: 0;
        height: 2px;
        bottom: -4px;
        left: 0;
        background-color: #00c9a7;
        transition: width 0.3s ease;
    }
    .viewer-title:hover::after {
        width: 100%;
    }
    .img-container {
        background: #161b22;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid #21262d;
        box-shadow: 0 0 15px rgba(0, 201, 167, 0.05);
        text-align: center;
    }
    .img-label {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        display: block;
    }
    
    /* Grad-CAM Legend */
    .heatmap-legend {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 1rem;
        font-size: 0.8rem;
        color: #8b949e;
    }
    .heatmap-bar {
        height: 8px;
        width: 60%;
        background: linear-gradient(90deg, #000080, #008000, #ffff00, #ff0000);
        border-radius: 4px;
    }

    /* Empty State */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
        color: #8b949e;
        animation: fadeIn 1s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .empty-state svg {
        width: 120px;
        height: 120px;
        fill: #21262d;
        margin-bottom: 2rem;
        transition: fill 0.3s ease;
    }
    .empty-state:hover svg {
        fill: #00c9a7;
    }
    .empty-state h2 {
        color: #e6edf3;
        margin-bottom: 0.5rem;
    }
    .bounce-arrow {
        font-size: 2rem;
        color: #00c9a7;
        margin-top: 2rem;
        animation: bounce 2s infinite;
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0) rotate(-90deg); }
        40% { transform: translateY(-15px) rotate(-90deg); }
        60% { transform: translateY(-7px) rotate(-90deg); }
    }

    /* Product Card */
    .product-card {
        background: #161b22;
        border: 1px solid #21262d;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .product-icon {
        font-size: 1.5rem;
        background: #21262d;
        padding: 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)




# ==========================================
# 1. MAIN APP HEADER
# ==========================================
status_class = "status-online" if is_online else "status-offline"
status_text = "SYSTEM ONLINE" if is_online else "API OFFLINE"
current_time = datetime.now().strftime("%H:%M:%S UTC")

st.markdown(f"""
<div class="header-container">
    <div class="scanning-line"></div>
    <div class="header-left">
        <h1 class="app-title">Zar3y Intelligence Platform</h1>
        <p class="app-tagline">Real-time crop pathology via Edge AI inference</p>
    </div>
    <div class="header-right" style="text-align: right;">
        <div class="status-badge">
            <div class="status-dot {status_class}"></div>
            {status_text} | {current_time}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not is_online:
    st.error("🚨 Inference Engine is disconnected. Please ensure the FastAPI backend is running on port 8000.")
    st.stop()


# ==========================================
# 2. UPLOAD CENTER SECTION
# ==========================================
st.markdown("<p style='font-variant:small-caps; color:#8b949e; letter-spacing:1px; font-weight:600;'>DATA INGESTION MODULE</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ==========================================
# EMPTY STATE
# ==========================================
if uploaded_file is None:
    st.markdown("""
    <div class="empty-state">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M17,8C8,10 5.9,16.17 3.82,21.34L5.71,22L6.66,19.7C7.14,19.87 7.64,20 8,20C19,20 22,3 22,3C21,5 14,5.25 9,6.25C4,7.25 2,11.5 2,13.5C2,15.5 3.75,17.25 3.75,17.25C7,8 17,8 17,8Z"/>
        </svg>
        <h2>Upload a leaf image to begin diagnosis</h2>
        <p>Supports JPG and PNG • Max 200MB • Instant AI analysis</p>
        <div class="bounce-arrow">↑</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ==========================================
# INFERENCE EXECUTION
# ==========================================
# Track recent uploads
file_mb = uploaded_file.size / (1024*1024)
if uploaded_file.name not in [x['name'] for x in st.session_state.recent_uploads]:
    st.session_state.recent_uploads.insert(0, {'name': uploaded_file.name, 'size': file_mb})
    if len(st.session_state.recent_uploads) > 3:
        st.session_state.recent_uploads.pop()

image = Image.open(uploaded_file)

with st.spinner("🔬 Running INT8 Inference..."):
    files = {"file": ("leaf.jpg", uploaded_file.getvalue(), "image/jpeg")}
    try:
        response = requests.post(PREDICT_URL, files=files, timeout=20)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Inference Exception: {e}")
        st.stop()

# Data Preparation
class_name = result["class_name"]
display_name = result["display_name"]
confidence = result["confidence"]
inference_ms = result["inference_ms"]
description = result["description"]
action = result["next_step"]
gradcam_base64 = result["grad_cam_base64"]
probs = result.get("probabilities", {})

severity, sev_color = get_severity_details(class_name)
crop_name = extract_crop_name(class_name)

short_diag = display_name.split("___")[1].replace("_", " ") if "___" in display_name else display_name
if len(short_diag) > 18:
    short_diag = short_diag[:15] + "..."


# ==========================================
# 3. AI METRICS ROW
# ==========================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🌿 Primary Diagnosis</div>
        <div class="metric-value" style="color: {sev_color};">{short_diag}</div>
        <div class="metric-title" style="font-size: 0.75rem; font-weight: 400; text-transform: none;">Crop: {crop_name}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    conf_pct = confidence * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🎯 Confidence</div>
        <div class="metric-value metric-data">{conf_pct:.1f}%</div>
        <div class="progress-track">
            <div class="progress-fill" style="width: {conf_pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">⚡ Inference Latency</div>
        <div class="metric-value metric-data">{inference_ms} ms</div>
        <div class="metric-title" style="font-size: 0.75rem; font-weight: 400; text-transform: none;">Hardware: CPU (TFLite)</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    bg_color = f"rgba({int(sev_color[1:3],16)}, {int(sev_color[3:5],16)}, {int(sev_color[5:7],16)}, 0.15)"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🚨 Severity Protocol</div>
        <div style="margin-top:auto; margin-bottom:auto;">
            <div class="metric-badge" style="background: {bg_color}; color: {sev_color}; border: 1px solid {sev_color};">
                {severity.upper()}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Layout for remaining sections
col_main, col_side = st.columns([1.2, 1], gap="large")

with col_main:
    # ==========================================
    # 4. EXPLAINABILITY VIEWER
    # ==========================================
    st.markdown("<div class='viewer-title'>🔍 Explainability Viewer <span title='Grad-CAM highlights the pixels the AI used to make its decision' style='cursor:help; font-size:1rem; color:#8b949e;'>ⓘ</span></div>", unsafe_allow_html=True)
    
    gradcam_img = Image.open(io.BytesIO(base64.b64decode(gradcam_base64)))
    
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("<div class='img-container'><span class='img-label'>Original Input</span>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with v2:
        st.markdown("<div class='img-container'><span class='img-label'>Grad-CAM Overlay</span>", unsafe_allow_html=True)
        st.image(gradcam_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("""
    <div class="heatmap-legend">
        <span>Low Attention</span>
        <div class="heatmap-bar"></div>
        <span>High Attention</span>
    </div>
    <br>
    """, unsafe_allow_html=True)

    # ==========================================
    # 7. PROBABILITY DISTRIBUTION CHART
    # ==========================================
    st.markdown("<div class='viewer-title' style='margin-top:1rem;'>📊 Probability Distribution</div>", unsafe_allow_html=True)
    st.markdown("<span style='color: #8b949e; font-size: 0.9rem;'>Top-5 predictions from MobileNetV3</span>", unsafe_allow_html=True)
    
    if probs:
        top_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True)[:5])
        df_probs = pd.DataFrame({
            "Disease": [d.split("___")[1].replace("_", " ") if "___" in d else d for d in top_probs.keys()],
            "Probability": [v * 100 for v in top_probs.values()]
        })
        
        fig = px.bar(
            df_probs, 
            x="Probability", 
            y="Disease", 
            orientation='h',
            color="Probability",
            color_continuous_scale=["#161b22", "#00c9a7", "#55efc4"],
            text_auto='.1f'
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", family="IBM Plex Mono"),
            xaxis=dict(showgrid=False, title="", visible=False),
            yaxis=dict(showgrid=False, title="", autorange="reversed", tickfont=dict(family="Outfit", size=13, color="#e6edf3")),
            coloraxis_showscale=False,
            height=250
        )
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False, texttemplate='%{x:.1f}%')
        st.plotly_chart(fig, use_container_width=True)

with col_side:
    # ==========================================
    # 5. DIAGNOSIS REPORT
    # ==========================================


    st.markdown("## 📑 Diagnosis Report")

    st.markdown(f"""
    ### 🌿 {display_name}

    **Description:**  
    {description}

    ---

    **Recommended Action:**  
    {action}
    """)



    # ==========================================
    # 6. AVAILABLE PRODUCTS SECTION
    # ==========================================
    st.markdown("<div class='viewer-title'>🛒 Treatment Products</div>", unsafe_allow_html=True)
    products = get_products_for_disease(class_name)
    for prod in products:
        st.markdown(f"""
        <div class="product-card">
            <div class="product-icon">{prod['icon']}</div>
            <div>
                <strong style="color:#e6edf3; font-size:0.95rem;">{prod['name']}</strong><br>
                <span style="color:#8b949e; font-size:0.85rem;">{prod['desc']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)


