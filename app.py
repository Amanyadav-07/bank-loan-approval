import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LOANIQ // CREDIT TERMINAL",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# BLOOMBERG TERMINAL DESIGN SYSTEM
# Philosophy: Pure black. Hard orange. Dense mono. CRT texture.
# Like a Bloomberg Terminal ate a 1980s mainframe and loved it.
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=VT323&display=swap');

/* ── DESIGN TOKENS ─────────────────────────────── */
:root {
    --black:      #000000;
    --void:       #030303;
    --panel:      #070707;
    --panel2:     #0c0c0c;
    --panel3:     #101010;
    --line:       #181818;
    --line2:      #222222;
    --orange:     #ff6600;
    --orange-dk:  #cc4400;
    --orange-lt:  #ff8844;
    --green:      #00ff41;
    --red:        #ff1a1a;
    --yellow:     #ffcc00;
    --ice:        #e2e2e2;
    --muted:      #777777;
    --dim:        #3a3a3a;
    --mono:       'Share Tech Mono', 'Courier New', monospace;
    --display:    'VT323', 'Share Tech Mono', monospace;
    --sans:       'Rajdhani', sans-serif;
}

/* ── KEYFRAMES ─────────────────────────────────── */
@keyframes blink {
    0%, 49% { opacity: 1; }
    50%, 100% { opacity: 0; }
}
@keyframes crtScan {
    0%   { transform: translateY(-10%); }
    100% { transform: translateY(110vh); }
}
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes barFill {
    from { width: 0%; }
}
@keyframes borderFlash {
    0%, 100% { border-color: var(--orange); }
    50%      { border-color: var(--orange-dk); }
}
@keyframes glitchText {
    0%,94%,100% { clip-path: none; transform: none; }
    95% { clip-path: polygon(0 20%, 100% 20%, 100% 21%, 0 21%);
          transform: translate(-2px, 0); }
    96% { clip-path: polygon(0 60%, 100% 60%, 100% 62%, 0 62%);
          transform: translate(2px, 0); }
    97% { clip-path: polygon(0 40%, 100% 40%, 100% 41%, 0 41%);
          transform: translate(-1px, 0); }
    98% { clip-path: none; transform: none; }
}
@keyframes pulseOrange {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,102,0,0); }
    50%      { box-shadow: 0 0 12px 2px rgba(255,102,0,0.15); }
}

/* ── GLOBAL BASE ───────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp {
    background-color: var(--black) !important;
    font-family: var(--mono) !important;
    color: var(--ice) !important;
}

/* CRT scanlines */
[data-testid="stApp"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 9999;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        transparent 0px, transparent 3px,
        rgba(0,0,0,0.12) 3px, rgba(0,0,0,0.12) 4px
    );
}
/* Moving CRT beam */
[data-testid="stApp"]::after {
    content: '';
    position: fixed;
    left: 0; right: 0; height: 200px;
    background: linear-gradient(
        180deg,
        transparent 0%,
        rgba(255, 102, 0, 0.012) 45%,
        rgba(255, 102, 0, 0.018) 50%,
        rgba(255, 102, 0, 0.012) 55%,
        transparent 100%
    );
    animation: crtScan 10s linear infinite;
    pointer-events: none;
    z-index: 9998;
}

/* ── MAIN CONTAINER ────────────────────────────── */
.block-container {
    padding: 0 1.8rem 4rem !important;
    max-width: 1600px !important;
}

/* ── TICKER BAR ────────────────────────────────── */
.bb-ticker-outer {
    background: var(--orange);
    height: 26px;
    overflow: hidden;
    display: flex;
    align-items: center;
    margin: 0 -1.8rem 1.6rem;
    border-bottom: 1px solid var(--orange-dk);
    animation: fadeIn 0.3s ease both;
}
.bb-ticker-track {
    display: inline-flex;
    gap: 0;
    white-space: nowrap;
    animation: ticker 28s linear infinite;
}
.bb-ticker-item {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    color: #000;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 0 32px;
}
.bb-ticker-item::before {
    content: '▸ ';
}

/* ── SIDEBAR ───────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--void) !important;
    border-right: 1px solid var(--line2) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 14px 14px !important;
}
[data-testid="stSidebar"] > div:first-child::before {
    content: '';
    display: block;
    height: 3px;
    background: var(--orange);
    margin: 0 -14px 20px;
}

/* Sidebar nav */
div[data-testid="stSidebar"] .stRadio > div {
    gap: 1px !important;
}
div[data-testid="stSidebar"] .stRadio label {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 9px 12px !important;
    border-radius: 0 !important;
    border: 1px solid transparent !important;
    border-left: 2px solid transparent !important;
    background: transparent !important;
    transition: color 0.15s, background 0.15s, border-color 0.15s !important;
    cursor: pointer !important;
}
div[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--orange) !important;
    background: rgba(255,102,0,0.05) !important;
    border-left-color: var(--orange) !important;
}

/* ── HEADINGS ──────────────────────────────────── */
h1, h2, h3 {
    font-family: var(--mono) !important;
    color: var(--ice) !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}
h1 { font-size: 1.3rem !important; font-weight: 400 !important; }
h2 { font-size: 1.1rem !important; font-weight: 400 !important; }
h3 { font-size: 0.95rem !important; font-weight: 400 !important; color: var(--muted) !important; }

/* ── NUMBER INPUT — 7-LAYER COMPLETE DARK FIX ───── */
[data-testid="stNumberInput"],
[data-testid="stNumberInput"] *,
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stNumberInput"] > div > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] input:focus,
[data-testid="stNumberInput"] input:hover,
[data-testid="stNumberInput"] input:active {
    background-color: var(--panel2) !important;
    background:       var(--panel2) !important;
    -webkit-appearance: none !important;
}
[data-testid="stNumberInput"] input:-webkit-autofill,
[data-testid="stNumberInput"] input:-webkit-autofill:hover,
[data-testid="stNumberInput"] input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0 1000px var(--panel2) inset !important;
    -webkit-text-fill-color: var(--orange) !important;
}
[data-testid="stNumberInput"] {
    background: var(--panel2) !important;
    border: 1px solid var(--line2) !important;
    border-radius: 0 !important;
    padding: 10px 13px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stNumberInput"]:focus-within {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 1px var(--orange) !important;
}
[data-testid="stNumberInput"] label {
    font-family: var(--mono) !important;
    font-size: 9px !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; color: var(--muted) !important;
}
[data-testid="stNumberInput"] input {
    color: var(--orange) !important;
    font-family: var(--mono) !important;
    font-size: 15px !important;
    border: 1px solid var(--line2) !important;
    border-radius: 0 !important;
    padding: 6px 10px !important;
    outline: none !important; box-shadow: none !important;
    caret-color: var(--orange) !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--orange) !important;
}
[data-testid="stNumberInput"] > div {
    border: 1px solid var(--line2) !important;
    border-radius: 0 !important; overflow: hidden !important;
}
[data-testid="stNumberInput"] button {
    background: var(--panel3) !important; color: var(--orange) !important;
    border: none !important; border-left: 1px solid var(--line2) !important;
    border-radius: 0 !important; font-size: 14px !important;
    transition: background 0.12s !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(255,102,0,0.12) !important;
}

/* ── SLIDER ────────────────────────────────────── */
[data-testid="stSlider"] {
    background: var(--panel2) !important;
    border: 1px solid var(--line2) !important;
    border-radius: 0 !important;
    padding: 10px 13px 8px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stSlider"]:hover { border-color: var(--orange-dk) !important; }
[data-testid="stSlider"] label {
    font-family: var(--mono) !important; font-size: 9px !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    font-family: var(--mono) !important; font-size: 9px !important;
    color: var(--dim) !important;
}

/* ── SELECTBOX ─────────────────────────────────── */
[data-testid="stSelectbox"] {
    background: var(--panel2) !important;
    border: 1px solid var(--line2) !important;
    border-radius: 0 !important;
    padding: 10px 13px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stSelectbox"]:hover { border-color: var(--orange-dk) !important; }
[data-testid="stSelectbox"] label {
    font-family: var(--mono) !important; font-size: 9px !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--panel3) !important;
    border: 1px solid var(--line2) !important;
    border-radius: 0 !important;
    color: var(--orange) !important;
    font-family: var(--mono) !important; font-size: 13px !important;
}

/* ── METRIC CARDS ──────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--panel2) !important;
    border: 1px solid var(--line2) !important;
    border-top: 2px solid var(--orange) !important;
    border-radius: 0 !important;
    padding: 14px 16px !important;
    transition: background 0.18s, border-color 0.18s !important;
    animation: fadeSlideUp 0.45s ease both !important;
}
[data-testid="stMetric"]:hover {
    background: rgba(255,102,0,0.04) !important;
    border-color: var(--orange) !important;
    animation: pulseOrange 1.8s ease infinite !important;
}
[data-testid="stMetricLabel"] p {
    font-family: var(--mono) !important; font-size: 9px !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--display) !important;
    font-size: 2rem !important; color: var(--orange) !important;
    letter-spacing: 1px !important;
}

/* ── BUTTON ────────────────────────────────────── */
.stButton > button {
    width: 100% !important;
    background: var(--black) !important;
    color: var(--orange) !important;
    border: 1px solid var(--orange) !important;
    border-radius: 0 !important;
    padding: 17px !important;
    font-family: var(--mono) !important;
    font-size: 11px !important; letter-spacing: 5px !important;
    text-transform: uppercase !important;
    transition: background 0.2s, box-shadow 0.2s, color 0.2s !important;
    position: relative !important; overflow: hidden !important;
}
.stButton > button:hover {
    background: rgba(255,102,0,0.1) !important;
    color: var(--orange-lt) !important;
    box-shadow: 0 0 24px rgba(255,102,0,0.18),
                inset 0 0 32px rgba(255,102,0,0.04) !important;
    animation: borderFlash 1.2s ease infinite !important;
}
.stButton > button:active {
    background: rgba(255,102,0,0.18) !important;
}

/* ── DIVIDER ───────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--line2) !important;
    margin: 20px 0 !important;
}

/* ── INFO ──────────────────────────────────────── */
[data-testid="stInfo"] {
    background: rgba(255,102,0,0.04) !important;
    border: 1px solid rgba(255,102,0,0.2) !important;
    border-left: 2px solid var(--orange) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important; font-size: 11px !important;
    color: rgba(226,226,226,0.75) !important;
}

/* ── DATAFRAME ─────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line2) !important;
    border-radius: 0 !important; overflow: hidden !important;
}

/* ── CAPTION ───────────────────────────────────── */
[data-testid="stCaptionContainer"] p, .stCaption {
    font-family: var(--mono) !important; font-size: 10px !important;
    color: var(--dim) !important; letter-spacing: 0.5px !important;
}

/* ── SCROLLBAR ─────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--black); }
::-webkit-scrollbar-thumb { background: var(--orange-dk); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--orange); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# CHART HELPER — no dict spreading, zero duplicate key risk
# ══════════════════════════════════════════════════════════════════════
def apply_chart(fig, height=360, title="", xtitle="", ytitle="", shapes=None):
    axis_cfg = dict(
        gridcolor="rgba(255,102,0,0.07)",
        zerolinecolor="rgba(255,102,0,0.22)",
        linecolor="#1e1e1e",
        tickfont=dict(
            family="Share Tech Mono, Courier New, monospace",
            size=10,
            color="#505050",
        ),
        showgrid=True,
    )
    layout_cfg = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#060606",
        font=dict(
            color="#555",
            family="Share Tech Mono, Courier New, monospace",
            size=11,
        ),
        title=dict(
            text=title,
            font=dict(
                color="#cccccc",
                family="Share Tech Mono, Courier New, monospace",
                size=13,
            ),
            x=0, xanchor="left",
            pad=dict(b=10),
        ),
        xaxis=dict(
            **axis_cfg,
            title=dict(
                text=xtitle,
                font=dict(color="#444", size=10),
            ),
        ),
        yaxis=dict(
            **axis_cfg,
            title=dict(
                text=ytitle,
                font=dict(color="#444", size=10),
            ),
        ),
        height=height,
        margin=dict(t=50, b=36, l=14, r=14),
        legend=dict(
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="#1e1e1e",
            borderwidth=1,
            font=dict(family="Share Tech Mono, Courier New, monospace", size=10, color="#777"),
        ),
        hoverlabel=dict(
            bgcolor="#0d0d0d",
            bordercolor="#ff6600",
            font=dict(
                family="Share Tech Mono, Courier New, monospace",
                size=11,
                color="#ff6600",
            ),
        ),
    )
    if shapes:
        layout_cfg["shapes"] = shapes
    fig.update_layout(**layout_cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════
# SAFE SHAP — handles list output (TreeExplainer) and 2D arrays
# ══════════════════════════════════════════════════════════════════════
def safe_shap(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.array(sv)
    if sv.ndim == 2:
        sv = sv[0]
    return sv


# ══════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS — native xgb.Booster, zero sklearn dependency
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    preprocessor  = joblib.load("preprocessor.pkl")
    booster       = xgb.Booster()
    booster.load_model("xgb_model.json")
    explainer     = joblib.load("shap_explainer.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return preprocessor, booster, explainer, feature_names

preprocessor, booster, explainer, feature_names = load_artifacts()


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv("Data/loan_data_cleaned.csv")

@st.cache_data
def load_sql_data():
    return (
        pd.read_csv("Data/sql_q1_education.csv"),
        pd.read_csv("Data/sql_q2_loan_purpose.csv"),
        pd.read_csv("Data/sql_q3_home_ownership.csv"),
        pd.read_csv("Data/sql_q4_income_band.csv"),
        pd.read_csv("Data/sql_q5_defaults.csv"),
    )

df = load_data()
q1, q2, q3, q4, q5 = load_sql_data()


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:22px; animation:fadeIn 0.5s ease both;">
        <div style="font-family:'VT323','Share Tech Mono',monospace;
                    font-size:32px; color:#ff6600; letter-spacing:4px; line-height:1;">
            LOAN<span style="color:#e2e2e2;">IQ</span>
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:8px;
                    color:#3a3a3a; letter-spacing:3px; text-transform:uppercase;
                    margin-top:4px;">
            CREDIT INTELLIGENCE TERMINAL
        </div>
        <div style="height:1px; background:rgba(255,102,0,0.35); margin-top:12px;"></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["▸  PREDICT APPROVAL", "▸  EDA INSIGHTS", "▸  SQL ANALYSIS"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="margin-top:28px; padding-top:16px;
                border-top:1px solid #151515;
                animation:fadeIn 0.8s ease both;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:8.5px;
                    color:#3a3a3a; letter-spacing:1.5px;
                    text-transform:uppercase; line-height:2.7;">
            ALGO &nbsp;&nbsp;&nbsp;&nbsp;:: XGBOOST<br>
            RECORDS &nbsp;:: 44,535<br>
            ROC-AUC &nbsp;:: 0.9771<br>
            ACCURACY :: 89.0%<br>
            THRESH &nbsp;&nbsp;:: 0.50
        </div>
    </div>
    <div style="margin-top:18px; padding:11px 13px;
                background:#050505;
                border:1px solid rgba(255,102,0,0.18);
                border-left:2px solid #ff6600;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:8px;
                    color:#ff6600; letter-spacing:1.5px; text-transform:uppercase;
                    margin-bottom:5px;">
            <span style="animation:blink 1s step-end infinite;">█</span>
            &nbsp;CRITICAL SIGNAL
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                    color:#666; line-height:1.8;">
            PRIOR DEFAULT<br>
            = 0% APPROVAL<br>
            N = 22,593
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TICKER BAR — renders on every page
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bb-ticker-outer">
    <div class="bb-ticker-track">
        <span class="bb-ticker-item">LOANIQ CREDIT TERMINAL v3.0</span>
        <span class="bb-ticker-item">XGBOOST ENGINE ACTIVE</span>
        <span class="bb-ticker-item">ROC-AUC: 0.9771</span>
        <span class="bb-ticker-item">DATASET: 44,535 RECORDS</span>
        <span class="bb-ticker-item">SHAP EXPLAINABILITY: ON</span>
        <span class="bb-ticker-item">ACCURACY: 89.0%</span>
        <span class="bb-ticker-item">PRIOR DEFAULT = 0% APPROVAL</span>
        <span class="bb-ticker-item">BUILT BY AMAN YADAV</span>
        <span class="bb-ticker-item">LOANIQ CREDIT TERMINAL v3.0</span>
        <span class="bb-ticker-item">XGBOOST ENGINE ACTIVE</span>
        <span class="bb-ticker-item">ROC-AUC: 0.9771</span>
        <span class="bb-ticker-item">DATASET: 44,535 RECORDS</span>
        <span class="bb-ticker-item">SHAP EXPLAINABILITY: ON</span>
        <span class="bb-ticker-item">ACCURACY: 89.0%</span>
        <span class="bb-ticker-item">PRIOR DEFAULT = 0% APPROVAL</span>
        <span class="bb-ticker-item">BUILT BY AMAN YADAV</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS — reusable HTML components
# ══════════════════════════════════════════════════════════════════════
def page_header(module_tag, title, subtitle):
    st.markdown(
        '<div style="margin-bottom:18px; animation:fadeSlideUp 0.4s ease both;">'
        '<div style="font-family:Share Tech Mono,monospace; font-size:9px; color:#ff6600;'
        ' letter-spacing:3px; text-transform:uppercase; margin-bottom:7px;">'
        '<span style="animation:blink 1s step-end infinite;">█</span>'
        ' &nbsp;MODULE :: ' + module_tag +
        '</div>'
        '<div style="font-family:VT323,Share Tech Mono,monospace; font-size:36px;'
        ' color:#e2e2e2; letter-spacing:4px; text-transform:uppercase;'
        ' animation:glitchText 12s ease infinite;">' + title +
        '</div>'
        '<div style="font-family:Share Tech Mono,monospace; font-size:10px;'
        ' color:#444; letter-spacing:2px; margin-top:5px;">' + subtitle +
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

def bb_label(text):
    """Bloomberg-style uppercase mono label — returned as HTML string."""
    return (
        '<div style="font-family:Share Tech Mono,Courier New,monospace;'
        ' font-size:9px; color:#555; letter-spacing:2px;'
        ' text-transform:uppercase; margin-bottom:5px;">' + text + '</div>'
    )

def section_rule(label):
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace; font-size:9px;'
        ' color:#444; letter-spacing:2px; text-transform:uppercase;'
        ' margin-bottom:10px;">'
        '┌─ ' + label + ' ────────────────────────────────</div>',
        unsafe_allow_html=True,
    )

def query_label(tag, text, color="#ff6600"):
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace; font-size:10px;'
        ' color:' + color + '; letter-spacing:1.5px; text-transform:uppercase;'
        ' margin-bottom:6px;">' + tag + ' :: ' + text + '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT APPROVAL
# ══════════════════════════════════════════════════════════════════════
if page == "▸  PREDICT APPROVAL":

    page_header(
        "LOAN DECISION ENGINE",
        "BANK LOAN APPROVAL",
        "REAL-TIME RISK ASSESSMENT  ::  XGBOOST  ::  SHAP EXPLAINABILITY",
    )
    st.divider()
    section_rule("BORROWER PROFILE INPUT")

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        person_age     = st.slider("AGE", 18, 80, 30, key="age")
        person_income  = st.number_input("ANNUAL INCOME ($)", 8000, 500000, 60000, step=1000, key="income")
        person_emp_exp = st.slider("EMPLOYMENT EXP (YRS)", 0, 50, 5, key="emp")
        credit_score   = st.slider("CREDIT SCORE", 390, 850, 650, key="cs")

    with c2:
        loan_amnt     = st.number_input("LOAN AMOUNT ($)", 500, 35000, 10000, step=500, key="loan")
        loan_int_rate = st.slider("INTEREST RATE (%)", 5.0, 20.0, 11.0, step=0.1, key="rate")

        # Compute loan % income (safe division)
        loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
        lti_pct_str = "{:.2%}".format(loan_percent_income)

        if loan_percent_income > 0.35:
            lti_col  = "#ff1a1a"
            lti_flag = "HIGH RISK"
        elif loan_percent_income > 0.20:
            lti_col  = "#ffcc00"
            lti_flag = "MODERATE"
        else:
            lti_col  = "#00ff41"
            lti_flag = "LOW RISK"

        # Pure string concat — ZERO f-string hex composites
        st.markdown(
            '<div style="background:#0c0c0c; border:1px solid #222222;'
            ' border-top:2px solid ' + lti_col + ';'
            ' padding:10px 13px; margin:4px 0 8px;">'
            + bb_label("LOAN % OF INCOME  [AUTO-COMPUTED]") +
            '<div style="display:flex; align-items:baseline; gap:10px; margin-top:2px;">'
            '<div style="font-family:VT323,Share Tech Mono,monospace;'
            ' font-size:34px; color:' + lti_col + '; letter-spacing:1px;">'
            + lti_pct_str +
            '</div>'
            '<div style="font-family:Share Tech Mono,monospace; font-size:9px;'
            ' color:' + lti_col + '; border:1px solid ' + lti_col + ';'
            ' padding:1px 7px; letter-spacing:1.5px;">'
            + lti_flag +
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        cb_person_cred_hist_length = st.slider("CREDIT HISTORY (YRS)", 2, 30, 5, key="ch")

    with c3:
        person_education               = st.selectbox("EDUCATION",      ["High School","Associate","Bachelor","Master","Doctorate"], key="edu")
        person_home_ownership          = st.selectbox("HOME OWNERSHIP",  ["RENT","OWN","MORTGAGE","OTHER"], key="home")
        loan_intent                    = st.selectbox("LOAN PURPOSE",    ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"], key="intent")
        previous_loan_defaults_on_file = st.selectbox("PRIOR DEFAULT?",  ["No","Yes"], key="def")

    # ── Engineered features ──────────────────────────────────────────
    loan_to_income             = loan_amnt / person_income if person_income > 0 else 0.0
    income_per_exp_year        = person_income / (person_emp_exp + 1)
    credit_score_to_loan_ratio = credit_score / loan_amnt if loan_amnt > 0 else 0.0
    previous_default_encoded   = 1 if previous_loan_defaults_on_file == "Yes" else 0

    input_df = pd.DataFrame([{
        "person_age":                     person_age,
        "person_income":                  person_income,
        "person_emp_exp":                 person_emp_exp,
        "loan_amnt":                      loan_amnt,
        "loan_int_rate":                  loan_int_rate,
        "loan_percent_income":            loan_percent_income,
        "cb_person_cred_hist_length":     cb_person_cred_hist_length,
        "credit_score":                   credit_score,
        "previous_loan_defaults_on_file": previous_default_encoded,
        "income_per_exp_year":            income_per_exp_year,
        "credit_score_to_loan_ratio":     credit_score_to_loan_ratio,
        "loan_to_income":                 loan_to_income,
        "person_education":               person_education,
        "person_home_ownership":          person_home_ownership,
        "loan_intent":                    loan_intent,
    }])

    st.divider()
    btn = st.button("▸  RUN CREDIT ASSESSMENT", use_container_width=True)

    if btn:
        # ── Inference ────────────────────────────────────────────────
        X_t  = preprocessor.transform(input_df)
        dm   = xgb.DMatrix(X_t)
        prob = float(booster.predict(dm)[0])

        # ── Pre-compute ALL display values — NO hex composites in HTML ──
        prob_pct  = "{:.1%}".format(prob)
        bar_width = "{:.1f}".format(prob * 100)
        cs_str    = "{:,}".format(credit_score)
        inc_str   = "${:,}".format(person_income)
        loan_str  = "${:,}".format(loan_amnt)
        lpi_str   = "{:.1%}".format(loan_percent_income)

        # Decision
        if prob >= 0.5:
            d_label = "APPROVED"
            d_color = "#00ff41"
            d_icon  = "[+]"
        else:
            d_label = "REJECTED"
            d_color = "#ff1a1a"
            d_icon  = "[-]"

        # Risk tier
        if prob >= 0.70:
            r_label = "LOW RISK"
            r_color = "#00ff41"
        elif prob >= 0.40:
            r_label = "MEDIUM RISK"
            r_color = "#ffcc00"
        else:
            r_label = "HIGH RISK"
            r_color = "#ff1a1a"

        # ASCII progress bar (40 chars)
        filled    = int(round(prob * 40))
        empty     = 40 - filled
        ascii_bar = "█" * filled + "░" * empty

        # Bar fill color
        bar_col = "#00ff41" if prob >= 0.70 else "#ffcc00" if prob >= 0.40 else "#ff1a1a"

        st.divider()

        # ── RESULT CARD — pure string concatenation, zero f-string ambiguity ──
        card = (
            '<div style="background:#040404; border:1px solid ' + d_color + ';'
            ' border-top:3px solid ' + d_color + '; padding:22px 26px 20px;'
            ' margin-bottom:20px; animation:fadeSlideUp 0.45s ease both;">'

            # Terminal header row
            '<div style="font-family:Share Tech Mono,Courier New,monospace;'
            ' font-size:9px; color:#333; letter-spacing:2px; text-transform:uppercase;'
            ' border-bottom:1px solid #151515; padding-bottom:10px; margin-bottom:18px;">'
            '&gt; CREDIT ASSESSMENT RESULT'
            ' &nbsp;::&nbsp; XGBOOST INFERENCE COMPLETE'
            ' &nbsp;::&nbsp; THRESHOLD 0.50'
            '</div>'

            # Three-column result grid
            '<div style="display:grid; grid-template-columns:1fr 1px 1fr 1px 1fr;'
            ' gap:0 28px; align-items:start;">'

            # — Col 1: Decision —
            '<div>'
            + bb_label("CREDIT DECISION") +
            '<div style="font-family:VT323,Share Tech Mono,monospace;'
            ' font-size:52px; color:' + d_color + '; letter-spacing:3px; line-height:1;'
            ' margin-top:4px;">'
            + d_icon + '&nbsp;' + d_label +
            '</div>'
            '</div>'

            # — Separator —
            '<div style="background:#1a1a1a; height:90px;"></div>'

            # — Col 2: Probability + ASCII bar —
            '<div>'
            + bb_label("APPROVAL PROBABILITY") +
            '<div style="font-family:VT323,Share Tech Mono,monospace;'
            ' font-size:64px; color:#ff6600; letter-spacing:1px; line-height:1;'
            ' margin-top:4px;">'
            + prob_pct +
            '</div>'
            '<div style="font-family:Share Tech Mono,Courier New,monospace;'
            ' font-size:11px; color:' + bar_col + '; margin-top:8px;'
            ' letter-spacing:0px;">'
            + ascii_bar +
            '</div>'
            '<div style="display:flex; justify-content:space-between; margin-top:3px;">'
            '<span style="font-family:Share Tech Mono,monospace;'
            ' font-size:8px; color:#2e2e2e;">0%</span>'
            '<span style="font-family:Share Tech Mono,monospace;'
            ' font-size:8px; color:#3a3a3a;">THRESH:50%</span>'
            '<span style="font-family:Share Tech Mono,monospace;'
            ' font-size:8px; color:#2e2e2e;">100%</span>'
            '</div>'
            '</div>'

            # — Separator —
            '<div style="background:#1a1a1a; height:90px;"></div>'

            # — Col 3: Risk + summary stats —
            '<div>'
            + bb_label("RISK CLASSIFICATION") +
            '<div style="font-family:VT323,Share Tech Mono,monospace;'
            ' font-size:30px; color:' + r_color + '; letter-spacing:3px;'
            ' line-height:1.2; margin-top:4px;">'
            + r_label +
            '</div>'
            '<div style="margin-top:14px;">'
            '<div style="font-family:Share Tech Mono,Courier New,monospace;'
            ' font-size:9px; color:#3a3a3a; letter-spacing:1px; line-height:2.2;">'
            'CREDIT SCORE &nbsp;&nbsp;&nbsp; ' + cs_str + '<br>'
            'INCOME &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ' + inc_str + '<br>'
            'LOAN AMT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ' + loan_str + '<br>'
            'LOAN/INCOME &nbsp;&nbsp;&nbsp; ' + lpi_str +
            '</div>'
            '</div>'
            '</div>'

            '</div>'   # end grid
            '</div>'   # end card
        )
        st.markdown(card, unsafe_allow_html=True)

        # ── Summary metrics row ──────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CREDIT SCORE",  "{:,}".format(credit_score))
        m2.metric("ANNUAL INCOME", "${:,}".format(person_income))
        m3.metric("LOAN AMOUNT",   "${:,}".format(loan_amnt))
        m4.metric("LOAN/INCOME",   "{:.1%}".format(loan_percent_income))

        st.info(
            "▸ SIGNAL: Prior loan default is the single strongest rejection predictor — "
            "0% approval across 22,593 defaulters in this dataset. Higher interest rates "
            "paradoxically correlate WITH approval: banks price risk into the rate."
        )

        # ── SHAP section ─────────────────────────────────────────────
        st.divider()
        st.markdown(
            '<div style="margin-bottom:14px; animation:fadeSlideUp 0.5s ease both;">'
            '<div style="font-family:Share Tech Mono,monospace; font-size:9px;'
            ' color:#ff6600; letter-spacing:3px; text-transform:uppercase;'
            ' margin-bottom:6px;">┌─ MODULE :: SHAP EXPLAINABILITY ────────────────────</div>'
            '<div style="font-family:VT323,Share Tech Mono,monospace; font-size:26px;'
            ' color:#e2e2e2; letter-spacing:3px; text-transform:uppercase;">'
            'DECISION FACTOR ANALYSIS</div>'
            '<div style="font-family:Share Tech Mono,monospace; font-size:9px;'
            ' color:#444; margin-top:4px; letter-spacing:1.5px;">'
            'GREEN = PUSHES TOWARD APPROVAL &nbsp;|&nbsp; RED = PUSHES TOWARD REJECTION'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Safe SHAP extraction
        shap_vals = safe_shap(explainer, X_t)
        shap_df = (
            pd.DataFrame({"Feature": feature_names, "SHAP": shap_vals})
            .assign(Abs=lambda d: d["SHAP"].abs())
            .sort_values("Abs", ascending=True)
            .tail(10)
        )
        bar_colors = ["#00ff41" if v > 0 else "#ff1a1a" for v in shap_df["SHAP"]]

        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP"],
            y=shap_df["Feature"],
            orientation="h",
            marker=dict(color=bar_colors, opacity=0.88, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>SHAP VALUE: %{x:.4f}<extra></extra>",
        ))
        fig_shap = apply_chart(
            fig_shap,
            height=400,
            title="TOP 10 DECISION FACTORS — SHAP VALUES",
            xtitle="<-- TOWARD REJECTION     |     TOWARD APPROVAL -->",
            shapes=[dict(
                type="line",
                x0=0, x1=0, y0=-0.5, y1=9.5,
                line=dict(color="rgba(255,102,0,0.45)", width=1, dash="dot"),
            )],
        )
        fig_shap.update_layout(bargap=0.25)
        st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif page == "▸  EDA INSIGHTS":

    page_header(
        "EXPLORATORY DATA ANALYSIS",
        "DATASET INTELLIGENCE",
        "44,500+ LOAN APPLICATIONS  ::  VISUAL PATTERN DISCOVERY",
    )
    st.divider()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TOTAL RECORDS",  "{:,}".format(len(df)))
    k2.metric("APPROVAL RATE",  "{:.2%}".format(df["loan_status"].mean()))
    k3.metric("AVG CREDIT",     "{:.0f}".format(df["credit_score"].mean()))
    k4.metric("AVG LOAN",       "${:,.0f}".format(df["loan_amnt"].mean()))

    st.divider()

    c1, c2 = st.columns(2, gap="small")

    with c1:
        home_df = (
            df.groupby("person_home_ownership")["loan_status"]
            .mean().reset_index()
            .rename(columns={"person_home_ownership": "Ownership", "loan_status": "Rate"})
        )
        home_df["Rate"] = (home_df["Rate"] * 100).round(2)
        fig1 = px.bar(
            home_df, x="Ownership", y="Rate",
            color="Rate",
            color_continuous_scale=[[0, "#0d0600"], [0.5, "#993d00"], [1, "#ff6600"]],
            labels={"Rate": "APPROVAL RATE (%)"},
        )
        fig1.update_coloraxes(showscale=False)
        fig1.update_traces(marker_line_width=0)
        st.plotly_chart(
            apply_chart(fig1, title="APPROVAL RATE BY HOME OWNERSHIP"),
            use_container_width=True,
        )

    with c2:
        intent_df = df["loan_intent"].value_counts().reset_index()
        intent_df.columns = ["Purpose", "Count"]
        fig2 = px.pie(
            intent_df, names="Purpose", values="Count", hole=0.42,
            color_discrete_sequence=[
                "#ff6600","#cc4400","#ff8844","#993300","#ffaa66","#662200"
            ],
        )
        fig2.update_traces(
            textfont_color="white",
            textfont_family="Share Tech Mono, Courier New",
            textfont_size=10,
        )
        st.plotly_chart(
            apply_chart(fig2, title="APPLICATIONS BY LOAN PURPOSE"),
            use_container_width=True,
        )

    c1, c2 = st.columns(2, gap="small")

    with c1:
        fig3 = px.histogram(
            df, x="credit_score", color="loan_status",
            barmode="overlay", nbins=50, opacity=0.82,
            color_discrete_map={0: "#ff1a1a", 1: "#00ff41"},
            labels={"loan_status": "STATUS", "credit_score": "CREDIT SCORE"},
        )
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(
            apply_chart(fig3, title="CREDIT SCORE DISTRIBUTION BY OUTCOME"),
            use_container_width=True,
        )

    with c2:
        sample = df.sample(n=2000, random_state=42).copy()
        sample["Status"] = sample["loan_status"].map({0: "REJECTED", 1: "APPROVED"})
        fig4 = px.scatter(
            sample, x="person_income", y="loan_amnt", color="Status",
            color_discrete_map={"REJECTED": "#ff1a1a", "APPROVED": "#00ff41"},
            labels={
                "person_income": "ANNUAL INCOME ($)",
                "loan_amnt":     "LOAN AMOUNT ($)",
            },
            opacity=0.42,
        )
        fig4.update_traces(marker_size=3)
        st.plotly_chart(
            apply_chart(fig4, title="INCOME vs LOAN AMOUNT — SCATTER"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — SQL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "▸  SQL ANALYSIS":

    page_header(
        "SQL BUSINESS INTELLIGENCE",
        "DATA-DRIVEN INSIGHTS",
        "5 BUSINESS QUERIES  ::  SQLITE  ::  44,500+ RECORDS",
    )
    st.divider()

    # Q1 — Rejection by purpose
    query_label("QUERY_01", "REJECTION RATE BY LOAN PURPOSE")
    fig_q2 = px.bar(
        q2.sort_values("rejection_rate_pct", ascending=True),
        x="rejection_rate_pct", y="loan_intent", orientation="h",
        color="rejection_rate_pct",
        color_continuous_scale=[[0, "#110000"], [0.5, "#aa1100"], [1, "#ff1a1a"]],
        labels={"rejection_rate_pct": "REJECTION RATE %", "loan_intent": ""},
    )
    fig_q2.update_coloraxes(showscale=False)
    fig_q2.update_traces(marker_line_width=0)
    st.plotly_chart(
        apply_chart(fig_q2, height=300, title="WHICH LOAN PURPOSE GETS REJECTED MOST?"),
        use_container_width=True,
    )

    st.divider()

    # Q2 — Income vs approval
    query_label("QUERY_02", "APPROVAL RATE BY INCOME BAND")
    st.caption("SIGNAL: LOWER-INCOME BORROWERS OFTEN APPROVED — SMALLER, MORE SERVICEABLE LOANS")
    fig_q4 = px.bar(
        q4, x="income_band", y="approval_rate_pct",
        color="approval_rate_pct",
        color_continuous_scale=[[0, "#050f04"], [0.5, "#006622"], [1, "#00ff41"]],
        labels={"approval_rate_pct": "APPROVAL RATE %", "income_band": "INCOME BAND"},
    )
    fig_q4.update_coloraxes(showscale=False)
    fig_q4.update_traces(marker_line_width=0)
    st.plotly_chart(
        apply_chart(fig_q4, height=300, title="DOES HIGHER INCOME GUARANTEE APPROVAL?"),
        use_container_width=True,
    )

    st.divider()

    # Q3 — Default finding [CRITICAL]
    query_label("QUERY_03 [CRITICAL]", "PRIOR DEFAULT vs APPROVAL RATE", color="#ff1a1a")

    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        fig_q5 = px.bar(
            q5, x="previous_loan_defaults_on_file", y="approval_rate_pct",
            color="approval_rate_pct",
            color_continuous_scale=[
                [0, "#110000"], [0.4, "#991100"], [0.8, "#ffcc00"], [1, "#00ff41"]
            ],
            labels={
                "approval_rate_pct":              "APPROVAL RATE %",
                "previous_loan_defaults_on_file": "PRIOR DEFAULT (0=NO  1=YES)",
            },
        )
        fig_q5.update_coloraxes(showscale=False)
        fig_q5.update_traces(marker_line_width=0)
        st.plotly_chart(
            apply_chart(fig_q5, height=300,
                        title="APPROVAL RATE: CLEAN HISTORY vs PRIOR DEFAULT"),
            use_container_width=True,
        )

    with c2:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        no_def  = float(q5[q5["previous_loan_defaults_on_file"] == 0]["approval_rate_pct"].values[0])
        yes_def = float(q5[q5["previous_loan_defaults_on_file"] == 1]["approval_rate_pct"].values[0])
        st.metric("CLEAN HISTORY",   "{:.1f}%".format(no_def))
        st.metric("PRIOR DEFAULTER", "{:.1f}%".format(yes_def))
        st.markdown(
            '<div style="margin-top:12px; padding:12px 14px;'
            ' background:#060000;'
            ' border:1px solid #ff1a1a;'
            ' border-left:2px solid #ff1a1a;">'
            '<div style="font-family:Share Tech Mono,monospace; font-size:8px;'
            ' color:#ff1a1a; letter-spacing:1.5px; text-transform:uppercase;'
            ' margin-bottom:6px;">'
            '<span style="animation:blink 0.8s step-end infinite;">█</span>'
            ' CRITICAL FINDING'
            '</div>'
            '<div style="font-family:Share Tech Mono,monospace; font-size:10px;'
            ' color:#666; line-height:1.9;">'
            'PRIOR DEFAULT<br>'
            '= 0% APPROVAL<br>'
            'ACROSS 22,593<br>'
            'APPLICANTS<br><br>'
            'STRONGER THAN<br>'
            'CREDIT SCORE,<br>'
            'INCOME, OR<br>'
            'LOAN SIZE'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Q4 — Credit score by ownership
    query_label("QUERY_04", "AVG CREDIT SCORE BY HOME OWNERSHIP & OUTCOME")
    fig_q3 = px.bar(
        q3, x="person_home_ownership", y="avg_credit_score",
        color="loan_status", barmode="group",
        color_discrete_map={0: "#ff1a1a", 1: "#00ff41"},
        labels={
            "avg_credit_score":      "AVG CREDIT SCORE",
            "person_home_ownership": "HOME OWNERSHIP",
            "loan_status":           "APPROVED",
        },
    )
    fig_q3.update_traces(marker_line_width=0)
    st.plotly_chart(
        apply_chart(fig_q3, height=300,
                    title="CREDIT SCORE BY OWNERSHIP & APPROVAL STATUS"),
        use_container_width=True,
    )

    st.divider()

    # Q5 — Education
    query_label("QUERY_05", "APPROVAL RATE BY EDUCATION LEVEL")
    st.caption(
        "SIGNAL: EDUCATION HAS NEAR-ZERO IMPACT — ALL LEVELS CLUSTER ~22%. "
        "BANKS CARE ABOUT FINANCIALS, NOT CREDENTIALS."
    )
    st.dataframe(q1, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    '<div style="display:flex; justify-content:space-between; align-items:center;'
    ' padding:4px 0 10px; animation:fadeIn 0.5s ease both;">'
    '<div style="font-family:Share Tech Mono,monospace; font-size:9px;'
    ' color:#2a2a2a; letter-spacing:1px;">'
    '&copy; 2026 &nbsp;::&nbsp; LOANIQ TERMINAL &nbsp;::&nbsp;'
    '<span style="color:#444;"> AMAN YADAV</span>'
    '</div>'
    '<div style="font-family:Share Tech Mono,monospace; font-size:9px;'
    ' color:#1e1e1e; letter-spacing:1px;">'
    'PYTHON &nbsp;::&nbsp; XGBOOST &nbsp;::&nbsp; SHAP &nbsp;::&nbsp;'
    ' STREAMLIT &nbsp;::&nbsp; SQLITE'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)