"""
datamind_app.py — DataMind ML Platform
Ultra-vibrant, colorful, interactive platform with Neural Network mapping,
speed-optimized training, and rich Deep Dive intelligence suite.
"""
from __future__ import annotations
import pathlib, sys, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.ml_pipeline import detect_task, prepare_data, train_all_models, predict_single
from src.chatbot import answer as bot_answer

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataMind — AI & ML Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# ULTRA-VIBRANT COLORFUL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: radial-gradient(circle at 10% 20%, #0d1127 0%, #060814 90%) !important;
    color: #f1f5f9;
}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #060814; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #ec4899, #8b5cf6, #3b82f6); border-radius: 4px; }

/* ── Vibrant Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: linear-gradient(90deg, rgba(236,72,153,0.1), rgba(139,92,246,0.1), rgba(59,130,246,0.1), rgba(16,185,129,0.1));
    border-radius: 16px;
    padding: 8px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 4px 20px rgba(139,92,246,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 700;
    font-size: 0.92rem;
    color: #94a3b8;
    border: none;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc;
    background: rgba(255,255,255,0.06);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 50%, #3b82f6 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 6px 20px rgba(236,72,153,0.45);
    transform: translateY(-1px);
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1329 0%, #080b18 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.2) !important;
    box-shadow: 4px 0 25px rgba(0,0,0,0.5);
}

/* ── Vibrant Buttons ── */
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    transition: all 0.25s ease;
    border: none;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 50%, #06b6d4 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px rgba(236,72,153,0.4);
    padding: 12px 28px;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 30px rgba(236,72,153,0.65);
    transform: translateY(-2px) scale(1.02);
}

/* ── Glowing Input Elements ── */
.stTextInput input, .stSelectbox div, .stMultiSelect div,
.stNumberInput input, .stTextArea textarea {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    transition: border 0.3s;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #ec4899 !important;
    box-shadow: 0 0 15px rgba(236,72,153,0.3) !important;
}

/* ── Vibrant Metrics ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(236,72,153,0.1), rgba(139,92,246,0.15));
    border: 1px solid rgba(236,72,153,0.3);
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

/* ── Creative UI Components ── */
.dm-hero {
    text-align: center;
    padding: 35px 0 15px;
}
.dm-hero h1 {
    font-size: 3.8rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #f43f5e 0%, #ec4899 25%, #a855f7 50%, #3b82f6 75%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px;
    filter: drop-shadow(0 2px 15px rgba(236,72,153,0.35));
}
.dm-hero p {
    color: #94a3b8;
    font-size: 1.15rem;
    font-weight: 500;
}

.dm-upload-wrap {
    border: 2px dashed #ec4899;
    border-radius: 24px;
    padding: 40px 30px 25px;
    background: radial-gradient(ellipse at center, rgba(236,72,153,0.15) 0%, rgba(139,92,246,0.05) 60%, transparent 80%);
    text-align: center;
    margin: 18px 0;
    box-shadow: 0 0 35px rgba(236,72,153,0.2);
}
.dm-upload-icon { font-size: 3.8rem; margin-bottom: 8px; filter: drop-shadow(0 0 15px #ec4899); }
.dm-upload-title { font-size: 1.5rem; font-weight: 800; color: #f472b6; margin-bottom: 6px; }

/* ── Stat Cards ── */
.dm-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 20px 0;
}
.dm-stat {
    background: linear-gradient(135deg, rgba(236,72,153,0.18) 0%, rgba(139,92,246,0.15) 50%, rgba(59,130,246,0.15) 100%);
    border: 1px solid rgba(236,72,153,0.35);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.dm-stat-num {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f472b6, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.dm-stat-label {
    font-size: 0.78rem;
    color: #cbd5e1;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-top: 5px;
}

/* ── Section Titles ── */
.dm-section-title {
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ec4899, #8b5cf6, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    border-left: 4px solid #ec4899;
    padding-left: 14px;
    margin: 26px 0 16px;
}

/* ── Badges ── */
.dm-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 700;
    margin: 2px 4px;
}
.db-pink   { background: rgba(236,72,153,0.2); color: #f472b6; border: 1px solid #ec4899; }
.db-purple { background: rgba(168,85,247,0.2); color: #c084fc; border: 1px solid #a855f7; }
.db-cyan   { background: rgba(6,182,212,0.2);  color: #22d3ee; border: 1px solid #06b6d4; }
.db-green  { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #10b981; }

/* ── Leaderboard Row ── */
.dm-leader-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    border-radius: 14px;
    margin: 8px 0;
    background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(139,92,246,0.06));
    border: 1px solid rgba(139,92,246,0.2);
    transition: all 0.2s;
}
.dm-leader-row:hover {
    border-color: #ec4899;
    box-shadow: 0 4px 15px rgba(236,72,153,0.2);
}
.dm-leader-rank { font-size: 1.4rem; font-weight: 900; width: 36px; }
.dm-leader-rank.gold   { color: #fbbf24; filter: drop-shadow(0 0 6px #fbbf24); }
.dm-leader-rank.silver { color: #e2e8f0; filter: drop-shadow(0 0 4px #e2e8f0); }
.dm-leader-rank.bronze { color: #f97316; filter: drop-shadow(0 0 4px #f97316); }
.dm-leader-name { flex: 1; font-weight: 700; font-size: 1.05rem; }
.dm-leader-score { font-size: 1.25rem; font-weight: 900; color: #ec4899; }

/* ── Deep Dive Chat Bubbles ── */
.dm-bubble-wrap { margin: 16px 0; }
.dm-bubble-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 6px;
}
.dm-bubble-user > div {
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    border-radius: 20px 20px 4px 20px;
    padding: 13px 22px;
    max-width: 82%;
    font-size: 0.96rem;
    font-weight: 500;
    color: #ffffff;
    box-shadow: 0 4px 18px rgba(236,72,153,0.35);
}
.dm-bubble-ai {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 6px;
}
.dm-bubble-ai > div {
    background: linear-gradient(135deg, rgba(30,41,59,0.85), rgba(15,23,42,0.9));
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 20px 20px 20px 4px;
    padding: 16px 24px;
    max-width: 85%;
    font-size: 0.96rem;
    line-height: 1.7;
    color: #f1f5f9;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
.dm-ai-label {
    font-size: 0.78rem;
    color: #38bdf8;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Data Brief Cards ── */
.dm-brief-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.7), rgba(15,23,42,0.85));
    border-radius: 16px;
    border: 1px solid rgba(139,92,246,0.3);
    padding: 22px;
    margin: 12px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
.dm-insight {
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 12px 18px;
    margin: 10px 0;
    font-size: 0.94rem;
    font-weight: 500;
}
.di-green  { background: rgba(16,185,129,0.12); border-color: #10b981; color: #6ee7b7; }
.di-orange { background: rgba(245,158,11,0.12); border-color: #f59e0b; color: #fde68a; }
.di-red    { background: rgba(239,68,68,0.12);  border-color: #ef4444; color: #fca5a5; }
.di-blue   { background: rgba(59,130,246,0.12); border-color: #3b82f6; color: #93c5fd; }
.di-pink   { background: rgba(236,72,153,0.12); border-color: #ec4899; color: #f472b6; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "df": None, "packed": None, "results": None,
        "target": None, "upload_name": "", "chat_history": [],
        "task_override": "auto",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
_init()

df: pd.DataFrame | None = st.session_state["df"]
results: dict | None    = st.session_state["results"]
packed: dict | None     = st.session_state["packed"]


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURE DIAGRAM GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_nn_mapping(n_inputs: int, task: str = "classification"):
    """Render a visual layered neural network graph using Plotly."""
    layers = [
        {"name": "Input Layer", "nodes": min(n_inputs, 8), "total": n_inputs, "color": "#38bdf8", "x": 0},
        {"name": "Hidden 1 (Dense-64)", "nodes": 6, "total": 64, "color": "#ec4899", "x": 1.5},
        {"name": "Hidden 2 (Dense-32)", "nodes": 5, "total": 32, "color": "#8b5cf6", "x": 3.0},
        {"name": "Output Layer", "nodes": 2 if task == "classification" else 1, "total": 2 if task == "classification" else 1, "color": "#10b981", "x": 4.5},
    ]

    fig = go.Figure()

    # Draw inter-layer synaptic connection lines
    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        y1_coords = np.linspace(-1, 1, l1["nodes"])
        y2_coords = np.linspace(-1, 1, l2["nodes"])
        for y1 in y1_coords:
            for y2 in y2_coords:
                fig.add_trace(go.Scatter(
                    x=[l1["x"], l2["x"]],
                    y=[y1, y2],
                    mode="lines",
                    line=dict(color="rgba(139,92,246,0.18)", width=1),
                    hoverinfo="none",
                    showlegend=False
                ))

    # Draw nodes
    for l in layers:
        y_coords = np.linspace(-1, 1, l["nodes"])
        texts = [f"Node {j+1}" for j in range(l["nodes"])]
        fig.add_trace(go.Scatter(
            x=[l["x"]] * l["nodes"],
            y=y_coords,
            mode="markers+text",
            marker=dict(size=26, color=l["color"], line=dict(color="#ffffff", width=2), symbol="circle"),
            text=texts,
            textposition="top center",
            textfont=dict(color="#f8fafc", size=9),
            name=f"{l['name']} ({l['total']} units)",
            hovertext=[f"{l['name']} (Total {l['total']} units)" for _ in y_coords],
            hoverinfo="text"
        ))

    fig.update_layout(
        title=f"🧠 Neural Network Layer & Topology Mapping ({task.title()})",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=380,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 20px">
        <div style="font-size:1.8rem;font-weight:900;background:linear-gradient(135deg, #ec4899, #8b5cf6, #38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ⚡ DataMind AI
        </div>
        <div style="font-size:0.75rem;color:#94a3b8;font-weight:700;letter-spacing:0.06em">
            HIGH-SPEED INTELLIGENT ML STUDIO
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(236,72,153,0.15), rgba(139,92,246,0.15));border:1px solid rgba(236,72,153,0.35);border-radius:14px;padding:12px 16px;margin-bottom:12px">
            <div style="color:#f472b6;font-weight:800;font-size:0.92rem">📂 {st.session_state['upload_name'] or 'Active Dataset'}</div>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:2px">{df.shape[0]:,} rows · {df.shape[1]} columns</div>
        </div>
        """, unsafe_allow_html=True)

    if results is not None:
        best = results.get("best_name", "—")
        task = results.get("task", "")
        col  = "Accuracy" if task == "classification" else "R²"
        mval = results["metrics"][col].astype(float).max() if col in results["metrics"].columns else None
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(16,185,129,0.15), rgba(6,182,212,0.15));border:1px solid rgba(16,185,129,0.35);border-radius:14px;padding:12px 16px;margin-bottom:12px">
            <div style="color:#34d399;font-weight:800;font-size:0.92rem">🏆 Top Model: {best}</div>
            <div style="color:#94a3b8;font-size:0.8rem;margin-top:2px">{col}: <b style="color:#38bdf8">{mval:.4f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:14px 0'>", unsafe_allow_html=True)

    st.markdown("### 💬 Live Assistant")
    st.caption("Ask quick questions about your data or models at any time")

    for item in st.session_state["chat_history"][-2:]:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:8px 12px;margin:4px 0;font-size:0.82rem">
            <span style="color:#ec4899;font-weight:700">YOU:</span> {item['q'][:50]}...
        </div>
        <div style="background:rgba(139,92,246,0.08);border-radius:10px;padding:8px 12px;margin:4px 0;font-size:0.82rem">
            <span style="color:#38bdf8;font-weight:700">AI:</span> {item['a'][:70]}...
        </div>
        """, unsafe_allow_html=True)

    sidebar_q = st.text_input("Quick question", placeholder="e.g. Which model is best?", label_visibility="collapsed", key="side_q")
    col_s1, col_s2 = st.columns([3, 1])
    if col_s1.button("Send ⚡", key="side_send", use_container_width=True):
        if sidebar_q.strip():
            ans = bot_answer(sidebar_q, st.session_state["df"], st.session_state["results"])
            st.session_state["chat_history"].append({"q": sidebar_q, "a": ans})
            st.rerun()
    if col_s2.button("🗑", key="side_clr"):
        st.session_state["chat_history"] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN VIEW: 5 TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Upload Data",
    "⚙️ Train Models",
    "📊 Visualise",
    "📋 Data Brief",
    "💬 Deep Dive",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="dm-hero">
        <h1>Welcome to DataMind Studio</h1>
        <p>Drop any CSV, Excel, or JSON dataset to ignite full-pipeline automated Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dm-upload-wrap">
        <div class="dm-upload-icon">⚡</div>
        <div class="dm-upload-title">Drag & Drop Your Dataset Here</div>
        <div style="color:#94a3b8;font-size:0.92rem">CSV · TSV · Excel (.xlsx, .xls) · JSON (Up to 200 MB)</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv", "tsv", "xlsx", "xls", "json"], label_visibility="collapsed", key="main_upload_t1")

    if uploaded:
        try:
            name = uploaded.name.lower()
            if name.endswith((".xlsx", ".xls")):
                df_new = pd.read_excel(uploaded)
            elif name.endswith(".json"):
                df_new = pd.read_json(uploaded)
            elif name.endswith(".tsv"):
                df_new = pd.read_csv(uploaded, sep="\t")
            else:
                df_new = pd.read_csv(uploaded)

            if len(df_new) > 100_000:
                df_new = df_new.sample(100_000, random_state=42).reset_index(drop=True)
                st.warning("⚡ Loaded 100,000 rows for high-performance responsive operations.")

            if st.session_state["upload_name"] != uploaded.name:
                st.session_state.update({
                    "df": df_new, "packed": None, "results": None,
                    "target": None, "upload_name": uploaded.name,
                })
                st.rerun()

        except Exception as exc:
            st.error(f"Error loading file: {exc}")

    df = st.session_state["df"]

    if df is not None:
        miss_total = int(df.isna().sum().sum())
        dup_total  = int(df.duplicated().sum())
        num_cols   = len(df.select_dtypes(include="number").columns)
        cat_cols   = df.shape[1] - num_cols

        st.markdown(f"""
        <div class="dm-stat-grid">
            <div class="dm-stat">
                <div class="dm-stat-num">{df.shape[0]:,}</div>
                <div class="dm-stat-label">Total Observations</div>
            </div>
            <div class="dm-stat">
                <div class="dm-stat-num">{df.shape[1]}</div>
                <div class="dm-stat-label">Feature Columns</div>
            </div>
            <div class="dm-stat">
                <div class="dm-stat-num">{miss_total:,}</div>
                <div class="dm-stat-label">Missing Values</div>
            </div>
            <div class="dm-stat">
                <div class="dm-stat-num">{dup_total:,}</div>
                <div class="dm-stat-label">Duplicate Rows</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="dm-section-title">📄 Dataset Preview & Table Glimpse</div>', unsafe_allow_html=True)
        st.dataframe(df.head(40), use_container_width=True, height=280)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="dm-section-title">🔢 Column Schema & Missingness</div>', unsafe_allow_html=True)
            info_df = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Missing": df.isna().sum(),
                "Missing %": (df.isna().mean() * 100).round(2),
                "Unique Values": df.nunique()
            })
            st.dataframe(info_df, use_container_width=True, height=260)
        with c2:
            st.markdown('<div class="dm-section-title">📈 Statistical Overview</div>', unsafe_allow_html=True)
            st.dataframe(df.describe().T.round(3), use_container_width=True, height=260)

        st.success("✨ **Dataset Loaded Successfully!** Switch over to **⚙️ Train Models** to launch high-speed training.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN MODELS & NEURAL NETWORK MAPPING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if df is None:
        st.info("📂 Please upload your dataset in **Tab 1** first.")
        st.stop()

    st.markdown("""
    <div style="padding:15px 0 10px">
        <div style="font-size:2rem;font-weight:900;background:linear-gradient(90deg,#ec4899,#8b5cf6,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            High-Speed Model Studio & Neural Mapping
        </div>
        <div style="color:#94a3b8">Multi-model concurrent training, hardware acceleration & topology visualization</div>
    </div>
    """, unsafe_allow_html=True)

    col_cfg, col_algo = st.columns([1, 1], gap="large")

    with col_cfg:
        st.markdown('<div class="dm-section-title">🎯 Target Column & Task Configuration</div>', unsafe_allow_html=True)
        tgt_opts = list(df.columns)
        default_idx = tgt_opts.index(st.session_state["target"]) if st.session_state["target"] in tgt_opts else 0
        target = st.selectbox("Select Target Feature to Predict", tgt_opts, index=default_idx, key="t2_target")
        st.session_state["target"] = target

        auto_task = detect_task(df[target].dropna())
        task_choice = st.radio("Task Mode", ["auto", "classification", "regression"], horizontal=True, key="t2_task_radio")
        effective_task = auto_task if task_choice == "auto" else task_choice

        badge = "db-pink" if effective_task == "classification" else "db-cyan"
        st.markdown(f"""
        <span class="dm-badge {badge}">
            {"🔥 Multi-Class / Binary Classification" if effective_task=="classification" else "⚡ Continuous Value Regression"}
        </span>
        """, unsafe_allow_html=True)

        st.markdown('<div class="dm-section-title" style="margin-top:18px">⚙️ Performance Options</div>', unsafe_allow_html=True)
        test_pct = st.slider("Validation Holdout Split %", 10, 40, 20, 5, key="t2_split")

    with col_algo:
        st.markdown('<div class="dm-section-title">🤖 Select Algorithms to Train</div>', unsafe_allow_html=True)

        if effective_task == "classification":
            algos = [
                "Logistic Regression", "Decision Tree", "Random Forest",
                "AdaBoost", "Gradient Boosting", "Neural Network (MLP)", "KNN", "SVM", "Naive Bayes"
            ]
        else:
            algos = [
                "Linear Regression", "Ridge Regression", "Lasso Regression",
                "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting",
                "Neural Network (MLP)", "KNN", "SVR"
            ]

        selected_algos = st.multiselect("Choose Algorithms (Train multiple simultaneously)", algos, default=algos[:6], key="t2_algos")

    if not selected_algos:
        st.warning("Select at least one algorithm above.")
    else:
        st.markdown(f"""
        <div style="background:rgba(236,72,153,0.12);border:1px solid #ec4899;border-radius:14px;padding:14px 20px;margin:12px 0">
            <b style="color:#f472b6">⚡ Accelerated Training Ready:</b> Training {len(selected_algos)} model(s) concurrently across available CPU cores.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Train Selected Models Now", type="primary", key="t2_train_btn"):
            with st.spinner("⚡ Training accelerated models..."):
                try:
                    packed = prepare_data(df, target, test_size=test_pct / 100)
                    results = train_all_models(packed)

                    # Filter to user selected algorithms
                    keep_f = {k: v for k, v in results["fitted"].items() if k in selected_algos}
                    keep_p = {k: v for k, v in results["predictions"].items() if k in selected_algos}
                    keep_m = results["metrics"].loc[[k for k in selected_algos if k in results["metrics"].index]]
                    sort_c = "Accuracy" if effective_task == "classification" else "R²"
                    best_n = keep_m[sort_c].astype(float).idxmax() if sort_c in keep_m.columns else list(keep_f.keys())[0]

                    results["fitted"] = keep_f
                    results["predictions"] = keep_p
                    results["metrics"] = keep_m
                    results["best_name"] = best_n

                    st.session_state["packed"] = packed
                    st.session_state["results"] = results
                    st.success(f"✨ Training finished in record time! Best model: **{best_n}**")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training error: {exc}")

    # ── Display Neural Network Mapping & Results ──────────────────────────────
    st.markdown("---")
    st.markdown('<div class="dm-section-title">🧠 Neural Network Layer & Synapse Topology Mapping</div>', unsafe_allow_html=True)
    n_feats = len(df.columns) - 1
    nn_fig = generate_nn_mapping(n_feats, task=effective_task)
    st.plotly_chart(nn_fig, use_container_width=True, key="t2_nn_map")

    if results is not None:
        st.markdown("---")
        st.markdown('<div class="dm-section-title">🏆 Model Leaderboard & Benchmark Scores</div>', unsafe_allow_html=True)
        task_r = results["task"]
        metrics = results["metrics"]
        best = results["best_name"]
        sort_col = "Accuracy" if task_r == "classification" else "R²"

        ranked = metrics.sort_values(sort_col, ascending=False) if sort_col in metrics.columns else metrics
        medals = ["🥇", "🥈", "🥉"]
        medal_cls = ["gold", "silver", "bronze"]
        num_cols_m = metrics.select_dtypes(include="number").columns.tolist()

        for i, (name, row) in enumerate(ranked.iterrows()):
            medal = medals[i] if i < 3 else f"#{i+1}"
            mc = medal_cls[i] if i < 3 else ""
            main_score = row[sort_col] if sort_col in row else 0.0
            other_scores = " &nbsp;·&nbsp; ".join(
                f"<span style='color:#94a3b8'>{c}: <b style='color:#38bdf8'>{row[c]:.4f}</b></span>"
                for c in num_cols_m if c != sort_col and not pd.isna(row.get(c, float("nan")))
            )
            st.markdown(f"""
            <div class="dm-leader-row">
                <div class="dm-leader-rank {mc}">{medal}</div>
                <div class="dm-leader-name">{name} {"<span class='dm-badge db-pink'>WINNER</span>" if name==best else ""}</div>
                <div>{other_scores}</div>
                <div class="dm-leader-score">{float(main_score):.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        fig_bar = px.bar(
            ranked.reset_index(), x="Model", y=sort_col,
            color=sort_col,
            color_continuous_scale=["#3b82f6", "#8b5cf6", "#ec4899"],
            title=f"Comparative Benchmark ({sort_col})",
            text_auto=".4f"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"), height=360, showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="t2_benchmark_bar")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUALISE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if df is None:
        st.info("Upload your dataset in Tab 1 to unlock visual diagnostics.")
        st.stop()

    st.markdown("""
    <div style="padding:15px 0 10px">
        <div style="font-size:2rem;font-weight:900;background:linear-gradient(90deg,#ec4899,#8b5cf6,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Visual Analytics & Exploratory Studio
        </div>
        <div style="color:#94a3b8">Interactive multi-dimensional charts, correlation grids, and distributions</div>
    </div>
    """, unsafe_allow_html=True)

    nums = list(df.select_dtypes(include="number").columns)
    cats = [c for c in df.columns if c not in nums]

    v_mode = st.radio(
        "Visualization Tool",
        ["🔥 Vibrant Correlation Matrix", "📊 Dynamic Distributions", "🔵 Multi-Feature Scatter", "🎻 Violin & Box Plots", "🌊 Feature Intensity Heatmap"],
        horizontal=True, key="t3_vmode"
    )
    st.markdown("---")

    P_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"), margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
    )

    if v_mode == "🔥 Vibrant Correlation Matrix":
        if len(nums) < 2:
            st.warning("Need at least 2 numeric features.")
        else:
            corr = df[nums].corr().round(3)
            fig_c = px.imshow(
                corr, text_auto=True, aspect="auto",
                color_continuous_scale="Magma",
                title="🔥 Pearson Correlation Heatmap"
            )
            fig_c.update_layout(height=max(420, len(nums) * 45), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(fig_c, use_container_width=True, key="t3_corr_chart")

    elif v_mode == "📊 Dynamic Distributions":
        if not nums:
            st.warning("No numeric columns.")
        else:
            sel = st.multiselect("Select columns", nums, default=nums[:min(4, len(nums))], key="t3_dist_cols")
            if sel:
                fig_d = make_subplots(rows=-(-len(sel) // 2), cols=2, subplot_titles=sel)
                for idx, col in enumerate(sel):
                    r, c = idx // 2 + 1, idx % 2 + 1
                    fig_d.add_trace(go.Histogram(x=df[col].dropna(), marker_color="#ec4899", opacity=0.8, name=col), row=r, col=c)
                fig_d.update_layout(height=260 * (-(-len(sel) // 2)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), showlegend=False)
                st.plotly_chart(fig_d, use_container_width=True, key="t3_hist_chart")

    elif v_mode == "🔵 Multi-Feature Scatter":
        if len(nums) < 2:
            st.warning("Need at least 2 numeric features.")
        else:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X-Axis", nums, index=0, key="t3_sc_x")
            y_col = c2.selectbox("Y-Axis", nums, index=1 if len(nums)>1 else 0, key="t3_sc_y")
            color_c = c3.selectbox("Color Mapping", ["None"] + list(df.columns), key="t3_sc_color")
            sample = df.sample(min(4000, len(df)), random_state=42)
            fig_sc = px.scatter(
                sample, x=x_col, y=y_col,
                color=None if color_c == "None" else color_c,
                color_continuous_scale="Plasma",
                title=f"{y_col} versus {x_col}",
                opacity=0.7
            )
            fig_sc.update_layout(height=460, **P_LAYOUT)
            st.plotly_chart(fig_sc, use_container_width=True, key="t3_scatter_chart")

    elif v_mode == "🎻 Violin & Box Plots":
        if not nums:
            st.warning("No numeric columns.")
        else:
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Metric Column", nums, key="t3_vio_y")
            x_col = c2.selectbox("Group By (Categorical)", ["None"] + cats, key="t3_vio_x")
            fig_v = px.violin(
                df.sample(min(5000, len(df)), random_state=42),
                y=y_col, x=None if x_col == "None" else x_col,
                color=None if x_col == "None" else x_col,
                box=True, points="outliers",
                color_discrete_sequence=px.colors.qualitative.Prism,
                title=f"Violin & Box Plot of {y_col}"
            )
            fig_v.update_layout(height=460, **P_LAYOUT)
            st.plotly_chart(fig_v, use_container_width=True, key="t3_violin_chart")

    elif v_mode == "🌊 Feature Intensity Heatmap":
        if not nums:
            st.warning("No numeric columns.")
        else:
            sample_h = df[nums].sample(min(250, len(df)), random_state=42)
            fig_h = px.imshow(sample_h.T, aspect="auto", color_continuous_scale="Viridis", title="Feature Intensity Grid Across Sampled Observations")
            fig_h.update_layout(height=max(320, len(nums) * 35), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(fig_h, use_container_width=True, key="t3_fheat_chart")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA BRIEF
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if df is None:
        st.info("Upload your dataset in Tab 1 to generate an analytical brief.")
        st.stop()

    st.markdown("""
    <div style="padding:15px 0 10px">
        <div style="font-size:2rem;font-weight:900;background:linear-gradient(90deg,#10b981,#38bdf8,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Automated Data Health Brief & Diagnostics
        </div>
        <div style="color:#94a3b8">Comprehensive quality metrics, outlier flags, and predictive readiness</div>
    </div>
    """, unsafe_allow_html=True)

    miss_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    dup_pct  = df.duplicated().sum() / len(df) * 100
    nums_df  = df.select_dtypes(include="number")

    n_outlier_cols = 0
    for c in nums_df.columns:
        s_col = nums_df[c].dropna()
        q1, q3 = s_col.quantile(0.25), s_col.quantile(0.75)
        if ((s_col < q1 - 1.5 * (q3 - q1)) | (s_col > q3 + 1.5 * (q3 - q1))).any():
            n_outlier_cols += 1

    health_score = max(0, 100 - miss_pct * 2.5 - dup_pct * 2 - (n_outlier_cols / max(len(nums_df.columns), 1)) * 10)
    score_clr = "#10b981" if health_score >= 80 else "#f59e0b" if health_score >= 60 else "#ef4444"

    c_left, c_right = st.columns([1, 2], gap="large")

    with c_left:
        st.markdown(f"""
        <div class="dm-brief-card" style="text-align:center">
            <div style="color:#94a3b8;font-size:0.85rem;font-weight:800;letter-spacing:0.1em;text-transform:uppercase">
                Data Readiness Index
            </div>
            <div style="font-size:5rem;font-weight:900;color:{score_clr};line-height:1.1;margin:10px 0;filter:drop-shadow(0 0 20px {score_clr}66)">
                {health_score:.0f}
            </div>
            <div style="color:{score_clr};font-size:1.1rem;font-weight:800">
                {"EXCELLENT (Production Ready)" if health_score>=80 else "GOOD (Minor Cleanup Recommended)" if health_score>=60 else "NEEDS PREPROCESSING"}
            </div>
            <div style="margin-top:20px;text-align:left;font-size:0.86rem">
                <div style="display:flex;justify-content:space-between;padding:4px 0"><span>Null Ratio:</span><b>{miss_pct:.2f}%</b></div>
                <div style="display:flex;justify-content:space-between;padding:4px 0"><span>Duplicate Rows:</span><b>{dup_pct:.2f}%</b></div>
                <div style="display:flex;justify-content:space-between;padding:4px 0"><span>Outlier Columns:</span><b>{n_outlier_cols} / {len(nums_df.columns)}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="dm-section-title">✨ Key Strategic Insights</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="dm-insight di-pink">
            📊 <b>Volume Check:</b> Processing <b>{df.shape[0]:,} records</b> across <b>{df.shape[1]} features</b>.
        </div>
        """, unsafe_allow_html=True)

        if miss_pct == 0:
            st.markdown("""
            <div class="dm-insight di-green">
                ✨ <b>Zero Missing Values:</b> Dataset contains complete records without any missing data.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="dm-insight di-orange">
                ⚠️ <b>Imputation Alert:</b> {miss_pct:.2f}% of cells are null. Auto-imputation is applied during preprocessing.
            </div>
            """, unsafe_allow_html=True)

        if dup_pct > 0:
            st.markdown(f"""
            <div class="dm-insight di-orange">
                ⚠️ <b>Duplicates Detected:</b> {df.duplicated().sum():,} duplicate rows ({dup_pct:.1f}%).
            </div>
            """, unsafe_allow_html=True)

        tgt = st.session_state.get("target")
        if tgt and tgt in df.columns:
            st.markdown(f"""
            <div class="dm-insight di-blue">
                🎯 <b>Target Variable:</b> Configured to predict <b>`{tgt}`</b>.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEEP DIVE (POWERED CHATBOT & INTERACTIVE ASSISTANT)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div style="padding:15px 0 10px">
        <div style="font-size:2rem;font-weight:900;background:linear-gradient(90deg,#ec4899,#8b5cf6,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            💬 DataMind Deep Dive & Interactive Query Studio
        </div>
        <div style="color:#94a3b8">Ask anything about your data, models, architecture, or diagnostics</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick action prompt pills
    st.markdown('<div class="dm-section-title" style="margin-top:10px">💡 Quick Deep Dive Questions</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    q_sugs = [
        "Show me the neural network mapping",
        "Which model has the highest accuracy?",
        "Are there any outliers in the data?",
        "How many missing values are present?",
        "What are the feature correlations?",
        "How do I improve model accuracy?",
        "Explain Random Forest vs Boosting",
        "Summarize this entire dataset"
    ]

    for idx, q_text in enumerate(q_sugs):
        col_target = [c1, c2, c3, c4][idx % 4]
        if col_target.button(q_text, key=f"t5_sug_{idx}", use_container_width=True):
            reply = bot_answer(q_text, st.session_state["df"], st.session_state["results"])
            st.session_state["chat_history"].append({"q": q_text, "a": reply})
            st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:20px 0'>", unsafe_allow_html=True)

    # Chat history display
    if not st.session_state["chat_history"]:
        st.markdown("""
        <div style="text-align:center;padding:50px 0;color:#64748b">
            <div style="font-size:3.5rem;filter:drop-shadow(0 0 15px rgba(236,72,153,0.3))">💬</div>
            <div style="font-size:1.15rem;font-weight:700;margin-top:12px;color:#cbd5e1">No queries in Deep Dive yet</div>
            <div style="font-size:0.9rem;color:#64748b">Type any question below or click any of the suggestion buttons above!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, item in enumerate(st.session_state["chat_history"]):
            st.markdown(f"""
            <div class="dm-bubble-wrap">
                <div class="dm-bubble-user">
                    <div>{item['q']}</div>
                </div>
                <div style="font-size:0.75rem;color:#94a3b8;text-align:right;margin:2px 6px">You</div>
                <div style="height:6px"></div>
                <div class="dm-ai-label">⚡ DATAMIND INTELLIGENCE</div>
                <div class="dm-bubble-ai">
                    <div>{item['a']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Input form for queries
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.form("t5_chat_form", clear_on_submit=True):
        q_user = st.text_input("Ask a question about your data or models:", placeholder="e.g. Which model performed best and why?", key="t5_user_input")
        btn_c1, btn_c2 = st.columns([5, 1])
        submitted = btn_c1.form_submit_button("Ask DataMind ⚡", type="primary", use_container_width=True)
        clear_chat = btn_c2.form_submit_button("Clear 🗑", use_container_width=True)

        if submitted and q_user.strip():
            rep = bot_answer(q_user, st.session_state["df"], st.session_state["results"])
            st.session_state["chat_history"].append({"q": q_user, "a": rep})
            st.rerun()

        if clear_chat:
            st.session_state["chat_history"] = []
            st.rerun()
