import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import time

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tic-Tac-Toe AI Arena",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

/* ── Root variables ── */
:root {
    --neon-blue:   #00d4ff;
    --neon-purple: #b347ea;
    --neon-green:  #00ff88;
    --neon-pink:   #ff006e;
    --neon-gold:   #ffd700;
    --dark-bg:     #0a0a1a;
    --card-bg:     #12122a;
    --card-border: #2a2a5a;
}

/* ── Global background ── */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2a 50%, #0a0a1a 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Animated starfield background ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(1px 1px at 10% 20%, rgba(0,212,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 60%, rgba(179,71,234,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 10%, rgba(0,255,136,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 80%, rgba(255,0,110,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 50%, rgba(0,212,255,0.2) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hero title ── */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #00d4ff, #b347ea, #00ff88, #ffd700, #00d4ff);
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s infinite linear;
    text-shadow: none;
    margin-bottom: 0.2rem;
}

@keyframes shimmer {
    0%   { background-position: 0%; }
    100% { background-position: 300%; }
}

.hero-sub {
    font-family: 'Orbitron', monospace;
    font-size: 0.95rem;
    text-align: center;
    color: #7a7aaa;
    letter-spacing: 0.15em;
    margin-bottom: 2rem;
}

/* ── Glassmorphism cards ── */
.glass-card {
    background: rgba(18, 18, 42, 0.85);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 30px rgba(0,212,255,0.08), inset 0 0 30px rgba(179,71,234,0.03);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.glass-card:hover {
    box-shadow: 0 0 50px rgba(0,212,255,0.2), inset 0 0 30px rgba(179,71,234,0.05);
    transform: translateY(-2px);
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(18, 18, 42, 0.9);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover { transform: scale(1.04); }
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.metric-label {
    font-size: 0.75rem;
    color: #7a7aaa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Neon borders by color ── */
.border-blue   { border: 1px solid rgba(0,212,255,0.5);   box-shadow: 0 0 20px rgba(0,212,255,0.15); }
.border-purple { border: 1px solid rgba(179,71,234,0.5);  box-shadow: 0 0 20px rgba(179,71,234,0.15); }
.border-green  { border: 1px solid rgba(0,255,136,0.5);   box-shadow: 0 0 20px rgba(0,255,136,0.15); }
.border-gold   { border: 1px solid rgba(255,215,0,0.5);   box-shadow: 0 0 20px rgba(255,215,0,0.15); }
.border-pink   { border: 1px solid rgba(255,0,110,0.5);   box-shadow: 0 0 20px rgba(255,0,110,0.15); }

/* ── Section headers ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: #00d4ff;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,212,255,0.3);
    letter-spacing: 0.1em;
}

/* ── Board cell ── */
.board-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    max-width: 260px;
    margin: 0 auto;
}
.board-cell {
    width: 80px; height: 80px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 12px;
    font-size: 2rem; font-weight: 900;
    cursor: pointer;
    transition: all 0.25s ease;
    font-family: 'Orbitron', monospace;
}
.cell-x {
    background: rgba(0,212,255,0.15);
    border: 2px solid rgba(0,212,255,0.6);
    color: #00d4ff;
    text-shadow: 0 0 15px #00d4ff;
}
.cell-o {
    background: rgba(179,71,234,0.15);
    border: 2px solid rgba(179,71,234,0.6);
    color: #b347ea;
    text-shadow: 0 0 15px #b347ea;
}
.cell-b {
    background: rgba(42,42,90,0.5);
    border: 2px dashed rgba(100,100,160,0.4);
    color: transparent;
}
.board-cell:hover { transform: scale(1.08); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d22 0%, #0a0a1a 100%);
    border-right: 1px solid rgba(0,212,255,0.15);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #a0a0cc;
    font-size: 0.85rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(18,18,42,0.8);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    color: #7a7aaa;
    padding: 8px 16px;
    transition: all 0.3s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(179,71,234,0.2));
    color: #00d4ff !important;
    border: 1px solid rgba(0,212,255,0.4);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(179,71,234,0.2));
    color: #00d4ff;
    border: 1px solid rgba(0,212,255,0.5);
    border-radius: 10px;
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    padding: 0.5rem 1.5rem;
    transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.35), rgba(179,71,234,0.35));
    box-shadow: 0 0 20px rgba(0,212,255,0.4);
    transform: translateY(-1px);
}

/* ── Prediction result banners ── */
.result-win {
    background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,212,255,0.15));
    border: 1px solid rgba(0,255,136,0.5);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
.result-lose {
    background: linear-gradient(135deg, rgba(255,0,110,0.15), rgba(179,71,234,0.15));
    border: 1px solid rgba(255,0,110,0.5);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 15px rgba(0,255,136,0.2); }
    50%      { box-shadow: 0 0 35px rgba(0,255,136,0.5); }
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 15px rgba(255,0,110,0.2); }
    50%      { box-shadow: 0 0 35px rgba(255,0,110,0.5); }
}

.result-text {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
}

/* ── Progress / loading ── */
.stProgress > div > div { background: linear-gradient(90deg, #00d4ff, #b347ea); }

/* ── DataFrame ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a1a; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Data helpers ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("tic_tac_toe.csv")
    return df


@st.cache_data
def preprocess(df):
    le = LabelEncoder()
    X = df.drop("class", axis=1).apply(lambda col: le.fit_transform(col))
    y = (df["class"] == "positive").astype(int)
    return X, y


@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        cv     = cross_val_score(mdl, X, y, cv=5, scoring="accuracy")
        results[name] = {
            "model":     mdl,
            "accuracy":  accuracy_score(y_test, y_pred),
            "cv_mean":   cv.mean(),
            "cv_std":    cv.std(),
            "y_test":    y_test,
            "y_pred":    y_pred,
            "X_test":    X_test,
        }
    return results, X_train, X_test, y_train, y_test


# ─── Plotly theme ────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a0a0cc", family="Inter"),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor="#12122a", bordercolor="#2a2a5a", font_color="#e0e0ff"),
)
NEON = ["#00d4ff", "#b347ea", "#00ff88", "#ffd700", "#ff006e", "#ff6b35"]


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-family:Orbitron,monospace; font-size:1.4rem; font-weight:900;
                    background:linear-gradient(90deg,#00d4ff,#b347ea);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            🎮 TTT AI ARENA
        </div>
        <div style='font-size:0.7rem; color:#555577; letter-spacing:0.15em; margin-top:4px;'>
            ENDGAME INTELLIGENCE
        </div>
    </div>
    <hr style='border-color:rgba(0,212,255,0.15); margin:0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    selected_model = st.selectbox(
        "🤖 Active Model",
        ["Random Forest", "Gradient Boosting", "Decision Tree", "K-Nearest Neighbors"],
    )
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#7a7aaa;'>DISPLAY OPTIONS</div>",
                unsafe_allow_html=True)
    show_raw    = st.checkbox("Show Raw Dataset", value=False)
    show_matrix = st.checkbox("Confusion Matrix", value=True)
    show_roc    = st.checkbox("ROC Curve",        value=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#555577; line-height:1.6;'>
        <b style='color:#7a7aaa;'>Dataset:</b> UCI Tic-Tac-Toe Endgame<br>
        <b style='color:#7a7aaa;'>Instances:</b> 958<br>
        <b style='color:#7a7aaa;'>Features:</b> 9 board positions<br>
        <b style='color:#7a7aaa;'>Classes:</b> X wins / X doesn't win
    </div>
    """, unsafe_allow_html=True)


# ─── Load data ───────────────────────────────────────────────────────────────
df = load_data()
X, y = preprocess(df)
results, X_train, X_test, y_train, y_test = train_models(X, y)
active = results[selected_model]

# ─── Hero header ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎮 TIC-TAC-TOE AI ARENA</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ENDGAME STATE INTELLIGENCE · MACHINE LEARNING ANALYSIS</div>',
            unsafe_allow_html=True)

# ─── KPI strip ───────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1, f"{active['accuracy']*100:.1f}%", "MODEL ACCURACY", "#00d4ff", "border-blue"),
    (k2, f"{active['cv_mean']*100:.1f}%",  "CV SCORE (5-FOLD)", "#b347ea", "border-purple"),
    (k3, "958",                             "ENDGAME STATES",   "#00ff88", "border-green"),
    (k4, "65.3%",                           "X WIN RATE",       "#ffd700", "border-gold"),
]
for col, val, lbl, color, bdr in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card {bdr}">
            <div class="metric-value" style="color:{color};">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main tabs ───────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "🤖 Model Analysis", "🎯 Live Predictor", "📈 Feature Intelligence"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-header">🎯 CLASS DISTRIBUTION</div>', unsafe_allow_html=True)
        class_counts = df["class"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["X Wins (Positive)", "X Doesn't Win (Negative)"],
            values=class_counts.values,
            hole=0.55,
            marker=dict(
                colors=["#00d4ff", "#b347ea"],
                line=dict(color="#0a0a1a", width=3),
            ),
            textinfo="percent+label",
            textfont=dict(color="#e0e0ff", size=12),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b style='font-size:16px'>958</b><br><span style='font-size:10px'>STATES</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#e0e0ff", size=14),
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=320,
                              legend=dict(font=dict(color="#a0a0cc")))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">📡 MODEL COMPARISON</div>', unsafe_allow_html=True)
        model_names = list(results.keys())
        accuracies  = [results[m]["accuracy"]*100 for m in model_names]
        cv_means    = [results[m]["cv_mean"]*100   for m in model_names]
        short_names = ["Rand. Forest", "Grad. Boost", "Dec. Tree", "KNN"]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=short_names, y=accuracies, name="Test Accuracy",
            marker=dict(color=NEON[0], opacity=0.85,
                        line=dict(color=NEON[0], width=1)),
            hovertemplate="<b>%{x}</b><br>Test Acc: %{y:.1f}%<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            x=short_names, y=cv_means, name="CV Mean",
            marker=dict(color=NEON[1], opacity=0.85,
                        line=dict(color=NEON[1], width=1)),
            hovertemplate="<b>%{x}</b><br>CV Mean: %{y:.1f}%<extra></extra>",
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT, height=320, barmode="group",
            yaxis=dict(title="Accuracy (%)", gridcolor="rgba(255,255,255,0.05)",
                       range=[80, 100]),
            legend=dict(font=dict(color="#a0a0cc")),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Board position frequency
    st.markdown('<div class="section-header">🕹️ BOARD POSITION ANALYSIS</div>', unsafe_allow_html=True)
    positions = [c for c in df.columns if c != "class"]
    pos_data  = {pos: df[pos].value_counts().to_dict() for pos in positions}
    pos_df    = pd.DataFrame(pos_data).T.fillna(0)

    fig_heat = go.Figure(go.Heatmap(
        z=pos_df.values,
        x=pos_df.columns,
        y=[p.replace("_", " ").title() for p in pos_df.index],
        colorscale=[[0, "#0a0a1a"], [0.5, "#b347ea"], [1, "#00d4ff"]],
        text=pos_df.values.astype(int),
        texttemplate="%{text}",
        hovertemplate="Position: %{y}<br>Symbol: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=280,
                           xaxis=dict(title="Symbol"),
                           yaxis=dict(title=""))
    st.plotly_chart(fig_heat, use_container_width=True)

    if show_raw:
        st.markdown('<div class="section-header">🗄️ RAW DATASET</div>', unsafe_allow_html=True)
        st.dataframe(df.style.applymap(
            lambda v: "color:#00d4ff" if v == "x" else
                      ("color:#b347ea" if v == "o" else "color:#555577")
        ), use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f'<div class="section-header">🤖 {selected_model.upper()} — DEEP ANALYSIS</div>',
                unsafe_allow_html=True)

    # Classification report
    report = classification_report(
        active["y_test"], active["y_pred"],
        target_names=["X Doesn't Win", "X Wins"], output_dict=True
    )
    r1, r2 = st.columns(2)
    metrics_data = [
        ("Precision (X Wins)", f"{report['X Wins']['precision']*100:.1f}%", "#00ff88", "border-green"),
        ("Recall (X Wins)",    f"{report['X Wins']['recall']*100:.1f}%",    "#00d4ff", "border-blue"),
        ("F1-Score",           f"{report['X Wins']['f1-score']*100:.1f}%",  "#b347ea", "border-purple"),
        ("CV Std Dev",         f"±{active['cv_std']*100:.2f}%",             "#ffd700", "border-gold"),
    ]
    for i, (lbl, val, color, bdr) in enumerate(metrics_data):
        col = r1 if i < 2 else r2
        with col:
            st.markdown(f"""
            <div class="metric-card {bdr}" style="margin-bottom:12px;">
                <div class="metric-value" style="color:{color};font-size:1.6rem;">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2)

    # Confusion matrix
    if show_matrix:
        with col_cm:
            st.markdown('<div class="section-header">🔷 CONFUSION MATRIX</div>',
                        unsafe_allow_html=True)
            cm = confusion_matrix(active["y_test"], active["y_pred"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale=[[0, "#0a0a1a"], [0.5, "#b347ea"], [1, "#00d4ff"]],
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["X Doesn't Win", "X Wins"],
                y=["X Doesn't Win", "X Wins"],
            )
            fig_cm.update_traces(textfont=dict(size=20, color="white"))
            fig_cm.update_layout(**PLOTLY_LAYOUT, height=320,
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    if show_roc:
        with col_roc:
            st.markdown('<div class="section-header">📈 ROC CURVE</div>',
                        unsafe_allow_html=True)
            mdl = active["model"]
            if hasattr(mdl, "predict_proba"):
                probs = mdl.predict_proba(active["X_test"])[:, 1]
            else:
                probs = active["y_pred"]
            fpr, tpr, _ = roc_curve(active["y_test"], probs)
            roc_auc     = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}",
                line=dict(color="#00d4ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random",
                line=dict(color="#555577", dash="dash", width=1),
            ))
            fig_roc.update_layout(
                **PLOTLY_LAYOUT, height=320,
                xaxis=dict(title="False Positive Rate",
                           gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="True Positive Rate",
                           gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(font=dict(color="#a0a0cc")),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # CV scores across all models
    st.markdown('<div class="section-header">📊 CROSS-VALIDATION SCORES</div>',
                unsafe_allow_html=True)
    cv_fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        cv_scores = cross_val_score(res["model"], X, y, cv=5, scoring="accuracy") * 100
        short = ["RF", "GB", "DT", "KNN"][i]
        cv_fig.add_trace(go.Box(
            y=cv_scores, name=short,
            marker_color=NEON[i],
            line_color=NEON[i],
            fillcolor=f"rgba({int(NEON[i][1:3],16)},{int(NEON[i][3:5],16)},{int(NEON[i][5:7],16)},0.15)",
            boxpoints="all", jitter=0.4, pointpos=0,
        ))
    cv_fig.update_layout(
        **PLOTLY_LAYOUT, height=350,
        yaxis=dict(title="Accuracy (%)", gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
    )
    st.plotly_chart(cv_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">🎯 LIVE BOARD PREDICTOR</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7a7aaa; font-size:0.85rem;'>Configure each board position below "
        "and the AI will predict whether <b style='color:#00d4ff'>X wins</b> "
        "this endgame state.</p>",
        unsafe_allow_html=True,
    )

    positions_map = {
        "top_left":      (0, 0), "top_middle":    (0, 1), "top_right":     (0, 2),
        "middle_left":   (1, 0), "middle_middle":  (1, 1), "middle_right":  (1, 2),
        "bottom_left":   (2, 0), "bottom_middle":  (2, 1), "bottom_right":  (2, 2),
    }
    pos_labels = {
        "top_left": "Top Left",       "top_middle": "Top Center",    "top_right": "Top Right",
        "middle_left": "Mid Left",    "middle_middle": "Center",     "middle_right": "Mid Right",
        "bottom_left": "Bot Left",    "bottom_middle": "Bot Center",  "bottom_right": "Bot Right",
    }

    col_controls, col_preview = st.columns([1, 1])

    board_state = {}
    with col_controls:
        st.markdown("**Set each position:**")
        grid_cols = [st.columns(3) for _ in range(3)]
        pos_list  = list(positions_map.keys())
        for idx, pos in enumerate(pos_list):
            row, col = idx // 3, idx % 3
            with grid_cols[row][col]:
                board_state[pos] = st.selectbox(
                    pos_labels[pos], ["x", "o", "b"],
                    key=f"sel_{pos}",
                    format_func=lambda v: {"x": "✕ X", "o": "○ O", "b": "· Blank"}[v],
                )

        quick = st.columns(3)
        with quick[0]:
            if st.button("⚡ X Wins Sample"):
                sample_positive = df[df["class"] == "positive"].iloc[0]
                for pos in pos_list:
                    st.session_state[f"sel_{pos}"] = sample_positive[pos]
                st.rerun()
        with quick[1]:
            if st.button("🎲 Random State"):
                sample = df.sample(1).iloc[0]
                for pos in pos_list:
                    st.session_state[f"sel_{pos}"] = sample[pos]
                st.rerun()
        with quick[2]:
            if st.button("🔄 Clear Board"):
                for pos in pos_list:
                    st.session_state[f"sel_{pos}"] = "b"
                st.rerun()

    with col_preview:
        st.markdown("**Board Preview:**")
        symbol_map = {"x": "✕", "o": "○", "b": ""}
        cells_html  = ""
        for pos in pos_list:
            sym = board_state[pos]
            cls = {"x": "cell-x", "o": "cell-o", "b": "cell-b"}[sym]
            cells_html += f'<div class="board-cell {cls}">{symbol_map[sym]}</div>'

        st.markdown(f"""
        <div style="background:rgba(18,18,42,0.8); border:1px solid rgba(0,212,255,0.2);
                    border-radius:16px; padding:1.5rem; display:flex;
                    justify-content:center; align-items:center; min-height:220px;">
            <div class="board-grid">{cells_html}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    pred_col = st.columns([1, 2, 1])[1]
    with pred_col:
        predict_btn = st.button("🔮 ANALYZE BOARD STATE", use_container_width=True)

    if predict_btn:
        with st.spinner("Running AI analysis..."):
            time.sleep(0.6)
            le = LabelEncoder()
            le.fit(["b", "o", "x"])
            input_vec = np.array([[le.transform([board_state[p]])[0] for p in pos_list]])
            mdl = active["model"]
            pred_class = mdl.predict(input_vec)[0]
            confidence = (
                mdl.predict_proba(input_vec)[0].max() * 100
                if hasattr(mdl, "predict_proba") else None
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if pred_class == 1:
            conf_txt = f"<br><span style='font-size:1rem;color:#00ff88;'>Confidence: {confidence:.1f}%</span>" if confidence else ""
            st.markdown(f"""
            <div class="result-win">
                <div class="result-text" style="color:#00ff88;">🏆 X WINS THIS STATE!</div>
                {conf_txt}
            </div>""", unsafe_allow_html=True)
        else:
            conf_txt = f"<br><span style='font-size:1rem;color:#ff006e;'>Confidence: {confidence:.1f}%</span>" if confidence else ""
            st.markdown(f"""
            <div class="result-lose">
                <div class="result-text" style="color:#ff006e;">❌ X DOES NOT WIN THIS STATE</div>
                {conf_txt}
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Feature Intelligence
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">📈 FEATURE INTELLIGENCE</div>',
                unsafe_allow_html=True)

    f1, f2 = st.columns(2)

    # Feature importance (for tree-based models)
    mdl = active["model"]
    if hasattr(mdl, "feature_importances_"):
        with f1:
            st.markdown("**Feature Importances**", unsafe_allow_html=False)
            feat_imp = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=True)
            fig_imp  = go.Figure(go.Bar(
                x=feat_imp.values * 100,
                y=[p.replace("_", " ").title() for p in feat_imp.index],
                orientation="h",
                marker=dict(
                    color=feat_imp.values,
                    colorscale=[[0, "#b347ea"], [0.5, "#00d4ff"], [1, "#00ff88"]],
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
            ))
            fig_imp.update_layout(
                **PLOTLY_LAYOUT, height=360,
                xaxis=dict(title="Importance (%)", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        with f1:
            st.info("Feature importances are not available for this model type.")

    # Win rate by position
    with f2:
        st.markdown("**X Win Rate by Position**", unsafe_allow_html=False)
        win_rates = {}
        for pos in [c for c in df.columns if c != "class"]:
            pos_df_x = df[df[pos] == "x"]
            if len(pos_df_x) > 0:
                win_rates[pos.replace("_", " ").title()] = (
                    (pos_df_x["class"] == "positive").mean() * 100
                )
        wr_series = pd.Series(win_rates).sort_values()
        fig_wr = go.Figure(go.Bar(
            x=wr_series.values,
            y=wr_series.index,
            orientation="h",
            marker=dict(
                color=wr_series.values,
                colorscale=[[0, "#b347ea"], [0.5, "#ffd700"], [1, "#00ff88"]],
                line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            ),
            hovertemplate="<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>",
        ))
        fig_wr.update_layout(
            **PLOTLY_LAYOUT, height=360,
            xaxis=dict(title="Win Rate (%)", gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_wr, use_container_width=True)

    # Symbol usage vs outcome
    st.markdown('<div class="section-header">🔬 SYMBOL DISTRIBUTION BY OUTCOME</div>',
                unsafe_allow_html=True)
    dfw = df[df["class"] == "positive"].drop("class", axis=1)
    dfl = df[df["class"] == "negative"].drop("class", axis=1)

    sym_win  = dfw.apply(pd.Series.value_counts).T.fillna(0)
    sym_lose = dfl.apply(pd.Series.value_counts).T.fillna(0)

    fig_sym = make_subplots(
        rows=1, cols=2,
        subplot_titles=["X Wins States", "X Doesn't Win States"],
    )
    for sym, color in [("x", "#00d4ff"), ("o", "#b347ea"), ("b", "#555577")]:
        if sym in sym_win.columns:
            fig_sym.add_trace(go.Bar(
                x=[p.replace("_", " ").title() for p in sym_win.index],
                y=sym_win[sym], name=f"'{sym}'",
                marker_color=color, opacity=0.85, showlegend=True,
            ), row=1, col=1)
        if sym in sym_lose.columns:
            fig_sym.add_trace(go.Bar(
                x=[p.replace("_", " ").title() for p in sym_lose.index],
                y=sym_lose[sym], name=f"'{sym}'",
                marker_color=color, opacity=0.85, showlegend=False,
            ), row=1, col=2)

    fig_sym.update_layout(
        **PLOTLY_LAYOUT, height=360, barmode="stack",
        legend=dict(font=dict(color="#a0a0cc")),
    )
    fig_sym.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_sym, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(0,212,255,0.1); margin-top:2rem;'>
<div style='text-align:center; font-family:Orbitron,monospace; font-size:0.7rem;
            color:#555577; letter-spacing:0.1em; padding:1rem 0;'>
    🎮 TIC-TAC-TOE AI ARENA · UCI ENDGAME DATASET · BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)
