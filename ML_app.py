# app.py
import time
import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Predictive Maintenance – Failure Type",
    page_icon="🔧",
    layout="wide",
)

# -------------------- theme / layout CSS -------------------
st.markdown("""
<style>
.main .block-container{padding-top:1rem; padding-bottom:2rem; max-width:1200px;}
.header {
  display:flex; justify-content:space-between; align-items:center;
  padding: 8px 14px; border:1px solid #1f2a44; border-radius:12px;
  background:#0f172a; margin-bottom:14px;
}
.header h1 {font-size:1.35rem; margin:0;}
.header .sub {color:#94a3b8; font-size:0.9rem; margin-top:4px;}
.badge {background:#0b1220; border:1px solid #324057; padding:6px 10px; border-radius:10px;}
.metrics .stMetric {background:#0f172a; border:1px solid #1f2a44; border-radius:14px; padding:14px;}
.metrics [data-testid="stMetricValue"] {font-size:1.4rem;}
.section-title {font-weight:600; margin:6px 0 6px 0;}
.small {color:#9aa4b2; font-size:0.9rem;}
hr {border:none; border-top:1px solid #1f2a44; margin:12px 0;}
.stButton>button {border-radius:10px; padding:9px 14px;}
</style>
""", unsafe_allow_html=True)

# ========================= MODEL & CONSTANTS =========================
MODEL_PATH = pathlib.Path("models/rf_failure_predictor.pkl")

FEATURE_ORDER = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

LABELS = {
    0: "No Failure",
    1: "TWF (Tool Wear Failure)",
    2: "HDF (Heat Dissipation Failure)",
    3: "PWF (Power Failure)",
    4: "OSF (Overstrain Failure)",
    5: "RNF (Random Failure)"
}

@st.cache_resource(show_spinner=False)
def safe_load_model(path: pathlib.Path):
    """Return (model, error_message)."""
    if not path.exists():
        return None, f"Model file not found: {path}"
    try:
        m = joblib.load(path.as_posix())
        if not hasattr(m, "predict"):
            return None, "Loaded object has no .predict(). Is this the right file?"
        return m, None
    except Exception as e:
        return None, f"Failed to load model: {type(e).__name__}: {e}"

# ---- Demo model (fallback) ----
class DummyModel:
    def __init__(self, n_classes=6, seed=42):
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.w = np.array([0.5, 0.7, 0.02, 0.08, 0.03])

    def _scores(self, X: np.ndarray):
        base = X @ self.w
        logits = np.vstack([
            -0.4 + 0.01*base,   # 0
            -0.8 + 0.02*base,   # 1
            -0.6 + 0.03*base,   # 2
            -0.7 + 0.025*base,  # 3
            -0.5 + 0.015*base,  # 4
            -1.0 + 0.02*base,   # 5
        ]).T
        return logits

    def predict_proba(self, X_df: pd.DataFrame):
        X = X_df.to_numpy(dtype=float)
        logits = self._scores(X)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X_df: pd.DataFrame):
        p = self.predict_proba(X_df)
        return np.argmax(p, axis=1)

def get_model_or_demo(path: pathlib.Path, use_demo: bool):
    real_model, err = safe_load_model(path)
    if real_model is not None:
        return real_model, True, None
    if use_demo:
        return DummyModel(), False, err
    return None, False, err

# ---- Single row & predict ----
def make_single_row(air, proc, rpm, torq, wear):
    return pd.DataFrame([{
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torq,
        "Tool wear [min]": wear
    }])[FEATURE_ORDER]

def predict_one(pipeline, X_df: pd.DataFrame):
    t0 = time.perf_counter()
    y = int(pipeline.predict(X_df)[0])
    proba = pipeline.predict_proba(X_df)[0] if hasattr(pipeline, "predict_proba") else None
    ms = (time.perf_counter()-t0)*1000.0
    return y, proba, ms

# ========================= Helpers untuk Batch CSV =========================
REQUIRED_COLS = {
    "air temperature [k]":    "Air temperature [K]",
    "process temperature [k]":"Process temperature [K]",
    "rotational speed [rpm]": "Rotational speed [rpm]",
    "torque [nm]":            "Torque [Nm]",
    "tool wear [min]":        "Tool wear [min]",
}

def make_template(n=20) -> pd.DataFrame:
    return pd.DataFrame({
        "Air temperature [K]":     np.round(np.random.uniform(295, 306, n), 2),
        "Process temperature [K]": np.round(np.random.uniform(305, 316, n), 2),
        "Rotational speed [rpm]":  np.random.randint(1160, 2860, n),
        "Torque [Nm]":             np.round(np.random.uniform(4, 80, n), 2),
        "Tool wear [min]":         np.random.randint(0, 260, n),
    })[FEATURE_ORDER]

def read_csv_smart(file) -> pd.DataFrame:
    file.seek(0)
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, engine="python")
    if df.shape[1] == 1:
        file.seek(0)
        try:
            df = pd.read_csv(file, sep=";", engine="python")
        except Exception:
            file.seek(0)
            df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df

def normalize_schema(df: pd.DataFrame):
    lower_map = {c.lower(): c for c in df.columns}
    renamed, missing = {}, []
    for want_lower, want_exact in REQUIRED_COLS.items():
        if want_lower in lower_map:
            orig = lower_map[want_lower]
            renamed[orig] = want_exact
        else:
            candidates = {c.replace(" ", "").lower(): c for c in df.columns}
            key = want_lower.replace(" ", "")
            if key in candidates:
                orig = candidates[key]
                renamed[orig] = want_exact
            else:
                missing.append(want_exact)
    df2 = df.rename(columns=renamed, errors="ignore")
    have = [c for c in FEATURE_ORDER if c in df2.columns]
    df2 = df2[have] if have else df2
    return df2, missing, renamed

# ========================= SIDEBAR (INPUTS) =========================
with st.sidebar:
    st.header("⚙️ Inputs")
    st.caption("Set sensor values, then click **Predict**.")
    air  = st.slider("Air temperature [K]",      295.0, 305.5, 300.0, 0.1)
    proc = st.slider("Process temperature [K]",  305.0, 315.5, 310.0, 0.1)
    rpm  = st.slider("Rotational speed [rpm]",   1160, 2860, 1450, 1)
    torq = st.slider("Torque [Nm]",              3.8,  80.0,  45.0, 0.1)
    wear = st.slider("Tool wear [min]",          0,    260,   100,  1)

    st.markdown("---")
    show_probs = st.toggle("Show probabilities", True)
    show_shap  = st.toggle("Explain with SHAP (optional)", False,
                           help="Requires `shap`; adds some latency.")

    st.markdown("---")
    demo_mode = st.toggle("Use demo model if real model missing", value=True,
                          help="Aktif: pakai simulasi agar app bisa dicoba tanpa file .pkl")

    st.markdown("---")
    st.caption("📄 Batch prediction (CSV)")
    up = st.file_uploader("Upload CSV", type=["csv"],
                          help="Columns: " + ", ".join(FEATURE_ORDER), key="uploader_sidebar")

# ========================= LOAD MODEL =========================
model, is_real, model_err = get_model_or_demo(MODEL_PATH, use_demo=demo_mode)
model_ready = model is not None

# ========================= HEADER STRIP =========================
st.markdown(
    f"""
    <div class="header">
      <div>
        <h1>🔧 Predictive Maintenance – Failure Type Classifier</h1>
        <div class="sub">Real-time inference & batch scoring • Random Forest pipeline</div>
      </div>
      <div class="badge">
        {'🟢 Real Model Loaded' if (model_ready and is_real) else ('🟡 Demo Model Active' if model_ready else '🔴 Model Missing')}
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ========================= SINGLE PREDICTION =========================
st.markdown('<div class="section-title">Single Prediction</div>', unsafe_allow_html=True)
X_one = make_single_row(air, proc, rpm, torq, wear)
pred_btn = st.button("Predict", key="predict_one", use_container_width=False, disabled=not model_ready)

y = None; proba = None; ms = None
if pred_btn and model_ready:
    y, proba, ms = predict_one(model, X_one)

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.markdown('<div class="metrics">', unsafe_allow_html=True)
    label_text = "—" if y is None else LABELS.get(y, str(y))
    st.metric("Predicted class", label_text)
    st.caption("" if y is None else f"Class ID: {y}")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metrics">', unsafe_allow_html=True)
    conf = (np.max(proba)*100 if proba is not None else None)
    st.metric("Confidence", "—" if conf is None else f"{conf:.1f}%")
    st.caption("Max predicted probability")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metrics">', unsafe_allow_html=True)
    st.metric("Latency", "—" if ms is None else f"{ms:.1f} ms")
    st.caption("End-to-end inference time")
    st.markdown('</div>', unsafe_allow_html=True)

if show_probs and proba is not None:
    st.markdown("#### Class probabilities")
    prob_df = pd.DataFrame({"Class":[LABELS[i] for i in range(len(proba))], "Probability":proba}) \
                 .sort_values("Probability", ascending=False)
    st.bar_chart(prob_df.set_index("Class"))

st.markdown("---")

# ========================= BATCH PREDICTION =========================
st.markdown('<div class="section-title">Batch Prediction</div>', unsafe_allow_html=True)

# Template download
if st.button("⬇️ Download CSV template", key="dl_template_btn"):
    tmpl = make_template(20)
    st.download_button(
        label="Click to download",
        data=tmpl.to_csv(index=False).encode("utf-8"),
        file_name="predict_template.csv",
        mime="text/csv",
        key="dl_template_file"
    )

uploaded_here = st.file_uploader(
    "Or upload CSV here (optional)",
    type=["csv"],
    key="uploader_center",
    help="Headers can be case-insensitive; we’ll normalize them."
)

file_obj = uploaded_here or up

if file_obj is None:
    st.info("Upload a CSV or use the template to run batch predictions.")
else:
    raw = read_csv_smart(file_obj)
    norm, missing, renamed = normalize_schema(raw)

    with st.expander("Preview & schema check", expanded=False):
        st.write("First 5 rows (raw):")
        st.dataframe(raw.head(), use_container_width=True)
        if renamed:
            st.write("Renamed → required names:", renamed)
        st.write("Columns after normalization:", list(norm.columns))
        if missing:
            st.error(f"Missing required columns: {missing}")

    if not missing:
        run_batch = st.button("Run batch predict", key="run_batch_btn", disabled=not model_ready)
        if run_batch:
            if not model_ready:
                st.warning("Model not loaded – batch prediction disabled.")
            else:
                norm = norm[FEATURE_ORDER]
                preds = model.predict(norm)
                out = norm.copy()
                out["Prediction_ID"]    = preds
                out["Prediction_Label"] = [LABELS.get(int(p), str(int(p))) for p in preds]

                st.success(f"Done. Rows predicted: {len(out)}")
                st.dataframe(out.head(50), use_container_width=True)
                st.download_button(
                    "⬇️ Download results",
                    out.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv",
                    key="download_batch_file"
                )
