from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from text_utils import normalize_queries

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hinglish_intent_model.joblib"
METRICS_PATH = ROOT / "outputs" / "validation_metrics.json"
CLASS_REPORT_PATH = ROOT / "outputs" / "classwise_validation_report.csv"
MISTAKES_PATH = ROOT / "outputs" / "validation_misclassified.csv"

st.set_page_config(
    page_title="BhashaSense | Hinglish Intent AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e8eefc; --muted:#93a2bf; --panel:#111b31; --line:rgba(148,163,184,.16); --mint:#5eead4; --blue:#7aa7ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 15% 0%, rgba(57,92,160,.23), transparent 35%), linear-gradient(145deg,#080d19 0%,#0b1426 52%,#0a1020 100%); color:var(--ink); }
.block-container { max-width:1420px; padding-top:1.65rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0c1528 0%,#0b1221 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.55rem .65rem; border-radius:10px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(122,167,255,.10); }
.stTextArea textarea, .stTextInput input { background:#0c172b!important; color:var(--ink)!important; border:1px solid rgba(148,163,184,.25)!important; border-radius:13px!important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color:var(--mint)!important; box-shadow:0 0 0 1px var(--mint)!important; }
.stButton button, .stDownloadButton button, .stFormSubmitButton button { border-radius:11px; border:1px solid rgba(94,234,212,.35); background:linear-gradient(110deg,#145b66,#24528b); color:white; font-weight:700; min-height:2.6rem; }
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover { border-color:var(--mint); color:white; }
[data-testid="stMetric"] { background:rgba(17,27,49,.88); border:1px solid var(--line); border-radius:16px; padding:15px 17px; }
[data-testid="stMetricLabel"] { color:#9aa9c4!important; }
[data-testid="stMetricValue"] { color:#f1f5ff!important; font-family:'Space Grotesk',sans-serif; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:13px; overflow:hidden; }
hr { border-color:var(--line); }
.eyebrow { color:var(--mint); text-transform:uppercase; letter-spacing:.17em; font-size:.72rem; font-weight:700; margin-bottom:.5rem; }
.hero { background:linear-gradient(115deg,rgba(23,48,79,.96),rgba(18,31,55,.92) 58%,rgba(17,49,62,.9)); border:1px solid rgba(122,167,255,.22); border-radius:22px; padding:28px 30px; margin-bottom:20px; box-shadow:0 18px 50px rgba(0,0,0,.18); }
.hero h1 { font-family:'Space Grotesk',sans-serif; color:#f4f7ff; font-size:2.25rem; line-height:1.1; margin:.1rem 0 .7rem 0; }
.hero p { color:#b3c2dc; max-width:780px; font-size:1rem; margin:0; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; color:#a8fff0; background:rgba(94,234,212,.1); border:1px solid rgba(94,234,212,.24); font-size:.72rem; font-weight:700; letter-spacing:.04em; }
.panel { background:rgba(17,27,49,.84); border:1px solid var(--line); border-radius:17px; padding:18px 20px; }
.panel-title { color:#f1f5ff; font-family:'Space Grotesk',sans-serif; font-size:1.06rem; font-weight:700; margin-bottom:.35rem; }
.muted { color:var(--muted); font-size:.88rem; }
.prediction { background:linear-gradient(130deg,rgba(17,76,77,.38),rgba(29,54,94,.38)); border:1px solid rgba(94,234,212,.28); border-radius:19px; padding:22px; }
.prediction .big { font-family:'Space Grotesk',sans-serif; font-size:2.1rem; font-weight:700; color:#f3fffd; }
.prediction .sub { color:#a9c3d2; font-size:.87rem; }
.small-note { color:#9baac2; font-size:.78rem; }
footer { visibility:hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained intent model...")
def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is missing. Keep models/hinglish_intent_model.joblib "
            "inside this project folder."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data

def load_metrics() -> dict[str, Any]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data

def load_class_report() -> pd.DataFrame:
    if CLASS_REPORT_PATH.exists():
        return pd.read_csv(CLASS_REPORT_PATH)
    return pd.DataFrame()


def intent_name(label: int) -> str:
    return "Out-of-Scope · Class 55" if int(label) == 55 else f"Intent #{int(label)}"


def predict_one(model: Any, text: str) -> tuple[int, list[dict[str, Any]], float | None]:
    normalized = normalize_queries([text])
    prediction = int(model.predict(normalized)[0])
    scores = model.decision_function(normalized)
    scores = np.asarray(scores)
    if scores.ndim == 2:
        row = scores[0]
    else:
        # Defensive handling for a binary classifier; this project has 56 classes.
        row = np.array([-float(scores[0]), float(scores[0])])
    class_ids = np.asarray(model.classes_, dtype=int)
    order = np.argsort(row)[::-1][:5]
    ranked = [
        {"Intent": intent_name(int(class_ids[i])), "Class ID": int(class_ids[i]), "SVM decision score": float(row[i])}
        for i in order
    ]
    margin = float(row[order[0]] - row[order[1]]) if len(order) > 1 else None
    return prediction, ranked, margin


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def append_history(query: str, prediction: int, margin: float | None) -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "query": query,
        "class_id": prediction,
        "intent": intent_name(prediction),
        "margin": margin,
    })
    st.session_state.history = st.session_state.history[:20]


# Load the bundled, already-trained artifact.
try:
    ARTIFACT = load_model()
    MODEL = ARTIFACT["model"]
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info("Check that the project was extracted completely and that the models folder is present.")
    st.stop()

METRICS = load_metrics()
REPORT = load_class_report()

with st.sidebar:
    st.markdown("<div class='eyebrow'>DATA SPRINT 2.0</div>", unsafe_allow_html=True)
    st.markdown("## 🧠 BhashaSense")
    st.caption("Hinglish intent intelligence")
    st.markdown("<span class='pill'>● MODEL LOADED</span>", unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Live classifier", "Batch prediction", "Model insights"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Model stack**")
    st.caption("Word + character TF-IDF")
    st.caption("Linear Support Vector Machine")
    st.caption("56 intent classes")
    st.markdown("<div class='small-note'>Romanized Hinglish · Devanagari · ASR-noisy text</div>", unsafe_allow_html=True)

if page == "Overview":
    hero(
        "MULTILINGUAL VOICE INTENT CLASSIFIER",
        "Understand the intent behind every query.",
        "A local AI interface for testing Romanized Hinglish, Devanagari Hindi, and noisy speech-to-text queries with your trained Data Sprint 2.0 model.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training utterances", "16,936")
    c2.metric("Intent classes", "56")
    c3.metric("Test utterances", "4,912")
    c4.metric("Validation accuracy", f"{METRICS.get('accuracy', 0):.2%}" if METRICS else "—")

    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("<div class='panel'><div class='panel-title'>Your AI workspace</div><p class='muted'>Run one query at a time, upload a CSV for bulk classification, or inspect the model's validation results.</p></div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("#### How to use")
        st.markdown("1. Open **Live classifier** and type a query in Hindi, Hinglish, or Devanagari.\n2. View the predicted class and the top alternative classes.\n3. Use **Batch prediction** to classify many rows from a CSV.\n4. Check **Model insights** for validation metrics and class-wise performance.")
    with right:
        st.markdown("<div class='panel'><div class='panel-title'>Model information</div><p class='muted'>The model uses word unigrams/bigrams and character n-grams (2–6), weighted toward character patterns to better tolerate spelling variation.</p><hr/><p class='muted'><b>Class 55</b> is reserved for Out-of-Scope queries. The dataset provides numeric labels for the other classes, so they are displayed as Intent #0–#54 rather than guessed names.</p></div>", unsafe_allow_html=True)

elif page == "Live classifier":
    hero(
        "LIVE INFERENCE",
        "Ask in your language.",
        "Enter a short voice-transcribed query. The classifier will return one of 56 intent classes and rank the closest alternatives.",
    )
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        st.markdown("<div class='panel-title'>Query input</div>", unsafe_allow_html=True)
        with st.form("live_prediction_form", clear_on_submit=False):
            query = st.text_area(
                "Type your query",
                placeholder="Example: kal subah 7 baje ka alarm laga do…",
                height=145,
                help="You can enter Romanized Hinglish, Hindi in Devanagari, or ASR-style text.",
            )
            submitted = st.form_submit_button("✦  Detect intent", use_container_width=True)
        st.caption("Tip: Keep the sentence close to what a user would naturally say to a voice assistant.")

        if submitted:
            if not query.strip():
                st.warning("Please enter a query before running the classifier.")
            else:
                pred, ranked, margin = predict_one(MODEL, query)
                append_history(query, pred, margin)
                st.session_state["latest_result"] = {
                    "query": query, "prediction": pred, "ranked": ranked, "margin": margin
                }

        result = st.session_state.get("latest_result")
        if result:
            st.write("")
            pred = result["prediction"]
            st.markdown(
                f"<div class='prediction'><div class='eyebrow'>PREDICTED INTENT</div>"
                f"<div class='big'>{intent_name(pred)}</div>"
                f"<div class='sub'>Predicted class ID: {pred} · {('Out-of-Scope query' if pred == 55 else 'Supported intent class')}</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown("**Top alternative classes**")
            st.dataframe(pd.DataFrame(result["ranked"]), hide_index=True, use_container_width=True)
            st.caption("SVM decision scores rank the model's class choices; they are not calibrated probabilities.")

    with right:
        st.markdown("<div class='panel-title'>Recent predictions</div>", unsafe_allow_html=True)
        history = st.session_state.get("history", [])
        if history:
            history_df = pd.DataFrame(history)[["time", "query", "intent"]]
            st.dataframe(history_df, hide_index=True, use_container_width=True, height=370)
            if st.button("Clear history", key="clear_history"):
                st.session_state.history = []
                st.rerun()
        else:
            st.markdown("<div class='panel'><div class='panel-title'>No predictions yet</div><p class='muted'>Your recent queries and predicted classes will appear here after you run the first prediction.</p></div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("<div class='panel'><div class='panel-title'>Supported input styles</div><p class='muted'>Romanized Hinglish<br/>Devanagari Hindi<br/>Noisy ASR transcriptions</p></div>", unsafe_allow_html=True)

elif page == "Batch prediction":
    hero(
        "BULK CLASSIFICATION",
        "Classify an entire file at once.",
        "Upload a CSV containing a query_text column. When example_id is present, the app can produce the exact two-column competition submission format.",
    )
    up = st.file_uploader("Upload CSV file", type=["csv"], help="Required column: query_text. Optional column: example_id.")
    if up is not None:
        try:
            df = pd.read_csv(up)
            if "query_text" not in df.columns:
                st.error("This CSV must contain a query_text column.")
            elif len(df) == 0:
                st.warning("The uploaded CSV has no rows.")
            else:
                st.success(f"Loaded {len(df):,} queries.")
                st.dataframe(df.head(8), hide_index=True, use_container_width=True)
                if st.button("Run batch prediction", type="primary", use_container_width=True):
                    texts = df["query_text"].fillna("").astype(str).tolist()
                    normalized = normalize_queries(texts)
                    predictions = MODEL.predict(normalized).astype(int)
                    if "example_id" in df.columns:
                        result_df = pd.DataFrame({"example_id": df["example_id"], "target": predictions})
                        filename = "submission.csv"
                    else:
                        result_df = pd.DataFrame({
                            "query_text": texts,
                            "predicted_target": predictions,
                            "predicted_intent": [intent_name(x) for x in predictions],
                        })
                        filename = "intent_predictions.csv"
                    st.session_state["batch_result"] = result_df
                    st.session_state["batch_filename"] = filename
                batch_result = st.session_state.get("batch_result")
                if batch_result is not None:
                    st.write("")
                    st.markdown("#### Prediction results")
                    b1, b2 = st.columns(2)
                    b1.metric("Rows classified", f"{len(batch_result):,}")
                    b2.metric("Distinct classes predicted", int(batch_result.iloc[:, -1].nunique()))
                    st.dataframe(batch_result.head(20), hide_index=True, use_container_width=True)
                    st.download_button(
                        "⬇ Download predictions CSV",
                        data=batch_result.to_csv(index=False).encode("utf-8"),
                        file_name=st.session_state.get("batch_filename", "intent_predictions.csv"),
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")

    st.markdown("<div class='panel'><div class='panel-title'>Competition file checklist</div><p class='muted'>For Data Sprint 2.0, upload test.csv. It already contains example_id and query_text. The download will contain example_id and target, with IDs kept in the uploaded order.</p></div>", unsafe_allow_html=True)

elif page == "Model insights":
    hero(
        "MODEL MONITORING",
        "Know how the classifier performs.",
        "Validation metrics below come from a stratified 80/20 split of the supplied training set. They are validation estimates, not leaderboard results.",
    )
    if METRICS:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{METRICS.get('accuracy', 0):.2%}")
        c2.metric("Macro F1", f"{METRICS.get('macro_f1', 0):.4f}")
        c3.metric("Weighted F1", f"{METRICS.get('weighted_f1', 0):.4f}")
        c4.metric("Class 55 F1", f"{METRICS.get('class_55_f1', 0):.4f}")
        st.caption(f"Validation rows: {METRICS.get('validation_rows', '—'):,} · Random seed: {METRICS.get('random_seed', '—')} · Model: {METRICS.get('model', '—')}")
    else:
        st.warning("Validation metrics file not found in outputs/.")

    if not REPORT.empty:
        st.write("")
        st.markdown("#### Class-wise validation performance")
        mode = st.radio("Show classes", ["Lowest F1", "Highest F1", "All classes"], horizontal=True)
        sorted_report = REPORT.sort_values("f1_score", ascending=(mode == "Lowest F1"))
        if mode != "All classes":
            sorted_report = sorted_report.head(12)
        chart = px.bar(
            sorted_report,
            x="f1_score",
            y=sorted_report["target"].astype(str).map(lambda x: "Out-of-Scope · 55" if x == "55" else f"Intent #{x}"),
            orientation="h",
            labels={"x": "F1 score", "y": "Class"},
            range_x=[0, 1],
            template="plotly_dark",
        )
        chart.update_layout(
            height=max(360, 32 * len(sorted_report)),
            margin=dict(l=10, r=20, t=15, b=15),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce7fb"),
        )
        chart.update_traces(marker_color="#5eead4", marker_line_width=0)
        st.plotly_chart(chart, use_container_width=True)
        with st.expander("View full class-wise metrics table"):
            st.dataframe(REPORT.sort_values("target"), hide_index=True, use_container_width=True)
    else:
        st.info("Class-wise report not found. Run train.py to generate validation outputs.")

    if MISTAKES_PATH.exists():
        with st.expander("Inspect validation mistakes"):
            mistakes = pd.read_csv(MISTAKES_PATH)
            st.caption(f"{len(mistakes):,} validation examples were misclassified by the saved validation run.")
            st.dataframe(mistakes.head(100), hide_index=True, use_container_width=True)

st.divider()
st.markdown(
    "<div class='small-note'>BhashaSense · Data Sprint 2.0 · Predictions are model outputs, not guaranteed interpretations. "
    "Numeric class labels are shown because the supplied dataset does not include human-readable names for Classes 0–54.</div>",
    unsafe_allow_html=True,
)
