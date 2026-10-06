"""Streamlit triage demo.  Run:  streamlit run src/triage/app.py   (or: python -m triage.cli app)

- Model selector lists ONLY models that are actually available.
- Classical models load from models/ (local) or deploy_model/ (committed, used on Streamlit Cloud).
- The LLM option appears only when a free provider is configured (key in env or Streamlit secrets,
  or a reachable local Ollama); otherwise the sidebar explains why it is disabled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `triage` importable on Streamlit Cloud

import streamlit as st  # noqa: E402

from triage import config  # noqa: E402

config.load_dotenv()

CLASSICAL = {"tfidf_logreg": "TF-IDF + Logistic Regression", "tfidf_xgboost": "TF-IDF + XGBoost"}
SAMPLES = [
    "my bag never arrived and nobody at the baggage counter can tell me where it is",
    "stuck on the tarmac for 3 hours, flight delayed again. unacceptable",
    "your website keeps erroring when I try to change my booking",
    "the flight attendant was rude and ignored my request for help",
]


def _model_path(name: str) -> Path | None:
    for base in (config.MODELS, config.DEPLOY_MODEL):
        p = base / f"{name}.joblib"
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_classical(name: str):
    from triage.models.classical import ClassicalModel

    return ClassicalModel.load(_model_path(name))


@st.cache_resource(show_spinner="Contacting LLM provider...")
def load_llm():
    """Returns (classifier | None, reason | None). No fabricated fallbacks."""
    from triage.models import llm

    try:
        provider = llm.make_provider()
    except llm.ProviderUnavailable as exc:
        return None, str(exc)
    status = json.loads((config.RESULTS / "llm_status.json").read_text()) if (config.RESULTS / "llm_status.json").exists() else {}
    mode = status.get("chosen_mode", "zero_shot")
    pcfg = config.providers()[provider.name]
    exp = config.experiment()["llm"]
    limiter = llm.RateLimiter(pcfg["limits"], exp["safety_factor"])
    clf = llm.LLMClassifier(
        provider,
        mode,
        llm.load_prompt(),
        config.class_names(),
        limiter,
        config.CACHE,
        exp["max_retries"],
        exp["max_output_tokens"],
        exp["max_wait_s"],
    )
    return clf, None


def available_models() -> tuple[dict, dict]:
    models, unavailable = {}, {}
    for key, title in CLASSICAL.items():
        if _model_path(key):
            models[title] = ("classical", key)
        else:
            unavailable[title] = "model file not found - run `python -m triage.cli train-classical`"
    clf, reason = load_llm()
    if clf:
        models[f"LLM ({clf.provider.name}: {clf.provider.model}, {clf.mode})"] = ("llm", clf)
    else:
        unavailable["LLM (free provider)"] = reason
    return models, unavailable


def main() -> None:
    st.set_page_config(page_title="Airline complaint triage", page_icon="✈️", layout="centered")
    st.title("✈️ Airline complaint triage")
    st.caption("Demo / portfolio project - not production-ready. Classifies a complaint tweet and shows the queue it would be routed to.")
    models, unavailable = available_models()
    with st.sidebar:
        st.header("Models")
        for title, why in unavailable.items():
            st.warning(f"**{title}** unavailable: {why}")
        if not models:
            st.error("No model is available. Train one first.")
    if not models:
        st.stop()

    choice = st.selectbox("Model", list(models))
    if "text" not in st.session_state:
        st.session_state["text"] = SAMPLES[0]
    cols = st.columns(len(SAMPLES))
    for i, s in enumerate(SAMPLES):
        if cols[i].button(f"Example {i + 1}", help=s):
            st.session_state["text"] = s
    text = st.text_area("Customer tweet", key="text", height=110)

    if st.button("Predict", type="primary"):
        if not text.strip():
            st.info("Enter some text first.")
            st.stop()
        kind, handle = models[choice]
        try:
            if kind == "classical":
                out = load_classical(handle).predict_one(text)
                out["reason"] = None
            else:
                r = handle.classify(text)
                out = {"category": r.category, "confidence": r.confidence, "reason": r.reason, "parse_failure": r.parse_failure}
        except Exception as exc:  # noqa: BLE001 - show provider problems in the UI
            st.error(f"Prediction failed: {exc}")
            st.stop()
        route = config.routing()[out["category"]]
        st.subheader(f"Category: `{out['category']}`")
        conf = out.get("confidence")
        st.metric("Confidence", "n/a" if conf is None else f"{conf:.0%}")
        if out.get("reason"):
            st.write(f"**Reason (LLM, self-reported):** {out['reason']}")
        if out.get("parse_failure"):
            st.warning("The LLM output could not be parsed; fell back to `other`.")
        st.success(f"**Route to:** {route['queue']}  |  priority {route['priority']}  |  target response {route['target_response']}")
        if "probabilities" in out:
            st.bar_chart(out["probabilities"], horizontal=True)
    st.caption("Routing queues are illustrative (configs/routing.yaml).")


main()
