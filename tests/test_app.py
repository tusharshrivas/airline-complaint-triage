import pytest

from triage import config

HAVE_MODEL = (config.MODELS / "tfidf_logreg.joblib").exists() or (config.DEPLOY_MODEL / "tfidf_logreg.joblib").exists()


@pytest.mark.skipif(not HAVE_MODEL, reason="no trained model: run `python -m triage.cli train-classical`")
def test_app_starts_and_predicts_sample_tweet(monkeypatch):
    from streamlit.testing.v1 import AppTest

    for k in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    at = AppTest.from_file(str(config.ROOT / "src" / "triage" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("LLM" in w.value and "unavailable" in w.value for w in at.sidebar.warning)  # LLM disabled with a message
    at.text_area[0].set_value("my bag never arrived and nobody can tell me where it is")
    at.button[len(at.button) - 1].click().run()  # the 'Predict' button is the last one
    assert not at.exception
    assert any("Category" in s.value for s in at.subheader)
    assert any("Route to" in s.value for s in at.success)
