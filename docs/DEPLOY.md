# Deployment (all free)

> Status of verification is listed at the end. Items marked **(not verified)** could not be executed in the build sandbox.

## A. Local
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                             # makes `python -m triage.cli` work from the project folder
cp .env.example .env                                         # fill in what you have; never commit .env (Windows: copy)

# data: put Tweets.csv at data/raw/Tweets.csv, or set DATA_PATH, or set KAGGLE_USERNAME/KAGGLE_KEY
python -m triage.cli prepare-data
python -m triage.cli train-classical        # ~10 min on 1 CPU core (XGBoost grid dominates)
python -m triage.cli run-llm                # needs GROQ_API_KEY / GEMINI_API_KEY or a running Ollama; skips cleanly otherwise
python -m triage.cli evaluate               # writes results/*, docs/tradeoffs.md and the generated README blocks
python -m triage.cli export-deploy-model    # copies the validation-best classical model to deploy_model/
python -m triage.cli app                    # http://localhost:8501
pytest -q                                   # no network, keys or dataset required
```
Free LLM options (pick one via `LLM_PROVIDER` in `.env`):
- **groq** (default): free key at https://console.groq.com. Free-plan limits (checked 2026-10-06): 30 RPM, 14.4K requests/day, 6K tokens/min, **500K tokens/day** for `llama-3.1-8b-instant`.
  A full run takes several days of quota; re-run `run-llm` daily - the cache resumes where it stopped.
- **gemini**: free key at https://aistudio.google.com/apikey, model `gemini-2.5-flash-lite` (≈1,000 requests/day).
- **ollama**: install from https://ollama.com, `ollama pull qwen2.5:3b`, `LLM_PROVIDER=ollama`. No key and no quota; speed depends on your machine (record its specs - `run-llm` logs CPU/RAM).

## B. Docker (Docker + Compose are free) **(not verified)**
```bash
cp .env.example .env            # optional; without keys the LLM option is simply disabled
docker compose up --build       # Streamlit on http://localhost:8501
```
`models/`, `data/`, `results/`, `cache/` are mounted as volumes. The image bundles `deploy_model/`, so the app also works with no mounted `models/`.
Run `export-deploy-model` before building so `deploy_model/` exists. To reach Ollama running on the host from the container, `OLLAMA_HOST` defaults to `http://host.docker.internal:11434`.

## C. Streamlit Community Cloud (free for public GitHub repos) **(not verified)**
1. Run the pipeline locally, then `python -m triage.cli export-deploy-model` and commit `deploy_model/` (≈2.3 MB for logistic regression; GitHub's file limit is 100 MB, so size is not a problem).
   Do **not** commit `data/`, `models/`, `.env` (all git-ignored).
2. Push to a public GitHub repo (confirm the dataset license and the tweet excerpts in `results/error_analysis.md` first - see ASSUMPTIONS.md section 2).
3. https://share.streamlit.io -> *New app* -> repository/branch -> **Main file path: `src/triage/app.py`** -> *Advanced settings*: Python 3.12 and Secrets:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy. The app loads `deploy_model/*.joblib` (it looks in `models/` first, then `deploy_model/`).

**Why commit the model instead of training at startup:** training needs `Tweets.csv`, which should not be redistributed (license) and would make cold starts slow; the model is small and deterministic.
**Why Ollama is not used in the cloud deployment:** Streamlit Community Cloud runs your app in a small shared container; it cannot run an Ollama daemon, has no GPU and limited RAM, and cannot reach a `localhost` on your laptop.
The cloud app therefore uses the classical models and, if a secret is set, a free hosted API (Groq/Gemini). The shared free quota is per project, so a public demo can exhaust it - the app shows the error instead of faking a prediction.

## What was verified in the build sandbox
- Local pipeline (A) steps `prepare-data`, `train-classical`, `evaluate`, `export-deploy-model`, `pytest`, and the Streamlit app via Streamlit's `AppTest` harness.
- Not verifiable there: live Groq/Gemini/Ollama calls, Docker build, Streamlit Cloud deployment, Kaggle download.
