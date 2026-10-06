# Architecture

```mermaid
flowchart LR
    RAW[("Tweets.csv<br/>data/raw (git-ignored)")] --> LOAD["data.py<br/>dedupe tweet_id, filter negative,<br/>drop 'Can't Tell', map to 6 classes"]
    LOAD --> CLEAN["text_clean.py<br/>strip URLs/@mentions,<br/>repair source artefact, keep negations/emoji"]
    CLEAN --> DEDUP["dedupe identical cleaned text"]
    DEDUP --> SPLIT["stratified 70/15/15, seed 42<br/>data/processed/*_ids.txt"]

    SPLIT -->|train| FEAT["TF-IDF word 1-2 + char 3-5<br/>(fit on train only)"]
    FEAT --> LR["Logistic Regression<br/>C tuned on val"]
    FEAT --> XGB["XGBoost<br/>depth/lr tuned on val, early stop"]
    LR --> CP["classical predictions<br/>results/predictions/*.csv"]
    XGB --> CP
    LR --> MODELS[("models/ -> deploy_model/")]

    SPLIT -->|val subsample| PCHOICE["prompt choice<br/>zero-shot vs few-shot (val only)"]
    PCHOICE --> LLM["llm.py: groq | gemini | ollama<br/>JSON output, cache, rate limit, backoff"]
    SPLIT -->|test| LLM
    LLM --> LP["LLM predictions<br/>results/predictions/llm_test.csv"]

    CP --> EVAL["evaluate / report.py<br/>P/R/F1, confusion, bootstrap CI,<br/>error analysis, tradeoffs"]
    LP --> EVAL
    EVAL --> OUT[("results/metrics.json<br/>comparison.md, confusion_*.png,<br/>error_analysis.md, README block")]

    MODELS --> APP["app.py (Streamlit)<br/>model selector + routing queue"]
    LLM -.->|key in env/secrets| APP
    ROUTE[["configs/routing.yaml"]] --> APP
```

Key invariants
- The test split is identical for every model (saved as tweet_id lists) and is only *evaluated* in `evaluate`.
- The vectoriser is fitted on the training split only.
- LLM calls are cached on disk by (provider, model, prompt version + hash, mode, input): reruns are free and resumable across days.
- If the LLM branch cannot run, the pipeline records why and reports classical results only.
