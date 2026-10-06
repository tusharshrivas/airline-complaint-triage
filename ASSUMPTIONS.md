# ASSUMPTIONS

Every assumption, deviation from the original spec, and thing that could **not** be verified is recorded here.
Run-time facts (dates, seeds, models, timings) live in `results/RUN_LOG.md`; results live in `results/metrics.json`.

## 1. Build environment (what was and was not reachable)
- Built on 2026-10-06 in a sandbox (Linux, 1 CPU core, ~4 GB RAM, Python 3.12) whose network allow-list contained PyPI and GitHub but **not**
  api.groq.com, generativelanguage.googleapis.com, kaggle.com, huggingface.co or a local Ollama (all returned HTTP 403 / connection refused).
- Consequence: **the LLM branch could not execute during the build.** The code path is implemented and unit-tested with mocked HTTP, but no LLM
  metric exists, and none is invented. `results/RUN_LOG.md` records the skip. To produce LLM results run `python -m triage.cli run-llm` on a machine
  with a free key or Ollama (see HANDOFF.md).
- Docker, Streamlit Community Cloud and the live provider APIs could not be exercised from the sandbox (see the checklist in the final report).

## 2. Dataset
- Spec: Kaggle "Twitter US Airline Sentiment" (CrowdFlower / Figure Eight), `Tweets.csv`, 14,640 rows.
- Acquisition used here: Kaggle was blocked, so `Tweets.csv` was copied from public GitHub mirrors. Four independent repositories hold a
  byte-identical file (SHA-256 `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e`, 14,640 rows, expected 15 columns), which strongly suggests it
  is the original release, but provenance is not independently proven. The code's primary path is `DATA_PATH` / `data/raw/Tweets.csv` / the Kaggle REST API
  (the Kaggle download path is implemented with plain HTTP but **was not tested**: no Kaggle access).
- **License: NOT VERIFIED.** I could not open the Kaggle page. My recollection is that it is listed under a Creative Commons non-commercial, share-alike license
  (CC BY-NC-SA 4.0) and that tweet text is also subject to Twitter/X terms. **You must confirm the license before publishing**; if it is non-commercial/share-alike
  this affects redistribution (hence `data/raw/`, `data/processed/` are git-ignored, and only a small trained model plus aggregate results are meant to be published).
  `results/error_analysis.md` quotes ~60 tweets; confirm that is acceptable or remove it before publishing.
- Row accounting (from `RUN_LOG.md`): 14,640 raw -> 14,485 after dropping 155 duplicate `tweet_id`s -> 9,082 negative with a reason -> 7,906 after removing
  "Can't Tell" -> 7,901 after dropping 5 duplicate cleaned texts.

## 3. Label mapping (`configs/label_map.yaml`)
| class | source `negativereason` | why |
|---|---|---|
| customer_service | Customer Service Issue, Flight Attendant Complaints | both are about how staff treat/handle the customer; matches the spec |
| delay_or_cancellation | Late Flight, Cancelled Flight | both are schedule disruption and go to the same operational desk |
| baggage | Lost Luggage, Damaged Luggage | same team (baggage services) |
| booking_or_fare | Flight Booking Problems | only booking-related reason in the data |
| flight_crew_or_flight | Bad Flight | the in-flight experience. **The name is the spec's; flight-attendant complaints are in `customer_service`**, so "crew" is misleading - consider renaming to `in_flight_experience` |
| other | longlines (+ any unmapped reason) | long lines/crowding don't fit the other queues; `other` also doubles as the LLM's parse-failure fallback |
- "Can't Tell" is dropped (annotators could not decide). Classes are imbalanced (`other` is ~2% of rows), so per-class metrics on `other` rest on few test tweets; macro-F1 is noisy.

## 4. Text cleaning and a data artefact found during the build
- Removed: URLs, @mentions (airline handles carry the brand, not the complaint type), HTML entities unescaped, whitespace collapsed. Kept: negations, emojis, punctuation,
  hashtag words, case (see `text_clean.py` docstring).
- **Source-text artefact (deviation from the spec, deliberate):** ~16% of usable tweets contain label strings that were injected into the text by a find-and-replace
  bug in the released file: `cancel`->"Cancelled Flight" (e.g. "Cancelled Flightled flight"), `late`->"Late Flight" ("Late Flightr"), `book`->"Flight Booking Problems"
  ("reFlight Booking Problems"). Real tweets never contain these strings, so models trained on them would face train/serve skew, and they partly mirror the label.
  `clean_text` repairs exactly these three substitutions by default. `ablation.py` quantifies the effect for logistic regression (repaired vs raw text) and reports
  accuracy on affected rows; the numbers are in `results/ablation_artifacts.json` / `comparison.md`. Assumption: the replaced substrings were exactly `cancel`, `late`, `book`
  (inferred from the examples; case of the original letters is lost, which is irrelevant after lower-casing).
  **Measured outcome (`results/ablation_artifacts.json`): no detectable effect on logistic regression** - test macro-F1 0.592 (repaired) vs 0.590 (raw), validation 0.615 vs 0.618, i.e. within noise.
  So the repair was a precaution against train/serve skew; the reported conclusions do not depend on it. (Only LR was ablated; XGBoost and the LLM were not.)

## 5. Splitting
- Deduplicated first by `tweet_id`, then by exact cleaned lower-cased text (keeps the first row), so retweet-like duplicates cannot straddle train/test.
  No further label-noise handling is done (the first occurrence of a duplicated text wins).
- Stratified on the 6-class label, seed 42, 70/15/15 (two chained `train_test_split` calls). IDs are saved to `data/processed/{train,val,test}_ids.txt`; the frame is sorted by
  `tweet_id` before splitting so the result does not depend on file row order (tested).

## 6. Classical models
- One TF-IDF featuriser (word 1-2-grams + `char_wb` 3-5-grams, sublinear tf, `min_df=2`, vocabulary caps 30k/60k in `experiment.yaml`) fitted on **train only**.
  The caps/min_df are my choices to keep training feasible on one CPU core; they were not tuned.
- Logistic regression: `C` in {0.1, 1, 10} on validation macro-F1, `class_weight="balanced"`. XGBoost: `max_depth` in {3, 6} x `learning_rate` in {0.15, 0.3}
  (a deliberately small grid and capped rounds, because a first, larger setting took >6 min per config on the 1-core build machine; XGBoost may therefore be under-tuned relative to logistic regression), up to 150 rounds (colsample_bytree 0.3, max_bin 64), early stopping (20) on validation log-loss, balanced sample weights, selection on
  validation macro-F1. Validation is therefore used for early stopping *and* model selection (mild optimism on val numbers only; test is untouched).
- Models are **not** refit on train+val. "Best classical model" for the bootstrap = highest *validation* macro-F1 (never chosen on test).
- Train time = vectoriser fit + final estimator fit; the grid-search time is stored separately. Latency = mean over 300 single-tweet predictions on 1 CPU core
  (clean + featurise + predict) - a serving-style number, not batch throughput.

## 7. LLM branch
- Rate limits were read from the providers' public documentation via web search on 2026-10-06 (the doc sites were not directly reachable from the sandbox). They change; re-check.
| provider | model used (default) | RPM | RPD | TPM | TPD | source |
|---|---|---|---|---|---|---|
| groq | `llama-3.1-8b-instant` (the code confirms it via `GET /openai/v1/models`; falls back to `llama-3.3-70b-versatile`) | 30 | 14,400 | 6,000 | 500,000 | console.groq.com/docs/rate-limits (free plan) |
| gemini | `gemini-2.5-flash-lite` | 10 used (docs table: 15) | 1,000 | 250,000 | - | ai.google.dev/gemini-api/docs/rate-limits; sources disagree on RPM, I used the lower value |
| ollama | `qwen2.5:3b` (or `llama3.2:3b`) | local | local | local | local | no remote limits |
- The client uses 80% of these (`safety_factor`). **Groq's 500K tokens/day is the binding constraint**: zero-shot is ~0.45K tokens/call and few-shot ~1.3K tokens/call (estimates from prompt length; `run-llm` prints its own plan), so the
  300-tweet prompt-choice subsample for both prompts plus the 1,186-tweet test set needs roughly 3 days (if zero-shot wins) to 5-6 days (if few-shot wins) of Groq free-tier token quota at the 80% safety margin. `run-llm` therefore stops cleanly when the daily budget is hit
  and **resumes from the disk cache** on re-run; the evaluation handles partial coverage by scoring every model on the same covered subset and flagging it.
- Zero- vs few-shot is chosen on a stratified **validation** subsample of 300 tweets (seed 42), never on test; ties go to zero-shot. The test split is classified once with the chosen prompt.
- The LLM sees the same cleaned text as the classical models (mentions/URLs removed, artefact repaired).
- Not verified against the live APIs: Groq `response_format=json_object` acceptance for the chosen model, Gemini `thinkingConfig` handling, Ollama `format=json` behaviour.
  The adapters follow the providers' documented request shapes and are tested only with mocked HTTP.

## 8. Routing table
`configs/routing.yaml` queue names, priorities and SLAs are **illustrative inventions for the demo**, not taken from any airline.

## 9. Deployment
- Streamlit Community Cloud cannot receive the dataset (license, size, secrecy of nothing but license), so the app uses a small trained model committed under `deploy_model/`
  (exported with `python -m triage.cli export-deploy-model`). `models/` stays git-ignored per the spec.
- Ollama is a local daemon and is not available on Streamlit Community Cloud, so the cloud app offers the classical models and (with a secret) Groq/Gemini only.
