# RUN_LOG

Append-only record of every pipeline step actually executed. Written by the code, not by hand.

## 2026-10-06 06:26:58 UTC - prepare-data

- command: `python -m triage.cli prepare-data`
- input: `/home/claude/airline-complaint-triage/data/raw/Tweets.csv` sha256=ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e
- seed=42 split={'train': 0.7, 'val': 0.15, 'test': 0.15}
- row accounting: {'raw_rows': 14640, 'after_tweet_id_dedupe': 14485, 'negative_with_reason': 9082, 'after_excluding_cant_tell': 7906, 'after_dropping_empty_text': 7906, 'after_text_dedupe': 7901, 'class_counts': {'customer_service': 3357, 'delay_or_cancellation': 2478, 'baggage': 792, 'flight_crew_or_flight': 575, 'booking_or_fare': 522, 'other': 177}}
- split sizes: train=5530, val=1185, test=1186
- split fingerprint (sha256[:16] of the 3 id lists): fd80f801e7ba465f

## 2026-10-06 06:26:59 UTC - prepare-data

- command: `python -m triage.cli prepare-data`
- input: `/home/claude/airline-complaint-triage/data/raw/Tweets.csv` sha256=ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e
- seed=42 split={'train': 0.7, 'val': 0.15, 'test': 0.15}
- row accounting: {'raw_rows': 14640, 'after_tweet_id_dedupe': 14485, 'negative_with_reason': 9082, 'after_excluding_cant_tell': 7906, 'after_dropping_empty_text': 7906, 'after_text_dedupe': 7901, 'class_counts': {'customer_service': 3357, 'delay_or_cancellation': 2478, 'baggage': 792, 'flight_crew_or_flight': 575, 'booking_or_fare': 522, 'other': 177}}
- split sizes: train=5530, val=1185, test=1186
- split fingerprint (sha256[:16] of the 3 id lists): fd80f801e7ba465f

## 2026-10-06 06:39:16 UTC - prepare-data

- command: `python -m triage.cli prepare-data`
- input: `/home/claude/airline-complaint-triage/data/raw/Tweets.csv` sha256=ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e
- seed=42 split={'train': 0.7, 'val': 0.15, 'test': 0.15}
- row accounting: {'raw_rows': 14640, 'after_tweet_id_dedupe': 14485, 'negative_with_reason': 9082, 'after_excluding_cant_tell': 7906, 'after_dropping_empty_text': 7906, 'after_text_dedupe': 7901, 'class_counts': {'customer_service': 3357, 'delay_or_cancellation': 2478, 'baggage': 792, 'flight_crew_or_flight': 575, 'booking_or_fare': 522, 'other': 177}}
- split sizes: train=5530, val=1185, test=1186
- split fingerprint (sha256[:16] of the 3 id lists): fd80f801e7ba465f

## 2026-10-06 11:57:33 UTC - artifact ablation (post-hoc analysis, LR only)

- command: `python -m triage.cli train-classical`
- {"repaired_text": {"C": 1, "val_macro_f1": 0.6146694930176766, "test_macro_f1": 0.5920720934296221}, "raw_unrepaired_text": {"C": 1, "val_macro_f1": 0.6180718955836662, "test_macro_f1": 0.5900129652857814}, "test_rows_with_artifact": 208, "test_rows_total": 1186, "artifact_share_all_rows": 0.15922035185419567, "repaired_model_accuracy_on_artifact_rows": 0.7788461538461539, "repaired_model_accuracy_on_clean_rows": 0.6983640081799591}

## 2026-10-06 11:57:33 UTC - train-classical

- command: `python -m triage.cli train-classical`
- seed=42; features=47694; vectoriser fit 0.8s (train split only)
- tfidf_logreg: params={'C': 1} val_macro_f1=0.6147 train_time_s=2.7 tuning_total_s=6.1 size_bytes=2324084 latency_ms_mean=1.63 (n=300 single predictions, 1 CPU)
- tfidf_xgboost: params={'max_depth': 3, 'learning_rate': 0.3, 'best_iteration': 148} val_macro_f1=0.6020 train_time_s=251.9 tuning_total_s=1035.6 size_bytes=618407 latency_ms_mean=1.47 (n=300 single predictions, 1 CPU)
- best classical by validation macro-F1: tfidf_logreg
- environment: {'python': '3.12.3', 'platform': 'Linux-6.18.44-fc-v70-x86_64-with-glibc2.39', 'cpu_count': 1, 'packages': {'scikit-learn': '1.8.0', 'xgboost': '3.4.1', 'pandas': '3.0.2', 'numpy': '2.4.4', 'requests': '2.33.1', 'streamlit': '1.65.0', 'matplotlib': '3.10.8', 'PyYAML': '6.0.3'}}

## 2026-10-06 12:01:49 UTC - run-llm SKIPPED (LLM branch did not execute)

- command: `python -m triage.cli run-llm`
- provider=groq
- reason: GROQ_API_KEY is not set
- No LLM numbers exist; none are fabricated.

## 2026-10-06 12:01:51 UTC - evaluate

- command: `python -m triage.cli evaluate`
- models evaluated: ['tfidf_logreg', 'tfidf_xgboost']; items=1186/1186; partial LLM coverage=False
- best classical by validation macro-F1: tfidf_logreg
- verdict: No LLM-vs-classical comparison is available because the LLM branch did not complete (see RUN_LOG.md).
- wrote results/metrics.json, comparison.md, confusion_*.png, error_analysis.md, docs/tradeoffs.md, README blocks

## 2026-10-06 12:01:59 UTC - environment note: LLM branch could not run in the build sandbox

- command: `curl probes (see ASSUMPTIONS.md section 1)`
- Probes on 2026-10-06 from the build sandbox: api.groq.com/openai/v1/models -> HTTP 403; generativelanguage.googleapis.com -> HTTP 403; kaggle.com -> HTTP 403; huggingface.co -> HTTP 403; localhost:11434 (Ollama) -> connection refused.
- So even with a key the LLM branch could not have executed here. No LLM results exist; none are fabricated. Run `python -m triage.cli run-llm` on a machine with a free key or Ollama.

## 2026-10-06 12:02:57 UTC - evaluate

- command: `python -m triage.cli evaluate`
- models evaluated: ['tfidf_logreg', 'tfidf_xgboost']; items=1186/1186; partial LLM coverage=False
- best classical by validation macro-F1: tfidf_logreg
- verdict: No LLM-vs-classical comparison is available because the LLM branch did not complete (see RUN_LOG.md).
- wrote results/metrics.json, comparison.md, confusion_*.png, error_analysis.md, docs/tradeoffs.md, README blocks

## 2026-10-06 15:04:25 UTC - prepare-data

- command: `python -m triage.cli prepare-data`
- input: `C:\Users\tusha\Downloads\airline-complaint-triage\airline-complaint-triage\data\raw\Tweets.csv` sha256=ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e
- seed=42 split={'train': 0.7, 'val': 0.15, 'test': 0.15}
- row accounting: {'raw_rows': 14640, 'after_tweet_id_dedupe': 14485, 'negative_with_reason': 9082, 'after_excluding_cant_tell': 7906, 'after_dropping_empty_text': 7906, 'after_text_dedupe': 7901, 'class_counts': {'customer_service': 3357, 'delay_or_cancellation': 2478, 'baggage': 792, 'flight_crew_or_flight': 575, 'booking_or_fare': 522, 'other': 177}}
- split sizes: train=5530, val=1185, test=1186
- split fingerprint (sha256[:16] of the 3 id lists): fd80f801e7ba465f

## 2026-10-06 15:24:55 UTC - artifact ablation (post-hoc analysis, LR only)

- command: `python -m triage.cli train-classical`
- {"repaired_text": {"C": 1, "val_macro_f1": 0.6146694930176766, "test_macro_f1": 0.5920720934296221}, "raw_unrepaired_text": {"C": 1, "val_macro_f1": 0.6180718955836662, "test_macro_f1": 0.5900129652857814}, "test_rows_with_artifact": 208, "test_rows_total": 1186, "artifact_share_all_rows": 0.15922035185419567, "repaired_model_accuracy_on_artifact_rows": 0.7788461538461539, "repaired_model_accuracy_on_clean_rows": 0.6983640081799591}

## 2026-10-06 15:24:55 UTC - train-classical

- command: `python -m triage.cli train-classical`
- seed=42; features=47694; vectoriser fit 0.7s (train split only)
- tfidf_logreg: params={'C': 1} val_macro_f1=0.6147 train_time_s=2.9 tuning_total_s=6.9 size_bytes=2323951 latency_ms_mean=2.00 (n=300 single predictions, 1 CPU)
- tfidf_xgboost: params={'max_depth': 3, 'learning_rate': 0.3, 'best_iteration': 147} val_macro_f1=0.5986 train_time_s=250.2 tuning_total_s=1083.5 size_bytes=619700 latency_ms_mean=1.65 (n=300 single predictions, 1 CPU)
- best classical by validation macro-F1: tfidf_logreg
- environment: {'python': '3.12.6', 'platform': 'Windows-11-10.0.26200-SP0', 'cpu_count': 16, 'packages': {'scikit-learn': '1.8.0', 'xgboost': '3.4.1', 'pandas': '3.0.2', 'numpy': '2.4.4', 'requests': '2.33.1', 'streamlit': '1.65.0', 'matplotlib': '3.10.8', 'PyYAML': '6.0.3'}}

## 2026-10-06 15:44:53 UTC - evaluate

- command: `python -m triage.cli evaluate`
- models evaluated: ['tfidf_logreg', 'tfidf_xgboost']; items=1186/1186; partial LLM coverage=False
- best classical by validation macro-F1: tfidf_logreg
- verdict: No LLM-vs-classical comparison is available because the LLM branch did not complete (see RUN_LOG.md).
- wrote results/metrics.json, comparison.md, confusion_*.png, error_analysis.md, docs/tradeoffs.md, README blocks
