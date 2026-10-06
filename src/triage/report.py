"""Builds every reported artefact from recorded predictions: metrics.json, comparison.md,
confusion PNGs, error_analysis.md, tradeoffs.md and the generated blocks of README.md.

Nothing here invents numbers: all values come from results/predictions/*.csv and the JSON
files written by `train-classical` / `run-llm`.
"""

from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from triage import config, data, error_analysis, runlog, stats  # noqa: E402
from triage.evaluate import compute_metrics, load_predictions  # noqa: E402

CLASSICAL = ["tfidf_logreg", "tfidf_xgboost"]
TITLES = {"tfidf_logreg": "TF-IDF + Logistic Regression", "tfidf_xgboost": "TF-IDF + XGBoost"}
NA = "n/a"


def _read_json(name: str, default=None):
    p = config.RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def _llm_title(status: dict) -> str:
    return f"LLM ({status.get('provider', '?')}: {status.get('model', '?')}, {status.get('chosen_mode', '?')})"


def plot_confusion(metrics: dict, title: str, path) -> None:
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{norm[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if norm[i, j] > 0.55 else "black",
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\n(rows normalised; n={int(cm.sum())})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _f(x, nd=3):
    return NA if x is None else f"{x:.{nd}f}"


def results_table(models: dict, llm_status: dict) -> str:
    head = (
        "| Model | Macro precision | Macro recall | Macro F1 | Weighted F1 | Accuracy | Parse failures (LLM) | "
        "Mean latency ms | Train time s (classical) | Rate-limited retries (LLM) |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for m in models.values():
        x = m["metrics"]
        rows.append(
            f"| {m['title']} | {x['macro_precision']:.3f} | {x['macro_recall']:.3f} | {x['macro_f1']:.3f} | "
            f"{x['weighted_f1']:.3f} | {x['accuracy']:.3f} | {m.get('parse_failures', NA)} | "
            f"{_f(m.get('latency_ms_mean'), 1)} | {_f(m.get('train_time_s'), 1)} | {m.get('rate_limited_retries', NA)} |"
        )
    if "llm" not in models:
        rows.append(
            f"| LLM ({llm_status.get('provider', 'n/a')}) | not run | not run | not run | not run | not run | not run | not run | {NA} | not run |"
        )
    return head + "\n" + "\n".join(rows)


def per_class_table(m: dict) -> str:
    out = ["| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for lab, v in m["metrics"]["per_class"].items():
        out.append(f"| {lab} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    return "\n".join(out)


def _verdict(boot: dict | None, best_title: str, llm_title: str) -> str:
    if not boot:
        return "No LLM-vs-classical comparison is available because the LLM branch did not complete (see RUN_LOG.md)."
    d, lo, hi = boot["point_estimate"], boot["ci95_low"], boot["ci95_high"]
    base = (
        f"Macro-F1 difference ({best_title} minus {llm_title}) = {d:+.3f}, 95% bootstrap CI [{lo:+.3f}, {hi:+.3f}] "
        f"({boot['n_resamples']} resamples, seed {boot['seed']}, n={boot['n_items']}). The interval "
    )
    if boot["includes_zero"]:
        return base + "**includes zero**: this test set does not show a reliable difference in macro-F1 between them."
    return base + ("**excludes zero**: the best classical model scored higher." if d > 0 else "**excludes zero**: the LLM scored higher.")


def build_all() -> dict:
    labels = config.class_names()
    exp = config.experiment()
    frames = data.load_split_frames()
    test = frames["test"]
    texts = dict(zip(test["tweet_id"].astype(int), test["text_clean"]))
    train_info = _read_json("classical_train.json")
    if not train_info:
        raise SystemExit("results/classical_train.json missing - run `train-classical` first.")
    llm_status = _read_json("llm_status.json", {"status": "not_run", "reason": "run-llm has not been executed"})

    preds = {n: load_predictions(n) for n in CLASSICAL}
    for n, p in preds.items():
        if p is None:
            raise SystemExit(f"missing predictions for {n}; run `train-classical`.")
    llm_df = load_predictions("llm") if llm_status.get("status") in ("ok", "partial") else None
    test_ids = set(test["tweet_id"].astype(int))
    eval_ids = test_ids & set(llm_df["tweet_id"].astype(int)) if llm_df is not None else test_ids
    partial = llm_df is not None and len(eval_ids) < len(test_ids)

    def subset(df):
        return df[df["tweet_id"].astype(int).isin(eval_ids)].sort_values("tweet_id").reset_index(drop=True)

    models: dict[str, dict] = {}
    for n in CLASSICAL:
        info = train_info["models"][n]
        df = subset(preds[n])
        models[n] = {
            "title": TITLES[n],
            "kind": "classical",
            "n_eval": len(df),
            "metrics": compute_metrics(df["y_true"], df["y_pred"], labels),
            "metrics_full_test": compute_metrics(preds[n]["y_true"], preds[n]["y_pred"], labels) if partial else None,
            "train_time_s": info["train_time_s"],
            "tuning_total_s": info["tuning_total_s"],
            "model_size_bytes": info["model_size_bytes"],
            "latency_ms_mean": info["latency_ms_mean"],
            "best_params": info["best_params"],
            "val_macro_f1": info["val_macro_f1"],
        }
    llm_title = None
    if llm_df is not None:
        df = subset(llm_df)
        llm_title = _llm_title(llm_status)
        lat = df.loc[df["latency_ms"] > 0, "latency_ms"]
        models["llm"] = {
            "title": llm_title,
            "kind": "llm",
            "n_eval": len(df),
            "metrics": compute_metrics(df["y_true"], df["y_pred"], labels),
            "parse_failures": int(df["parse_failure"].sum()),
            "repaired_after_retry": int(df["repaired"].sum()),
            "rate_limited_retries": int(df["rate_limited_retries"].sum()),
            "latency_ms_mean": float(lat.mean()) if len(lat) else None,
            "prompt_tokens_total": int(df["prompt_tokens"].sum()),
            "completion_tokens_total": int(df["completion_tokens"].sum()),
            "provider": llm_status.get("provider"),
            "model_id": llm_status.get("model"),
            "prompt_mode": llm_status.get("chosen_mode"),
            "prompt_version": llm_status.get("prompt_version"),
            "prompt_choice": llm_status.get("prompt_choice"),
        }

    best_classical = train_info["best_by_val"]
    boot = None
    if llm_df is not None:
        a, b = subset(preds[best_classical]), subset(llm_df)
        assert (a["tweet_id"].values == b["tweet_id"].values).all()
        boot = stats.bootstrap_macro_f1_diff(
            a["y_true"], a["y_pred"], b["y_pred"], labels, exp["bootstrap"]["n_resamples"], exp["bootstrap"]["seed"]
        )
        boot["a"], boot["b"] = best_classical, "llm"
    verdict = _verdict(boot, TITLES[best_classical], llm_title or "LLM")

    ablation = _read_json("ablation_artifacts.json")
    top_feats = _read_json("logreg_top_features.json", {})
    splits = data.load_splits()
    metrics_json = {
        "meta": {
            "generated_utc": runlog.utc_now(),
            "command": "python -m triage.cli evaluate",
            "seed": exp["seed"],
            "environment": runlog.env_snapshot(),
            "split_fingerprint": data.splits_fingerprint(splits),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "test_class_counts": {k: int(v) for k, v in test["label"].value_counts().items()},
            "n_test": int(len(test)),
            "n_evaluated_in_comparison": len(eval_ids),
            "llm_partial_coverage": bool(partial),
            "best_classical_by_validation": best_classical,
        },
        "labels": labels,
        "models": models,
        "llm_status": llm_status,
        "bootstrap_best_classical_minus_llm": boot,
        "verdict": verdict,
        "artifact_ablation": ablation,
    }
    (config.RESULTS / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    for n, m in models.items():
        plot_confusion(m["metrics"], m["title"], config.RESULTS / f"confusion_{n}.png")

    # ---- error analysis
    n_ex = exp["error_analysis"]["n_examples"]
    parts = [
        "# Error analysis",
        "",
        f"{n_ex} misclassified test examples per model, allocated to the most common confusions "
        f"(seed {exp['seed']}). Tweets are shown after cleaning. Source labels come from a single crowd-annotation pass, "
        "so some 'errors' are annotation noise or genuinely ambiguous tweets.",
        "",
    ]
    ea_sources = {n: subset(preds[n]) for n in CLASSICAL}
    if llm_df is not None:
        ea_sources["llm"] = subset(llm_df)
    for n, df in ea_sources.items():
        parts.append(error_analysis.render(models[n]["title"], df, texts, n_ex, exp["seed"]))
    (config.RESULTS / "error_analysis.md").write_text("\n".join(parts), encoding="utf-8")

    # ---- comparison.md
    comp = [
        "# Model comparison",
        "",
        f"Generated {metrics_json['meta']['generated_utc']} from `results/predictions/*.csv`. "
        f"Test items evaluated: {len(eval_ids)} of {len(test)}."
        + (" **LLM coverage is partial; all models are scored on the same covered subset.**" if partial else ""),
        "",
        results_table(models, llm_status),
        "",
        "Latency: classical = single-tweet prediction on 1 CPU core (n=300, includes cleaning + featurising); "
        "LLM = provider round-trip recorded at call time (excludes client-side rate-limit sleeps; free-tier queueing varies). "
        "Train time = vectoriser fit + final estimator fit (hyper-parameter search time is in `classical_train.json`).",
        "",
    ]
    if "llm" not in models:
        comp += [
            f"> **LLM branch did not execute** - status `{llm_status.get('status')}`: {llm_status.get('reason', 'see RUN_LOG.md')}",
            "",
        ]
    comp += ["## Bootstrap CI (best classical minus LLM)", "", verdict, ""]
    for n, m in models.items():
        comp += [f"## Per-class metrics - {m['title']}", "", per_class_table(m), "", f"![confusion {n}](confusion_{n}.png)", ""]
    if ablation:
        a, r = ablation["repaired_text"], ablation["raw_unrepaired_text"]
        comp += [
            "## Sensitivity: source-text artefact (logistic regression only, post-hoc)",
            "",
            f"{ablation['artifact_share_all_rows']:.1%} of usable tweets contain an injected label string "
            f"({ablation['test_rows_with_artifact']} of {ablation['test_rows_total']} test tweets). "
            f"TF-IDF+LR test macro-F1: repaired text **{a['test_macro_f1']:.3f}** vs raw unrepaired text **{r['test_macro_f1']:.3f}** "
            f"(validation: {a['val_macro_f1']:.3f} vs {r['val_macro_f1']:.3f}). Accuracy of the repaired model on test rows "
            f"that had the artefact: {_f(ablation['repaired_model_accuracy_on_artifact_rows'])}, on the rest: {_f(ablation['repaired_model_accuracy_on_clean_rows'])}.",
            "",
        ]
    if top_feats:
        comp += [
            "## Interpretability: top n-grams per class (logistic regression weights)",
            "",
            "| class | top features (w = word n-gram, c = char n-gram) |",
            "|---|---|",
        ]
        comp += [f"| {c} | {', '.join(f'`{t}`' for t, _ in fs)} |" for c, fs in top_feats.items()]
        comp.append("")
    (config.RESULTS / "comparison.md").write_text("\n".join(comp), encoding="utf-8")

    _write_tradeoffs(models, llm_status, boot, verdict, best_classical)
    _update_readme(models, llm_status, verdict, ea_sources, best_classical, ablation, partial, len(eval_ids), len(test))

    runlog.log(
        "evaluate",
        [
            f"models evaluated: {list(models)}; items={len(eval_ids)}/{len(test)}; partial LLM coverage={partial}",
            f"best classical by validation macro-F1: {best_classical}",
            f"verdict: {verdict}",
            "wrote results/metrics.json, comparison.md, confusion_*.png, error_analysis.md, docs/tradeoffs.md, README blocks",
        ],
        "python -m triage.cli evaluate",
    )
    print(results_table(models, llm_status))
    print("\n" + verdict)
    return metrics_json


def _write_tradeoffs(models, llm_status, boot, verdict, best) -> None:
    lines = [
        "# Tradeoffs",
        "",
        "Generated by `python -m triage.cli evaluate`. Section 1 contains only measured values from this repository's runs "
        "(`results/metrics.json`). Section 2 is general engineering context and is labelled as such - it is not a measured result.",
        "",
        "## 1. Measured in this project",
        "",
        "| Dimension | " + " | ".join(m["title"] for m in models.values()) + " |",
        "|---|" + "---|" * len(models),
    ]

    def row(label, fn):
        return f"| {label} | " + " | ".join(fn(m) for m in models.values()) + " |"

    lines += [
        row("Macro-F1 (test)", lambda m: f"{m['metrics']['macro_f1']:.3f}"),
        row("Accuracy (test)", lambda m: f"{m['metrics']['accuracy']:.3f}"),
        row("Mean latency / prediction", lambda m: f"{_f(m.get('latency_ms_mean'), 1)} ms"),
        row("Train time", lambda m: f"{_f(m.get('train_time_s'), 1)} s" if m["kind"] == "classical" else "none (no training)"),
        row("Artefact size", lambda m: f"{m['model_size_bytes'] / 1e6:.2f} MB" if m["kind"] == "classical" else "none (hosted model)"),
        row("Money cost", lambda m: "$0 (local CPU)" if m["kind"] == "classical" else "$0 on the free tier, but rate-limited"),
        row(
            "Interpretability",
            lambda m: (
                "linear weights -> top n-grams per class (see comparison.md)"
                if m["title"].endswith("Logistic Regression")
                else (
                    "feature importances only"
                    if m["kind"] == "classical"
                    else "free-text 'reason' per prediction (self-reported, not a faithful explanation)"
                )
            ),
        ),
    ]
    lines.append("")
    if "llm" not in models:
        lines += [
            f"**LLM columns are absent: the LLM branch did not run** (`{llm_status.get('status')}`: {llm_status.get('reason', 'see RUN_LOG.md')}). "
            "No LLM accuracy, latency or reliability numbers are claimed.",
            "",
        ]
    else:
        m = models["llm"]
        lines += [
            f"LLM reliability: {m['parse_failures']} parse failures, {m['repaired_after_retry']} repaired by the retry, "
            f"{m['rate_limited_retries']} rate-limited retries; {m['prompt_tokens_total']:,} prompt + {m['completion_tokens_total']:,} completion tokens "
            f"for {m['n_eval']} test tweets.",
            "",
        ]
    lines += [
        f"Accuracy comparison: {verdict}",
        "",
        "## 2. General considerations (not measured here)",
        "",
        "- **Cost at scale:** the classical models run on a CPU for effectively zero marginal cost; a hosted LLM costs per token once you leave the free tier, "
        "and free tiers have request/token caps (see `configs/providers.yaml`).",
        "- **Maintenance:** classical models need labelled data and periodic retraining but are deterministic and pinned; an LLM needs prompt maintenance "
        "and can change behaviour when the provider updates or retires a model.",
        "- **Data needs:** the classical models required ~5.5k labelled tweets; a zero-shot LLM needs none (few-shot needs a handful).",
        "- **Privacy/latency:** the LLM option sends tweet text to a third party (or needs a local model); the classical model keeps data in-process.",
        "- **Interpretability:** an LLM's `reason` is generated text, not a faithful account of the decision; linear weights are inspectable.",
        "",
    ]
    (config.ROOT / "docs" / "tradeoffs.md").write_text("\n".join(lines), encoding="utf-8")


def _replace_block(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {tag}:START -->).*?(<!-- {tag}:END -->)", re.S)
    if not pat.search(text):
        return text
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text)


def _update_readme(models, llm_status, verdict, ea_sources, best, ablation, partial, n_eval, n_test) -> None:
    path = config.ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    note = [
        f"_All numbers below are copied from `results/metrics.json` (generated {runlog.utc_now()}). "
        f"Test items evaluated: {n_eval} of {n_test}._",
        "",
    ]
    if partial:
        note.append("**LLM coverage is partial; every model is scored on the same covered subset.**\n")
    res = "\n".join(note) + "\n" + results_table(models, llm_status) + "\n\n"
    if "llm" not in models:
        res += f"> **The LLM branch did not run** (`{llm_status.get('status')}`): {llm_status.get('reason', 'see results/RUN_LOG.md')}\n>\n> No LLM numbers are reported or implied.\n\n"
    res += f"**Best classical model (chosen on validation macro-F1):** {TITLES[best]}.\n\n**Statistical comparison:** {verdict}\n\nPer-class tables: `results/comparison.md`; confusion matrices: `results/confusion_*.png`."
    text = _replace_block(text, "RESULTS", res)

    tr = ["See `docs/tradeoffs.md` (measured values + clearly separated general considerations).", ""]
    for m in models.values():
        tr.append(
            f"- {m['title']}: macro-F1 {m['metrics']['macro_f1']:.3f}, latency {_f(m.get('latency_ms_mean'), 1)} ms"
            + (f", train {m['train_time_s']:.1f} s, model {m['model_size_bytes'] / 1e6:.2f} MB" if m["kind"] == "classical" else "")
        )
    text = _replace_block(text, "TRADEOFFS", "\n".join(tr))

    ea = ["Full tables with 20 examples per model: `results/error_analysis.md`.", ""]
    for n, df in ea_sources.items():
        pairs = error_analysis.confusion_pairs(df).head(3)
        ea.append(f"- {models[n]['title']}: most common confusions - " + "; ".join(f"{t} -> {p} ({c})" for (t, p), c in pairs.items()))
    if ablation:
        a, r = ablation["repaired_text"], ablation["raw_unrepaired_text"]
        ea.append(
            f"- Data artefact: {ablation['artifact_share_all_rows']:.1%} of usable tweets contain injected label strings; "
            f"LR test macro-F1 {a['test_macro_f1']:.3f} (repaired) vs {r['test_macro_f1']:.3f} (raw)."
        )
    text = _replace_block(text, "ERRORS", "\n".join(ea))
    path.write_text(text, encoding="utf-8")
