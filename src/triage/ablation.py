"""Post-hoc sensitivity analysis for the Tweets.csv substitution artefact (logistic regression only).

Compares TF-IDF+LR trained/evaluated on (a) repaired text (the pipeline default) and
(b) raw, unrepaired text. This is reported as analysis; it never feeds a modelling decision.
"""

from __future__ import annotations

import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from triage import config, data, runlog
from triage.models.classical import ClassicalModel, Featurizer
from triage.text_clean import clean_text, has_artifact


def _fit_eval(variant_repair: bool, frames, labels, cfg) -> dict:
    idx = {lab: i for i, lab in enumerate(labels)}
    tx = {k: [clean_text(t, repair_artifacts=variant_repair) for t in frames[k]["text"]] for k in ("train", "val", "test")}
    y = {k: frames[k]["label"].map(idx).values for k in ("train", "val", "test")}
    feat = Featurizer(cfg).fit(tx["train"])
    X = {k: feat.transform(tx[k]) for k in tx}
    best = None
    for C in cfg["logreg"]["C_grid"]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=cfg["logreg"]["max_iter"], random_state=42).fit(
            X["train"], y["train"]
        )
        score = f1_score(y["val"], clf.predict(X["val"]), average="macro", zero_division=0)
        if best is None or score > best[0]:
            best = (score, C, clf)
    score, C, clf = best
    test_f1 = f1_score(y["test"], clf.predict(X["test"]), average="macro", zero_division=0)
    return {"C": C, "val_macro_f1": float(score), "test_macro_f1": float(test_f1)}


def run_ablation() -> dict:
    cfg = config.experiment()["classical"]
    labels = config.class_names()
    frames = data.load_split_frames()
    out = {
        "repaired_text": _fit_eval(True, frames, labels, cfg),
        "raw_unrepaired_text": _fit_eval(False, frames, labels, cfg),
    }
    te = frames["test"]
    mask = te["text"].map(has_artifact).values
    out["test_rows_with_artifact"] = int(mask.sum())
    out["test_rows_total"] = int(len(te))
    out["artifact_share_all_rows"] = float(
        sum(has_artifact(t) for f in frames.values() for t in f["text"]) / sum(len(f) for f in frames.values())
    )
    # main (repaired) LR model: accuracy on rows that had artefacts vs rows that did not
    model = ClassicalModel.load(config.MODELS / "tfidf_logreg.joblib")
    pred = model.predict(te["text"].tolist())
    correct = te["label"].values == pred
    out["repaired_model_accuracy_on_artifact_rows"] = float(correct[mask].mean()) if mask.any() else None
    out["repaired_model_accuracy_on_clean_rows"] = float(correct[~mask].mean())
    (config.RESULTS / "ablation_artifacts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    runlog.log("artifact ablation (post-hoc analysis, LR only)", [json.dumps(out)], "python -m triage.cli train-classical")
    return out
