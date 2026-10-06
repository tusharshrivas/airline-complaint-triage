"""TF-IDF + logistic regression and TF-IDF + XGBoost.

Both models share ONE vectoriser fitted on the training split only (no leakage from
validation/test vocabulary or IDF statistics). Hyper-parameters are chosen on the
validation split; the test split is touched once, by `evaluate`.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from triage import config, data, runlog
from triage.evaluate import compute_metrics
from triage.text_clean import clean_text

# Keeps contractions ("don't") and single punctuation/emoji tokens; no stop-word removal.
TOKEN_PATTERN = r"(?u)\b\w[\w']*\b|[^\w\s]"
MODEL_NAMES = {"logreg": "tfidf_logreg", "xgboost": "tfidf_xgboost"}


class Featurizer:
    """word 1-2-gram + char_wb 3-5-gram TF-IDF, sublinear tf, stacked side by side."""

    def __init__(self, cfg: dict):
        w, c = cfg["tfidf_word"], cfg["tfidf_char"]
        self.word = TfidfVectorizer(
            ngram_range=tuple(w["ngram_range"]),
            min_df=w["min_df"],
            max_features=w["max_features"],
            sublinear_tf=True,
            lowercase=True,
            token_pattern=TOKEN_PATTERN,
            dtype=np.float32,
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(c["ngram_range"]),
            min_df=c["min_df"],
            max_features=c["max_features"],
            sublinear_tf=True,
            lowercase=True,
            dtype=np.float32,
        )

    def fit(self, texts):
        self.word.fit(texts)
        self.char.fit(texts)
        return self

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    @property
    def n_features(self) -> int:
        return len(self.word.vocabulary_) + len(self.char.vocabulary_)


class ClassicalModel:
    """Self-contained artefact: featurizer + estimator + label order. Accepts RAW tweets."""

    def __init__(self, name: str, featurizer: Featurizer, estimator, labels: list[str]):
        self.name, self.featurizer, self.estimator, self.labels = name, featurizer, estimator, labels

    def predict_proba(self, raw_texts) -> np.ndarray:
        cleaned = [clean_text(t) for t in raw_texts]
        return self.estimator.predict_proba(self.featurizer.transform(cleaned))

    def predict(self, raw_texts) -> list[str]:
        return [self.labels[i] for i in self.predict_proba(raw_texts).argmax(axis=1)]

    def predict_one(self, raw_text: str) -> dict:
        proba = self.predict_proba([raw_text])[0]
        order = np.argsort(-proba)
        return {
            "category": self.labels[int(order[0])],
            "confidence": float(proba[order[0]]),
            "probabilities": {self.labels[i]: float(proba[i]) for i in order},
        }

    def top_features(self, k: int = 8) -> dict[str, list[tuple[str, float]]]:
        """Highest-weight n-grams per class (logistic regression only) - interpretability."""
        if not hasattr(self.estimator, "coef_"):
            return {}
        names = np.array(
            [f"w:{t}" for t in self.featurizer.word.get_feature_names_out()]
            + [f"c:{t}" for t in self.featurizer.char.get_feature_names_out()]
        )
        out = {}
        for i, lab in enumerate(self.labels):
            idx = np.argsort(-self.estimator.coef_[i])[:k]
            out[lab] = [(str(names[j]), float(self.estimator.coef_[i][j])) for j in idx]
        return out

    def save(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path.stat().st_size

    @staticmethod
    def load(path: Path) -> ClassicalModel:
        return joblib.load(path)


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def train_logreg(Xtr, ytr, Xva, yva, cfg):
    """Tune C on validation macro-F1; class_weight='balanced'."""
    tried, best = [], None
    for C in cfg["logreg"]["C_grid"]:
        t0 = time.perf_counter()
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=cfg["logreg"]["max_iter"], random_state=42).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        score = _macro_f1(yva, clf.predict(Xva))
        tried.append({"C": C, "val_macro_f1": score, "fit_s": fit_s})
        print(f"  logreg C={C}: val macro-F1={score:.4f} {fit_s:.1f}s", flush=True)
        if best is None or score > best[0]:
            best = (score, clf, {"C": C}, fit_s)
    return best, tried


def train_xgboost(Xtr, ytr, Xva, yva, cfg, n_classes):
    """Tune max_depth x learning_rate on validation macro-F1, early stopping on validation logloss."""
    x = cfg["xgboost"]
    weights = compute_sample_weight("balanced", ytr)  # mirrors class_weight='balanced'
    tried, best = [], None
    ckpt_dir = config.ROOT / "cache" / "xgb_grid"  # resumable: each finished config is checkpointed
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for depth in x["max_depth_grid"]:
        for lr in x["learning_rate_grid"]:
            ckpt = ckpt_dir / f"d{depth}_lr{lr}_n{x['n_estimators']}.joblib"
            if ckpt.exists():
                clf, fit_s = joblib.load(ckpt)
            else:
                t0 = time.perf_counter()
                clf = XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    tree_method="hist",
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=x["n_estimators"],
                    early_stopping_rounds=x["early_stopping_rounds"],
                    eval_metric="mlogloss",
                    n_jobs=x["n_jobs"],
                    random_state=42,
                    subsample=0.9,
                    colsample_bytree=0.3,
                    max_bin=64,
                )
                clf.fit(Xtr, ytr, sample_weight=weights, eval_set=[(Xva, yva)], verbose=False)
                fit_s = time.perf_counter() - t0
                joblib.dump((clf, fit_s), ckpt)
            score = _macro_f1(yva, clf.predict(Xva))
            tried.append(
                {"max_depth": depth, "learning_rate": lr, "val_macro_f1": score, "best_iteration": int(clf.best_iteration), "fit_s": fit_s}
            )
            print(f"  xgb depth={depth} lr={lr}: val macro-F1={score:.4f} iters={clf.best_iteration} fit={fit_s:.0f}s", flush=True)
            if best is None or score > best[0]:
                best = (score, clf, {"max_depth": depth, "learning_rate": lr, "best_iteration": int(clf.best_iteration)}, fit_s)
    return best, tried


def _time_single_predictions(model: ClassicalModel, texts: list[str]) -> float:
    """Mean wall-clock ms per prediction when tweets arrive one at a time (serving pattern)."""
    model.predict_one(texts[0])  # warm-up
    t0 = time.perf_counter()
    for t in texts:
        model.predict_one(t)
    return (time.perf_counter() - t0) / len(texts) * 1000


def run_training() -> dict:
    cfg = config.experiment()["classical"]
    labels = config.class_names()
    idx = {lab: i for i, lab in enumerate(labels)}
    frames = data.load_split_frames()
    tr, va, te = frames["train"], frames["val"], frames["test"]
    ytr, yva = tr["label"].map(idx).values, va["label"].map(idx).values

    t0 = time.perf_counter()
    feat = Featurizer(cfg).fit(tr["text_clean"].tolist())
    vec_s = time.perf_counter() - t0
    Xtr, Xva = feat.transform(tr["text_clean"].tolist()), feat.transform(va["text_clean"].tolist())

    summary: dict = {"n_features": feat.n_features, "vectorizer_fit_s": vec_s, "models": {}}
    runs = {
        "logreg": lambda: train_logreg(Xtr, ytr, Xva, yva, cfg),
        "xgboost": lambda: train_xgboost(Xtr, ytr, Xva, yva, cfg, len(labels)),
    }
    config.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for key, run in runs.items():
        t_grid = time.perf_counter()
        (score, clf, params, fit_s), tried = run()
        grid_s = time.perf_counter() - t_grid
        name = MODEL_NAMES[key]
        model = ClassicalModel(name, feat, clf, labels)
        size = model.save(config.MODELS / f"{name}.joblib")

        # Predictions for the held-out splits. Test predictions are produced here but only
        # *evaluated* in `evaluate`; nothing above used them for any decision.
        for split_name, frame in (("val", va), ("test", te)):
            proba = model.predict_proba(frame["text"].tolist())
            pred = proba.argmax(axis=1)
            out = pd.DataFrame(
                {
                    "tweet_id": frame["tweet_id"],
                    "y_true": frame["label"],
                    "y_pred": [labels[i] for i in pred],
                    "confidence": proba.max(axis=1),
                }
            )
            out.to_csv(config.PREDICTIONS / f"{name}_{split_name}.csv", index=False)
        latency_ms = _time_single_predictions(model, te["text"].tolist()[:300])

        val_metrics = compute_metrics(va["label"], [labels[i] for i in model.predict_proba(va["text"].tolist()).argmax(1)], labels)
        summary["models"][name] = {
            "best_params": params,
            "val_macro_f1": val_metrics["macro_f1"],
            "tuning_grid": tried,
            "final_fit_s": fit_s,
            "tuning_total_s": grid_s,
            "train_time_s": vec_s + fit_s,  # vectoriser fit + final estimator fit
            "model_size_bytes": size,
            "latency_ms_mean": latency_ms,
            "latency_n": 300,
        }
        print(
            f"{name}: val macro-F1={val_metrics['macro_f1']:.4f} params={params} "
            f"fit={fit_s:.1f}s size={size / 1e6:.2f}MB latency={latency_ms:.2f}ms"
        )

    best = max(summary["models"], key=lambda n: summary["models"][n]["val_macro_f1"])
    summary["best_by_val"] = best
    summary["selection_rule"] = "highest validation macro-F1 (test split not used)"
    (config.RESULTS / "classical_train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from triage.ablation import run_ablation  # local import: ablation imports this module

    run_ablation()
    lr_feats = ClassicalModel.load(config.MODELS / "tfidf_logreg.joblib").top_features(6)
    (config.RESULTS / "logreg_top_features.json").write_text(json.dumps(lr_feats, indent=2), encoding="utf-8")
    runlog.log(
        "train-classical",
        [f"seed=42; features={feat.n_features}; vectoriser fit {vec_s:.1f}s (train split only)"]
        + [
            f"{n}: params={m['best_params']} val_macro_f1={m['val_macro_f1']:.4f} train_time_s={m['train_time_s']:.1f} "
            f"tuning_total_s={m['tuning_total_s']:.1f} size_bytes={m['model_size_bytes']} "
            f"latency_ms_mean={m['latency_ms_mean']:.2f} (n=300 single predictions, 1 CPU)"
            for n, m in summary["models"].items()
        ]
        + [f"best classical by validation macro-F1: {best}", f"environment: {runlog.env_snapshot()}"],
        "python -m triage.cli train-classical",
    )
    return summary


def export_deploy_model() -> Path:
    """Copy the validation-selected classical model to deploy_model/ (small, committable)."""
    summary = json.loads((config.RESULTS / "classical_train.json").read_text())
    name = summary["best_by_val"]
    config.DEPLOY_MODEL.mkdir(exist_ok=True)
    dest = config.DEPLOY_MODEL / f"{name}.joblib"
    shutil.copy(config.MODELS / f"{name}.joblib", dest)
    return dest
