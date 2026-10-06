"""Pick misclassified test examples, prioritising the most common confusions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_pairs(df: pd.DataFrame) -> pd.Series:
    err = df[df["y_true"] != df["y_pred"]]
    return err.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False, kind="stable")


def select_examples(df: pd.DataFrame, n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Allocate the n slots to (true, predicted) confusion pairs in proportion to their error
    counts (largest-remainder), so the most common confusions get the most examples."""
    err = df[df["y_true"] != df["y_pred"]]
    if err.empty:
        return err
    pairs = confusion_pairs(df)
    n = min(n, len(err))
    share = pairs / pairs.sum() * n
    quota = np.floor(share).astype(int)
    remainder = (share - quota).sort_values(ascending=False, kind="stable")
    for key in remainder.index[: n - int(quota.sum())]:
        quota[key] += 1
    rng = np.random.default_rng(seed)
    picks = []
    for (t, p), q in quota.items():
        if q <= 0:
            continue
        group = err[(err["y_true"] == t) & (err["y_pred"] == p)]
        picks.append(group.iloc[rng.permutation(len(group))[:q]])
    return pd.concat(picks).reset_index(drop=True)


def _esc(s: object) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def render(model_title: str, df: pd.DataFrame, texts: dict[int, str], n: int = 20, seed: int = 42) -> str:
    pairs = confusion_pairs(df)
    total_err = int((df["y_true"] != df["y_pred"]).sum())
    out = [
        f"## {model_title}",
        "",
        f"{total_err} errors out of {len(df)} test items ({total_err / len(df):.1%}). "
        f"Showing {min(n, total_err)} examples, allocated to confusion pairs in proportion to how common they are.",
        "",
        "| true -> predicted | errors |",
        "|---|---|",
    ]
    out += [f"| {t} -> {p} | {c} |" for (t, p), c in pairs.head(8).items()]
    out += ["", "| # | tweet_id | true | predicted | conf | tweet (cleaned) | LLM reason |", "|---|---|---|---|---|---|---|"]
    sel = select_examples(df, n, seed)
    for i, r in enumerate(sel.itertuples(), 1):
        conf = "" if pd.isna(r.confidence) else f"{r.confidence:.2f}"
        reason = _esc(getattr(r, "reason", "")) if "reason" in sel.columns and isinstance(getattr(r, "reason", ""), str) else ""
        out.append(f"| {i} | {r.tweet_id} | {r.y_true} | {r.y_pred} | {conf} | {_esc(texts.get(int(r.tweet_id), ''))} | {reason} |")
    return "\n".join(out) + "\n"
