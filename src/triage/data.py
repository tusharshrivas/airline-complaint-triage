"""Dataset acquisition, filtering, label mapping and the leakage-safe stratified split."""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from triage import config
from triage.text_clean import clean_text

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/crowdflower/twitter-airline-sentiment"
REQUIRED_COLUMNS = {"tweet_id", "airline_sentiment", "negativereason", "airline", "text", "tweet_created"}
SPLITS = ("train", "val", "test")

MISSING_DATA_README = """# Dataset not found

This project does not ship the dataset (license: see ASSUMPTIONS.md). Provide `Tweets.csv`
("Twitter US Airline Sentiment", CrowdFlower / Figure Eight, 14,640 tweets) in ONE of these ways:

1. Download it from https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment and
   save it as `data/raw/Tweets.csv`, or
2. Set `DATA_PATH=/path/to/Tweets.csv` in `.env`, or
3. Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` and re-run `python -m triage.cli prepare-data`
   (the code downloads it through the Kaggle REST API).

Then re-run `python -m triage.cli prepare-data`. No data is ever fabricated.
"""


class DatasetMissing(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_kaggle(dest: Path) -> Path:
    """Download via the Kaggle REST API (HTTP basic auth with KAGGLE_USERNAME/KAGGLE_KEY)."""
    import requests

    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        raise DatasetMissing("KAGGLE_USERNAME/KAGGLE_KEY not set")
    resp = requests.get(KAGGLE_URL, auth=(user, key), timeout=120, allow_redirects=True)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith("tweets.csv"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))
    return dest


def ensure_raw() -> Path:
    """Return a path to Tweets.csv, or write data/README.md and raise DatasetMissing."""
    env_path = os.environ.get("DATA_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    default = config.DATA_RAW / "Tweets.csv"
    if default.exists():
        return default
    try:
        return download_kaggle(default)
    except Exception as exc:  # noqa: BLE001 - any failure falls through to instructions
        config.DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
        (config.DATA_RAW.parent / "README.md").write_text(MISSING_DATA_README, encoding="utf-8")
        raise DatasetMissing(
            f"Tweets.csv not found and Kaggle download unavailable ({exc}). Instructions written to data/README.md."
        ) from exc


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def reason_to_class(label_cfg: dict) -> dict[str, str]:
    return {reason: cls for cls, reasons in label_cfg["classes"].items() for reason in reasons}


def build_labeled(df: pd.DataFrame, label_cfg: dict | None = None, dedupe_text: bool = True):
    """Filter to usable negative tweets, map to 6 classes, de-duplicate. Returns (df, stats)."""
    label_cfg = label_cfg or config.label_map()
    mapping = reason_to_class(label_cfg)
    stats: dict = {"raw_rows": int(len(df))}

    df = df.drop_duplicates(subset="tweet_id", keep="first")
    stats["after_tweet_id_dedupe"] = int(len(df))

    neg = df[(df["airline_sentiment"] == "negative") & df["negativereason"].notna()]
    stats["negative_with_reason"] = int(len(neg))
    neg = neg[~neg["negativereason"].isin(label_cfg.get("excluded_reasons", []))]
    stats["after_excluding_cant_tell"] = int(len(neg))

    out = neg[["tweet_id", "airline", "negativereason", "text"]].copy()
    out["label"] = out["negativereason"].map(mapping).fillna(label_cfg.get("unmapped_to", "other"))
    out["text_clean"] = out["text"].map(clean_text)
    out = out[out["text_clean"].str.len() > 0]
    stats["after_dropping_empty_text"] = int(len(out))

    if dedupe_text:
        key = out["text_clean"].str.lower()
        out = out[~key.duplicated(keep="first")]
    stats["after_text_dedupe"] = int(len(out))
    out = out.sort_values("tweet_id").reset_index(drop=True)
    stats["class_counts"] = {k: int(v) for k, v in out["label"].value_counts().items()}
    return out, stats


def stratified_split(df: pd.DataFrame, seed: int, fractions: dict[str, float]) -> dict[str, list[int]]:
    """Stratified train/val/test split returned as sorted tweet_id lists.

    The frame is sorted by tweet_id first so the result never depends on row order.
    """
    df = df.sort_values("tweet_id").reset_index(drop=True)
    rest = 1.0 - fractions["train"]
    train, temp = train_test_split(df, test_size=rest, stratify=df["label"], random_state=seed)
    val_share = fractions["val"] / (fractions["val"] + fractions["test"])
    val, test = train_test_split(temp, train_size=val_share, stratify=temp["label"], random_state=seed)
    return {
        "train": sorted(int(i) for i in train["tweet_id"]),
        "val": sorted(int(i) for i in val["tweet_id"]),
        "test": sorted(int(i) for i in test["tweet_id"]),
    }


def save_splits(labeled: pd.DataFrame, splits: dict[str, list[int]], out_dir: Path | None = None) -> None:
    out_dir = out_dir or config.DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_dir / "labeled.csv", index=False)
    for name, ids in splits.items():
        (out_dir / f"{name}_ids.txt").write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")


def load_splits(out_dir: Path | None = None) -> dict[str, list[int]]:
    out_dir = out_dir or config.DATA_PROCESSED
    return {name: [int(x) for x in (out_dir / f"{name}_ids.txt").read_text().split()] for name in SPLITS}


def load_split_frames(out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Return {'train','val','test'} DataFrames built from the saved tweet_id lists."""
    out_dir = out_dir or config.DATA_PROCESSED
    labeled = pd.read_csv(out_dir / "labeled.csv").set_index("tweet_id", drop=False)
    labeled["text_clean"] = labeled["text_clean"].fillna("")
    return {name: labeled.loc[ids].reset_index(drop=True) for name, ids in load_splits(out_dir).items()}


def splits_fingerprint(splits: dict[str, list[int]]) -> str:
    h = hashlib.sha256()
    for name in SPLITS:
        h.update(name.encode())
        h.update(",".join(map(str, splits[name])).encode())
    return h.hexdigest()[:16]
