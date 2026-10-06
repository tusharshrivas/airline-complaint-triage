import numpy as np
import pandas as pd
import pytest

from triage import config, data

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["a", "b", "c", "d", "e", "f"], size=n, p=[0.4, 0.25, 0.12, 0.1, 0.08, 0.05])
    return pd.DataFrame({"tweet_id": np.arange(1000, 1000 + n), "label": labels})


def test_no_tweet_id_in_two_splits():
    s = data.stratified_split(synthetic(), 42, FRACTIONS)
    sets = [set(v) for v in s.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(x) for x in sets) == 600


def test_split_sizes_and_stratification():
    df = synthetic()
    s = data.stratified_split(df, 42, FRACTIONS)
    assert abs(len(s["train"]) / 600 - 0.70) < 0.02 and abs(len(s["test"]) / 600 - 0.15) < 0.02
    overall = df["label"].value_counts(normalize=True)
    test = df[df.tweet_id.isin(s["test"])]["label"].value_counts(normalize=True)
    assert (overall - test.reindex(overall.index).fillna(0)).abs().max() < 0.03


def test_split_identical_across_two_runs_and_row_order():
    df = synthetic()
    a = data.stratified_split(df, 42, FRACTIONS)
    b = data.stratified_split(df.copy(), 42, FRACTIONS)
    c = data.stratified_split(df.sample(frac=1, random_state=1), 42, FRACTIONS)
    assert a == b == c
    assert data.stratified_split(df, 7, FRACTIONS) != a


@pytest.mark.skipif(
    not (config.DATA_PROCESSED / "test_ids.txt").exists(),
    reason="dataset not prepared: run `python -m triage.cli prepare-data` (needs Tweets.csv)",
)
def test_saved_real_splits_are_disjoint():
    s = data.load_splits()
    assert not (set(s["train"]) & set(s["val"])) and not (set(s["train"]) & set(s["test"])) and not (set(s["val"]) & set(s["test"]))
    frames = data.load_split_frames()
    texts = [set(f["text_clean"].str.lower()) for f in frames.values()]
    assert not (texts[0] & texts[2]) and not (texts[0] & texts[1]), "identical cleaned text across splits"
