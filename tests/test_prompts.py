import pytest

from triage import config, data
from triage.models import llm

pytestmark = pytest.mark.skipif(
    not (config.DATA_PROCESSED / "train_ids.txt").exists(),
    reason="dataset not prepared: run `python -m triage.cli prepare-data` (needs Tweets.csv)",
)


def test_every_few_shot_example_comes_from_train_and_matches_its_label():
    frames = data.load_split_frames()
    train = frames["train"].set_index("tweet_id")
    other_ids = set(frames["val"]["tweet_id"]) | set(frames["test"]["tweet_id"])
    for ex in llm.load_prompt().fewshot:
        assert ex["tweet_id"] in train.index and ex["tweet_id"] not in other_ids
        assert train.loc[ex["tweet_id"], "label"] == ex["category"]
        assert train.loc[ex["tweet_id"], "text_clean"] == ex["text"]
