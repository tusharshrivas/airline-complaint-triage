import yaml

from triage import config
from triage.data import build_labeled, reason_to_class

REASONS = [
    "Customer Service Issue",
    "Late Flight",
    "Cancelled Flight",
    "Lost Luggage",
    "Bad Flight",
    "Flight Booking Problems",
    "Flight Attendant Complaints",
    "longlines",
    "Damaged Luggage",
]


def test_exactly_six_classes():
    assert len(config.class_names()) == 6
    assert set(config.class_names()) == {
        "customer_service",
        "delay_or_cancellation",
        "baggage",
        "booking_or_fare",
        "flight_crew_or_flight",
        "other",
    }


def test_every_known_reason_is_mapped_once():
    cfg = yaml.safe_load(open(config.CONFIGS / "label_map.yaml"))
    flat = [r for rs in cfg["classes"].values() for r in rs]
    assert len(flat) == len(set(flat)), "a reason is mapped to two classes"
    mapping = reason_to_class(cfg)
    assert all(r in mapping for r in REASONS)
    assert mapping["Flight Attendant Complaints"] == "customer_service"
    assert mapping["longlines"] == "other"
    assert "Can't Tell" not in mapping and "Can't Tell" in cfg["excluded_reasons"]


def test_filter_and_map_on_synthetic_frame():
    import pandas as pd

    df = pd.DataFrame(
        {
            "tweet_id": [1, 2, 3, 4, 5, 5],
            "airline_sentiment": ["negative", "negative", "negative", "positive", "negative", "negative"],
            "negativereason": ["Late Flight", "Can't Tell", None, None, "Lost Luggage", "Lost Luggage"],
            "airline": ["United"] * 6,
            "text": ["a late", "b", "c", "d", "bag lost", "bag lost"],
        }
    )
    out, stats = build_labeled(df)
    assert list(out["tweet_id"]) == [1, 5]
    assert list(out["label"]) == ["delay_or_cancellation", "baggage"]
    assert stats["after_tweet_id_dedupe"] == 5
