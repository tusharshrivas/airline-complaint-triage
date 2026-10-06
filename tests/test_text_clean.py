from triage.text_clean import clean_text, has_artifact


def test_removes_urls_mentions_and_collapses_whitespace():
    assert clean_text("@united   my bag  is lost http://t.co/abc123 \n  help") == "my bag is lost help"


def test_keeps_negations_emojis_and_punctuation():
    out = clean_text("@AmericanAir NOT cancelled?! 😡 don't care")
    assert "NOT" in out and "😡" in out and "don't" in out and "?!" in out


def test_hashtag_word_kept_hash_dropped():
    assert clean_text("#worstairline ever") == "worstairline ever"


def test_html_entities_unescaped():
    assert clean_text("fix &amp; refund") == "fix & refund"


def test_none_and_nan_safe():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_repairs_known_source_artifacts():
    assert clean_text("Cancelled Flightled flight") == "cancelled flight"
    assert clean_text("a few days Late Flight to me") == "a few days late to me"
    assert clean_text("reFlight Booking Problems it") == "rebook it"


def test_artifact_repair_can_be_disabled_and_detected():
    raw = "Cancelled Flightled flight"
    assert clean_text(raw, repair_artifacts=False) == raw
    assert has_artifact(raw) and not has_artifact("cancelled flight")
