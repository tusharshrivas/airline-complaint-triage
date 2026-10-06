"""Tweet cleaning shared by every model (classical and LLM see the same text).

What we remove, and why
- URLs: t.co links are unique tokens with no category signal; they only add noise/overfit.
- @mentions: almost every tweet starts with the airline handle (@united, @AmericanAir ...).
  The handle identifies the airline, not the complaint type, and would let models learn
  airline-specific shortcuts. Removing it also hides the brand from the LLM.
- HTML entities (&amp;) are unescaped; runs of whitespace are collapsed.

Source-data repair (found while building this project, see ASSUMPTIONS.md)
- The released Tweets.csv contains a find-and-replace artefact: the substrings "cancel", "late"
  and "book" were replaced by the label names "Cancelled Flight", "Late Flight" and
  "Flight Booking Problems" (e.g. "Cancelled Flightled flight", "reFlight Booking Problems",
  "Late Flightr"). ~16% of the usable tweets are affected. Those strings never occur in real
  tweets (train/serve skew) and partly mirror the label (leakage). `repair_artifacts=True`
  (default) inverts the three substitutions exactly; `repair_artifacts=False` reproduces the raw
  text and is used only for the sensitivity ablation.

What we deliberately KEEP
- Negations ("not", "never", "n't", "no"): "flight NOT cancelled" must differ from "cancelled".
  No stop-word removal is applied anywhere.
- Emojis and punctuation ("!!!", "?"): they carry tone and are tokenised by the vectorisers.
- Hashtag words (only the leading '#' is dropped) and original casing (vectorisers lower-case
  on their own; the LLM sees the original casing).
"""

from __future__ import annotations

import html
import re

_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_HASH = re.compile(r"#(?=\w)")
_SPACE = re.compile(r"\s+")
# Injected label strings -> the original word stems they replaced (case-sensitive: that is how they appear).
_ARTIFACTS = (("Cancelled Flight", "cancel"), ("Late Flight", "late"), ("Flight Booking Problems", "book"))


def clean_text(text: object, repair_artifacts: bool = True) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = html.unescape(str(text))
    if repair_artifacts:
        for injected, original in _ARTIFACTS:
            s = s.replace(injected, original)
    s = _URL.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = _HASH.sub("", s)
    return _SPACE.sub(" ", s).strip()


def has_artifact(text: object) -> bool:
    """True if the raw tweet contains one of the injected label strings (see module docstring)."""
    s = html.unescape(str(text))
    return any(injected in s for injected, _ in _ARTIFACTS)
