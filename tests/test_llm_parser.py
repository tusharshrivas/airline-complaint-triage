"""LLM parsing, retry, cache and rate-limit logic. Everything is mocked: no network, no keys."""

import json

import pytest

from triage.models import llm

LABELS = ["customer_service", "delay_or_cancellation", "baggage", "booking_or_fare", "flight_crew_or_flight", "other"]


# ---------------------------------------------------------------- parser
def test_valid_json():
    out = llm.parse_llm_output('{"category": "baggage", "confidence": 0.9, "reason": "lost bag"}', LABELS)
    assert out == {"category": "baggage", "confidence": 0.9, "reason": "lost bag"}


def test_json_in_code_fence_and_extra_text():
    assert llm.parse_llm_output('```json\n{"category": "other", "confidence": 0.5, "reason": "x"}\n```', LABELS)["category"] == "other"
    assert llm.parse_llm_output('Sure! {"category": "baggage", "confidence": 1, "reason": "r"} done', LABELS)["category"] == "baggage"


@pytest.mark.parametrize("bad", ["", "not json", "{broken", '{"cat": "baggage"}', "[1, 2]", '{"category": 5}'])
def test_invalid_json_returns_none(bad):
    assert llm.parse_llm_output(bad, LABELS) is None


def test_unknown_label_is_invalid_not_remapped():
    assert llm.parse_llm_output('{"category": "refund", "confidence": 0.9, "reason": "x"}', LABELS) is None


def test_label_normalisation_and_confidence_handling():
    out = llm.parse_llm_output('{"category": "Delay or Cancellation", "confidence": 85, "reason": "r"}', LABELS)
    assert out["category"] == "delay_or_cancellation" and out["confidence"] == 0.85
    assert llm.parse_llm_output('{"category": "other", "confidence": "high"}', LABELS)["confidence"] is None


# ---------------------------------------------------------------- fake provider
class FakeProvider(llm.BaseProvider):
    name, model = "fake", "fake-1"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, system, user, max_tokens):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return llm.ChatResponse(r, 100, 20)


def prompt():
    return llm.Prompt("t", "SYSTEM", [{"text": "bag lost", "category": "baggage", "confidence": 0.9, "reason": "r"}], "h")


def clf(provider, tmp_path=None, **kw):
    return llm.LLMClassifier(provider, "zero_shot", prompt(), LABELS, None, tmp_path, sleep=lambda s: None, **kw)


GOOD = '{"category": "baggage", "confidence": 0.9, "reason": "r"}'


def test_repair_retry_succeeds():
    p = FakeProvider(["garbage", GOOD])
    res = clf(p).classify("my bag is lost")
    assert res.category == "baggage" and res.repaired and not res.parse_failure and p.calls == 2


def test_two_invalid_outputs_count_as_parse_failure_and_fall_back_to_other():
    p = FakeProvider(["garbage", '{"category": "nonsense"}'])
    res = clf(p).classify("whatever")
    assert res.parse_failure and res.category == "other" and p.calls == 2


def test_disk_cache_makes_rerun_free(tmp_path):
    p = FakeProvider([GOOD])
    first = clf(p, tmp_path).classify("my bag is lost")
    second = clf(p, tmp_path).classify("my bag is lost")
    assert p.calls == 1 and second.cached and second.category == first.category == "baggage"


def test_cache_key_depends_on_mode_and_model(tmp_path):
    a = clf(FakeProvider([]), tmp_path)
    b = llm.LLMClassifier(FakeProvider([]), "few_shot", prompt(), LABELS, None, tmp_path)
    assert a._key("x") != b._key("x")


def test_429_is_retried_with_backoff_and_counted():
    sleeps = []
    p = FakeProvider([llm.ProviderHTTPError(429, "slow down", 2.0), llm.ProviderHTTPError(503, "oops"), GOOD])
    c = llm.LLMClassifier(p, "zero_shot", prompt(), LABELS, None, None, sleep=sleeps.append)
    res = c.classify("bag")
    assert res.category == "baggage" and res.rate_limited_retries == 1 and res.other_retries == 1
    assert sleeps[0] == 2.0 and sleeps[1] == 4.0  # Retry-After honoured, then exponential backoff


def test_gives_up_after_max_retries():
    p = FakeProvider([llm.ProviderHTTPError(500, "x")] * 5)
    with pytest.raises(llm.LLMRequestFailed):
        clf(p, max_retries=3).classify("bag")
    assert p.calls == 4


def test_long_retry_after_is_treated_as_daily_quota():
    p = FakeProvider([llm.ProviderHTTPError(429, "TPD limit", 3600.0)])
    with pytest.raises(llm.DailyQuotaExhausted):
        clf(p).classify("bag")


def test_non_retryable_error_fails_fast():
    p = FakeProvider([llm.ProviderHTTPError(401, "bad key")])
    with pytest.raises(llm.LLMRequestFailed):
        clf(p).classify("bag")
    assert p.calls == 1


# ---------------------------------------------------------------- prompt modes
def test_few_shot_adds_examples_zero_shot_does_not():
    pr = prompt()
    assert "Examples:" not in pr.system_for("zero_shot") and "bag lost" in pr.system_for("few_shot")


def test_real_prompt_file_loads_with_12_shots():
    pr = llm.load_prompt()
    assert len(pr.fewshot) == 12 and {s["category"] for s in pr.fewshot} == set(LABELS)
    assert "ONE category" in pr.system and "Examples:" not in pr.system


# ---------------------------------------------------------------- mocked HTTP adapters
class Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, json.dumps(body)

    def json(self):
        return self._body


def test_groq_adapter_request_and_parsing(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return Resp(200, {"choices": [{"message": {"content": GOOD}}], "usage": {"prompt_tokens": 50, "completion_tokens": 9}})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    prov = llm.OpenAICompatProvider("groq", "https://api.groq.com/openai/v1", "KEY", "llama-3.1-8b-instant")
    out = prov.chat("sys", "user", 120)
    assert seen["url"].endswith("/chat/completions") and seen["body"]["temperature"] == 0
    assert seen["headers"]["Authorization"] == "Bearer KEY" and out.prompt_tokens == 50 and out.completion_tokens == 9


def test_http_429_raises_with_retry_after(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp(429, {"error": "rl"}, {"Retry-After": "7"}))
    with pytest.raises(llm.ProviderHTTPError) as e:
        llm.OpenAICompatProvider("groq", "https://x", "k", "m").chat("s", "u", 10)
    assert e.value.status == 429 and e.value.retry_after == 7.0


def test_gemini_and_ollama_adapters(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        if "generateContent" in url:
            assert headers["x-goog-api-key"] == "K"
            return Resp(
                200,
                {
                    "candidates": [{"content": {"parts": [{"text": GOOD}]}}],
                    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
                },
            )
        return Resp(200, {"message": {"content": GOOD}, "prompt_eval_count": 11, "eval_count": 4})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    g = llm.GeminiProvider("https://generativelanguage.googleapis.com/v1beta", "K", "gemini-2.5-flash-lite").chat("s", "u", 50)
    o = llm.OllamaProvider("http://localhost:11434", "qwen2.5:3b").chat("s", "u", 50)
    assert g.text == GOOD and g.prompt_tokens == 7 and o.text == GOOD and o.completion_tokens == 4


def test_no_key_means_provider_unavailable_not_fake_results(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("triage.config.getenv", lambda name, default=None: None)
    with pytest.raises(llm.ProviderUnavailable, match="GROQ_API_KEY"):
        llm.make_provider("groq")


# ---------------------------------------------------------------- rate limiter
def test_rate_limiter_sleeps_to_respect_rpm():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    rl = llm.RateLimiter({"rpm": 2, "tpm": None, "rpd": None, "tpd": None}, safety=1.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        rl.wait(10)
    assert len(slept) == 1 and 59 < slept[0] < 61  # third call must wait for the window to roll


def test_rate_limiter_token_budget_and_daily_quota():
    now = [0.0]
    rl = llm.RateLimiter(
        {"rpm": None, "tpm": 1000, "rpd": 3, "tpd": None}, safety=1.0, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s)
    )
    rl.wait(600)
    rl.wait(600)  # exceeds 1000 tpm -> must sleep until the first expires
    assert now[0] >= 59
    rl.wait(100)
    with pytest.raises(llm.DailyQuotaExhausted):
        rl.wait(100)


def test_rate_limiter_persists_daily_ledger(tmp_path):
    path = tmp_path / "usage.json"
    a = llm.RateLimiter({"rpd": 10}, 1.0, path)
    a.wait(50)
    b = llm.RateLimiter({"rpd": 10}, 1.0, path)
    assert b.day["requests"] == 1 and b.day["tokens"] == 50
