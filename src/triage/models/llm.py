"""Provider-agnostic LLM classifier (groq | gemini | ollama) using plain HTTP.

Design points
- One interface (`BaseProvider.chat`) behind three adapters; keys come from the environment only.
- Strict JSON output {"category","confidence","reason"}; one repair retry, then it counts as a
  parse failure and falls back to "other".
- temperature 0 and a disk cache keyed by (provider, model, prompt version+hash, mode, input),
  so reruns are free, reproducible and *resumable* across days when a free-tier daily quota runs out.
- Rate limiting reads limits from configs/providers.yaml (filled from provider docs), keeps a
  sliding-window RPM/TPM limiter plus a persisted daily ledger, and backs off exponentially on
  429/5xx (max 3 retries), honouring Retry-After.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from triage import config, data, runlog
from triage.evaluate import compute_metrics
from triage.text_clean import clean_text

# ----------------------------------------------------------------------------- errors


class ProviderUnavailable(RuntimeError):
    """No key / host unreachable / model not offered. The LLM branch is skipped, never faked."""


class DailyQuotaExhausted(RuntimeError):
    """A daily request/token quota is used up; rerun later - the cache makes this resumable."""


class LLMRequestFailed(RuntimeError):
    """A request kept failing after the allowed retries."""


class ProviderHTTPError(RuntimeError):
    def __init__(self, status: int, body: str = "", retry_after: float | None = None):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status, self.body, self.retry_after = status, body, retry_after


# ----------------------------------------------------------------------------- prompt


@dataclass
class Prompt:
    version: str
    system: str
    fewshot: list[dict]
    content_hash: str

    def system_for(self, mode: str) -> str:
        if mode == "zero_shot" or not self.fewshot:
            return self.system
        shots = "\n".join(
            f"Tweet: {ex['text']}\nAnswer: "
            + json.dumps({"category": ex["category"], "confidence": ex["confidence"], "reason": ex["reason"]})
            for ex in self.fewshot
        )
        return f"{self.system}\n\nExamples:\n{shots}"


def load_prompt(path: Path | None = None) -> Prompt:
    exp = config.experiment()["llm"]
    path = path or config.ROOT / exp["prompt_file"]
    raw = path.read_text(encoding="utf-8")
    m = re.search(r"^## SYSTEM\s*\n(.*?)(?=^## FEWSHOT\s*$)", raw, flags=re.S | re.M)
    system = (m.group(1) if m else raw).strip()
    fewshot = []
    header = re.search(r"^## FEWSHOT\s*$", raw, flags=re.M)
    for line in (raw[header.end() :] if header else "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            fewshot.append(json.loads(line))
    return Prompt(exp["prompt_version"], system, fewshot, hashlib.sha256(raw.encode()).hexdigest()[:10])


# ----------------------------------------------------------------------------- parsing


def parse_llm_output(text: str, labels: list[str]) -> dict | None:
    """Return {"category","confidence","reason"} or None when the output is unusable.

    Unknown categories are treated as invalid (-> repair retry), never silently remapped.
    """
    if not text or not text.strip():
        return None
    s = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    obj = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b > a:
            try:
                obj = json.loads(s[a : b + 1])
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("category"), str):
        return None
    cat = re.sub(r"[\s\-]+", "_", obj["category"].strip().lower())
    if cat not in labels:
        return None
    conf = obj.get("confidence")
    try:
        conf = float(conf)
        conf = conf / 100 if 1 < conf <= 100 else conf
        conf = min(max(conf, 0.0), 1.0)
    except (TypeError, ValueError):
        conf = None
    return {"category": cat, "confidence": conf, "reason": str(obj.get("reason", ""))[:300]}


# ----------------------------------------------------------------------------- providers


@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _check(resp: requests.Response) -> requests.Response:
    if resp.status_code >= 400:
        ra = resp.headers.get("Retry-After")
        try:
            retry_after = float(ra) if ra else None
        except ValueError:
            retry_after = None
        raise ProviderHTTPError(resp.status_code, resp.text, retry_after)
    return resp


class BaseProvider:
    name = "base"
    model = ""

    def chat(self, system: str, user: str, max_tokens: int) -> ChatResponse:  # pragma: no cover
        raise NotImplementedError


class OpenAICompatProvider(BaseProvider):
    """Groq (OpenAI-compatible endpoint)."""

    def __init__(self, name: str, endpoint: str, api_key: str, model: str, timeout: int = 60):
        self.name, self.endpoint, self.api_key, self.model, self.timeout = name, endpoint.rstrip("/"), api_key, model, timeout

    def chat(self, system, user, max_tokens):
        resp = _check(
            requests.post(
                f"{self.endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
        )
        body = resp.json()
        usage = body.get("usage") or {}
        return ChatResponse(body["choices"][0]["message"]["content"], usage.get("prompt_tokens"), usage.get("completion_tokens"))


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, endpoint: str, api_key: str, model: str, timeout: int = 60):
        self.endpoint, self.api_key, self.model, self.timeout = endpoint.rstrip("/"), api_key, model, timeout

    def chat(self, system, user, max_tokens):
        gen = {"temperature": 0, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"}
        if "2.5-flash" in self.model and "lite" not in self.model:
            gen["thinkingConfig"] = {"thinkingBudget": 0}  # keep the free-tier token budget for the answer
        resp = _check(
            requests.post(
                f"{self.endpoint}/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": gen,
                },
                timeout=self.timeout,
            )
        )
        body = resp.json()
        cands = body.get("candidates") or []
        parts = (cands[0].get("content", {}).get("parts") if cands else None) or [{}]
        usage = body.get("usageMetadata") or {}
        return ChatResponse("".join(p.get("text", "") for p in parts), usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: int = 300):
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout

    def chat(self, system, user, max_tokens):
        resp = _check(
            requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "num_predict": max_tokens, "seed": 42},
                },
                timeout=self.timeout,
            )
        )
        body = resp.json()
        return ChatResponse(body.get("message", {}).get("content", ""), body.get("prompt_eval_count"), body.get("eval_count"))


def _get_json(url: str, **kw) -> dict:
    try:
        return _check(requests.get(url, timeout=20, **kw)).json()
    except (requests.RequestException, ProviderHTTPError) as exc:
        raise ProviderUnavailable(f"cannot reach {url.split('?')[0]}: {exc}") from exc


def make_provider(name: str | None = None) -> BaseProvider:
    """Build a provider or raise ProviderUnavailable with a human-readable reason."""
    name = (name or config.getenv("LLM_PROVIDER") or "groq").lower()
    cfgs = config.providers()
    if name not in cfgs:
        raise ProviderUnavailable(f"unknown provider '{name}' (expected one of {list(cfgs)})")
    cfg = cfgs[name]
    override = config.getenv("LLM_MODEL")
    prefs = [override] if override else cfg["model_preference"]

    if name == "ollama":
        host = config.getenv(cfg["host_env"]) or cfg["endpoint"]
        tags = _get_json(f"{host.rstrip('/')}/api/tags")
        have = {m["name"] for m in tags.get("models", [])}
        for m in prefs:
            if m in have or f"{m}:latest" in have:
                return OllamaProvider(host, m)
        raise ProviderUnavailable(f"Ollama reachable but none of {prefs} pulled (have: {sorted(have)}). Run `ollama pull {prefs[0]}`.")

    key = config.getenv(cfg["key_env"])
    if not key:
        raise ProviderUnavailable(f"{cfg['key_env']} is not set")
    if name == "groq":
        listing = _get_json(f"{cfg['endpoint']}/models", headers={"Authorization": f"Bearer {key}"})
        available = {m["id"] for m in listing.get("data", [])}
        for m in prefs:
            if m in available:
                return OpenAICompatProvider("groq", cfg["endpoint"], key, m)
        raise ProviderUnavailable(f"none of {prefs} is listed by Groq GET /models (available: {sorted(available)[:15]}...)")
    listing = _get_json(f"{cfg['endpoint']}/models", headers={"x-goog-api-key": key})
    available = {m["name"].removeprefix("models/") for m in listing.get("models", [])}
    for m in prefs:
        if m in available:
            return GeminiProvider(cfg["endpoint"], key, m)
    raise ProviderUnavailable(f"none of {prefs} is listed by the Gemini API")


# ----------------------------------------------------------------------------- rate limiting


class RateLimiter:
    """Sliding-window RPM/TPM limiter + persisted daily request/token ledger."""

    def __init__(self, limits: dict | None, safety: float = 0.8, state_path: Path | None = None, clock=time.monotonic, sleep=time.sleep):
        limits = limits or {}

        def scale(v):
            return max(1, int(v * safety)) if v else None

        self.rpm, self.tpm = scale(limits.get("rpm")), scale(limits.get("tpm"))
        self.rpd, self.tpd = scale(limits.get("rpd")), scale(limits.get("tpd"))
        self.clock, self.sleep, self.state_path = clock, sleep, state_path
        self.window: deque[list] = deque()
        self.slept_s = 0.0
        self.day = {"date": self._today(), "requests": 0, "tokens": 0}
        if state_path and state_path.exists():
            saved = json.loads(state_path.read_text())
            if saved.get("date") == self.day["date"]:
                self.day = saved

    @staticmethod
    def _today() -> str:
        return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")

    def _persist(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.day))

    def wait(self, est_tokens: int) -> list:
        if self.day["date"] != self._today():
            self.day = {"date": self._today(), "requests": 0, "tokens": 0}
        if self.tpm and est_tokens > self.tpm:
            raise ProviderUnavailable(f"one request (~{est_tokens} tokens) exceeds the per-minute token limit ({self.tpm}); use zero-shot")
        if self.rpd and self.day["requests"] + 1 > self.rpd:
            raise DailyQuotaExhausted(f"daily request budget {self.rpd} used")
        if self.tpd and self.day["tokens"] + est_tokens > self.tpd:
            raise DailyQuotaExhausted(f"daily token budget {self.tpd} used ({self.day['tokens']} so far)")
        while True:
            now = self.clock()
            while self.window and self.window[0][0] <= now - 60:
                self.window.popleft()
            need = 0.0
            if self.rpm and len(self.window) >= self.rpm:
                need = max(need, self.window[0][0] + 60 - now)
            if self.tpm:
                total, expiry = sum(w[1] for w in self.window) + est_tokens, 0.0
                for t, tok in self.window:
                    if total <= self.tpm:
                        break
                    total -= tok
                    expiry = t + 60 - now
                if total > self.tpm:
                    expiry = 60.0
                need = max(need, expiry)
            if need <= 0:
                break
            self.sleep(need + 0.05)
            self.slept_s += need + 0.05
        entry = [self.clock(), est_tokens]
        self.window.append(entry)
        self.day["requests"] += 1
        self.day["tokens"] += est_tokens
        self._persist()
        return entry

    def record(self, entry: list, actual_tokens: int | None) -> None:
        if actual_tokens:
            self.day["tokens"] += actual_tokens - entry[1]
            entry[1] = actual_tokens
            self._persist()


# ----------------------------------------------------------------------------- classifier

REPAIR_MSG = (
    "Your previous reply was not valid. Reply again with ONLY one JSON object with keys "
    '"category" (one of: {labels}), "confidence" (0-1) and "reason". No other text.'
)


@dataclass
class LLMResult:
    category: str
    confidence: float | None = None
    reason: str = ""
    parse_failure: bool = False
    repaired: bool = False
    latency_ms: float = 0.0
    rate_limited_retries: int = 0
    other_retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""
    cached: bool = field(default=False, compare=False)


class LLMClassifier:
    def __init__(
        self,
        provider: BaseProvider,
        mode: str,
        prompt: Prompt,
        labels: list[str],
        limiter: RateLimiter | None = None,
        cache_dir: Path | None = None,
        max_retries: int = 3,
        max_tokens: int = 120,
        max_wait_s: float = 120,
        sleep=time.sleep,
    ):
        self.provider, self.mode, self.prompt, self.labels = provider, mode, prompt, labels
        self.limiter, self.cache_dir, self.max_retries = limiter, cache_dir, max_retries
        self.max_tokens, self.max_wait_s, self.sleep = max_tokens, max_wait_s, sleep
        self.system = prompt.system_for(mode)

    # -- cache
    def _key(self, text: str) -> str:
        blob = json.dumps(
            [self.provider.name, self.provider.model, self.prompt.version, self.prompt.content_hash, self.mode, text], ensure_ascii=False
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str) -> LLMResult | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            res = LLMResult(**json.loads(p.read_text()))
            res.cached = True
            return res
        return None

    def _cache_put(self, key: str, res: LLMResult) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(asdict(res), ensure_ascii=False))

    # -- network with backoff
    def _call(self, user: str, stats: dict) -> ChatResponse:
        est = int((len(self.system) + len(user)) / 3.2) + self.max_tokens
        for attempt in range(self.max_retries + 1):
            entry = self.limiter.wait(est) if self.limiter else None
            t0 = time.perf_counter()
            try:
                resp = self.provider.chat(self.system, user, self.max_tokens)
            except ProviderHTTPError as exc:
                retryable = exc.status in (408, 429) or exc.status >= 500
                if not retryable:
                    raise LLMRequestFailed(str(exc)) from exc
                wait = exc.retry_after if exc.retry_after else min(60.0, 2.0 * 2**attempt)
                if exc.status == 429:
                    stats["rl"] += 1
                    if wait > self.max_wait_s:
                        raise DailyQuotaExhausted(f"429 asks to wait {wait:.0f}s: {exc.body[:120]}") from exc
                else:
                    stats["other"] += 1
                if attempt == self.max_retries:
                    raise LLMRequestFailed(f"gave up after {self.max_retries} retries: {exc}") from exc
                self.sleep(wait)
                continue
            except (requests.ConnectionError, requests.Timeout) as exc:
                stats["other"] += 1
                if attempt == self.max_retries:
                    raise LLMRequestFailed(f"gave up after {self.max_retries} retries: {exc}") from exc
                self.sleep(min(60.0, 2.0 * 2**attempt))
                continue
            stats["latency"] += (time.perf_counter() - t0) * 1000
            used = (resp.prompt_tokens or 0) + (resp.completion_tokens or 0)
            stats["pt"] += resp.prompt_tokens or 0
            stats["ct"] += resp.completion_tokens or 0
            if self.limiter:
                self.limiter.record(entry, used or None)
            return resp
        raise LLMRequestFailed("unreachable")  # pragma: no cover

    def classify(self, raw_text: str) -> LLMResult:
        text = clean_text(raw_text)
        key = self._key(text)
        hit = self._cache_get(key)
        if hit:
            return hit
        stats = {"rl": 0, "other": 0, "latency": 0.0, "pt": 0, "ct": 0}
        user = f"Tweet: {text}"
        resp = self._call(user, stats)
        parsed, repaired = parse_llm_output(resp.text, self.labels), False
        raw = resp.text
        if parsed is None:  # one repair retry
            repaired = True
            repair = f"{user}\n\n" + REPAIR_MSG.format(labels=", ".join(self.labels))
            resp = self._call(repair, stats)
            raw = resp.text
            parsed = parse_llm_output(resp.text, self.labels)
        res = LLMResult(
            category=parsed["category"] if parsed else "other",
            confidence=parsed["confidence"] if parsed else None,
            reason=parsed["reason"] if parsed else "",
            parse_failure=parsed is None,
            repaired=repaired and parsed is not None,
            latency_ms=stats["latency"],
            rate_limited_retries=stats["rl"],
            other_retries=stats["other"],
            prompt_tokens=stats["pt"],
            completion_tokens=stats["ct"],
            raw=raw[:500],
        )
        self._cache_put(key, res)
        return res


# ----------------------------------------------------------------------------- pipeline


def _machine_specs() -> dict:
    mem = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem = f"{int(line.split()[1]) / 1024 / 1024:.1f} GiB"
                break
    except OSError:
        pass
    snap = runlog.env_snapshot()
    return {"cpu_count": snap["cpu_count"], "ram": mem, "platform": snap["platform"]}


def _write_status(status: dict) -> None:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "llm_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def _run_split(clf: LLMClassifier, frame: pd.DataFrame, limit: int | None = None):
    """Classify rows until done / limit / daily quota. Returns (rows, finished, message)."""
    rows, new_calls = [], 0
    for rec in frame.itertuples():
        try:
            if limit is not None and new_calls >= limit and clf._cache_get(clf._key(clean_text(rec.text))) is None:
                return rows, False, f"--limit {limit} reached"
            res = clf.classify(rec.text)
        except DailyQuotaExhausted as exc:
            return rows, False, f"daily quota exhausted: {exc}"
        except LLMRequestFailed as exc:
            print(f"  request failed for tweet {rec.tweet_id}: {exc}")
            continue
        new_calls += 0 if res.cached else 1
        rows.append(
            {
                "tweet_id": rec.tweet_id,
                "y_true": rec.label,
                "y_pred": res.category,
                "confidence": res.confidence,
                "reason": res.reason,
                "parse_failure": res.parse_failure,
                "repaired": res.repaired,
                "latency_ms": res.latency_ms,
                "rate_limited_retries": res.rate_limited_retries,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            }
        )
        if new_calls and new_calls % 25 == 0 and not res.cached:
            print(f"  {len(rows)}/{len(frame)} classified ({new_calls} new calls)", flush=True)
    return rows, True, "complete"


def plan_estimate(prompt: Prompt, n_val: int, n_test: int, limits: dict, safety: float) -> str:
    avg_user = 45

    def toks(mode):
        return int((len(prompt.system_for(mode)) / 3.2) + avg_user + 40)

    zs, fs = toks("zero_shot"), toks("few_shot")
    need_tokens = n_val * (zs + fs) + n_test * max(zs, fs)
    parts = [f"~{zs} tokens/call zero-shot, ~{fs} few-shot; worst case ~{need_tokens:,} tokens, {2 * n_val + n_test:,} calls"]
    if limits.get("tpd"):
        parts.append(f"=> >= {need_tokens / (limits['tpd'] * safety):.1f} day(s) at the daily token cap")
    if limits.get("rpd"):
        parts.append(f"/ >= {(2 * n_val + n_test) / (limits['rpd'] * safety):.1f} day(s) at the daily request cap")
    return " ".join(parts)


def run_pipeline(limit: int | None = None, provider_name: str | None = None) -> int:
    cmd = "python -m triage.cli run-llm" + (f" --limit {limit}" if limit else "")
    exp = config.experiment()
    llm_cfg = exp["llm"]
    pname = (provider_name or config.getenv("LLM_PROVIDER") or "groq").lower()
    base_status = {"provider": pname, "date_utc": runlog.utc_now(), "seed": exp["seed"]}
    try:
        provider = make_provider(pname)
    except ProviderUnavailable as exc:
        _write_status({**base_status, "status": "skipped", "reason": str(exc)})
        runlog.log(
            "run-llm SKIPPED (LLM branch did not execute)",
            [f"provider={pname}", f"reason: {exc}", "No LLM numbers exist; none are fabricated."],
            cmd,
        )
        print(f"[run-llm] skipped: {exc}")
        return 0

    pcfg = config.providers()[pname]
    labels = config.class_names()
    prompt = load_prompt()
    frames = data.load_split_frames()
    val = frames["val"]
    n_sub = min(llm_cfg["val_subsample"], len(val))
    val_sub, _ = train_test_split(val, train_size=n_sub, stratify=val["label"], random_state=exp["seed"])
    val_sub = val_sub.sort_values("tweet_id")
    test = frames["test"]
    limiter = RateLimiter(
        pcfg["limits"], llm_cfg["safety_factor"], config.CACHE / f"_usage_{pname}_{provider.model.replace('/', '_').replace(':', '_')}.json"
    )
    est = plan_estimate(prompt, len(val_sub), len(test), pcfg["limits"], llm_cfg["safety_factor"])
    print(f"[run-llm] provider={pname} model={provider.model}\n[run-llm] plan: {est}")
    status = {
        **base_status,
        "model": provider.model,
        "limits_used": pcfg["limits"],
        "limits_source": pcfg.get("limits_source"),
        "limits_checked": pcfg.get("limits_checked"),
        "safety_factor": llm_cfg["safety_factor"],
        "prompt_version": prompt.version,
        "prompt_hash": prompt.content_hash,
        "machine": _machine_specs() if pname == "ollama" else None,
        "plan": est,
    }

    def make_clf(mode):
        return LLMClassifier(
            provider,
            mode,
            prompt,
            labels,
            limiter,
            config.CACHE,
            llm_cfg["max_retries"],
            llm_cfg["max_output_tokens"],
            llm_cfg["max_wait_s"],
        )

    # 1) prompt choice on the validation subsample only
    choice = {}
    for mode in llm_cfg["prompt_modes"]:
        print(f"[run-llm] validation subsample ({len(val_sub)}) - {mode}")
        rows, done, msg = _run_split(make_clf(mode), val_sub)
        if not done:
            _write_status({**status, "status": "partial_prompt_choice", "reason": f"{mode}: {msg}"})
            runlog.log(
                "run-llm PARTIAL (prompt choice incomplete)",
                [
                    f"provider={pname} model={provider.model}",
                    f"{mode}: {len(rows)}/{len(val_sub)} done; stopped: {msg}",
                    "Re-run the same command later; the disk cache resumes where it stopped.",
                ],
                cmd,
            )
            print(f"[run-llm] stopped during prompt choice: {msg}. Re-run later to resume.")
            return 0
        df = pd.DataFrame(rows)
        m = compute_metrics(df["y_true"], df["y_pred"], labels)
        choice[mode] = {
            "val_macro_f1": m["macro_f1"],
            "val_accuracy": m["accuracy"],
            "n": m["n"],
            "parse_failures": int(df["parse_failure"].sum()),
            "mean_prompt_tokens": float(df["prompt_tokens"].mean()),
        }
    chosen = max(llm_cfg["prompt_modes"], key=lambda md: (choice[md]["val_macro_f1"], md == "zero_shot"))
    status["prompt_choice"] = {**choice, "chosen": chosen, "rule": "highest validation macro-F1; tie -> zero_shot (cheaper)"}
    runlog.log(
        "run-llm prompt choice (validation subsample only)",
        [
            f"provider={pname} model={provider.model} subsample n={len(val_sub)} seed={exp['seed']}",
            *[f"{md}: {v}" for md, v in choice.items()],
            f"chosen: {chosen}",
        ],
        cmd,
    )

    # 2) final test evaluation with the chosen prompt
    print(f"[run-llm] test split ({len(test)}) with {chosen}")
    rows, done, msg = _run_split(make_clf(chosen), test, limit)
    df = pd.DataFrame(rows)
    config.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    if len(df):
        df.to_csv(config.PREDICTIONS / "llm_test.csv", index=False)
    status.update(
        {
            "status": "ok" if done and len(df) == len(test) else "partial",
            "chosen_mode": chosen,
            "test_classified": int(len(df)),
            "test_total": int(len(test)),
            "stop_reason": msg,
            "rate_limiter_slept_s": round(limiter.slept_s, 1),
        }
    )
    _write_status(status)
    runlog.log(
        "run-llm test run",
        [
            f"provider={pname} model={provider.model} mode={chosen} seed={exp['seed']}",
            f"classified {len(df)}/{len(test)} test tweets; stop reason: {msg}",
            f"parse failures={int(df['parse_failure'].sum()) if len(df) else 0}; "
            f"rate-limited retries={int(df['rate_limited_retries'].sum()) if len(df) else 0}",
            f"limits used: {pcfg['limits']} x safety {llm_cfg['safety_factor']} (source {pcfg.get('limits_source')}, checked {pcfg.get('limits_checked')})",
        ],
        cmd,
    )
    if status["status"] == "partial":
        print(f"[run-llm] partial ({msg}). Re-run later to resume from the cache.")
    return 0
