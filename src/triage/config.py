"""Paths, .env loading and YAML config access. Secrets are read from the environment only."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("TRIAGE_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DEPLOY_MODEL = ROOT / "deploy_model"
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
CACHE = ROOT / "cache" / "llm"
PROMPTS = ROOT / "prompts"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader. Never overrides variables that are already set."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value and key.strip() not in os.environ:
            os.environ[key.strip()] = value


@cache
def load_yaml(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def label_map() -> dict:
    return load_yaml("label_map.yaml")


def experiment() -> dict:
    return load_yaml("experiment.yaml")


def routing() -> dict:
    return load_yaml("routing.yaml")["queues"]


def providers() -> dict:
    return load_yaml("providers.yaml")


def class_names() -> list[str]:
    """Class order is the order in label_map.yaml (used everywhere for matrices/tables)."""
    return list(label_map()["classes"].keys())


def getenv(name: str, default: str | None = None) -> str | None:
    """Env var, falling back to Streamlit secrets when running inside Streamlit."""
    value = os.environ.get(name)
    if value:
        return value
    try:  # pragma: no cover - only inside streamlit
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default
