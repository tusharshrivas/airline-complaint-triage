"""Append-only run log: results/RUN_LOG.md. Every number in the README traces back to it."""

from __future__ import annotations

import datetime as dt
import os
import platform
import sys
from importlib import metadata

from triage import config

PACKAGES = ["scikit-learn", "xgboost", "pandas", "numpy", "requests", "streamlit", "matplotlib", "PyYAML"]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def env_snapshot() -> dict:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
    }


def log(section: str, lines: list[str], command: str | None = None) -> None:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS / "RUN_LOG.md"
    if not path.exists():
        path.write_text(
            "# RUN_LOG\n\nAppend-only record of every pipeline step actually executed. Written by the code, not by hand.\n",
            encoding="utf-8",
        )
    block = [f"\n## {utc_now()} - {section}\n"]
    if command:
        block.append(f"- command: `{command}`")
    block.extend(f"- {line}" for line in lines)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(block) + "\n")
