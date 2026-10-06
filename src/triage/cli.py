"""Command line entry point: python -m triage.cli <command>."""

from __future__ import annotations

import argparse
import subprocess
import sys

from triage import config, runlog


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from triage import data

    config.load_dotenv()
    exp = config.experiment()
    try:
        raw = data.ensure_raw()
    except data.DatasetMissing as exc:
        print(f"[prepare-data] STOPPED: {exc}")
        runlog.log("prepare-data STOPPED", [str(exc)], "python -m triage.cli prepare-data")
        return 2
    df = data.load_raw(raw)
    labeled, stats = data.build_labeled(df, dedupe_text=exp["dedupe_cleaned_text"])
    splits = data.stratified_split(labeled, exp["seed"], exp["split"])
    data.save_splits(labeled, splits)
    fp = data.splits_fingerprint(splits)
    lines = [
        f"input: `{raw}` sha256={data.sha256_file(raw)}",
        f"seed={exp['seed']} split={exp['split']}",
        f"row accounting: {stats}",
        "split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()),
        f"split fingerprint (sha256[:16] of the 3 id lists): {fp}",
    ]
    runlog.log("prepare-data", lines, "python -m triage.cli prepare-data")
    print("\n".join(lines))
    return 0


def cmd_train_classical(args: argparse.Namespace) -> int:
    from triage.models import classical

    config.load_dotenv()
    result = classical.run_training()
    print(f"best classical model by validation macro-F1: {result['best_by_val']}")
    return 0


def cmd_run_llm(args: argparse.Namespace) -> int:
    from triage.models import llm

    config.load_dotenv()
    return llm.run_pipeline(limit=args.limit, provider_name=args.provider)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from triage import report

    config.load_dotenv()
    report.build_all()
    return 0


def cmd_export_deploy_model(args: argparse.Namespace) -> int:
    from triage.models import classical

    path = classical.export_deploy_model()
    print(f"exported {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    app_path = config.ROOT / "src" / "triage" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path), *args.extra])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data", help="load, filter, map, split").set_defaults(func=cmd_prepare_data)
    sub.add_parser("train-classical", help="TF-IDF + LR / XGBoost").set_defaults(func=cmd_train_classical)
    p = sub.add_parser("run-llm", help="zero/few-shot LLM classifier (free provider)")
    p.add_argument("--limit", type=int, default=None, help="max test tweets to classify this run")
    p.add_argument("--provider", default=None, help="groq | gemini | ollama (default: $LLM_PROVIDER or groq)")
    p.set_defaults(func=cmd_run_llm)
    sub.add_parser("evaluate", help="metrics, CI, plots, reports").set_defaults(func=cmd_evaluate)
    sub.add_parser("export-deploy-model", help="copy the best classical model to deploy_model/").set_defaults(func=cmd_export_deploy_model)
    p = sub.add_parser("app", help="launch the Streamlit demo")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_app)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
