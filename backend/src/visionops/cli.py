"""CPU video pipeline commands: python -m visionops.cli --help."""

import argparse
import json
import logging
import os
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from visionops.config import Settings, get_settings
from visionops.db.session import build_engine
from visionops.pipelines.baseline import run_baseline
from visionops.pipelines.ingest import ingest_batch, initialize_database
from visionops.video.dataset import prepare_dataset
from visionops.video.process import VideoError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file", type=Path, help="Explicit .env file; exported variables take priority"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    dataset = sub.add_parser("prepare-dataset")
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--local", type=Path)
    dataset.add_argument("--source-url")
    dataset.add_argument("--license", dest="license_name")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("files", nargs="+", type=Path)
    baseline = sub.add_parser("baseline")
    baseline.add_argument("source", type=Path)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--preset", default="medium")
    baseline.add_argument(
        "--code-revision", help="Record an exact revision; append -dirty if appropriate"
    )
    baseline.add_argument("--vmaf", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = None
    try:
        if args.env_file:
            if not args.env_file.is_file():
                parser.error(f"Environment file not found: {args.env_file}")
            settings = Settings(_env_file=args.env_file)
            for key, value in settings.model_dump().items():
                os.environ.setdefault(f"VISIONOPS_{key.upper()}", str(value))
        get_settings.cache_clear()
        if args.command == "prepare-dataset":
            print(
                prepare_dataset(
                    args.output,
                    local=args.local,
                    source_url=args.source_url,
                    license_name=args.license_name,
                )
            )
            return 0
        engine = build_engine()
        if args.command == "ingest":
            initialize_database(engine)
            result = ingest_batch(args.files, engine)
            print(result.model_dump_json(indent=2))
            return int(bool(result.errors))
        report = run_baseline(
            args.source,
            args.output,
            engine,
            preset=args.preset,
            code_revision=args.code_revision,
            vmaf=args.vmaf,
        )
        print(json.dumps(report, indent=2, allow_nan=False))
        return int(any(item["status"] != "completed" for item in report["results"]))
    except (VideoError, OSError, ValueError, SQLAlchemyError) as exc:
        logging.error(json.dumps({"event": "pipeline_failed", "error": str(exc)}))
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
