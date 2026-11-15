"""Offline study CLI; python -m visionops.ml.cli --help."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from visionops.ml.config import load_config
from visionops.ml.data import SourceSample
from visionops.ml.inference import load_bundle
from visionops.ml.optimization import select_crf
from visionops.pipelines.ingest import content_hash
from visionops.video.features import FeatureSettings, VideoFeatureExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--count", type=int, default=12)
    for command in ("prepare", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("config", type=Path)
        sub.add_argument(
            "--code-revision",
            required=True,
            help="Exact revision, with -dirty suffix for uncommitted code",
        )
    predict = commands.add_parser("recommend")
    predict.add_argument("model", type=Path)
    predict.add_argument("source", type=Path)
    predict.add_argument("--minimum-ssim", type=float, required=True)
    predict.add_argument(
        "--domain",
        choices=["public_video", "synthetic_video"],
        required=True,
        help="Explicit nonsensitive input classification",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "synthetic":
            from visionops.ml.prepare import generate_sources

            print(generate_sources(args.output, count=args.count))
        elif args.command in {"prepare", "train"}:
            config = load_config(args.config)
            if args.command == "prepare":
                from visionops.ml.prepare import prepare_measurements

                prepare_measurements(
                    config.manifest,
                    config.database,
                    config.output / "measurements",
                    args.code_revision,
                )
                print("Phase 2 measurements completed")
            else:
                from visionops.ml.study import run_study

                report = run_study(config, code_revision=args.code_revision)
                print(
                    json.dumps(
                        {
                            "report": str(Path(report["output_directory"]) / "report.json"),
                            "test": report["test"],
                            "policies": report["policies"]["summary"],
                            "mlflow_run_id": report["mlflow_run_id"],
                        },
                        indent=2,
                        allow_nan=False,
                    )
                )
        else:
            bundle = load_bundle(args.model)
            if bundle.metadata.get("sensitive", True):
                raise ValueError(
                    "Restricted model requires a dedicated local-domain inference adapter"
                )
            source = SourceSample(
                "inference",
                content_hash(args.source),
                args.source,
                domain=args.domain,
                sensitive=False,
            )
            extractor = VideoFeatureExtractor(
                FeatureSettings(**bundle.metadata["feature_settings"])
            )
            if extractor.version != bundle.metadata["extractor_version"]:
                raise ValueError("Extractor version differs from the training artifact")
            result = select_crf(
                bundle, extractor.extract(source), args.minimum_ssim, bundle.metadata["candidates"]
            )
            print(json.dumps(asdict(result), indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        # No paths, dataset records or stack traces in this boundary's error log.
        logging.error(
            "Study command failed (%s); check local inputs and configuration", type(exc).__name__
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
