"""Offline research orchestration; no API, agent, medical or GPU dependency."""

import hashlib
import importlib.metadata
import json
import uuid

import joblib

from visionops.ml.comparison import compare_policies
from visionops.ml.config import StudyConfig
from visionops.ml.data import build_table, split_groups
from visionops.ml.evaluation import regression_metrics
from visionops.ml.inference import predict_values
from visionops.ml.tracking import track_public_run
from visionops.ml.training import train_regressors
from visionops.ml.video_dataset import Phase2VideoAdapter
from visionops.video.features import VideoFeatureExtractor


def run_study(config: StudyConfig, *, code_revision: str | None = None) -> dict:
    dataset = Phase2VideoAdapter(config.manifest, config.database, tuple(config.candidates)).load()
    extractor = VideoFeatureExtractor(config.features)
    table = build_table(dataset, extractor)
    split = split_groups(
        table,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    bundle, report = train_regressors(
        table, split, ("ssim", "bitrate_bps"), forest=config.forest, seed=config.seed
    )
    bundle.metadata.update(
        candidates=config.candidates,
        feature_settings=config.features.model_dump(),
        code_revision=code_revision,
        library_versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "opencv-python-headless", "joblib")
        },
    )
    # Models are frozen after validation selection. Test outcomes are never used to refit.
    predictions = predict_values(bundle, table.x[split.test])
    report["test"] = {
        target: regression_metrics(
            table.targets[target][split.test], values, [split.groups[i] for i in split.test]
        )
        for target, values in predictions.items()
    }
    report["policies"] = compare_policies(
        table,
        split,
        bundle,
        minimum_ssim=config.minimum_ssim,
        candidates=config.candidates,
        fixed_crf=config.fixed_crf,
    )
    report["metadata"] = bundle.metadata
    report["dataset"] = {
        "source_count": len(dataset.sources),
        "rows": len(table.x),
        "kind": "synthetic_only"
        if all(s.domain == "synthetic_video" for s in dataset.sources)
        else "public_video",
        "digest": hashlib.sha256(
            json.dumps(
                [
                    (o.source_id, o.condition, o.targets, o.measurements)
                    for o in dataset.observations
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "split_groups": {
            name: len({split.groups[i] for i in getattr(split, name)})
            for name in ("train", "validation", "test")
        },
    }
    output = config.output / str(uuid.uuid4())
    output.mkdir(parents=True, exist_ok=False)
    bundle.metadata["dataset_digest"] = report["dataset"]["digest"]
    joblib.dump(bundle, output / "model.joblib")
    (output / "interpretation.json").write_text(
        json.dumps(report["validation_permutation_importance"], indent=2, allow_nan=False) + "\n"
    )
    # Local snapshot preserves exact measured targets, partitions and feature values.
    labels = {
        int(i): name for name in ("train", "validation", "test") for i in getattr(split, name)
    }
    snapshot = [
        {
            "source_id": identity,
            "source_hash": next(s.content_hash for s in dataset.sources if s.source_id == identity),
            "condition": int(table.conditions[i]),
            "split": labels[i],
            "features": dict(zip(table.feature_names, table.x[i].tolist(), strict=True)),
            "targets": {name: float(values[i]) for name, values in table.targets.items()},
            "measurements": next(
                o.measurements
                for o in dataset.observations
                if o.source_id == identity and o.condition == table.conditions[i]
            ),
        }
        for i, identity in enumerate(table.source_ids)
    ]
    (output / "dataset.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    report["mlflow_run_id"] = None
    if config.tracking:
        parameters = {
            "seed": config.seed,
            "code_revision": code_revision or "unspecified",
            "dataset_kind": report["dataset"]["kind"],
            "source_count": len(dataset.sources),
            "row_count": len(table.x),
            "dataset_digest": report["dataset"]["digest"],
            "feature_version": extractor.version,
            "feature_settings": config.features.model_dump_json(),
            "forest": config.forest.model_dump_json(),
            "quality_threshold": config.minimum_ssim,
            "candidates": str(config.candidates),
            "library_versions": json.dumps(bundle.metadata["library_versions"]),
            **{
                f"{name}_groups": value for name, value in report["dataset"]["split_groups"].items()
            },
            **{
                f"selected_{target}": name
                for target, name in bundle.metadata["selected_models"].items()
            },
        }
        metrics = {
            f"test_{target}_{metric}": value
            for target, scores in report["test"].items()
            for metric, value in scores.items()
        }
        metrics.update(
            {
                f"policy_{policy}_{metric}": value
                for policy, scores in report["policies"]["summary"].items()
                for metric, value in scores.items()
            }
        )
        for target, models in report["validation"].items():
            for model, scores in models.items():
                metrics.update(
                    {f"validation_{target}_{model}_{key}": value for key, value in scores.items()}
                )
        report["mlflow_run_id"] = track_public_run(
            dataset,
            config.tracking_directory,
            parameters=parameters,
            metrics=metrics,
            artifacts=[output / "model.joblib", output / "interpretation.json"],
        )
    report["output_directory"] = str(output)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    return report
