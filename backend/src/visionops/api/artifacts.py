"""Read-only projections of configured public reports and local MLflow run metadata."""

import re
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from uuid import UUID

import yaml
from pydantic import ValidationError

from visionops.api.publication import Publication, read_json, safe_revision
from visionops.api.schemas import MLflowReference, RecommendationResponse, StudyResponse
from visionops.config import Settings

FEATURE_NAMES = {
    "spatial_complexity",
    "temporal_complexity",
    "motion_intensity",
    "edge_density",
    "texture_complexity",
    "mean_brightness",
    "brightness_variation",
    "condition",
}


@dataclass
class StudyRecord:
    response: StudyResponse
    directory: Path = field(repr=False)
    digest: str = field(repr=False)


def mlflow_reference(run_id, directory: Path | None) -> MLflowReference:
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
        return MLflowReference(status="not_recorded")
    reference = MLflowReference(run_id=run_id)
    if directory is None or not directory.is_dir() or directory.is_symlink():
        return reference
    try:
        for experiment in sorted(islice(directory.iterdir(), 1000)):
            if not experiment.name.isdigit() or experiment.is_symlink() or not experiment.is_dir():
                continue
            path = experiment / run_id / "meta.yaml"
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size > 65536
                or not path.resolve().is_relative_to(directory.resolve())
            ):
                continue
            data = yaml.safe_load(path.read_text())
            if data.get("run_id") == run_id:
                reference.status = {
                    1: "running",
                    2: "scheduled",
                    3: "finished",
                    4: "failed",
                    5: "killed",
                }.get(data.get("status"), "unknown")
                return reference
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        pass
    return reference


def read_study(directory: Path, settings: Settings, published: Publication) -> StudyRecord | None:
    try:
        if directory.is_symlink() or str(UUID(directory.name)) != directory.name:
            return None
        root = settings.dashboard_reports
        raw = read_json(directory / "report.json", root)
        metadata, dataset = raw["metadata"], raw["dataset"]
        if metadata.get("sensitive") is not False or dataset["kind"] not in {
            "synthetic_only",
            "public_video",
        }:
            return None
        version = metadata["extractor_version"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", version):
            return None
        threshold = None
        try:
            threshold = read_json(directory / "config.json", root).get("minimum_ssim")
        except (OSError, ValueError, KeyError):
            pass
        model = directory / "model.joblib"
        model_exists = model.is_file() and not model.is_symlink()
        status = (
            "configured"
            if re.fullmatch(
                r"[a-f0-9]{64}", settings.dashboard_trusted_models.get(directory.name, "")
            )
            else "metadata_only"
        )
        recommendations = []
        names = {s.name for s in published.sources.values()}
        for row in raw.get("policies", {}).get("per_source", []):
            if row.get("source_id") not in names:
                continue
            predicted = row["predicted_recommendation"]
            feasible = predicted.get("crf") is not None
            measured = row.get("content_aware") or {}
            recommendations.append(
                RecommendationResponse(
                    source_name=row["source_id"],
                    crf=predicted.get("crf"),
                    predicted_ssim=predicted.get("predicted_ssim"),
                    predicted_bitrate_bps=predicted.get("predicted_bitrate_bps"),
                    constraint_status="predicted_feasible"
                    if feasible
                    else "no_predicted_candidate",
                    explanation="Predicted feasibility requires measurement."
                    if feasible
                    else "No candidate is predicted to meet the quality constraint.",
                    measured_ssim=measured.get("measured_ssim"),
                    measured_bitrate_bps=measured.get("measured_bitrate_bps"),
                )
            )
        importance = {
            target: {name: value for name, value in values.items() if name in FEATURE_NAMES}
            for target, values in raw.get("validation_permutation_importance", {}).items()
            if target in {"ssim", "bitrate_bps"}
        }
        response = StudyResponse(
            id=directory.name,
            dataset_kind=dataset["kind"],
            source_count=dataset["source_count"],
            row_count=dataset["rows"],
            seed=metadata["seed"],
            code_revision=safe_revision(metadata.get("code_revision")),
            selected_models=metadata["selected_models"],
            extractor_version=version,
            model_status=status if model_exists else "missing",
            minimum_ssim=threshold,
            mlflow=mlflow_reference(raw.get("mlflow_run_id"), settings.dashboard_mlflow_directory),
            split_groups=dataset["split_groups"],
            test_metrics=raw["test"],
            policies=raw["policies"]["summary"],
            feature_importance=importance,
            recommendations=recommendations,
        )
        return StudyRecord(response, directory, dataset.get("digest", ""))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ValidationError):
        return None


def studies(settings: Settings, published: Publication) -> list[StudyRecord]:
    root = settings.dashboard_reports
    if root is None or not root.is_dir() or root.is_symlink() or published.status != "ready":
        return []
    try:
        found = [
            read_study(directory, settings, published)
            for directory in sorted(islice(root.iterdir(), 1000))
            if directory.is_dir()
        ]
        return sorted(
            [record for record in found if record is not None],
            key=lambda record: (record.directory / "report.json").stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return []
