"""Allowlisted aggregate tracking for explicitly public/synthetic datasets only."""

from pathlib import Path

from visionops.ml.data import Dataset


def track_public_run(
    dataset: Dataset, directory: Path, *, parameters: dict, metrics: dict, artifacts: list[Path]
) -> str:
    if dataset.restricted:
        raise ValueError(
            "Restricted/medical datasets and model artifacts must not be sent to MLflow"
        )
    # Local file store only. Never send manifests, paths, identities or per-source reports.
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=directory.resolve().as_uri())
    experiment = client.get_experiment_by_name("encoding-optimization")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment("encoding-optimization")
    )
    run = client.create_run(experiment_id)
    run_id = run.info.run_id
    try:
        allowed = {
            "seed",
            "code_revision",
            "dataset_kind",
            "source_count",
            "row_count",
            "train_groups",
            "validation_groups",
            "test_groups",
            "dataset_digest",
            "selected_ssim",
            "selected_bitrate_bps",
            "feature_version",
            "forest",
            "feature_settings",
            "quality_threshold",
            "candidates",
            "library_versions",
        }
        if set(parameters) - allowed:
            raise ValueError("Unapproved tracking parameter")
        for name, value in parameters.items():
            client.log_param(run_id, name, value)
        for name, value in metrics.items():
            if value is not None:
                client.log_metric(run_id, name, float(value))
        for artifact in artifacts:
            # Runner creates only public model bundles and aggregate interpretation JSON here.
            if artifact.name not in {"model.joblib", "interpretation.json"}:
                raise ValueError("Unapproved tracking artifact")
            client.log_artifact(run_id, str(artifact))
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    return run_id
