import importlib
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from visionops.ml.config import ForestSettings, StudyConfig, load_config
from visionops.ml.data import (
    Dataset,
    Observation,
    Partition,
    SourceSample,
    build_table,
    split_groups,
)
from visionops.ml.evaluation import regression_metrics
from visionops.ml.inference import ModelBundle, load_bundle, predict_values
from visionops.ml.optimization import select_crf
from visionops.ml.tracking import track_public_run
from visionops.ml.training import train_regressors


class ArrayExtractor:
    feature_names = ("density", "variation")
    version = "test-domain-v1"

    def extract(self, source):
        number = int(source.source_id.split("-")[-1])
        return {"density": number / 10, "variation": (number % 3) / 3}


@pytest.fixture
def learning_table(tmp_path):
    sources = tuple(
        SourceSample(
            f"source-{i}", f"hash-{i}", tmp_path / str(i), domain="synthetic_video", sensitive=False
        )
        for i in range(12)
    )
    rows = tuple(
        Observation(
            s.source_id,
            crf,
            {"ssim": 1 - crf / 1000 - i / 2000, "bitrate_bps": 100000 - crf * 1000 + i * 500},
        )
        for i, s in enumerate(sources)
        for crf in (18, 23, 28)
    )
    return build_table(Dataset(sources, rows), ArrayExtractor())


def test_dataset_features_extracted_once_and_no_target_columns(learning_table):
    assert learning_table.x.shape == (36, 3)
    assert learning_table.feature_names == ("density", "variation", "condition")
    for i in range(0, 36, 3):
        assert np.all(learning_table.x[i : i + 3, :2] == learning_table.x[i, :2])


def test_group_splits_deterministic_and_disjoint(learning_table):
    first, second = split_groups(learning_table), split_groups(learning_table)
    for name in ("train", "validation", "test"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    sets = [
        {learning_table.source_ids[i] for i in indices}
        for indices in (first.train, first.validation, first.test)
    ]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert set.union(*sets) == {s.source_id for s in learning_table.dataset.sources}


def test_patient_study_and_hash_connected_groups(tmp_path):
    sources = tuple(
        SourceSample(
            f"source-{i}",
            "same" if i < 2 else f"hash-{i}",
            tmp_path / str(i),
            domain="medical",
            patient_group="patient-a" if i in (1, 2) else f"p-{i}",
            study_group="study-a" if i in (2, 3) else f"s-{i}",
        )
        for i in range(12)
    )
    observations = tuple(Observation(s.source_id, 23, {"custom_target": 0.5}) for s in sources)
    table = build_table(Dataset(sources, observations), ArrayExtractor())
    split = split_groups(table)
    mapping = {table.source_ids[i]: split.groups[i] for i in range(12)}
    assert len({mapping[f"source-{i}"] for i in range(4)}) == 1
    assert table.dataset.restricted


def test_reject_duplicate_conditions_and_too_few_groups(learning_table):
    dataset = learning_table.dataset
    with pytest.raises(ValueError, match="Duplicate"):
        build_table(
            Dataset(dataset.sources, (*dataset.observations, dataset.observations[0])),
            ArrayExtractor(),
        )
    small = build_table(Dataset(dataset.sources[:2], dataset.observations[:6]), ArrayExtractor())
    with pytest.raises(ValueError, match="five"):
        split_groups(small)


def test_training_only_fits_train_and_roundtrips(learning_table, tmp_path):
    split = split_groups(learning_table)
    fitted_inputs = []

    class RecordingDummy(DummyRegressor):
        def fit(self, x, y, sample_weight=None):
            fitted_inputs.append(x.copy())
            return super().fit(x, y, sample_weight=sample_weight)

    bundle, report = train_regressors(
        learning_table,
        split,
        ("ssim", "bitrate_bps"),
        forest=ForestSettings(n_estimators=4),
        factories={"dummy": RecordingDummy},
    )
    assert len(fitted_inputs) == 2
    assert all(np.array_equal(x, learning_table.x[split.train]) for x in fitted_inputs)
    assert "test" not in report
    assert "validation_permutation_importance" in report
    # Serialize a standard estimator bundle, not the local test class.
    standard, _ = train_regressors(
        learning_table, split, ("ssim", "bitrate_bps"), forest=ForestSettings(n_estimators=4)
    )
    path = tmp_path / "model.joblib"
    joblib.dump(standard, path)
    loaded = load_bundle(path)
    for key, values in predict_values(standard, learning_table.x[split.test]).items():
        assert np.array_equal(values, predict_values(loaded, learning_table.x[split.test])[key])


def test_training_rejects_overlapping_partition(learning_table):
    split = split_groups(learning_table)
    bad = Partition(split.train, split.validation, split.train, split.groups)
    with pytest.raises(ValueError, match="exactly once"):
        train_regressors(learning_table, bad, ("ssim",), forest=ForestSettings(n_estimators=2))


def test_regression_metrics_meaningful_r2():
    scores = regression_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 4], list("abcde"))
    assert scores["mae"] == pytest.approx(0.2)
    assert scores["rmse"] == pytest.approx(np.sqrt(0.2))
    assert scores["r2"] == 0.9
    assert regression_metrics([1, 2, 3], [1, 2, 2], list("abc"))["r2"] is None
    assert regression_metrics([1, 1], [1, 1], ["a", "b"])["r2"] is None
    assert regression_metrics([1, 2], [1, 2], ["a", "a"])["r2"] is None
    with pytest.raises(ValueError):
        regression_metrics([1], [np.nan], ["a"])


class CandidateModel:
    def __init__(self, values):
        self.values = values

    def predict(self, x):
        return np.asarray([self.values[int(crf)] for crf in x[:, -1]])


def candidate_bundle():
    return ModelBundle(
        {
            "ssim": CandidateModel({18: 0.99, 23: 0.98, 28: 0.96}),
            "bitrate_bps": CandidateModel({18: 200, 23: 100, 28: 150}),
        },
        ("density", "condition"),
        {"candidates": [18, 23, 28]},
    )


def test_crf_chooses_lowest_predicted_bitrate_not_largest_crf():
    result = select_crf(candidate_bundle(), {"density": 0.5}, 0.97, [18, 23, 28])
    assert (
        result.crf == 23 and result.predicted_ssim == 0.98 and result.predicted_bitrate_bps == 100
    )
    assert "measurement required" in result.explanation
    impossible = select_crf(candidate_bundle(), {"density": 0.5}, 0.999, [18, 23, 28])
    assert impossible.crf is None and impossible.predicted_ssim is None


@pytest.mark.parametrize(
    "threshold, grid",
    [(0, [18]), (1.1, [18]), (0.9, []), (0.9, [18, 18]), (0.9, [51]), (0.9, [18.0])],
)
def test_constraint_and_grid_validation(threshold, grid):
    with pytest.raises(ValueError):
        select_crf(candidate_bundle(), {"density": 0.5}, threshold, grid)


def test_inference_module_has_no_training_import():
    source = Path(importlib.import_module("visionops.ml.inference").__file__).read_text()
    assert "from visionops.ml.training" not in source
    assert "from visionops.ml.comparison" not in source


def test_medical_tracking_blocked_before_mlflow_import(tmp_path):
    dataset = Dataset(
        (
            SourceSample(
                "opaque",
                "hash",
                tmp_path / "private",
                domain="medical",
                patient_group="opaque-patient",
            ),
        ),
        (),
    )
    with patch.dict("sys.modules", {"mlflow.tracking": None}):
        with pytest.raises(ValueError, match="must not be sent"):
            track_public_run(dataset, tmp_path / "mlruns", parameters={}, metrics={}, artifacts=[])
    assert not (tmp_path / "mlruns").exists()


def test_experiment_yaml_strict_and_relative(tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(
        "manifest: manifest.json\ndatabase: video.db\noutput: artifacts\n"
        "tracking_directory: runs\nseed: 42\n"
    )
    config = load_config(path)
    assert config.manifest == tmp_path / "manifest.json"
    assert config.seed == 42
    for values in (
        {"seed": 7},
        {"candidates": [18, 23]},
        {"unknown": True},
        {"validation_fraction": 0.8, "test_fraction": 0.3},
    ):
        with pytest.raises(ValueError):
            StudyConfig(manifest="a", database="b", output="c", tracking_directory="d", **values)


def test_oracle_measured_test_only_and_abstention(learning_table):
    from visionops.ml.comparison import compare_policies

    split = split_groups(learning_table)
    bundle = candidate_bundle()
    # A compatible two-feature schema is required; model doubles read only condition.
    bundle.feature_names = learning_table.feature_names
    result = compare_policies(
        learning_table, split, bundle, minimum_ssim=0.999, candidates=[18, 23, 28], fixed_crf=23
    )
    assert result["summary"]["content_aware"]["abstention_rate"] == 1
    assert result["summary"]["content_aware"]["violation_rate_among_selected"] is None
    assert result["summary"]["content_aware"]["mean_bitrate_reduction_percent_vs_fixed"] is None
    assert {r["source_id"] for r in result["per_source"]} == {
        learning_table.source_ids[i] for i in split.test
    }


def test_local_mlflow_store_finishes_and_reuses_experiment(tmp_path):
    from mlflow.tracking import MlflowClient

    dataset = Dataset(
        (
            SourceSample(
                "public", "hash", tmp_path / "source", domain="synthetic_video", sensitive=False
            ),
        ),
        (),
    )
    store = tmp_path / "tracking"
    for _ in range(2):
        run_id = track_public_run(
            dataset, store, parameters={"seed": 42}, metrics={"test_mae": 0.1}, artifacts=[]
        )
        run = MlflowClient(tracking_uri=store.as_uri()).get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.params == {"seed": "42"}
        assert run.data.metrics == {"test_mae": 0.1}
    assert not (store / "artifacts").exists()


def test_domain_metric_and_estimator_can_be_injected(learning_table):
    split = split_groups(learning_table)

    def task_metric(y, prediction, groups):
        return {"task_similarity": float(1 / (1 + np.mean(np.abs(y - prediction))))}

    bundle, scores = train_regressors(
        learning_table,
        split,
        ("ssim",),
        forest=ForestSettings(n_estimators=2),
        factories={"domain_adapter": lambda: DummyRegressor(strategy="mean")},
        metric_fn=task_metric,
        selection_metric="task_similarity",
        higher_is_better=True,
    )
    assert bundle.metadata["selected_models"] == {"ssim": "domain_adapter"}
    assert "task_similarity" in scores["validation"]["ssim"]["domain_adapter"]


def test_oracle_uses_lowest_measured_feasible_candidate(learning_table):
    from visionops.ml.comparison import compare_policies

    split = split_groups(learning_table)
    bundle = candidate_bundle()
    bundle.feature_names = learning_table.feature_names
    result = compare_policies(
        learning_table, split, bundle, minimum_ssim=0.975, candidates=[18, 23, 28], fixed_crf=23
    )
    for row in result["per_source"]:
        feasible = [
            i
            for i in split.test
            if learning_table.source_ids[i] == row["source_id"]
            and learning_table.targets["ssim"][i] >= 0.975
        ]
        expected = min(feasible, key=lambda i: learning_table.targets["bitrate_bps"][i])
        assert row["oracle"]["crf"] == learning_table.conditions[expected]
        assert row["oracle"]["measured_ssim"] >= 0.975
    assert result["summary"]["oracle"]["violation_rate_among_selected"] == 0


def test_complete_study_saves_snapshot_and_tracks_aggregates(learning_table, tmp_path):
    from visionops.ml.study import run_study

    config = StudyConfig(
        manifest=tmp_path / "not-read",
        database=tmp_path / "not-read.db",
        output=tmp_path / "artifacts",
        tracking_directory=tmp_path / "tracking",
        forest=ForestSettings(n_estimators=4),
    )
    with patch("visionops.ml.study.Phase2VideoAdapter") as adapter:
        adapter.return_value.load.return_value = learning_table.dataset
        with patch("visionops.ml.study.VideoFeatureExtractor", return_value=ArrayExtractor()):
            report = run_study(config, code_revision="test-only")
    import json

    output = Path(report["output_directory"])
    rows = json.loads((output / "dataset.json").read_text())
    assert len(rows) == 36
    assert all("source_hash" in row for row in rows)
    assert report["dataset"]["split_groups"] == {"train": 6, "validation": 3, "test": 3}
    assert report["test"]["ssim"]["r2"] is None
    assert report["mlflow_run_id"]
    assert load_bundle(output / "model.joblib").feature_names == learning_table.feature_names


def test_recommend_cli_honors_input_domain_without_training(monkeypatch, capsys):
    import sys

    from visionops.ml.cli import main

    bundle = candidate_bundle()
    bundle.metadata.update(sensitive=False, feature_settings={}, extractor_version="test-version")

    class InferenceExtractor:
        version = "test-version"

        def extract(self, sample):
            assert sample.domain == "synthetic_video"
            return {"density": 0.5}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study",
            "recommend",
            "trusted.joblib",
            "sample.mp4",
            "--minimum-ssim",
            ".97",
            "--domain",
            "synthetic_video",
        ],
    )
    with (
        patch("visionops.ml.cli.load_bundle", return_value=bundle),
        patch("visionops.ml.cli.content_hash", return_value="hash"),
        patch("visionops.ml.cli.VideoFeatureExtractor", return_value=InferenceExtractor()),
    ):
        assert main() == 0
    import json

    assert json.loads(capsys.readouterr().out)["crf"] == 23


def test_negative_valid_ssim_is_infeasible_not_a_fabricated_score():
    bundle = candidate_bundle()
    bundle.models["ssim"] = CandidateModel({18: -0.1, 23: -0.2, 28: -0.3})
    result = select_crf(bundle, {"density": 0.5}, 0.9, [18, 23, 28])
    assert result.crf is None and result.predicted_ssim is None
