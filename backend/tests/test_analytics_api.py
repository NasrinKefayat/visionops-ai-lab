import hashlib
import json
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest
from fastapi.testclient import TestClient

from visionops.api.main import create_app
from visionops.config import Settings, get_settings
from visionops.db.models import (
    EncodingConfiguration,
    EncodingRecord,
    Experiment,
    QualityRecord,
    Video,
)
from visionops.db.session import build_session_factory


def identity(name):
    return str(uuid5(NAMESPACE_URL, name))


@pytest.fixture
def dashboard(tmp_path, engine):
    with build_session_factory(engine).begin() as session:
        for number in (1, 2, 3):
            session.add(
                Video(
                    id=number,
                    sha256=str(number) * 64,
                    source_path="/private/PATIENT_SECRET/source.mp4",
                    filename="PATIENT_SECRET.mp4",
                    duration_seconds=2,
                    width=320,
                    height=240,
                    fps=24,
                    codec="h264",
                    bitrate_bps=2000,
                    bitrate_source="container",
                    file_size_bytes=10000,
                    pixel_format="yuv420p",
                    ingested_at=datetime.now(UTC),
                )
            )
        for crf in (18, 23, 28):
            session.add(
                EncodingConfiguration(
                    fingerprint=str(crf) * 32, parameters={"crf": crf, "preset": "medium"}
                )
            )
        session.flush()
        for name, video, crf, status, ssim, psnr, infinite, bitrate in [
            ("a", 1, 18, "completed", 0.99, 40, False, 1000),
            ("b", 1, 23, "completed", 0.98, None, True, 800),
            ("c", 1, 28, "failed", None, None, False, 400),
            ("d", 2, 23, "completed", 0.97, 35, False, 500),
            ("medical", 3, 23, "completed", 0.1, 1, False, 5),
        ]:
            session.add(
                Experiment(
                    id=identity(name),
                    run_id=identity(f"run-{video}"),
                    video_id=video,
                    config_id=str(crf) * 32,
                    status=status,
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    ffmpeg_version="contains /private/PATIENT_SECRET",
                    ffprobe_version="secret",
                    seed=42,
                    code_revision="b8414e8",
                    error="/private/PATIENT_SECRET" if status == "failed" else None,
                )
            )
            session.flush()
            session.add(
                EncodingRecord(
                    experiment_id=identity(name),
                    output_path="/private/PATIENT_SECRET/out.mp4",
                    output_sha256="e" * 64,
                    elapsed_seconds=0.2,
                    bitrate_bps=bitrate,
                    file_size_bytes=5000,
                    duration_seconds=2,
                )
            )
            if ssim is not None:
                session.add(
                    QualityRecord(
                        experiment_id=identity(name),
                        psnr_db=psnr,
                        psnr_infinite=infinite,
                        ssim=ssim,
                        vmaf=None,
                        frame_count=48,
                    )
                )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "scene-1",
                        "content_hash": "1" * 64,
                        "domain": "synthetic_video",
                        "sensitive": False,
                    },
                    {
                        "source_id": "scene-2",
                        "content_hash": "2" * 64,
                        "domain": "public_video",
                        "sensitive": False,
                    },
                    {
                        "source_id": "PATIENT_SECRET",
                        "content_hash": "3" * 64,
                        "domain": "medical",
                        "sensitive": True,
                    },
                ]
            }
        )
    )
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        dashboard_manifest=manifest,
        dashboard_reports=tmp_path / "reports",
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        yield client, settings, tmp_path


def test_pagination_filters_and_metadata(dashboard):
    client, _, _ = dashboard
    first = client.get("/api/v1/videos?limit=1").json()
    assert first["total"] == 2 and len(first["items"]) == 1
    second = client.get("/api/v1/videos?offset=1&limit=1").json()
    assert second["items"][0]["id"] != first["items"][0]["id"]
    assert client.get("/api/v1/videos?dataset=public_video").json()["total"] == 1
    assert client.get("/api/v1/videos?search=scene-2").json()["total"] == 1
    video = client.get("/api/v1/videos/1").json()
    assert video["available_crfs"] == [18, 23, 28]
    assert video["width"] == 320 and video["duration_seconds"] == 2
    assert "PATIENT_SECRET" not in json.dumps([first, second, video])
    assert "source_path" not in video and "filename" not in video


def test_quality_compression_and_partial_results(dashboard):
    client, _, _ = dashboard
    rows = client.get("/api/v1/videos/1/quality").json()
    assert rows["total"] == 3
    completed = next(r for r in rows["items"] if r["experiment_id"] == identity("a"))
    assert completed["compression_ratio"] == 2
    assert completed["bitrate_reduction_percent"] == 50
    assert completed["measured_ssim"] == 0.99 and completed["measured_psnr_db"] == 40
    infinite = next(r for r in rows["items"] if r["experiment_id"] == identity("b"))
    assert infinite["psnr_infinite"] and infinite["measured_psnr_db"] is None
    failed = next(r for r in rows["items"] if r["status"] == "failed")
    assert failed["measured_ssim"] is None and failed["error_code"] == "processing_failed"
    assert "PATIENT_SECRET" not in json.dumps(rows)
    assert client.get("/api/v1/videos/1/comparison?crf=23").json()["total"] == 1
    assert client.get(f"/api/v1/videos/1/encodings?run_id={identity('run-2')}").json()["total"] == 0


def test_experiments_and_monitoring(dashboard):
    client, _, _ = dashboard
    assert client.get("/api/v1/experiments").json()["total"] == 3
    assert client.get("/api/v1/experiments?status=failed").json()["total"] == 1
    detail = client.get(f"/api/v1/experiments/{identity('a')}").json()
    assert detail["crf"] == 18 and detail["seed"] == 42
    metrics = client.get("/api/v1/monitoring").json()
    assert metrics["source_videos"] == 2 and metrics["completed_encodes"] == 3
    assert metrics["average_measured_ssim"] == pytest.approx(0.98)
    assert metrics["average_measured_psnr_db"] == 37.5
    assert metrics["infinite_psnr_count"] == 1
    assert metrics["status_counts"] == {"completed": 3, "failed": 1}


@pytest.mark.parametrize(
    "url",
    [
        "/api/v1/videos?limit=0",
        "/api/v1/videos?offset=-1",
        "/api/v1/videos?limit=101",
        "/api/v1/videos?dataset=medical",
        "/api/v1/videos/1/quality?crf=52",
        "/api/v1/experiments?status=secret",
        "/api/v1/experiments/not-a-uuid",
        "/api/v1/models/not-a-uuid",
    ],
)
def test_validation_errors_are_consistent(dashboard, url):
    response = dashboard[0].get(url)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "url",
    [
        "/api/v1/videos/999",
        "/api/v1/videos/3",
        f"/api/v1/experiments/{identity('medical')}",
        f"/api/v1/models/{identity('absent')}",
    ],
)
def test_missing_and_restricted_records_indistinguishable(dashboard, url):
    response = dashboard[0].get(url)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert "PATIENT_SECRET" not in response.text


def test_unavailable_database_does_not_create_files(tmp_path):
    missing = tmp_path / "missing.db"
    settings = Settings(database_url=f"sqlite:///{missing}")
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/v1/videos").json()["available"] is False
        metrics = client.get("/api/v1/monitoring").json()
        assert metrics["source_videos"] is None and metrics["average_measured_ssim"] is None
        assert client.get("/api/v1/videos/1").status_code == 503
        assert client.get("/api/v1/models").json()["items"] == []
    assert not missing.exists()


def test_database_dependency_is_readonly(dashboard):
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from visionops.api.dependencies import get_database

    generator = get_database(dashboard[1])
    session = next(generator)
    try:
        with pytest.raises(OperationalError):
            session.execute(text("DELETE FROM videos"))
    finally:
        generator.close()


def create_report(settings, root):
    run = identity("study")
    directory = settings.dashboard_reports / run
    directory.mkdir(parents=True)
    scores = {
        "test_sources": 1,
        "selected_sources": 1,
        "abstention_rate": 0,
        "violation_rate_among_selected": 0,
        "mean_bitrate_reduction_percent_vs_fixed": -20,
        "mean_absolute_distance_from_oracle_percent": 5,
        "oracle_comparable_sources": 1,
    }
    raw = {
        "metadata": {
            "sensitive": False,
            "seed": 42,
            "selected_models": {"ssim": "dummy", "bitrate_bps": "dummy"},
            "extractor_version": "opencv-gray-v1",
            "code_revision": "b8414e8",
            "secret": "PATIENT_SECRET",
        },
        "dataset": {
            "kind": "synthetic_only",
            "source_count": 2,
            "rows": 6,
            "digest": "f" * 64,
            "split_groups": {"train": 1, "validation": 1, "test": 1},
            "path": "/private/PATIENT_SECRET",
        },
        "test": {
            "ssim": {"mae": 0.01, "rmse": 0.02, "r2": None},
            "bitrate_bps": {"mae": 1, "rmse": 2, "r2": None},
        },
        "policies": {
            "summary": {name: scores for name in ("fixed", "content_aware", "oracle")},
            "per_source": [
                {
                    "source_id": "scene-1",
                    "predicted_recommendation": {
                        "crf": 18,
                        "predicted_ssim": 0.995,
                        "predicted_bitrate_bps": 1000,
                        "explanation": "/private/PATIENT_SECRET",
                    },
                    "content_aware": {"measured_ssim": 0.99, "measured_bitrate_bps": 1000},
                }
            ],
        },
        "validation_permutation_importance": {
            "ssim": {
                "condition": {"mean": 0.02, "std": 0.01},
                "/private/PATIENT_SECRET": {"mean": 3, "std": 1},
            }
        },
        "mlflow_run_id": "a" * 32,
        "output_directory": "/private/PATIENT_SECRET",
    }
    (directory / "report.json").write_text(json.dumps(raw))
    (directory / "config.json").write_text(
        json.dumps({"minimum_ssim": 0.99, "secret": "PATIENT_SECRET"})
    )
    (directory / "model.joblib").write_bytes(b"do not load this in GET")
    from visionops.api.artifacts import FEATURE_NAMES

    (directory / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "scene-1",
                    "source_hash": "1" * 64,
                    "features": {name: 0.5 for name in FEATURE_NAMES},
                }
            ]
        )
    )
    return run, directory, raw


def test_readonly_public_report_projection_and_model_unavailable(dashboard, monkeypatch):
    client, settings, root = dashboard
    run, directory, raw = create_report(settings, root)
    import joblib

    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("GET loaded a model"))
    for endpoint in ("studies", "models"):
        listing = client.get(f"/api/v1/{endpoint}").json()
        assert listing["total"] == 1 and listing["items"][0]["model_status"] == "metadata_only"
        assert "recommendations" not in listing["items"][0]
    response = client.get(f"/api/v1/studies/{run}")
    assert response.status_code == 200
    assert "PATIENT_SECRET" not in response.text and "/private" not in response.text
    assert response.json()["recommendations"][0]["predicted_ssim"] == 0.995
    assert response.json()["recommendations"][0]["measured_ssim"] == 0.99
    assert response.json()["mlflow"]["status"] == "unavailable"
    denied = client.post(
        f"/api/v1/models/{run}/recommend", json={"video_id": 1, "minimum_ssim": 0.99}
    )
    assert denied.status_code == 503 and denied.json()["error"]["code"] == "inference_unavailable"
    raw["metadata"]["sensitive"] = True
    (directory / "report.json").write_text(json.dumps(raw))
    assert client.get("/api/v1/studies").json()["total"] == 0


def test_digest_pinned_lightweight_inference_and_abstention(dashboard):
    import joblib
    import numpy as np
    from sklearn.dummy import DummyRegressor

    from visionops.api.artifacts import FEATURE_NAMES
    from visionops.ml.inference import ModelBundle

    client, settings, root = dashboard
    run, directory, _ = create_report(settings, root)
    names = (*sorted(FEATURE_NAMES - {"condition"}), "condition")
    x = np.zeros((2, len(names)))
    bundle = ModelBundle(
        {
            "ssim": DummyRegressor().fit(x, [0.995, 0.995]),
            "bitrate_bps": DummyRegressor().fit(x, [1000, 1000]),
        },
        names,
        {
            "sensitive": False,
            "dataset_digest": "f" * 64,
            "extractor_version": "opencv-gray-v1",
            "candidates": [18, 23, 28],
        },
    )
    path = directory / "model.joblib"
    joblib.dump(bundle, path)
    settings.dashboard_trusted_models = {run: hashlib.sha256(path.read_bytes()).hexdigest()}
    response = client.post(
        f"/api/v1/models/{run}/recommend", json={"video_id": 1, "minimum_ssim": 0.99}
    )
    assert response.status_code == 200
    assert (
        response.json()["crf"] == 18
        and response.json()["constraint_status"] == "predicted_feasible"
    )
    assert response.json()["measured_ssim"] is None
    impossible = client.post(
        f"/api/v1/models/{run}/recommend", json={"video_id": 1, "minimum_ssim": 1}
    )
    assert impossible.json()["crf"] is None
    assert impossible.json()["constraint_status"] == "no_predicted_candidate"
    assert (
        client.post(
            f"/api/v1/models/{run}/recommend",
            json={"video_id": 1, "minimum_ssim": 0.99, "candidates": [51]},
        ).status_code
        == 422
    )
    path.write_bytes(b"changed after administrator trust")
    assert (
        client.post(
            f"/api/v1/models/{run}/recommend", json={"video_id": 1, "minimum_ssim": 0.99}
        ).status_code
        == 422
    )


def test_inference_body_does_not_accept_filesystem_paths(dashboard):
    client, settings, root = dashboard
    run, _, _ = create_report(settings, root)
    response = client.post(
        f"/api/v1/models/{run}/recommend",
        json={"video_id": 1, "minimum_ssim": 0.99, "source_path": "/private/PATIENT_SECRET"},
    )
    assert response.status_code == 422 and "PATIENT_SECRET" not in response.text


def test_symlink_reports_and_malformed_files_are_not_published(dashboard):
    client, settings, root = dashboard
    run, directory, _ = create_report(settings, root)
    outside = root / "outside"
    outside.mkdir()
    (outside / "report.json").write_text((directory / "report.json").read_text())
    (settings.dashboard_reports / identity("symlink")).symlink_to(outside, target_is_directory=True)
    (directory / "report.json").write_text("not json")
    assert client.get("/api/v1/studies").json()["total"] == 0


def test_local_mlflow_status_is_readonly_and_allowlisted(dashboard):
    client, settings, root = dashboard
    run, _, _ = create_report(settings, root)
    settings.dashboard_mlflow_directory = root / "tracking"
    path = settings.dashboard_mlflow_directory / "123" / ("a" * 32) / "meta.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("run_id: " + "a" * 32 + "\nstatus: 3\nartifact_uri: /private/PATIENT_SECRET\n")
    result = client.get(f"/api/v1/studies/{run}")
    assert result.json()["mlflow"]["status"] == "finished"
    assert "PATIENT_SECRET" not in result.text
    path.write_text("invalid: [yaml")
    assert client.get(f"/api/v1/studies/{run}").json()["mlflow"]["status"] == "unavailable"


def test_corrupt_but_digest_pinned_model_is_reported_without_crashing(dashboard):
    client, settings, root = dashboard
    run, directory, _ = create_report(settings, root)
    settings.dashboard_trusted_models = {
        run: hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()
    }
    response = client.post(
        f"/api/v1/models/{run}/recommend", json={"video_id": 1, "minimum_ssim": 0.99}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "inference_unavailable"
    assert "PATIENT_SECRET" not in response.text


def test_malformed_publication_fails_closed_without_zero_metric_fallback(dashboard):
    client, settings, _ = dashboard
    settings.dashboard_manifest.write_text(json.dumps({"sources": ["PATIENT_SECRET"]}))
    assert client.get("/api/v1/videos").json()["available"] is False
    metrics = client.get("/api/v1/monitoring").json()
    assert metrics["publication_status"] == "unavailable"
    assert metrics["source_videos"] is None and metrics["average_measured_ssim"] is None
    assert "PATIENT_SECRET" not in json.dumps(metrics)


def test_database_errors_return_safe_consistent_response(dashboard, monkeypatch):
    from sqlalchemy.exc import OperationalError

    def fail(*_args):
        raise OperationalError("/private/PATIENT_SECRET", {}, Exception("credential-secret"))

    monkeypatch.setattr("visionops.api.routes.list_videos", fail)
    response = dashboard[0].get("/api/v1/videos")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_unavailable"
    assert "PATIENT_SECRET" not in response.text and "credential-secret" not in response.text


@pytest.mark.parametrize(
    "url", ["/api/v1/videos/0", "/api/v1/videos/-1/quality", "/api/v1/videos?offset=1000001"]
)
def test_bounded_numeric_parameters(dashboard, url):
    assert dashboard[0].get(url).status_code == 422


def test_readonly_api_startup_does_not_import_optional_ml_packages():
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'sklearn', 'cv2', 'joblib', 'mlflow'}:
            raise ImportError('Optional ML dependency intentionally unavailable')
sys.meta_path.insert(0, NoML())
from visionops.api.main import create_app
from fastapi.testclient import TestClient
from visionops.config import Settings, get_settings
app = create_app()
app.dependency_overrides[get_settings] = lambda: Settings(database_url='sqlite:////nonexistent/analytics.db')
with TestClient(app) as client:
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.get('/api/v1/models').json()['items'] == []
    assert client.get('/api/v1/videos').json()['available'] is False
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_unexpected_errors_keep_the_safe_error_envelope(dashboard, monkeypatch):
    def fail(*_args):
        raise RuntimeError("/private/PATIENT_SECRET")

    monkeypatch.setattr("visionops.api.routes.monitoring", fail)
    with TestClient(dashboard[0].app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/monitoring")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "PATIENT_SECRET" not in response.text
