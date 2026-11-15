import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from visionops.db.models import EncodingRecord, Experiment, QualityRecord, Video
from visionops.db.session import build_session_factory
from visionops.pipelines.baseline import run_baseline
from visionops.pipelines.ingest import ingest_batch, ingest_video
from visionops.video.dataset import prepare_dataset
from visionops.video.encoding import encode_video
from visionops.video.metadata import extract_metadata
from visionops.video.process import VideoError
from visionops.video.quality import evaluate_quality
from visionops.video.schemas import EncodingConfig


@pytest.mark.parametrize(
    "parameters",
    [
        {"crf": -1},
        {"crf": 52},
        {"crf": True},
        {"crf": "23"},
        {"preset": "medium; touch hacked"},
        {"codec": "hevc"},
        {"width": 320},
        {"height": 240},
        {"width": 321, "height": 240},
        {"threads": 0},
        {"width": 0, "height": 240},
    ],
)
def test_encoding_validation(parameters):
    with pytest.raises(ValidationError):
        EncodingConfig(**parameters)


def test_resolution_validation():
    assert EncodingConfig(width=160, height=120).height == 120


def test_foreign_keys_are_enabled(engine):
    with engine.connect() as connection:
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1
    with build_session_factory(engine)() as session:
        session.add(
            QualityRecord(
                experiment_id="missing",
                psnr_db=30,
                psnr_infinite=False,
                ssim=0.9,
                vmaf=None,
                frame_count=1,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_etl_duplicate_batch_and_persistence(synthetic_video, engine, tmp_path, caplog):
    caplog.set_level("INFO")
    first = ingest_video(synthetic_video, engine)
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(synthetic_video.read_bytes())
    second = ingest_video(copy, engine)
    assert not first.duplicate
    assert second.duplicate and first.video_id == second.video_id
    result = ingest_batch([tmp_path / "missing.mp4", synthetic_video], engine)
    assert len(result.processed) == 1 and len(result.errors) == 1
    with build_session_factory(engine)() as session:
        assert session.scalar(select(func.count()).select_from(Video)) == 1
        stored = session.get(Video, first.video_id)
        assert stored.width == 320 and stored.height == 240
    events = [json.loads(record.message)["event"] for record in caplog.records]
    assert "video_ingested" in events and "video_ingestion_failed" in events


def test_failed_metadata_leaves_no_record(engine, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with patch("visionops.pipelines.ingest.extract_metadata", side_effect=VideoError("bad video")):
        with pytest.raises(VideoError):
            ingest_video(bad, engine)
    with build_session_factory(engine)() as session:
        assert session.scalar(select(func.count()).select_from(Video)) == 0


def test_no_overwrite(synthetic_video, tmp_path):
    original = synthetic_video.read_bytes()
    with pytest.raises(VideoError, match="overwrite"):
        encode_video(synthetic_video, synthetic_video, EncodingConfig())
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep me")
    with pytest.raises(VideoError, match="overwrite"):
        encode_video(synthetic_video, output, EncodingConfig())
    assert output.read_bytes() == b"keep me"
    assert synthetic_video.read_bytes() == original


def test_encoding_failure_does_not_publish(synthetic_video, tmp_path):
    output = tmp_path / "failed.mp4"
    with patch("visionops.video.encoding.run_tool", side_effect=VideoError("encoder failed")):
        with pytest.raises(VideoError, match="encoder failed"):
            encode_video(synthetic_video, output, EncodingConfig())
    assert not output.exists()
    assert not list(tmp_path.glob(".visionops-*"))


def test_resize_encoding_and_quality_rejection(synthetic_video, tmp_path):
    encoded = encode_video(
        synthetic_video,
        tmp_path / "small.mp4",
        EncodingConfig(width=160, height=120, preset="ultrafast"),
    )
    assert (encoded.metadata.width, encoded.metadata.height) == (160, 120)
    assert encoded.metadata.codec == "h264"
    with pytest.raises(VideoError, match="matching dimensions"):
        evaluate_quality(synthetic_video, encoded.output_path)


def test_prepare_dataset_preserves_existing_files(synthetic_video):
    before = synthetic_video.read_bytes()
    with pytest.raises(VideoError, match="already exists"):
        prepare_dataset(synthetic_video.parent)
    assert synthetic_video.read_bytes() == before


def test_local_dataset_requires_attribution(synthetic_video, tmp_path):
    with pytest.raises(VideoError, match="attribution"):
        prepare_dataset(tmp_path / "local", local=synthetic_video)
    result = prepare_dataset(
        tmp_path / "attributed",
        local=synthetic_video,
        source_url="local synthetic fixture",
        license_name="synthetic",
    )
    assert result.read_bytes() == synthetic_video.read_bytes()


def test_baseline_end_to_end(synthetic_video, engine, tmp_path):
    report = run_baseline(
        synthetic_video, tmp_path / "runs", engine, preset="ultrafast", code_revision="pytest"
    )
    assert [r["config"]["crf"] for r in report["results"]] == [18, 23, 28]
    assert all(r["status"] == "completed" for r in report["results"]), report
    for result in report["results"]:
        assert result["encoding"]["elapsed_seconds"] > 0
        assert result["encoding"]["metadata"]["bitrate_bps"] > 0
        assert result["quality"]["psnr_db"] > 0
        assert 0 < result["quality"]["ssim"] <= 1
        assert result["quality"]["frame_count"] == 48
    with build_session_factory(engine)() as session:
        for table in (Experiment, EncodingRecord, QualityRecord):
            assert session.scalar(select(func.count()).select_from(table)) == 3
        assert session.scalar(select(func.count()).select_from(Video)) == 1
    saved = tmp_path / "runs" / report["run_id"] / "report.json"
    assert json.loads(saved.read_text()) == report
    same = evaluate_quality(synthetic_video, synthetic_video)
    assert same.psnr_infinite and same.psnr_db is None and same.ssim == 1
    assert extract_metadata(synthetic_video).fps == 24


def test_failed_experiments_are_recorded(synthetic_video, engine, tmp_path):
    with patch("visionops.pipelines.baseline.encode_video", side_effect=VideoError("test failure")):
        report = run_baseline(synthetic_video, tmp_path / "failed-runs", engine)
    assert all(item["status"] == "failed" for item in report["results"])
    with build_session_factory(engine)() as session:
        experiments = session.scalars(select(Experiment)).all()
        assert len(experiments) == 3
        assert all(e.error == "test failure" and e.completed_at is not None for e in experiments)
        assert session.scalar(select(func.count()).select_from(QualityRecord)) == 0


def test_quality_failure_retains_encoding_without_fabricated_metrics(
    synthetic_video, engine, tmp_path
):
    with patch(
        "visionops.pipelines.baseline.evaluate_quality", side_effect=VideoError("bad alignment")
    ):
        report = run_baseline(
            synthetic_video, tmp_path / "quality-failed", engine, preset="ultrafast"
        )
    assert all(item["status"] == "failed" and "encoding" in item for item in report["results"])
    assert all("quality" not in item for item in report["results"])
    with build_session_factory(engine)() as session:
        assert session.scalar(select(func.count()).select_from(EncodingRecord)) == 3
        assert session.scalar(select(func.count()).select_from(QualityRecord)) == 0
