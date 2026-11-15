from unittest.mock import patch

import pytest
from pydantic import ValidationError

from visionops.video.process import VideoError
from visionops.video.quality import evaluate_quality, frame_timeline, parse_psnr, parse_ssim
from visionops.video.schemas import QualityMetrics


def test_parse_metrics():
    assert parse_psnr("[Parsed_psnr_0] PSNR y:35 u:40 v:40 average:36.123 min:32 max:39") == (
        36.123,
        False,
    )
    assert parse_psnr("PSNR average:inf min:inf max:inf") == (None, True)
    assert parse_ssim("SSIM Y:0.9 U:0.99 V:0.99 All:0.9567 (13.6)") == 0.9567
    assert parse_ssim("SSIM All:1.000000 (inf)") == 1


@pytest.mark.parametrize(
    "parser, log",
    [
        (parse_psnr, "failed"),
        (parse_psnr, "average:nan"),
        (parse_ssim, "failed"),
        (parse_ssim, "All:1.1 "),
        (parse_ssim, "All:inf "),
    ],
)
def test_metrics_never_fabricated(parser, log):
    with pytest.raises(VideoError):
        parser(log)


def test_infinite_psnr_json_representation():
    metrics = QualityMetrics(psnr_infinite=True, ssim=1, frame_count=48)
    assert '"psnr_db":null' in metrics.model_dump_json()
    with pytest.raises(ValidationError):
        QualityMetrics(psnr_db=float("nan"), ssim=1, frame_count=1)
    with pytest.raises(ValidationError):
        QualityMetrics(psnr_db=20, psnr_infinite=True, ssim=1, frame_count=1)


def test_misaligned_frames_rejected(synthetic_video):
    with patch("visionops.video.quality.frame_timeline", side_effect=[[0, 0.04], [0, 0.08]]):
        with pytest.raises(VideoError, match="alignment"):
            evaluate_quality(synthetic_video, synthetic_video)


def test_different_frame_counts_rejected(synthetic_video):
    with patch("visionops.video.quality.frame_timeline", side_effect=[[0, 0.04], [0]]):
        with pytest.raises(VideoError, match="frame counts"):
            evaluate_quality(synthetic_video, synthetic_video)


@pytest.mark.parametrize(
    "payload",
    [
        '{"frames": []}',
        '{"frames": [{}]}',
        '{"frames": [{"best_effort_timestamp_time": "nan"}]}',
        '{"frames": [{"best_effort_timestamp_time": "1"}, {"best_effort_timestamp_time": "0"}]}',
    ],
)
def test_invalid_timeline(synthetic_video, payload):
    with patch("visionops.video.quality.run_tool") as tool:
        tool.return_value.stdout = payload
        with pytest.raises(VideoError, match="alignment"):
            frame_timeline(synthetic_video)


def test_optional_vmaf_capability_gate(synthetic_video):
    with patch("visionops.video.quality.run_tool") as tool:
        tool.return_value.stdout = "Available filters: psnr ssim"
        tool.return_value.stderr = "PSNR average:30 min:20 max:40 SSIM All:0.9 (10)"
        with patch("visionops.video.quality.frame_timeline", return_value=[0, 0.04]):
            with pytest.raises(VideoError, match="does not support libvmaf"):
                evaluate_quality(synthetic_video, synthetic_video, vmaf=True)
