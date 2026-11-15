import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from visionops.video.metadata import extract_metadata
from visionops.video.process import VideoError, run_tool
from visionops.video.schemas import VideoMetadata


@pytest.fixture
def probe_payload():
    return {
        "streams": [
            {
                "duration": "2",
                "width": 320,
                "height": 240,
                "avg_frame_rate": "24000/1001",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
            }
        ],
        "format": {"duration": "2", "bit_rate": "80000"},
    }


def test_metadata_fractional_fps_and_safe_arguments(tmp_path, probe_payload):
    path = tmp_path / "video ; $(echo nope).mp4"
    path.write_bytes(b"test")
    with patch("visionops.video.metadata.run_tool") as tool:
        tool.return_value.stdout = json.dumps(probe_payload)
        metadata = extract_metadata(path)
    assert metadata.fps == pytest.approx(24000 / 1001)
    assert metadata.filename == path.name
    assert metadata.bitrate_bps == 80000
    assert metadata.file_size_bytes == 4
    assert tool.call_args.args[0][-1] == str(path.resolve())


def test_metadata_estimates_missing_bitrate(tmp_path, probe_payload):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"test")
    del probe_payload["format"]["bit_rate"]
    with patch("visionops.video.metadata.run_tool") as tool:
        tool.return_value.stdout = json.dumps(probe_payload)
        result = extract_metadata(path)
    assert result.bitrate_bps == 16
    assert result.bitrate_source == "estimated"


@pytest.mark.parametrize("payload", ["not json", "{}", '{"streams": [], "format": {}}'])
def test_invalid_probe_result(tmp_path, payload):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"bad")
    with patch("visionops.video.metadata.run_tool") as tool:
        tool.return_value.stdout = payload
        with pytest.raises(VideoError, match="Invalid or unsupported"):
            extract_metadata(path)


def test_missing_and_empty_file(tmp_path):
    path = tmp_path / "empty.mp4"
    with pytest.raises(VideoError, match="does not exist"):
        extract_metadata(path)
    path.touch()
    with pytest.raises(VideoError, match="empty"):
        extract_metadata(path)


@pytest.mark.parametrize(
    "error, message",
    [
        (FileNotFoundError(), "Required executable not found"),
        (subprocess.TimeoutExpired("ffmpeg", 1), "timeout"),
        (subprocess.CalledProcessError(1, "ffmpeg", stderr="invalid media"), "invalid media"),
        (PermissionError("denied"), "Cannot execute"),
    ],
)
def test_tool_errors(error, message):
    with patch("visionops.video.process.subprocess.run", side_effect=error) as tool:
        with pytest.raises(VideoError, match=message):
            run_tool(["ffmpeg", "-version"])
    assert "shell" not in tool.call_args.kwargs
    assert tool.call_args.kwargs["check"] is True
    assert tool.call_args.kwargs["timeout"] > 0


def test_metadata_rejects_invalid_fields():
    with pytest.raises(ValidationError):
        VideoMetadata(
            filename="a",
            source_path=Path("a"),
            duration_seconds=-1,
            width=0,
            height=1,
            fps=0,
            codec="",
            bitrate_bps=0,
            bitrate_source="estimated",
            file_size_bytes=0,
            pixel_format="yuv420p",
        )
