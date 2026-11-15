"""Extract the first video stream using FFprobe; never execute a shell."""

import json
from fractions import Fraction
from pathlib import Path

from pydantic import ValidationError

from visionops.config import get_settings
from visionops.video.process import VideoError, run_tool, validate_video_file
from visionops.video.schemas import VideoMetadata


def extract_metadata(path: Path | str) -> VideoMetadata:
    path = validate_video_file(path)
    result = run_tool(
        [
            get_settings().ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        container = data["format"]
        if not isinstance(stream, dict) or not isinstance(container, dict):
            raise ValueError("Expected a video stream and container object")
        duration = float(stream.get("duration") or container["duration"])
        size = path.stat().st_size
        bitrate = container.get("bit_rate")
        has_bitrate = bitrate not in (None, "N/A", "0")
        return VideoMetadata(
            filename=path.name,
            source_path=path,
            duration_seconds=duration,
            width=stream["width"],
            height=stream["height"],
            fps=float(Fraction(stream["avg_frame_rate"])),
            codec=stream["codec_name"],
            bitrate_bps=int(bitrate) if has_bitrate else round(size * 8 / duration),
            bitrate_source="container" if has_bitrate else "estimated",
            file_size_bytes=size,
            pixel_format=stream["pix_fmt"],
        )
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, ValidationError) as exc:
        raise VideoError(f"Invalid or unsupported video metadata for {path}: {exc}") from exc
