"""Full-reference metrics with explicit decoded-frame timeline validation."""

import json
import math
import re
from pathlib import Path

from visionops.config import get_settings
from visionops.video.metadata import extract_metadata
from visionops.video.process import VideoError, run_tool, validate_video_file
from visionops.video.schemas import QualityMetrics

NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf)"


def parse_psnr(log: str) -> tuple[float | None, bool]:
    matches = re.findall(rf"average:({NUMBER})(?=\s|$)", log)
    if not matches:
        raise VideoError("FFmpeg did not report an aggregate PSNR")
    value = float(matches[-1])
    if value == math.inf:
        return None, True
    if not math.isfinite(value):
        raise VideoError("Invalid PSNR result")
    return value, False


def parse_ssim(log: str) -> float:
    matches = re.findall(rf"All:({NUMBER})(?=\s|$)", log)
    if not matches:
        raise VideoError("FFmpeg did not report an aggregate SSIM")
    value = float(matches[-1])
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise VideoError("Invalid SSIM result")
    return value


def frame_timeline(path: Path) -> list[float]:
    result = run_tool(
        [
            get_settings().ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        frames = json.loads(result.stdout)["frames"]
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
        if not timestamps or not all(math.isfinite(t) for t in timestamps):
            raise ValueError("Missing or nonfinite frame timestamps")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError("Non-increasing frame timestamps")
        return [t - timestamps[0] for t in timestamps]
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoError(f"Cannot establish decoded frame alignment for {path}: {exc}") from exc


def evaluate_quality(
    reference: Path | str, encoded: Path | str, *, vmaf: bool = False
) -> QualityMetrics:
    reference = validate_video_file(reference)
    encoded = validate_video_file(encoded)
    first, second = extract_metadata(reference), extract_metadata(encoded)
    if (first.width, first.height, first.pixel_format) != (
        second.width,
        second.height,
        second.pixel_format,
    ):
        raise VideoError("Quality evaluation requires matching dimensions and pixel formats")
    first_frames, second_frames = frame_timeline(reference), frame_timeline(encoded)
    if len(first_frames) != len(second_frames) or any(
        abs(a - b) > 0.0001 for a, b in zip(first_frames, second_frames, strict=False)
    ):
        raise VideoError("Videos have incompatible frame counts or relative timestamp alignment")
    if abs(first.duration_seconds - second.duration_seconds) > 0.001:
        raise VideoError("Videos have incompatible durations")
    prefix = (
        "[0:v:0]settb=AVTB,setpts=PTS-STARTPTS[ref];[1:v:0]settb=AVTB,setpts=PTS-STARTPTS[enc];"
    )

    def metric(filter_name: str) -> str:
        return run_tool(
            [
                get_settings().ffmpeg_binary,
                "-hide_banner",
                "-nostdin",
                "-i",
                str(reference),
                "-i",
                str(encoded),
                "-filter_complex_threads",
                "1",
                "-lavfi",
                prefix + f"[enc][ref]{filter_name}=shortest=1:repeatlast=0",
                "-an",
                "-f",
                "null",
                "-",
            ]
        ).stderr

    psnr, infinite = parse_psnr(metric("psnr"))
    ssim = parse_ssim(metric("ssim"))
    vmaf_score = None
    if vmaf:
        filters = run_tool([get_settings().ffmpeg_binary, "-hide_banner", "-filters"]).stdout
        if not re.search(r"\blibvmaf\b", filters):
            raise VideoError("VMAF requested but this FFmpeg build does not support libvmaf")
        log = metric("libvmaf")
        matches = re.findall(rf"VMAF score: ({NUMBER})", log)
        if not matches:
            raise VideoError("FFmpeg did not report a VMAF score")
        vmaf_score = float(matches[-1])
        if not math.isfinite(vmaf_score) or not 0 <= vmaf_score <= 100:
            raise VideoError("Invalid VMAF result")
    return QualityMetrics(
        psnr_db=psnr,
        psnr_infinite=infinite,
        ssim=ssim,
        vmaf=vmaf_score,
        frame_count=len(first_frames),
    )
