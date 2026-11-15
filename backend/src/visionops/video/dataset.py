"""Explicit small dataset preparation; never download data automatically."""

import json
import shutil
from pathlib import Path

from visionops.config import get_settings
from visionops.pipelines.ingest import content_hash
from visionops.video.metadata import extract_metadata
from visionops.video.process import VideoError, run_tool, tool_version, validate_video_file


def prepare_dataset(
    output: Path,
    *,
    local: Path | None = None,
    source_url: str | None = None,
    license_name: str | None = None,
) -> Path:
    """Generate a fixed synthetic clip, or copy a user-provided licensed local clip."""
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "sample.mp4"
    manifest = output / "manifest.json"
    if destination.exists() or manifest.exists():
        raise VideoError("Dataset output already exists; choose a fresh directory")
    if local is not None:
        local = validate_video_file(local)
        if not source_url or not license_name:
            raise VideoError("A local clip requires --source-url and --license attribution")
        metadata = extract_metadata(local)
        if metadata.duration_seconds > 30 or metadata.file_size_bytes > 50 * 1024 * 1024:
            raise VideoError(
                "Use a local sample no longer than 30 seconds and no larger than 50 MiB"
            )
        with destination.open("xb") as target, local.open("rb") as source:
            shutil.copyfileobj(source, target)
        provenance = {
            "source": source_url,
            "license": license_name,
            "original_filename": local.name,
        }
    else:
        run_tool(
            [
                get_settings().ffmpeg_binary,
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=24:duration=2",
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-preset",
                "medium",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(destination),
            ]
        )
        provenance = {
            "source": "FFmpeg lavfi testsrc2, 320x240, 24 fps, 2 seconds, lossless H.264",
            "license": "Synthetic test pattern; no third-party video footage",
        }
    provenance.update(
        sha256=content_hash(destination),
        seed=get_settings().random_seed,
        ffmpeg_version=tool_version(get_settings().ffmpeg_binary),
    )
    with manifest.open("x") as file:
        json.dump(provenance, file, indent=2)
        file.write("\n")
    return destination
