"""CPU H.264 encoding with exclusive publication of new output files."""

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from visionops.config import get_settings
from visionops.video.metadata import extract_metadata
from visionops.video.process import VideoError, run_tool, validate_video_file
from visionops.video.schemas import EncodingConfig, EncodingResult


def encode_video(source: Path | str, output: Path | str, config: EncodingConfig) -> EncodingResult:
    source = validate_video_file(source)
    output = Path(output).expanduser().absolute()
    if output.resolve() == source or output.exists() or output.is_symlink():
        raise VideoError(f"Refusing to overwrite a source or existing output: {output}")
    if output.suffix.lower() != ".mp4":
        raise VideoError("H.264 outputs must use the .mp4 extension")
    metadata = extract_metadata(source)
    if config.width is None and (metadata.width % 2 or metadata.height % 2):
        raise VideoError("Source dimensions must be even, or provide an even output resolution")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".visionops-", dir=output.parent) as directory:
        temporary = Path(directory) / "encoded.mp4"
        args = [
            get_settings().ffmpeg_binary,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-c:v",
            config.codec,
            "-crf",
            str(config.crf),
            "-preset",
            config.preset,
            "-threads",
            str(config.threads),
            "-pix_fmt",
            config.pixel_format,
            "-fps_mode",
            "passthrough",
        ]
        if config.width is not None:
            args += ["-vf", f"scale={config.width}:{config.height}:flags=bicubic"]
        args += ["-movflags", "+faststart", str(temporary)]
        started = time.perf_counter()
        run_tool(args)
        elapsed = time.perf_counter() - started
        encoded = extract_metadata(temporary)
        # Atomic no-clobber publication on the same filesystem, including concurrent writers.
        try:
            os.link(temporary, output)
        except OSError as exc:
            raise VideoError(
                f"Cannot publish output without overwriting files: {output}: {exc}"
            ) from exc
    encoded = encoded.model_copy(update={"filename": output.name, "source_path": output.resolve()})
    return EncodingResult(output_path=output.resolve(), elapsed_seconds=elapsed, metadata=encoded)
