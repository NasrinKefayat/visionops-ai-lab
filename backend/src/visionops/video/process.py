"""Safe, bounded external tool execution."""

import subprocess
from pathlib import Path

from visionops.config import get_settings


class VideoError(RuntimeError):
    """A file, tool, or video compatibility error suitable for a CLI message."""


def validate_video_file(path: Path | str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise VideoError(f"Video file does not exist or is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise VideoError(f"Video file is empty: {path}")
    return path


def run_tool(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=get_settings().video_timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise VideoError(
            f"Required executable not found: {args[0]}. Install FFmpeg/FFprobe."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoError(f"{args[0]} exceeded the configured timeout") from exc
    except subprocess.CalledProcessError as exc:
        raise VideoError(
            f"{args[0]} failed (exit {exc.returncode}): {(exc.stderr or '')[-4000:]}"
        ) from exc
    except OSError as exc:
        raise VideoError(f"Cannot execute {args[0]}: {exc}") from exc


def tool_version(executable: str) -> str:
    version = run_tool([executable, "-version"]).stdout.strip()
    if not version:
        raise VideoError(f"{executable} did not report a version")
    return version
