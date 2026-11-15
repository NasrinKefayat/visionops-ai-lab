"""Explicit public-data publication boundary; never import optional ML packages."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublicSource:
    name: str
    domain: str


@dataclass(frozen=True)
class Publication:
    sources: dict[str, PublicSource]
    status: str


def read_json(path: Path, root: Path | None = None, *, maximum_bytes: int = 2_000_000):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError("Unavailable artifact")
    if root is not None and (
        root.is_symlink() or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("Unavailable artifact")
    return json.loads(path.read_text())


def publication(path: Path | None) -> Publication:
    if path is None:
        return Publication({}, "not_configured")
    try:
        data = read_json(path)
        sources = {}
        names = set()
        for item in data["sources"]:
            if item.get("sensitive") is not False or item.get("domain") not in {
                "synthetic_video",
                "public_video",
            }:
                continue
            name, digest = item["source_id"], item["content_hash"]
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", name):
                continue
            if not re.fullmatch(r"[a-f0-9]{64}", digest) or digest in sources or name in names:
                raise ValueError("Ambiguous publication manifest")
            sources[digest] = PublicSource(name, item["domain"])
            names.add(name)
        return Publication(sources, "ready")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return Publication({}, "unavailable")


def safe_revision(value: str | None) -> str | None:
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{7,40}(?:-dirty)?", value)
        else None
    )
