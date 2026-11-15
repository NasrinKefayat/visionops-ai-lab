Phase 2: video data and encoding pipeline
================================================================================

Architecture
------------

The API and its ``GET /health`` endpoint are unchanged. The video workflow is a
standalone Python library and CLI, independent of FastAPI, ML frameworks, and GPUs.

* ``video/schemas.py``: Pydantic metadata, encoding configuration/results, metrics.
* ``video/process.py``: subprocess execution, file validation, timeout/tool errors.
* ``video/metadata.py``: first-video-stream FFprobe extraction.
* ``video/encoding.py``: CPU libx264 encoding and exclusive output publication.
* ``video/quality.py``: decoded timeline validation, PSNR/SSIM, optional VMAF.
* ``pipelines/ingest.py``: hashing, deterministic batch ordering, per-file ETL.
* ``pipelines/baseline.py``: three-CRF comparison, persistence and JSON reports.
* ``db/models.py``: relational records using the existing Base and session factory.

Settings are centralized in ``visionops.config.Settings``. Configure executable
names with ``VISIONOPS_FFMPEG_BINARY`` and ``VISIONOPS_FFPROBE_BINARY``; both default
to the PATH executables. ``VISIONOPS_VIDEO_TIMEOUT_SECONDS`` defaults to 300 per
subprocess. Install FFmpeg 5.1+ with libx264, PSNR and SSIM support. FFprobe must also
be available. Missing tools, failed commands, malformed metadata, and timeouts
produce explicit errors. No subprocess uses a shell.

ETL and schema
--------------

Raw file -> FFprobe metadata -> Pydantic validation -> normalized record -> SQLite.

Files are resolved to absolute paths. Metadata includes filename, duration in
seconds, width/height, average FPS (rational parsed as a float), codec, pixel
format, file size in bytes and container bitrate in bits/second. If FFprobe omits
bitrate, it is estimated as file bytes * 8 / duration and marked ``estimated``.
Container bitrate includes muxing overhead and any audio, rather than being an
elementary video-stream bitrate.

SHA-256 identifies source content, so ingestion of the same bytes at another path
reuses one video row. The first ingested path is retained; alternate locations
are not tracked. Files must remain unchanged during ingestion. Batch order is
sorted by resolved path; expected file errors are reported per item without
rolling back successful files. CLI batch commands exit nonzero if any file fails.
JSON event logs go to stderr; machine-readable CLI output goes to stdout.

The schema contains:

* ``videos``: unique source hash, first path, validated metadata, UTC ingestion time.
* ``encoding_configurations``: canonical configuration hash and JSON parameters.
* ``experiments``: run UUID, source/config foreign keys, status, UTC timestamps,
  complete FFmpeg/FFprobe version output, seed, optional code revision, error.
* ``encoding_results``: output hash/path, encoding wall time, bitrate, bytes,
  and duration; retained if encoding succeeds but quality evaluation fails.
* ``quality_metrics``: PSNR, SSIM, optional VMAF, and aligned frame count.

The existing engine factory now enables SQLite foreign keys. Table creation is
explicit in the CLI; API startup still has no schema side effects. SQLAlchemy
uses portable JSON, string, bigint, float and datetime types. SQLite stores UTC
values without preserving timezone metadata; interpret loaded timestamps as UTC.
There is no Alembic migration system yet: ``create_all`` only creates missing
tables, never upgrades an existing table. PostgreSQL requires a driver, a URL,
migrations, and verification in a future task. The MVP assumes one experiment
writer at a time; process interruption can leave a run marked ``running``.

Encoding and quality
--------------------

``encode_video(source, output, EncodingConfig(...))`` accepts H.264 CRF 0..51,
a whitelisted preset, a thread count, and optional even width and height supplied
together. Default: CRF 23, medium preset, one thread, native resolution, yuv420p.
The operation keeps only the first video stream, drops audio/subtitles and source
metadata, and retains frame timing. Requested scaling uses bicubic interpolation
and can change aspect ratio; choose dimensions deliberately. Wall time measures
the encoding subprocess, excluding metadata extraction and quality evaluation.

Encoding writes into a private temporary directory, then publishes with a
same-filesystem hard link that refuses existing destinations, including races.
Source files and existing outputs are never overwritten. The output filesystem
must support hard links. Failed temporary encodes are cleaned up.

Quality evaluation explicitly rejects mismatched dimensions, pixel formats,
durations, frame counts, or relative decoded presentation timestamps (100 us
frame tolerance; 1 ms duration tolerance). Time origins are normalized. It checks
the complete frame timeline before running FFmpeg's aggregate PSNR/SSIM filters;
repeat-last-frame behavior is disabled. This verifies timing alignment, not
semantic correspondence between unrelated videos. No automatic resizing, frame
truncation, or frame duplication is used to manufacture comparable metrics.
Resized encodes require a separately prepared, matching reference.

Lossless equality produces infinite PSNR. JSON and storage represent this as
``psnr_db=null, psnr_infinite=true`` rather than an invalid JSON Infinity value.
Metrics are never substituted after an error. VMAF is opt-in with ``--vmaf`` and
runs only if the installed FFmpeg advertises ``libvmaf``. Its model availability
must also be valid; errors are reported, not replaced with synthetic scores.
The current cloud FFmpeg has no libvmaf, so VMAF was not measured.

Preparing a small dataset
-------------------------

No data download happens automatically. All examples run from the repository root
with dependencies installed by ``cd backend && uv sync --frozen --extra dev``.
Install the tools on Debian/Ubuntu with ``sudo apt-get install ffmpeg``; macOS can
use ``brew install ffmpeg``. Docker and CI install FFmpeg explicitly.

For an offline, fixed synthetic smoke dataset::

   backend/.venv/bin/python -m visionops.cli prepare-dataset --output data/synthetic

This generates two seconds of FFmpeg ``testsrc2`` (320x240, 24 FPS, 48 frames),
encoded losslessly with libx264 CRF 0, medium preset and one thread. No third-party
footage is used. ``manifest.json`` records generation settings, tool version,
seed 42 and the content hash. Reusing a dataset directory is refused.

For a public footage example, obtain the Blender Foundation Sintel trailer from
its official source manually. The film is distributed under Creative Commons
Attribution 3.0. See https://www.sintel.org/ and
https://creativecommons.org/licenses/by/3.0/ for attribution/license terms.
Official trailer directory: https://download.blender.org/durian/trailer/ .
The following commands are explicit user actions, not part of setup or CI::

   mkdir -p data/public
   curl --fail --location --output data/public/sintel-trailer.mp4 \
     https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4
   ffmpeg -n -i data/public/sintel-trailer.mp4 -t 2 -an \
     -c:v libx264 -crf 0 -preset medium -threads 1 -pix_fmt yuv420p \
     data/public/sintel-2s.mp4
   backend/.venv/bin/python -m visionops.cli prepare-dataset \
     --output data/sintel-sample --local data/public/sintel-2s.mp4 \
     --source-url https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4 \
     --license "Sintel (c) Blender Foundation | durian.blender.org | CC BY 3.0; trimmed/re-encoded"

Review the download size before obtaining any public source; do not use large
datasets for this MVP. The preparation command accepts a local clip of at most
30 seconds and 50 MiB, copies without overwriting, and records user-supplied
source/license attribution and SHA-256. It does not independently verify a license.
The public-source download was not executed during this task. Any other local
sample can use the same interface with its actual source and license.

Baseline experiment
-------------------

From the repository root, after generating the synthetic dataset::

   VISIONOPS_DATABASE_URL=sqlite:///./data/video.db \
     backend/.venv/bin/python -m visionops.cli ingest data/synthetic/sample.mp4
   VISIONOPS_DATABASE_URL=sqlite:///./data/video.db \
     backend/.venv/bin/python -m visionops.cli baseline data/synthetic/sample.mp4 \
     --output artifacts/baseline --code-revision "$(git rev-parse HEAD)-dirty"

Use the exact clean commit SHA once the implementation is committed; ``-dirty``
explicitly indicates uncommitted code and is not a complete source snapshot.
For configuration from a local env file, supply ``--env-file .env`` before the
subcommand. Exported variables take priority. SQLite paths are relative to the
working directory, and the database parent directory must exist.

The baseline fixes CRFs to 18, 23, 28; defaults to medium, one CPU thread,
yuv420p and native dimensions. Each invocation creates a fresh run UUID with
three experiments and output files plus ``report.json`` under that run directory.
A report checkpoint is atomically replaced after each completed/failed candidate.
Experiments record configuration and tool versions; timing varies with hardware
and load. Seed 42 is recorded for provenance; libx264 has no Python random seed
input. Bitwise repeatability requires the same input bytes, FFmpeg/libx264 build,
settings, platform and threading. Reports contain measured results, never promises
of cross-machine identity. Datasets, databases and artifacts stay ignored by Git.

Tests cover validation, safe subprocess errors, metadata parsing, duplicate ETL,
foreign keys, encoding failures, alignment and metric parsing. CPU integration
tests generate local synthetic files, run all three CRFs and verify persisted
results. They skip explicitly if FFmpeg/FFprobe is missing; CI installs both.

Medical imaging, AI agents, MLflow integration, model training, dashboard
analytics and production services remain outside Phase 2.

Executed synthetic baseline
---------------------------
The local CLI experiment was executed with FFmpeg 7.1.5-0+deb13u1,
medium preset, one thread, native 320x240 yuv420p and 48 frames.
No public footage or VMAF evaluation was used.
Run ID: ``72630784-c80f-4c5a-a90a-7d184ddea31c``; source SHA-256:
``b945c0bb99efce704a46670dc9ec7106cfd2ab68667e5ae526e452d4d7c0611a``.
The source revision was recorded as ``4a20f2a-dirty`` (uncommitted Phase 2).
The ignored full report and outputs are under
``artifacts/baseline/<run-id>/``; the database is
``data/phase2-baseline.db``.

.. list-table:: Measured results (container bitrate includes muxing overhead)
   :header-rows: 1

   * - CRF
     - Bitrate (bps)
     - Bytes
     - Encoding seconds
     - PSNR (dB)
     - SSIM
   * - 18
     - 405584
     - 101396
     - 0.2801
     - 47.696540
     - 0.997666
   * - 23
     - 278808
     - 69702
     - 0.2387
     - 42.592218
     - 0.993613
   * - 28
     - 170856
     - 42714
     - 0.1697
     - 38.103287
     - 0.983461

A second CLI invocation retained one source row and three configurations,
with six total experiments/results/metrics. Output sizes and quality values
matched the first invocation; encoding time varied. This small synthetic
check demonstrates the pipeline, not generalization to real video content.
