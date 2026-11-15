# VisionOps-AI

A modular foundation for AI-powered video and medical image analysis, built as a software engineering and research portfolio project. **Phase 2 implements CPU video metadata ETL, H.264 encoding, and PSNR/SSIM evaluation.** Training, medical inference, agents, and a populated dashboard remain future work.

## Repository

```text
backend/
  pyproject.toml, uv.lock    # lightweight runtime + optional ML extras
  src/visionops/
    api/                    # FastAPI transport
    config.py               # validated environment settings
    db/                     # SQLAlchemy engine/session factories and video experiment tables
    video/                  # validated metadata, FFmpeg encoding, quality, dataset preparation
    medical/                # future medical imaging
    pipelines/              # standalone ETL and reproducible CRF baseline
    cli.py                  # dataset, ingest and baseline commands
    ml/                     # prediction/tracking contracts, training/evaluation placeholders
    assistant/              # typed tool request/result contracts
  tests/
frontend/                   # React + TypeScript + Vite
  src/
docs/                       # Sphinx documentation
.github/workflows/ci.yml
compose.yaml
PROJECT_CONTEXT.md
```

## Local development

Requirements: Python 3.11+ (CI uses 3.12), uv, Node.js 22+ and npm. Run from the repository root:

```bash
cp .env.example .env
cd backend
uv sync --frozen --extra dev
uv run --frozen --extra dev uvicorn visionops.api.main:app --reload --env-file ../.env
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Backend: port 8000, `GET /health` returns `{"status":"ok"}`; API documentation is at `/docs`. Frontend: port 5173. The frontend is a static skeleton and does not call the backend yet. Health is process liveness, not database/model readiness.

`visionops.config.Settings` centralizes configuration. Environment variables use `VISIONOPS_`; `.env` is loaded explicitly by the Uvicorn command or Compose, not implicitly by all Python entry points. SQLite URLs are relative to the process working directory. The database engine/session factories are lazy: starting the API does not create a schema or database file. The video CLI explicitly creates missing Phase 2 tables; no migration framework is implemented yet. `VISIONOPS_API_PORT` is validated metadata; set Uvicorn's `--port` explicitly to change its listen port (Compose uses 8000).

## Validation

```bash
cd backend
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev ruff format --check .
uv run --frozen --extra dev pytest
cd ..
backend/.venv/bin/sphinx-build -W -b html docs docs/_build/html
cd frontend
npm run build
```

The frontend build includes TypeScript checking. CI runs the same backend lint/tests, documentation build, and frontend build; deployment is intentionally manual and not configured.

## Docker development

From the repository root, after copying `.env.example` to `.env`:

```bash
docker compose up --build
# stop with Ctrl-C, then:
docker compose down
```

In a managed environment with an HTTPS inspection proxy, builds may need the environment CA mounted as a BuildKit secret in a temporary Dockerfile, with `PIP_CERT` / `NODE_EXTRA_CA_CERTS` pointing to that mount during dependency installation. Keep TLS verification enabled and certificates out of the repository and image layers. The standard Dockerfiles target ordinary local Docker installations.

These are development containers. Frontend runs Vite, not a production web server. Backend runs as a non-root user. Database storage is ephemeral until a durable volume and schema are designed. No source bind mounts or automatic container reload are configured.

## Research and ML roadmap

CPU and random seed **42** are defaults. Install optional extras only when a later phase needs them: `uv sync --frozen --extra dev --extra ml` for scikit-learn/MLflow, and `--extra torch` for PyTorch. PyTorch's accelerator build must be selected and verified against the target university cluster, driver, and scheduler; Phase 1 needs no CUDA or GPU. Optional ML packages are declared and locked, but not required or imported by the API.

Training and evaluation modules stay independent of FastAPI. `Predictor` supports replaceable model adapters and versioned outputs; `ExperimentTracker` provides an MLflow adapter boundary. The tool contracts prepare an assistant registry without executing arbitrary tools or implementing an agent. Dashboard charts and KPIs will consume future REST endpoints rather than access training or storage directly.

Future runs must record dataset/version and split, code revision, seed, configuration, metrics, environment, and model artifact version. Seed 42 is not itself a guarantee of deterministic GPU execution. Compare a simple baseline before adding deep models. Do not commit datasets, patient data, weights, credentials, local databases, or generated artifacts.

## Phase 2: run the video baseline

Install FFmpeg 5.1+ and FFprobe with libx264, PSNR and SSIM support (`sudo apt-get install ffmpeg` on Debian/Ubuntu, or `brew install ffmpeg` on macOS). No new Python packages, ML frameworks, or GPU are required.

From the repository root, after the backend installation above:

```bash
backend/.venv/bin/python -m visionops.cli prepare-dataset --output data/synthetic
VISIONOPS_DATABASE_URL=sqlite:///./data/video.db \
  backend/.venv/bin/python -m visionops.cli ingest data/synthetic/sample.mp4
VISIONOPS_DATABASE_URL=sqlite:///./data/video.db \
  backend/.venv/bin/python -m visionops.cli baseline data/synthetic/sample.mp4 \
  --output artifacts/baseline --code-revision "$(git rev-parse HEAD)-dirty"
```

The synthetic input is two seconds, 320x240, 24 FPS, lossless H.264. The baseline encodes **CRF 18, 23, 28**, medium preset, one CPU thread, native dimensions and yuv420p. It records source/output hashes, configuration, seed 42, full tool versions, UTC timestamps, bitrate, bytes, encoding time, PSNR and SSIM. Each invocation gets a fresh run directory with MP4s and `report.json`; successful/failed experiments are persisted in SQLite. Use a new dataset directory to prepare another sample. Use the actual clean code SHA after committing; `-dirty` marks uncommitted work, not a full source snapshot.

Metadata ingestion deduplicates by content SHA-256, even across filenames. SQLite uses the existing SQLAlchemy configuration; tables are `videos`, `encoding_configurations`, `experiments`, `encoding_results`, and `quality_metrics`. Foreign keys are enabled. `create_all` creates missing tables but does not migrate existing schemas. The MVP assumes one experiment writer. Video operations do not add API endpoints or change `/health`.

Encoding validates CRF/preset/even dimensions and refuses to overwrite inputs or existing outputs. It drops audio and measures video-container bitrate, including muxing overhead. Quality evaluation rejects mismatched dimensions, pixel formats, frame counts, durations or decoded relative frame timestamps. A resized output requires a matching reference; no automatic alignment hides incompatibility. Infinite PSNR is represented by a null value plus an explicit flag. VMAF is opt-in (`--vmaf`) and requires FFmpeg libvmaf support; it is not installed by default.

See [the pipeline documentation](docs/video_pipeline.rst) for architecture, schema, parameters, limitations and **public Sintel sample source/license instructions**. No public dataset is downloaded automatically; `prepare-dataset --local ... --source-url ... --license ...` also accepts small local clips (up to 30 seconds/50 MiB). Datasets, databases and outputs are ignored by Git. All CLI file paths and SQLite URLs are relative to the working directory. Supply `--env-file .env` before the subcommand to load local settings explicitly.

## Future work

Phase 3 is not implemented. Review the video baseline before adding read-only result APIs, real dashboard KPIs, content-aware optimization, or MLflow adapters. Medical imaging, model training and agent execution remain independent future components. This is not a clinically validated product.

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for decisions and status. MIT licensed.
