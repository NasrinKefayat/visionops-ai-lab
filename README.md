# VisionOps AI Lab

AI-Powered Video Quality Analysis & Content-Aware Encoding Optimization

A modular foundation for AI-powered video and medical image analysis, built as a software engineering and research portfolio project. **Phase 4 adds a read-only FastAPI analytics API and interactive React quality/optimization dashboard over existing measurements and research reports.** Phase 3 supplies CPU content features, grouped SSIM/bitrate regression, constrained CRF prediction and local MLflow tracking. Phase 2 video ETL/encoding/quality remains available. Medical inference, agents, deep models, and a populated dashboard remain future work.

## Repository

```text
backend/
  pyproject.toml, uv.lock    # lightweight runtime + optional ML extras
  src/visionops/
    api/                    # versioned routers, schemas, read-only queries/services, publication
    config.py               # validated environment settings
    db/                     # SQLAlchemy engine/session factories and video experiment tables
    video/                  # validated metadata, FFmpeg encoding, quality, dataset preparation
    medical/                # future medical imaging
    pipelines/              # standalone ETL and reproducible CRF baseline
    cli.py                  # dataset, ingest and baseline commands
    ml/                     # data, grouped training, inference, policies, tracking and study CLI
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

Backend: port 8000, `GET /health` returns `{"status":"ok"}`; API documentation is at `/docs`. Frontend: port 5173, with a same-origin Vite proxy to backend port 8000. The dashboard uses the versioned `/api/v1` endpoints; no data is shown until public/synthetic sources are explicitly published. Health is process liveness, not database/model readiness.

`visionops.config.Settings` centralizes configuration. Environment variables use `VISIONOPS_`; `.env` is loaded explicitly by the Uvicorn command or Compose, not implicitly by all Python entry points. SQLite URLs are relative to the process working directory. The database engine/session factories are lazy: starting the API does not create a schema or database file. The video CLI explicitly creates missing Phase 2 tables; no migration framework is implemented yet. `VISIONOPS_API_PORT` is validated metadata; set Uvicorn's `--port` explicitly to change its listen port (Compose uses 8000).

## Validation

```bash
cd backend
uv sync --frozen --extra dev --extra ml
uv run --frozen --extra dev --extra ml ruff check .
uv run --frozen --extra dev --extra ml ruff format --check .
uv run --frozen --extra dev --extra ml pytest
cd ..
backend/.venv/bin/sphinx-build -W -b html docs docs/_build/html
cd frontend
npm test
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

CPU and random seed **42** are defaults. Install optional extras for the Phase 3 offline workflow: `uv sync --frozen --extra dev --extra ml` for scikit-learn, OpenCV/NumPy, YAML and local MLflow client tracking, and `--extra torch` for PyTorch. PyTorch's accelerator build must be selected and verified against the target university cluster, driver, and scheduler; Phase 1 needs no CUDA or GPU. Optional ML packages are declared and locked, but not required or imported by the API.

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

## Phase 3: content-aware ML study

The offline workflow keeps dataset loading, features, grouped partitions, training, inference, policy evaluation and tracking separate. No API or frontend behavior changes. FFmpeg/FFprobe remain required for measurement; install the optional ML extra for research (no GPU/PyTorch/medical dependencies):

```bash
(cd backend && uv sync --frozen --extra dev --extra ml)
backend/.venv/bin/python -m visionops.ml.cli synthetic --output data/content-study --count 12
backend/.venv/bin/python -m visionops.ml.cli prepare research/encoding_optimization/synthetic.yaml \
  --code-revision "$(git rev-parse HEAD)-dirty"
backend/.venv/bin/python -m visionops.ml.cli train research/encoding_optimization/synthetic.yaml \
  --code-revision "$(git rev-parse HEAD)-dirty"
```

The generator refuses existing directories; reuse `train` to create a fresh run from the measured SQLite data. Use a clean revision once changes are committed. Sources are independently generated tiny synthetic scenes, not evidence of real-video generalization. Training compares DummyRegressor and RandomForestRegressor for SSIM and bitrate. Selection and permutation importance use validation only; frozen models are evaluated on held-out source groups. Inputs contain original-source features plus CRF, never encoded-output features or measured targets.

Reports, dataset snapshots, trusted-load-only model bundles and interpretation JSON live in ignored `artifacts/content-study/<uuid>/`. The YAML's local MLflow file store is `mlruns/encoding-optimization`; it logs allowed aggregate metadata/metrics and public model artifacts without an external server. The lightweight `mlflow-skinny` client is sufficient; an MLflow server/UI is not installed. `recommend` emits **predicted** SSIM/bitrate or explicitly abstains; measurements are required to verify a recommendation. R² is omitted for fewer than five independent evaluation groups or constant targets.

See [research methodology and feature definitions](docs/content_aware_ml.rst), [study configuration](research/encoding_optimization/synthetic.yaml), and [medical extension contract](docs/medical_extensibility.rst). To add real public videos, provide an ignored manifest with actual content hashes and parent-recording groups; no datasets are downloaded automatically. Use substantially more independent sources before making performance claims.

Medical source identities and patient/study group keys remain local and are excluded from features/tracking. Restricted data is rejected by the public-video adapters and MLflow tracking guard. Domain-specific loaders, features, estimators and evaluators can reuse the generic contracts; DICOM, CT, MRI, ultrasound and segmentation pipelines are documented future extensions, not implemented features. Keep all medical data/metadata and derived artifacts in approved external or ignored storage.

## Phase 4: launch the analytics dashboard

The dashboard has **Quality Analysis** (source metadata, measured quality, compression, timing and CRF charts) and **Optimization** (model metadata, saved predictions, policy comparisons and feature importance). Missing data is explicit; metrics are never filled with demo values or zero. Synthetic studies and predicted versus measured results are labeled separately.

After producing actual Phase 2/3 data using the commands above, start from the repository root:

```bash
export VISIONOPS_DATABASE_URL="sqlite:///$PWD/data/content-study/video.db"
export VISIONOPS_DASHBOARD_MANIFEST="$PWD/data/content-study/manifest.json"
export VISIONOPS_DASHBOARD_REPORTS="$PWD/artifacts/content-study"
export VISIONOPS_DASHBOARD_MLFLOW_DIRECTORY="$PWD/mlruns/encoding-optimization"
backend/.venv/bin/uvicorn visionops.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal, run `cd frontend && npm ci && npm run dev`. Without publication settings, the dashboard intentionally shows empty/unavailable states. For real footage, use an ignored local manifest with licensed originals, stable hashes and parent-recording groups, `domain: public_video`, and `sensitive: false`; run the existing prepare/train workflow against a copied YAML. No downloads, encoding or training occur through GET requests.

The API exposes paginated `/api/v1/videos`, source-scoped quality/encodings/comparison, `/experiments`, `/monitoring`, `/studies`, and `/models`. Local filesystem paths, raw filenames and private metadata are excluded. SQLite access is read-only. Model loading is off by default; optional POST inference requires an administrator-pinned trusted model digest, installed ML extras and existing cached source features. No paths or arbitrary commands are accepted from clients.

See [API, setup and safety details](docs/analytics_dashboard.rst) for endpoint parameters and optional inference configuration. `docker compose -f compose.yaml -f compose.analytics.yaml up --build` can publish the prepared example data through read-only mounts; default Compose needs no dataset. This local MVP has no authentication and is not a production deployment.

## Future work

Phase 5 has not started. Real-data validation, agents, production serving/authentication, medical pipelines, deep models and clinical validation remain future work.

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for decisions and status. MIT licensed.
