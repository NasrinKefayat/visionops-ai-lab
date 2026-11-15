# VisionOps-AI project context

## Goal and scope

A portfolio platform demonstrating modular backend/frontend engineering, applied computer vision, reproducible research, and future MLOps. Target audiences include software/AI internships and medical imaging PhD applications. Phase 1 provides infrastructure and Phase 2 provides a standalone CPU video baseline; keep the initial MVP small enough for approximately three days of work. This is a scope target, not a claim that future features are complete.

## Architecture and decisions

- Python `src` package in `backend`, React/TypeScript in `frontend`, Sphinx in `docs`.
- FastAPI is the transport layer. Training/evaluation stay in independent ML modules; no model loading, training, or schema creation at API startup.
- Pydantic settings use `VISIONOPS_` variables and CPU/seed 42 defaults. Scripts must load the root `.env` explicitly when needed.
- SQLAlchemy engine/session factories support SQLite without import-time side effects. Phase 2 adds explicit CLI-created metadata/experiment tables and content hashes; migrations and durable container volumes remain deferred.
- Framework-independent prediction and experiment tracker protocols allow replacing models and adding an MLflow adapter. Model outputs carry an explicit version.
- Assistant tool request/result schemas and a protocol are boundaries only. Registry, argument-specific validation, authorization, providers, and agent loop remain future work.
- MLflow, scikit-learn, and PyTorch are optional extras. No large ML frameworks are needed for Phase 1; cluster-specific CUDA installation and job orchestration are deferred.
- uv and npm lockfiles support repeatable installs. CI validates lint, tests, docs, and frontend build; no automatic deployment is configured.
- Docker Compose is for development. Processes and ephemeral databases are not production deployment or persistence solutions.

## Current status

Implemented: Phase 1 `/health`, central settings, lazy SQLite infrastructure, Ruff, React skeleton, Docker/Compose, CI, Sphinx, typed model/tracking/tool contracts; Phase 2 FFprobe metadata, content-addressed batch ETL, H.264 encoding, aligned PSNR/SSIM, optional capability-gated VMAF, SQLite experiment records, synthetic/local dataset preparation and a three-CRF baseline CLI.

Not implemented: medical ingestion, ML training/evaluation, populated dashboards/charts, MLflow adapter/server, model registry, AI provider/tool execution, authentication, migrations, deployment, clinical validation, or GPU jobs.

## Reproducibility and MLOps plan

Use seed 42 and document deterministic settings and hardware limitations. Each future experiment should capture dataset identity and split, code revision, dependency environment, full configuration, baseline comparison, metrics, and model artifact version. Keep datasets and weights outside Git. Use a future MLflow adapter for run tracking and registry integration, with separate promotion and evaluation gates. Do not equate configuration defaults with executed experiments.

## Phase 2 decisions and workflow

- Reuse the existing settings, Base, engine/session factories. Explicit `initialize_database` creates missing tables; API startup is unchanged. SQLite foreign keys are enabled per connection. No schema migration or PostgreSQL verification is claimed.
- `video/` owns file/tool/metadata/encoding/quality boundaries. `pipelines/` orchestrates content-addressed ingestion and experiments independently of FastAPI. CLI is `python -m visionops.cli`.
- Source SHA-256 prevents metadata duplicates across paths; the first path is retained. Batch ordering is sorted, per-file transactions isolate expected file failures, and logs are JSON events.
- Default encoding is libx264, medium, one CPU thread, yuv420p; the baseline fixes CRFs 18/23/28 and native dimensions. Audio is dropped. Container bitrate includes overhead. Optional explicit scaling is not evaluated against an incompatible native reference.
- PSNR/SSIM use FFmpeg only after full decoded frame timeline/dimension/pixel-format/duration checks. Infinite PSNR has an explicit flag. VMAF is opt-in and absent on this cloud FFmpeg.
- Tables: videos, encoding_configurations, experiments, encoding_results, quality_metrics. Generic SQLAlchemy types and explicit keys allow a later PostgreSQL migration. The MVP assumes one experiment writer. Interrupted experiments can remain `running`; recovery and schema migrations are future work.
- Record full FFmpeg/FFprobe versions, hashes, code revision supplied by the caller, seed 42, configuration, timestamps and measured results. Timing varies with hardware/load. Dirty revision labels do not snapshot uncommitted source.
- Dataset preparation generates a fixed synthetic sample or copies a small attributed local sample. Public Blender Sintel instructions and CC BY 3.0 attribution are in `docs/video_pipeline.rst`; no public download is automatic or claimed executed.
- No new Python dependency or GPU is needed. Docker and CI install the FFmpeg OS package. Settings include executable names and per-command timeouts.

## Next steps after review

Phase 3 has not started. Review the standalone baseline and data provenance first. Future work can add read-only experiment/result APIs, dashboard KPIs/charts, content-aware optimization and MLflow adapters. Keep ML training, medical imaging and agents separate until their own interfaces and tests are defined.

The user manages all commits, pushes, merges, rebases and pull requests. Phase 2 implementation stays uncommitted on the existing `feature/video-data-pipeline` branch; no branch operations are authorized as part of implementation.

## Phase 1 local validation

- Five Pytest tests passed: health, environment settings, invalid device, SQLite session, and prediction validation. One upstream Starlette/httpx TestClient deprecation warning remains.
- Ruff lint and formatting checks passed; Sphinx HTML built with warnings treated as errors.
- Frozen uv installation and `npm ci` completed; TypeScript and Vite production build passed.
- Native Uvicorn and Vite HTTP requests returned the expected health/HTML/transformed TSX content.
- Compose configuration validated. The ordinary container build hit the cloud proxy CA trust error. Both images then built with temporary Dockerfile copies using BuildKit CA secrets, without disabling TLS verification or embedding the CA. Compose started those images; backend health and frontend HTML/TSX requests passed. Validation containers were removed afterward.
- No GPU/ML experiment, browser automation, remote GitHub Actions run, or production deployment was performed. Docker Python installation currently resolves the declared version ranges; frozen uv installs are used for local development and CI.

## Phase 2 local validation

- Frozen `uv sync --frozen --extra dev` succeeded without new Python dependencies or ML frameworks.
- Final suite: 55 passed, no skips, one upstream Starlette/httpx TestClient deprecation warning. Coverage includes Phase 1 smoke tests, metadata/validation, tool errors/timeouts, content duplicate handling, SQLite foreign keys/persistence, safe encoding failures/resizing, alignment, metric parsing, optional VMAF gating, CLI exits and real three-CRF integration.
- Ruff lint and format checks passed. Sphinx HTML built with warnings treated as errors. Existing frontend TypeScript/Vite build and Compose configuration validation passed.
- Two native CLI baseline runs completed all CRFs with FFmpeg 7.1.5-0+deb13u1. Same file sizes and PSNR/SSIM values, differing wall times. Database: one video, three configurations, six experiments/encoding results/quality results. Exact measured values and run provenance are in the Sphinx video pipeline page and ignored reports.
- Final backend Docker image built with the FFmpeg OS package. Cloud HTTPS trust was supplied only through a temporary Dockerfile's BuildKit CA secret for pip, without disabling TLS or adding certificates to repository/image layers. A disposable non-root container completed three CPU CRF runs with metrics and persistence. Container `/health` returned HTTP 200 and expected JSON; validation containers were removed.
- VMAF is unavailable in the installed FFmpeg and was not measured. Public sample download, GPU execution, PostgreSQL, remote GitHub Actions, MLflow and production deployment were not run or claimed. Synthetic video is a pipeline sanity check, not a real-content benchmark.
- Phase 2 files are left unstaged and uncommitted for user review on `feature/video-data-pipeline`; no branches were created/switched and no commits, pushes, merges, rebases, resets, cherry-picks or PRs were performed.
