# VisionOps-AI project context

## Goal and scope

A portfolio platform demonstrating modular backend/frontend engineering, applied computer vision, reproducible research, and future MLOps. Target audiences include software/AI internships and medical imaging PhD applications. Phase 1 provides infrastructure only; keep the initial MVP small enough for approximately three days of work. This is a scope target, not a claim that future features are complete.

## Architecture and decisions

- Python `src` package in `backend`, React/TypeScript in `frontend`, Sphinx in `docs`.
- FastAPI is the transport layer. Training/evaluation stay in independent ML modules; no model loading, training, or schema creation at API startup.
- Pydantic settings use `VISIONOPS_` variables and CPU/seed 42 defaults. Scripts must load the root `.env` explicitly when needed.
- SQLAlchemy engine/session factories support SQLite without import-time side effects. Tables, migrations, data lineage, and durable container volumes are deferred.
- Framework-independent prediction and experiment tracker protocols allow replacing models and adding an MLflow adapter. Model outputs carry an explicit version.
- Assistant tool request/result schemas and a protocol are boundaries only. Registry, argument-specific validation, authorization, providers, and agent loop remain future work.
- MLflow, scikit-learn, and PyTorch are optional extras. No large ML frameworks are needed for Phase 1; cluster-specific CUDA installation and job orchestration are deferred.
- uv and npm lockfiles support repeatable installs. CI validates lint, tests, docs, and frontend build; no automatic deployment is configured.
- Docker Compose is for development. Processes and ephemeral databases are not production deployment or persistence solutions.

## Current status

Implemented: `/health`, central settings, lazy SQLite infrastructure, smoke/config/database/validation tests, Ruff, React skeleton, Docker/Compose, CI, Sphinx, typed model/tracking/tool contracts, and domain placeholders.

Not implemented: ingestion, preprocessing, model training/evaluation, populated dashboards/charts, MLflow adapter/server, model registry, AI provider/tool execution, authentication, migrations, deployment, clinical validation, or GPU jobs.

## Reproducibility and MLOps plan

Use seed 42 and document deterministic settings and hardware limitations. Each future experiment should capture dataset identity and split, code revision, dependency environment, full configuration, baseline comparison, metrics, and model artifact version. Keep datasets and weights outside Git. Use a future MLflow adapter for run tracking and registry integration, with separate promotion and evaluation gates. Do not equate configuration defaults with executed experiments.

## Next phase

1. Select a public/synthetic representative dataset and define acceptance criteria.
2. Add Pydantic ingestion records, SQLite metadata tables/migrations, and an idempotent ETL path.
3. Implement one CPU baseline and held-out evaluation independently of the API.
4. Record experiments through an MLflow adapter; define artifact/version and split metadata.
5. Expose read-only run/results endpoints, then add real dashboard KPIs and charts.
6. Add agents and university GPU jobs only after tested data/model boundaries exist.

The user manages all branch creation, commits, pushes, PRs, and merges. Phase 1 changes are intentionally left uncommitted on the existing temporary branch. See the implementation handoff for actual validation outcomes; no remote CI or future-task restoration is implied by local checks.

## Phase 1 local validation

- Five Pytest tests passed: health, environment settings, invalid device, SQLite session, and prediction validation. One upstream Starlette/httpx TestClient deprecation warning remains.
- Ruff lint and formatting checks passed; Sphinx HTML built with warnings treated as errors.
- Frozen uv installation and `npm ci` completed; TypeScript and Vite production build passed.
- Native Uvicorn and Vite HTTP requests returned the expected health/HTML/transformed TSX content.
- Compose configuration validated. The ordinary container build hit the cloud proxy CA trust error. Both images then built with temporary Dockerfile copies using BuildKit CA secrets, without disabling TLS verification or embedding the CA. Compose started those images; backend health and frontend HTML/TSX requests passed. Validation containers were removed afterward.
- No GPU/ML experiment, browser automation, remote GitHub Actions run, or production deployment was performed. Docker Python installation currently resolves the declared version ranges; frozen uv installs are used for local development and CI.
