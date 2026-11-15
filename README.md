# VisionOps-AI

A modular foundation for AI-powered video and medical image analysis, built as a software engineering and research portfolio project. **Phase 1 is a scaffold:** no analysis, training, medical inference, agent loop, or populated dashboard is implemented.

## Repository

```text
backend/
  pyproject.toml, uv.lock    # lightweight runtime + optional ML extras
  src/visionops/
    api/                    # FastAPI transport
    config.py               # validated environment settings
    db/                     # SQLAlchemy engine/session factories
    video/, medical/        # domain placeholders
    pipelines/              # ETL placeholder
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

`visionops.config.Settings` centralizes configuration. Environment variables use `VISIONOPS_`; `.env` is loaded explicitly by the Uvicorn command or Compose, not implicitly by all Python entry points. SQLite URLs are relative to the process working directory. The database engine/session factories are lazy: starting the API does not create a schema or database file. No migrations or application tables exist yet. `VISIONOPS_API_PORT` is validated metadata; set Uvicorn's `--port` explicitly to change its listen port (Compose uses 8000).

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

## Starting Phase 2

Review Phase 1 and manage branches/commits manually. Then define a small public or synthetic dataset, a validated ETL schema and SQLite metadata tables, and one CPU baseline with a fixed split and seed 42. Add tests for ingestion and evaluation before connecting API endpoints and dashboard KPIs. Implement the MLflow adapter and artifact/version metadata when real experiment tracking is introduced. Keep video and medical tasks independently scoped; this scaffold is not a clinically validated product.

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for decisions and status. MIT licensed.
