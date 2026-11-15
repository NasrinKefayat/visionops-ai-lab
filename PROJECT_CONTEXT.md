# VisionOps-AI project context

## Goal and scope

A portfolio platform demonstrating modular backend/frontend engineering, applied computer vision, reproducible research, and future MLOps. Target audiences include software/AI internships and medical imaging PhD applications. Phase 1 provides infrastructure, Phase 2 a standalone CPU video baseline, and Phase 3 an offline content-aware ML study, and Phase 4 a read-only analytics API/dashboard; keep the initial MVP small enough for approximately three days of work. This is a scope target, not a claim that future features are complete.

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

Implemented: Phase 1 `/health`, central settings, lazy SQLite infrastructure, Ruff, React skeleton, Docker/Compose, CI, Sphinx, typed model/tracking/tool contracts; Phase 2 FFprobe metadata, content-addressed batch ETL, H.264 encoding, aligned PSNR/SSIM, optional capability-gated VMAF, SQLite experiment records, synthetic/local dataset preparation and a three-CRF baseline CLI; Phase 3 original-source OpenCV/NumPy features, grouped regression datasets, Dummy/RandomForest comparison, constrained CRF prediction, measured policy/Oracle evaluation, validation permutation importance, saved bundles and allowlisted local MLflow tracking.

Not implemented: medical ingestion/models, deep learning, external MLflow server/UI, model registry, AI provider/tool execution, authentication, migrations, deployment, clinical validation, or GPU jobs.

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

## Phase 3 architecture and decisions

- Dataset adapters, source feature extractors, grouped tables, training, inference, policy comparison and tracking are separate offline modules. The existing Phase 2 database/model/session and video encoding/quality implementations are reused unchanged.
- `SourceSample` stores source identity/content hash, a private path, domain and sensitivity, plus optional opaque patient/study groups. Connected-component grouping unions hash/patient/study links before splitting, so aliases and all CRF variants remain together. Missing privacy declarations default to restricted.
- The Phase 2 adapter is explicitly public/synthetic video only, verifies original hashes, selects latest completed compatible medium/native/one-thread measurements per CRF, and exports a frozen local snapshot. Medical/restricted sources are rejected before Phase 2's existing path logs. Domain adapters/features can be supplied through lightweight protocols.
- Features: deterministic grayscale Sobel complexity, frame difference variation/magnitude, Canny density, histogram entropy and brightness statistics, plus CRF. No encoded frames or measured quality/bitrate/identity/group fields enter model inputs. One-frame inputs have zero temporal evidence. Resolution/FPS effects are not input covariates; the synthetic demo is a fixed acquisition regime, not a general multi-resolution model.
- Seed 42 grouped 60/20/20 protocol uses ceiling validation/test counts. At least five independent groups are required, with substantially more recommended. Fit Dummy and RandomForest on train only; choose by validation MAE, no refitting. Validation-only permutation importance is descriptive, not causal. MAE/RMSE on frozen held-out test; R² is withheld below five independent evaluation groups or for constant targets.
- CRF selection uses predictions only and abstains when no predicted-feasible candidate exists. Measured test-only Oracle is strictly retrospective. Policy reports retain abstention/violation denominators, negative bitrate reductions and distance from feasible Oracle. Predictions are not measurement guarantees.
- YAML centralizes study paths, features, forest parameters, candidate grid and threshold. Models record feature schema/settings/version, library versions, seed, code revision, selected estimators and dataset digest. Source paths and grouping keys never go to MLflow. Load joblib files only from trusted sources.
- ML extra now installs CPU scikit-learn, NumPy, OpenCV headless, PyYAML, joblib and mlflow-skinny. Local MLflow file tracking needs no server or API key. API runtime stays lightweight; torch/CUDA remain optional and uninstalled.
- MLflow refuses restricted/medical data, including trained artifacts, before client creation. Public tracking allowlists aggregate metadata/metrics and model/interpretation artifacts; manifests, source identities, per-source data and local source paths are excluded.
- Generic training accepts alternative estimators, metrics, selection criteria and interpretation scoring. Future scalar medical research can reuse those contracts with a dedicated private loader/extractor. DICOM, CT, MRI, ultrasound, tensor segmentation/Dice and GPU task adapters are documented, not implemented. Video PSNR/SSIM stay separate from future medical task metrics.
- Medical data/metadata/derived artifacts belong in approved external or ignored private storage. Git/Docker ignore common imaging/model extensions as additional safeguards, not complete PHI detection. No medical dataset was obtained and no medical dependency was added.

## Phase 4 architecture and decisions

- Versioned `/api/v1` routers, Pydantic projections, SQLAlchemy queries, service aggregation, dependency-injected read-only SQLite access and sanitized errors preserve the existing `/health`, CLIs and research methodology.
- Public-data publication is opt-in: only manifest entries with explicitly false sensitivity and public/synthetic video domains authorize DB rows by source content hash. Source aliases are displayed; raw filenames, file paths, patient/study keys, raw errors and unapproved report fields are excluded. No medical presentation pipeline is implemented.
- Existing saved Phase 3 reports are projected read-only from configured UUID directories. Malformed/private/symlink reports are excluded; no GET deserializes models, encodes, trains, creates a database/schema or starts MLflow. Local MLflow metadata is read in a bounded path-free ID/status projection without a tracking client/server.
- Optional POST inference requires a server-configured exact SHA-256 of a trusted local model. Verified in-memory bytes, privacy/provenance and original-source cached features are checked; client paths are forbidden. No subprocess, decoding, training or tracking occurs. A digest establishes identity, not the safety of untrusted pickle code. Default inference stays disabled.
- React Quality Analysis scopes measured charts/tables to a selected source/run/page with CRF/source/domain filters and encoding detail selection. Optimization labels predictions separately from measurements and exposes selected models, saved recommendations, validation importance, measured fixed/ML/Oracle summaries, coverage, violations and negative reductions. No new research results are fabricated.
- Reusable KPI/state/pagination components retain null metrics as missing, explicit infinity and failure states. Abortable requests prevent stale-source data races. Responsive SVG charts avoid a runtime chart dependency; Vitest/Testing Library cover practical frontend flows. Vite proxies API requests locally and to the backend Compose service.
- `compose.analytics.yaml` provides optional read-only mounts for already prepared example data. The API image keeps ML extras optional; host inference uses the documented ML environment. This is a public-data local MVP, not authenticated production serving.
- JSON/directory/model reads are bounded for this small study. Large-scale indexing, arbitrary artifact browsing, user uploads, background workers, authorization, deployment and medical processing remain out of scope.

## Next steps after review

Phase 5 has not started. Validate on larger licensed real cohorts without changing held-out methodology, then consider authorization, indexed artifact metadata, background operations, real deployment, agents and dedicated medical interfaces in separate work.

The user manages Git manually. Phase 4 changes remain unstaged and uncommitted on the existing `feature/analytics-dashboard` branch. No branch creation/switches, commits, pushes, merges, rebases, resets or PRs are part of implementation.

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

## Phase 3 executed study and validation

- Dataset actually used: 12 independently generated synthetic 160x120, 12 FPS, one-second scenes; 36 completed Phase 2 CRF measurements. No public/medical download. Group split: 6 train / 3 validation / 3 test sources (18/9/9 observations), seed 42.
- Models actually fitted: DummyRegressor and 64-tree/max-depth-8 RandomForest for each of SSIM and bitrate; validation selected RandomForest for both. Selected models were not refitted on validation/test.
- Test SSIM MAE 0.00819742, RMSE 0.01333903; bitrate MAE 34542.65 bps, RMSE 40982.24 bps. R² withheld because only three independent test groups exist.
- At threshold .99: fixed CRF 23 violated measured SSIM on 2/3 test sources; content-aware and Oracle 0/3. Content-aware bitrate reduction versus fixed was -36.54% (an increase), Oracle -24.64%; content-aware absolute distance from feasible Oracle 11.91%. No bitrate gain/generalization is claimed.
- MLflow local run `00da791593564a9cbee391ee9fdaaa7a` verified FINISHED, with public model/interpretation artifacts and aggregate metadata/metrics. Final report/model/snapshot: `artifacts/content-study/5f7408dc-e150-4690-830a-b3705a7b284c/`. Tracking store: `mlruns/encoding-optimization`. Early local layout attempts failed; MLflow-managed artifact locations fixed the issue and a repeated-store regression test passes.
- Source provenance is `340ea5f-dirty`; outputs are not a clean committed source snapshot. Full settings, versions, measurement IDs, source hashes, dataset snapshot and split labels are preserved locally. Synthetic results are pipeline checks, not generalization/clinical evidence.
- Frozen ML/dev install, Ruff lint/format, Sphinx with warnings treated as errors, frontend TypeScript/Vite build and existing Phase 2 CPU baseline tests were executed. Final Pytest: 84 passed, no skips, three upstream deprecation warnings (Starlette/httpx and MLflow/Pydantic). Ruff and format checks passed for 46 Python files; all existing Phase 2 smoke/integration tests remain passing.
- No GPU, PyTorch, medical reader, external MLflow service, remote CI run or Phase 4 feature was executed/implemented. Files remain uncommitted for manual Git review.

## Phase 4 local validation

- Full Pytest suite: 114 passed, no skips, three upstream deprecation warnings (Starlette/httpx and MLflow/Pydantic). The existing Phase 2/3 tests, including real CPU encodes and grouped learning, remain passing. New API tests cover responses, validation, pagination, publication/privacy, SQL integration/readonly enforcement, finite/infinite/missing metrics, model-unavailable/corrupt/digest mismatch/abstention behavior, MLflow status, sanitized errors and API startup without optional ML imports.
- Frontend: clean npm install, seven Vitest/React Testing Library tests passed, TypeScript/Vite build passed. Tests cover empty/loading/error/retry states, actual API-contract rendering, filtering, encoding detail, prediction labels, inference abstention and keyboard chart interaction.
- Ruff lint and format checks passed for 55 Python files. Sphinx HTML built with warnings treated as errors. Default Compose and the optional analytics read-only overlay both validated; Docker images were not rebuilt for Phase 4.
- Chromium desktop (1440px) and mobile (390px) checks used actual backend data: 12 existing synthetic source videos, 36 completed encodes and three completed saved reports. Source/CRF selection, encoding inspection, optimization navigation/labels and responsive layout passed, with no page errors or horizontal page overflow. Browser screenshots are local validation artifacts outside Git.
- Existing trusted RandomForest bundle `5f7408dc-e150-4690-830a-b3705a7b284c` was tested through optional POST inference with an in-memory digest allowlist: CRF 18, predicted SSIM 0.9930176718750008 and bitrate 124755.25 bps for synthetic-00 at threshold .99; threshold 1.0 returned explicit abstention. No measured metric was attached to the fresh prediction, and the default running configuration kept inference disabled.
- No new research result, source dataset, training protocol, medical pipeline, GPU dependency, external MLflow service or Phase 5 feature was introduced. Validation reused ignored existing synthetic data/reports and temporary test fixtures. Changes remain unstaged/uncommitted for user review.
