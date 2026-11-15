# VisionOps-AI project context

## Goal and scope

A portfolio platform demonstrating modular backend/frontend engineering, applied computer vision, reproducible research, and future MLOps. Target audiences include software/AI internships and medical imaging PhD applications. Phase 1 provides infrastructure, Phase 2 a standalone CPU video baseline, and Phase 3 an offline content-aware ML study; keep the initial MVP small enough for approximately three days of work. This is a scope target, not a claim that future features are complete.

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

Not implemented: medical ingestion/models, deep learning, populated dashboards/charts, external MLflow server/UI, model registry, AI provider/tool execution, authentication, migrations, deployment, clinical validation, or GPU jobs.

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

## Next steps after review

Phase 4 has not started. Collect licensed real sources with independent recording groups, freeze a larger protocol and validate acquisition/domain shift before performance claims. Later work can add result APIs/dashboard integration, agents, MLflow registry, uncertainty/fallback policies and dedicated medical/GPU task adapters.

The user manages Git manually. Phase 3 changes remain unstaged and uncommitted on the existing `feature/content-aware-ml` branch. No branch switches/creation, commits, pushes, merges, rebases, resets or PRs are part of implementation.

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
