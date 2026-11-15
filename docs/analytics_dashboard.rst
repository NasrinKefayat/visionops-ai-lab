Phase 4: analytics API and dashboard
================================================================================

Scope and architecture
--------------------------------------------------------------------------------

The dashboard displays existing public/synthetic Phase 2 measurements and Phase 3
reports. It does not train, encode, change the research protocol, create datasets
or substitute demo metrics when data is missing. The existing ``GET /health``
and offline CLIs remain unchanged.

Backend responsibilities are separated:

* ``api/routes.py``: versioned HTTP routes and validated parameters.
* ``api/schemas.py``: finite Pydantic response models, explicit measured/predicted
  names, paginated envelopes and error documentation.
* ``api/dependencies.py``: request-scoped, dependency-injected SQLite sessions,
  opened with URI ``mode=ro``; no schema/database creation during GET requests.
* ``api/queries.py``: SQLAlchemy queries and allowlisted path-free projections.
* ``api/publication.py``: explicitly approved nonsensitive source identities.
* ``api/artifacts.py``: bounded JSON projections and read-only MLflow run metadata.
* ``api/services.py``: KPIs and optional digest-pinned inference on cached features.
* ``api/errors.py``: consistent sanitized errors, without paths, input values,
  driver diagnostics or raw processing exceptions.

The React frontend has Quality Analysis and Optimization views, a typed API client,
an abortable fetch hook, reusable KPI/state/pagination components, and responsive
SVG scatter/bar charts. Chart interactions support point hover, keyboard focus,
and encoding selection. No charting runtime dependency was added. Requests are
cancelled on source/filter changes so old-source responses do not replace new
ones. The Vite development proxy forwards ``/api`` to the backend, avoiding CORS
configuration or exposing server filesystem settings to the browser.

API reference
--------------------------------------------------------------------------------

All data endpoints below use the ``/api/v1`` prefix. List endpoints accept
``offset`` (0..1,000,000) and ``limit`` (1..100, default 20).

.. list-table:: Implemented endpoints
   :header-rows: 1

   * - Method and path
     - Behavior
   * - GET /videos
     - Published videos; optional dataset kind and source-name search.
   * - GET /videos/{video_id}
     - Safe source metadata, available CRFs and run IDs.
   * - GET /videos/{video_id}/encodings
     - Paginated measured encoding/configuration records; CRF/run filters.
   * - GET /videos/{video_id}/quality
     - Same measured projection for charting PSNR/SSIM/bitrate/time/ratios.
   * - GET /videos/{video_id}/comparison
     - Same source-scoped projection for configuration comparison tables.
   * - GET /experiments
     - Paginated encodings, completed by default; video/CRF/status filters.
   * - GET /experiments/{experiment_id}
     - Safe configuration, timestamps, provenance and measured outcome details.
   * - GET /monitoring
     - Aggregate KPIs, published experiment statuses and availability.
   * - GET /studies and GET /models
     - Paginated public saved-study/model metadata, newest report first.
   * - GET /studies/{study_id} and GET /models/{study_id}
     - Selected models, test metrics, importance, saved recommendations and policies.
   * - POST /models/{study_id}/recommend
     - Optional trusted-model prediction from a published video's cached features.

List envelopes include ``items``, ``total``, ``offset``, ``limit``, ``available``
and ``message``. Missing datasets return explicit empty/unavailable envelopes.
Missing/unpublished records return the same 404, so private records are not
revealed. Invalid requests return 422; unavailable data/inference returns 503
where appropriate. Errors have ``{"error": {"code": ..., "message": ...}}``;
validation errors include field locations without rejected input values.
OpenAPI is available at the existing ``/docs`` route.

No source/output paths, raw filenames, private grouping keys, arbitrary report
fields, raw errors or secrets are returned. The approved manifest source alias is
the display name. All chart comparisons are scoped to one selected source/run;
charts show the current result page and tables retain the experiment/run identity.

Metrics and presentation
--------------------------------------------------------------------------------

Measured SSIM/PSNR come directly from stored Phase 2 quality records. File size,
bitrate and encoding duration come from stored encoding results. Compression ratio
is original container bytes / output container bytes. Bitrate reduction versus
source is 100 * (1 - output bitrate / source bitrate). Audio is dropped by the
encoding pipeline, and container bitrate includes muxing overhead; these ratios
are not elementary video-only codec efficiency claims.

Failed/partial encodes retain available measurements and explicit status, never
inventing missing quality values. Infinite PSNR uses its existing flag and null
finite score; tables show infinity, finite charts omit it, and average finite
PSNR reports the excluded infinite count. Missing metrics show a dash, not zero.
Counts are zero only for a known empty published cohort; unavailable data is null.
KPIs summarize completed published experiments. Scatter axes are labeled with
units and scale to the selected data; SSIM is bounded to its valid range.

Saved Phase 3 predictions and measured outcomes have different columns. Synthetic
studies are labeled prominently. Public-video studies may contain a mixture of
public/synthetic sources; that label does not establish real-only evaluation.
The displayed study report is not re-evaluated or retrained by the API.

Policy charts and tables retain measured violation rates, abstention, coverage,
negative bitrate reductions and retrospective Oracle distance. In particular, the
existing synthetic study's ML policy used more bitrate than fixed CRF while fixed
CRF violated SSIM on some sources. The UI does not claim this was a bitrate gain.
Oracle is a measured test-only retrospective comparator, not an inference input.
Feature permutation importance is validation-only sensitivity, not causation.

Publishing data and launching locally
--------------------------------------------------------------------------------

From the repository root, install the existing dependencies::

   cd backend
   uv sync --frozen --extra dev --extra ml
   cd ..
   cd frontend
   npm ci
   cd ..

To generate actual local data, use the existing Phase 3 commands. These perform
real FFmpeg measurements and model fitting outside the API; no automatic startup
or GET request does this. The generator refuses an existing source directory::

   backend/.venv/bin/python -m visionops.ml.cli synthetic \
     --output data/content-study --count 12
   backend/.venv/bin/python -m visionops.ml.cli prepare \
     research/encoding_optimization/synthetic.yaml \
     --code-revision "$(git rev-parse HEAD)-dirty"
   backend/.venv/bin/python -m visionops.ml.cli train \
     research/encoding_optimization/synthetic.yaml \
     --code-revision "$(git rev-parse HEAD)-dirty"

Reuse existing measurements/reports if they are already present. Use a clean exact
revision after committing. No new experiment was claimed or fabricated for
Phase 4; validation used the existing local synthetic measurements and reports.

Publication is opt-in, even if a SQLite database exists. From the repository root::

   export VISIONOPS_DATABASE_URL="sqlite:///$PWD/data/content-study/video.db"
   export VISIONOPS_DASHBOARD_MANIFEST="$PWD/data/content-study/manifest.json"
   export VISIONOPS_DASHBOARD_REPORTS="$PWD/artifacts/content-study"
   export VISIONOPS_DASHBOARD_MLFLOW_DIRECTORY="$PWD/mlruns/encoding-optimization"
   backend/.venv/bin/uvicorn visionops.api.main:app --host 127.0.0.1 --port 8000

In another terminal::

   cd frontend
   npm run dev

The frontend uses port 5173. Its proxy defaults to backend port 8000. To change the
server-side proxy target, set ``VISIONOPS_API_PROXY_TARGET`` when starting Vite;
that value is not embedded in client code. Relative backend paths resolve against
the backend process working directory. A root ``.env`` may be loaded explicitly
with Uvicorn's ``--env-file .env`` when launching from the root.

For real videos, create an ignored local manifest with actual original hashes,
unique safe source aliases, ``domain: public_video``, and ``sensitive: false``.
Retain correct parent-recording groups, provenance and license. Configure a copied
Phase 3 YAML and run the existing prepare/train workflow, or publish existing
Phase 2 measurements for Quality Analysis alone. The API never downloads data or
reads the raw source paths from a client request. Dataset paths and private fields
in the manifest are ignored by the publication projection.

Restricted/medical records default to unpublished. Medical videos, DICOM, CT,
MRI and ultrasound processing remain unimplemented. Do not relabel sensitive data
as public. Publication declarations require real review; this is not automatic
PHI detection. Keep private metadata, data and models in approved ignored/external
storage. Future medical adapters need their own authorized presentation layer.

Safe local artifacts and optional inference
--------------------------------------------------------------------------------

Saved reports are read only from administrator-configured directories with UUID
child names. Symlinks and malformed/private reports are excluded. JSON reads are
bounded to 2 MB and directory scans to 1,000 entries in this MVP. Only known model,
metric, importance and policy fields are projected. No endpoint accepts artifact
paths, executes arbitrary commands, serves model bytes, or permits filesystem
browsing.

MLflow status is optional: the API reads a known run's bounded ``meta.yaml`` under
a configured local file store and returns only its ID/status. It never starts an
MLflow server/client, creates a store, or returns artifact URIs/params/tags. Missing
or incompatible stores yield ``unavailable`` and leave research reports usable.

Inference is disabled by default. GETs never load joblib models. An administrator
can pin a trusted locally produced model's exact SHA-256 in
``VISIONOPS_DASHBOARD_TRUSTED_MODELS``, a JSON dictionary keyed by study UUID.
Joblib is executable deserialization: a digest establishes identity, not trust.
Only pin artifacts whose code/provenance you trust, never uploaded/untrusted files.
For example, obtain the digest locally::

   sha256sum artifacts/content-study/STUDY_UUID/model.joblib

Then set the backend environment before startup::

   export VISIONOPS_DASHBOARD_TRUSTED_MODELS='{"STUDY_UUID":"ACTUAL_SHA256"}'

The ML extra must be installed for inference; the default Docker API image keeps
it optional. POST bodies contain only ``video_id``, ``minimum_ssim`` and optional
candidate CRFs, not file paths. Inference uses bounded cached original-source
features from that same study snapshot. Model bytes are checked before loading
from the verified in-memory payload; provenance, privacy, feature schema and
candidate grid are checked. No FFprobe, encoding, training, feature decoding or
tracking is triggered. Missing features/models/dependencies are explicit errors.
No predicted-feasible candidate returns null values and a no-candidate status,
without silently relaxing the constraint. Fresh predictions have no measured
quality assigned to them.

Docker and validation
--------------------------------------------------------------------------------

Default Compose starts an empty dashboard until data is explicitly published.
After preparing the example data, optional read-only mounts are provided::

   docker compose -f compose.yaml -f compose.analytics.yaml config --quiet
   docker compose -f compose.yaml -f compose.analytics.yaml up --build

The frontend proxy points to the backend Compose service. These remain development
containers, with no authentication or production hardening. Do not expose the
local public-data service to untrusted networks. POST inference has no queue/rate
limit and is intended for small administrator-trusted models only.

Checks used for this phase include all existing Phase 2/3 tests plus API tests,
Vitest/React Testing Library tests, Ruff, TypeScript/Vite, Sphinx and Compose
configuration validation. Chromium checks exercise actual API data at desktop
and mobile sizes. No new research methodology, GPU dependency, medical pipeline,
external tracking server or Phase 5 feature was added.
