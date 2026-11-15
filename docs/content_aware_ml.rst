Phase 3: content-aware encoding research
================================================================================

Research question and scope
--------------------------------------------------------------------------------

Can source-content features predict SSIM/bitrate well enough to choose the
lowest-bitrate measured-grid CRF subject to a minimum predicted SSIM?

This is an offline CPU study, not an API, clinical system, deep model, agent or
production optimizer. Phase 2 video metadata, encoding and quality code is reused
without changing its behavior. There are no new API endpoints or frontend screens.

Architecture
--------------------------------------------------------------------------------

* ``ml/data.py``: source identity, dataset/feature interfaces, tabular observations,
  feature assembly and connected-component group splits. No file decoding here.
* ``ml/video_dataset.py``: adapter for completed Phase 2 measurements. Uses source
  hashes, not encoded outputs, to locate original features and preserve identity.
* ``video/features.py``: deterministic OpenCV/NumPy source feature extraction.
* ``ml/training.py``: train-only fitting and validation-only model selection.
* ``ml/inference.py``: trusted artifact loading and prediction; no training import.
* ``ml/optimization.py``: prediction-only constrained CRF selection, no Oracle data.
* ``ml/evaluation.py`` and ``comparison.py``: regression metrics and retrospective
  measured test-set policy comparison, separate from FFmpeg's quality metrics.
* ``ml/tracking.py``: allowlisted local MLflow tracking, blocked for restricted data.
* ``ml/prepare.py``, ``study.py`` and ``cli.py``: explicit offline orchestration.
* ``research/encoding_optimization/synthetic.yaml``: strict study configuration.

Dataset loading, feature extraction, fitting, inference and evaluation are separate
boundaries. ``DatasetAdapter`` and ``FeatureExtractor`` accept domain-specific
implementations. Training accepts estimator factories, a metric function,
selection criterion/direction and permutation scoring. Estimators implement a
scikit-learn-compatible fit/predict interface. A future PyTorch/CUDA adapter can
implement that interface without putting training into FastAPI. Tensor/segmentation
workflows still need a dedicated task adapter; this release implements scalar
regression only. No torch, CUDA, medical codec or imaging package is installed.

Feature definitions
--------------------------------------------------------------------------------

The first video stream is decoded by OpenCV. Original file bytes are checked
against their content hash before and after extraction. A fixed resize width
(default 160, aspect ratio preserved) and grayscale conversion precede features.
One CPU thread is used for OpenCV. Feature settings/version and library versions
are saved with the model; inference reconstructs the same extractor settings.

Uniform sampling uses unique evenly spaced frame indices, including first and
last, capped at ``max_frames`` (default 12). Stride sampling starts at frame zero
and takes every configured stride up to that cap. One-frame videos are supported:
temporal and motion features become zero, meaning no temporal evidence, not proof
of no motion. Missing/undecodable requested frames fail explicitly.

.. list-table:: Feature definitions
   :header-rows: 1

   * - Feature
     - Definition
   * - Spatial complexity
     - Mean per-frame standard deviation of Sobel gradient magnitude on gray/255.
   * - Temporal complexity
     - Mean standard deviation of signed differences between sampled gray/255 frames.
   * - Motion intensity
     - Mean absolute difference between sampled gray/255 frames.
   * - Edge density
     - Mean fraction of Canny edge pixels (thresholds 100/200 on 8-bit gray).
   * - Texture complexity
     - Mean Shannon entropy of each frame's 256-bin intensity histogram, in bits.
   * - Mean brightness
     - Mean grayscale intensity across sampled frames, normalized to [0, 1].
   * - Brightness variation
     - Standard deviation of sampled frame mean brightness values.

These are lightweight proxies, not optical flow or perceptual/diagnostic features.
Histogram entropy ignores spatial arrangement. Frame differences conflate motion,
scene changes and illumination; sampling gaps change their interpretation. Resize,
bit depth, decoder build and frame indexing affect features. Values must be finite. Original resolution/FPS are not input covariates in this
demo; use a matched acquisition regime or extend the extractor before making
claims on heterogeneous resolutions, frame rates or medical acquisitions.
The model inputs are these seven features plus CRF; SSIM, PSNR, measured bitrate,
encoded frames, source IDs and grouping keys are never model inputs.

Dataset and split protocol
--------------------------------------------------------------------------------

The manifest preserves an opaque source ID, content hash, local source path,
domain, sensitivity flag and optional opaque parent/patient/study grouping keys.
No dataset is downloaded automatically. A public manifest record looks like::

   {
     "source_id": "public-scene-001",
     "content_hash": "actual_sha256_of_original_file",
     "path": "originals/scene-001.mp4",
     "domain": "public_video",
     "sensitive": false,
     "study_group": "parent-film-001"
   }

Put records in ``{"sources": [...]}`` in an ignored local JSON manifest. Paths
resolve relative to that manifest. Use multiple independent scenes/films and
retain license/provenance alongside the dataset. Clips from one parent recording
must share a study grouping key, even if their hashes differ. Do not pretend
crops/transcodes of one clip are independent sources. At least five independent
groups are enforced; substantially more real groups (e.g. 30+) are recommended.
Medical/restricted input is refused by the Phase 2 video adapter.

``prepare`` runs the existing medium/one-thread/native-resolution CRF 18/23/28
baseline for each original. The loader reads completed compatible SQLite rows;
if repeated measurements exist it selects the latest completed compatible result
with an experiment-ID tie breaker. Targets are SSIM and container bitrate (which
includes muxing overhead); PSNR and tool versions remain auxiliary measurements.
The saved local dataset snapshot fixes experiment IDs, hashes, features, CRFs,
targets, PSNR and split labels so later DB additions do not change that artifact.

Groups are connected components sharing a source content hash, patient key or
study key. All encoded variants, aliases and linked recordings stay together.
Group keys never become input features. Sorted groups are shuffled with NumPy
seed 42; validation/test each take ceil(20% of groups), training gets the rest.
Splitting is unstratified, and can produce domain imbalance in small datasets.
For the 12-source demo this yields 6 training, 3 validation and 3 test sources
(18/9/9 encoded observations). Group membership is checked before fitting.

Training, interpretation and evaluation
--------------------------------------------------------------------------------

For each target, fit a mean DummyRegressor and a 64-tree RandomForestRegressor
(max depth 8, min leaf 1, seed 42, one job) using training rows only. Select by
validation MAE; deterministic ties prefer the model name in lexical order. Do not
refit on validation or test. The selected fitted models form a versioned joblib
bundle. Only load trusted joblib artifacts: deserialization can execute code.

Permutation importance uses validation rows only, five repeats, seed 42 and
negative MAE scoring. Results include mean/std for each feature and CRF in
``interpretation.json``. They describe model sensitivity, not causal effects.
Correlated features and few independent groups make importances unstable.

After selection, evaluate the frozen models on held-out test rows. Report MAE
and RMSE, with bitrate in bits/second. R-squared is omitted for constant targets
or fewer than five independent evaluation groups; even that minimum does not
establish statistical power. Report group counts alongside metrics. No test
outcome is used for model selection or threshold tuning. Infrastructure-debug
reruns use the same frozen protocol; they are not independent replications.

The prediction-only selector searches trained candidate CRFs and chooses the
lowest predicted bitrate satisfying minimum predicted SSIM. It supports a subset
of the trained grid and refuses extrapolation/invalid predictions. If none is
predicted feasible, it abstains with null recommendation and explanation. Predicted
feasibility never guarantees measured quality; deployments would need a separate
measurement/fallback policy, not silent threshold relaxation.

Retrospective comparison uses only test-source measured results:

* Fixed CRF: configured constant (default 23).
* Content-aware: the prediction-only recommendation, then measured lookup.
* Oracle: lowest measured bitrate satisfying measured SSIM, after prediction.

Oracle outcomes never enter training, inference, features or model selection.
Policy summaries include abstention, measured violation rate among selected
sources, mean per-source percentage bitrate reduction versus fixed CRF, and mean
absolute percentage bitrate distance from feasible Oracle. Missing selections or
infeasible Oracle cases yield nulls and explicit comparison counts. Reductions
use each policy's selected subset; compare coverage and denominators, not just
percentages. Negative reduction means an increase, which can be needed to satisfy
quality when fixed CRF is itself infeasible. These are not production guarantees.

Reproduction
--------------------------------------------------------------------------------

From the repository root (FFmpeg/FFprobe already installed)::

   cd backend
   uv sync --frozen --extra dev --extra ml
   cd ..
   backend/.venv/bin/python -m visionops.ml.cli synthetic \
     --output data/content-study --count 12
   backend/.venv/bin/python -m visionops.ml.cli prepare \
     research/encoding_optimization/synthetic.yaml \
     --code-revision "$(git rev-parse HEAD)-dirty"
   backend/.venv/bin/python -m visionops.ml.cli train \
     research/encoding_optimization/synthetic.yaml \
     --code-revision "$(git rev-parse HEAD)-dirty"

The synthetic command refuses an existing output directory. To repeat fitting,
reuse the measured SQLite database and run ``train`` again; each run writes a new
UUID output directory. To add real videos, prepare an ignored manifest as above,
copy the YAML and set manifest/database/output paths, then run prepare and train.
YAML paths resolve relative to the YAML file. No dataset URI is implicitly fetched.
Use a clean exact code revision after committing; a dirty suffix is not a source
snapshot. Keep the dataset/lockfile/FFmpeg versions with any reported results.

``dataset.json``, ``model.joblib``, ``config.json``, ``interpretation.json`` and
``report.json`` are under ``artifacts/content-study/<uuid>/``. Source features and
measured outcomes are local artifacts, not committed data. Recommendations on a
new explicitly public video use the saved feature pipeline::

   backend/.venv/bin/python -m visionops.ml.cli recommend \
     artifacts/content-study/<uuid>/model.joblib path/to/public-source.mp4 \
     --minimum-ssim 0.99 --domain public_video

The lightweight ML extra uses mlflow-skinny for client/local file tracking,
scikit-learn, NumPy, OpenCV headless, PyYAML and joblib. No MLflow server/API key,
GPU, pandas/pyarrow stack or medical imaging framework is required. The existing
Docker API image stays lightweight and does not install the ML extra; run the
offline workflow in its documented Python environment. CI installs ML extras.

MLflow and privacy boundary
--------------------------------------------------------------------------------

The YAML tracking directory is an explicit local file store:
``mlruns/encoding-optimization`` in the example. MlflowClient writes allowed
aggregate counts/dataset digest, seed, revision, model parameters, feature settings,
library versions, validation/test metrics and policy summaries. Only public model
and aggregate interpretation artifacts are logged. Manifests, raw frames, local
source paths, source IDs, per-source reports and grouping keys are not sent.
The file store manages artifact locations itself. No external server is contacted.

Any source marked sensitive, a medical domain, or an unrecognized domain makes a
dataset restricted. Tracking refuses it before importing/creating an MLflow client.
The default video feature/Phase 2 adapter also refuses restricted samples before
file decoding or Phase 2 path logging. Private adapters can use the generic data,
split, estimator and metric contracts locally, with tracking disabled. Model
artifacts trained on restricted data are sensitive too. Do not route medical
files through existing public-video CLI commands.

See :doc:`medical_extensibility` for the future medical architecture.

Executed synthetic study
--------------------------------------------------------------------------------

Twelve independently seeded synthetic one-second scenes (160x120, 12 FPS) were
actually generated locally; all 36 Phase 2 CRF measurements completed. No public
or medical dataset was downloaded. Train/validation/test contain 6/3/3 sources.
DummyRegressor and RandomForestRegressor were fitted for each of two targets;
validation MAE selected RandomForest for both. No test outcome changed the forest
parameters or quality threshold.

The final local report is under
``artifacts/content-study/5f7408dc-e150-4690-830a-b3705a7b284c/``.
MLflow run ``00da791593564a9cbee391ee9fdaaa7a`` was verified FINISHED with
``model.joblib`` and ``interpretation.json`` artifacts. The source revision was
``340ea5f-dirty``; this is uncommitted work, not a clean reproducibility snapshot.
Initial tracking-layout failures were corrected by using MLflow-managed artifact
locations, with a regression test for repeated store use. Debug reruns used the
same data, seed and model parameters, not independent replications.

.. list-table:: Frozen selected-model test metrics, nine rows from three sources
   :header-rows: 1

   * - Target
     - MAE
     - RMSE
     - R-squared
   * - SSIM
     - 0.00819742
     - 0.01333903
     - Withheld: fewer than five independent groups
   * - Bitrate (bits/second)
     - 34542.65
     - 40982.24
     - Withheld: fewer than five independent groups

At threshold 0.99, all policies selected all three test sources. Fixed CRF 23
violated measured SSIM on two of three; content-aware and Oracle had zero
violations on these three sources. Content-aware mean bitrate reduction versus
fixed CRF was **-36.54%** (a 36.54% increase), compared with Oracle's **-24.64%**.
Mean absolute distance from feasible Oracle was 11.91% for content-aware and
17.98% for fixed CRF. Fixed CRF was cheaper partly because it violated quality;
this result does not demonstrate a bitrate saving or real-content generalization.
No confidence intervals, clinical results or production gains are claimed.
