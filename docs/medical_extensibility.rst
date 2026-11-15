Future medical video and imaging compatibility
================================================================================

Phase 3 provides extension contracts, not a medical pipeline or clinical model.
No medical datasets are downloaded, and no pydicom, nibabel, MONAI or heavy imaging
framework is added. Existing FFmpeg encoding and PSNR/SSIM behavior is preserved.

Separation of responsibilities
--------------------------------------------------------------------------------

``SourceSample`` preserves an opaque source ID and content identity, a local path,
domain, sensitivity, and optional patient/study groups. File decoding is delegated
to a ``DatasetAdapter``/``FeatureExtractor``, not assumed by grouped splitting or
training. The implemented public-video adapter reads Phase 2 SQLAlchemy records;
it is not a reader for medical datasets. Missing privacy declarations default to
restricted, not public. Medical/DICOM/CT/MRI/ultrasound domain labels require an
opaque patient grouping key. Study keys allow additional recording-level linkage.

Future adapters may resolve volumes, series or multi-frame sources, perform
approved local de-identification and produce domain-specific features/targets.
Loading must remain separate from feature extraction, training, inference and
evaluation. Keep protected metadata and the identity mapping inside approved local
storage. Do not use paths, filenames, IDs or clinical labels as encoding features.

Splits union content identity, patient and study links before assigning partitions.
This prevents a patient's studies, duplicated volumes, or a recording's clips
from appearing on both sides of a split. A future adapter must supply trustworthy
stable grouping keys; the code cannot infer patient relationships from video bytes.
Missing/incorrect keys can still leak data. Prospective/site/time validation may
also be required by a particular research protocol.

Domain extension points
--------------------------------------------------------------------------------

* Medical videos: domain feature extractor for acquisition timing, overlays and
  modality properties; encoding/quality operations only on an approved export.
* DICOM: future adapter must respect series identity, photometric interpretation,
  transfer syntax, rescale values, orientation/spacing and private/burned-in data.
* CT/MRI: use volume-aware spacing/intensity preprocessing and patient/site splits,
  not consumer-video grayscale assumptions or arbitrary per-slice partitions.
* Ultrasound: preserve frame timing and acquisition settings, distinguish moving
  anatomy from probe motion, and group clips from the same patient/exam.

None of those readers, preprocessing steps or research models is implemented.
The core tabular training function accepts alternative feature schemas, compatible
estimator factories and a domain metric/selection rule without edits. GPU models
can later expose fit/predict through an adapter with explicit device configuration,
seed and version metadata. Segmentation/3D tensor workflows require an appropriate
task adapter and inference shape contract; the current scalar regressors are not
a segmentation framework.

Video quality and medical task metrics stay separate. PSNR/SSIM/VMAF belong to
full-reference encoding quality. Future metrics such as ``medical.segmentation.dice``
belong to a medical evaluator and must never be substituted for SSIM fields or
used to claim diagnostic preservation. Encoding improvements do not establish
clinical usefulness, and this project makes no clinical performance claim.

Storage and tracking
--------------------------------------------------------------------------------

Use approved storage outside the checkout, or ignored ``private/``, ``sensitive/``,
``datasets/`` or ``data/`` locations. Git/Docker ignore common DICOM/NIfTI/volume
extensions and model artifacts as additional protection. Ignore rules cannot
recognize every medical video or metadata file; never put protected CSV/JSON,
images, video, raw identifiers or linkage maps in tracked research/config files.

Core training and grouping do not log records or paths. The public study adapters
refuse restricted inputs before Phase 2's existing path logs can be used.
MLflow tracking refuses restricted datasets entirely, including their fitted
model artifacts, feature tables, hashes and aggregate metadata. There is no
"anonymized upload" option in this MVP. Restricted research must remain local
with tracking disabled and a dedicated private adapter. Artifact publication and
clinical data governance are outside this release.
