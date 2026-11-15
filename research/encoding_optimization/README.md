# Content-aware encoding study

Question: can original-source content features select a lower-bitrate H.264 CRF while meeting minimum predicted SSIM?

See [methodology](../../docs/content_aware_ml.rst) and [medical extension contracts](../../docs/medical_extensibility.rst). `synthetic.yaml` is the tested CPU-only protocol: seed 42, CRFs 18/23/28, medium preset, one encoding thread, native dimensions, minimum SSIM 0.99. Source data and outputs are ignored local files, not research assets to commit.

Run from the repository root:

```bash
(cd backend && uv sync --frozen --extra dev --extra ml)
backend/.venv/bin/python -m visionops.ml.cli synthetic --output data/content-study --count 12
backend/.venv/bin/python -m visionops.ml.cli prepare research/encoding_optimization/synthetic.yaml --code-revision "$(git rev-parse HEAD)-dirty"
backend/.venv/bin/python -m visionops.ml.cli train research/encoding_optimization/synthetic.yaml --code-revision "$(git rev-parse HEAD)-dirty"
```

Reuse `train` with the same measured database for a fresh artifact directory; the source generator refuses to overwrite an existing directory. Use the clean exact revision after manually committing. To use real public sources, create an ignored local manifest with source ID/hash/path, `domain: public_video`, `sensitive: false`, and a shared study key for clips from one recording. Use at least five independent groups (30+ recommended), document license/source, update a copied YAML, then run prepare/train. No automatic downloads happen.

Synthetic examples validate execution and leakage boundaries. They do not establish generalization or clinically useful compression. Report measured threshold violations, abstention and negative bitrate reductions honestly; fixed CRF can itself violate the constraint. Do not tune on held-out test outcomes.
