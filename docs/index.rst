VisionOps-AI
============

Phase 1 provides a development scaffold, not a trained analysis system.

Architecture
------------

* ``backend/src/visionops/api``: FastAPI transport and liveness endpoint.
* ``db`` and ``config``: SQLAlchemy infrastructure and validated settings.
* ``video``, ``medical``, ``pipelines``: independent future domain components.
* ``ml``: replaceable prediction and experiment tracking contracts; training
  and evaluation remain independent of HTTP request handling.
* ``assistant``: tool contracts for a future validated agent tool registry.
* ``frontend``: React and TypeScript dashboard skeleton.

Development
-----------

See the repository README for setup, validation, and service commands.
Settings use the ``VISIONOPS_`` prefix. CPU and seed 42 are defaults.
No framework is imported or model loaded by the health endpoint.

Research and MLOps roadmap
----------------------------------------

Future experiments must record dataset identity and split, code revision,
configuration, seed, metrics, and model artifact version. Seed 42 alone does
not guarantee reproducibility on every device. Baselines and evaluation
protocols must be defined before training. MLflow adapters, artifact storage,
model registry integration, and deployment gates are future work.
