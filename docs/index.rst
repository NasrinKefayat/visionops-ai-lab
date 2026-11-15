VisionOps-AI
============

Phase 4 adds a read-only analytics API and interactive quality/optimization dashboard.
No clinical workflow, deep model or production optimizer is implemented.

.. toctree::
   :maxdepth: 2

   video_pipeline
   content_aware_ml
   medical_extensibility
   analytics_dashboard

Architecture
------------

* ``backend/src/visionops/api``: FastAPI transport and liveness endpoint.
* ``db`` and ``config``: SQLAlchemy infrastructure and validated settings.
* ``video`` and ``pipelines``: standalone metadata, ETL and encoding experiments.
* ``medical``: future independent medical imaging domain.
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
protocols must be defined before training. Local MLflow tracking is implemented for explicitly public/synthetic experiments.
Model registry integration, production artifact storage and deployment gates remain
future work.
