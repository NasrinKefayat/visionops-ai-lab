"""Local dataset contracts and leakage-resistant grouped partitions.

No file decoding, target measurement, training or tracking happens in this module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SourceSample:
    source_id: str
    content_hash: str
    path: Path = field(repr=False)
    domain: str = "restricted"
    patient_group: str | None = field(default=None, repr=False)
    study_group: str | None = field(default=None, repr=False)
    sensitive: bool = True

    def __post_init__(self):
        if not self.source_id or not self.content_hash:
            raise ValueError("Source identity and content hash are required")
        if (
            self.domain in {"medical", "medical_video", "dicom", "ct", "mri", "ultrasound"}
            and not self.patient_group
        ):
            raise ValueError("Medical samples require an opaque patient grouping key")


@dataclass(frozen=True)
class Observation:
    source_id: str
    condition: int
    targets: dict[str, float]
    measurements: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    sources: tuple[SourceSample, ...]
    observations: tuple[Observation, ...]

    @property
    def restricted(self) -> bool:
        return any(
            s.sensitive or s.domain not in {"public_video", "synthetic_video"} for s in self.sources
        )


class DatasetAdapter(Protocol):
    def load(self) -> Dataset: ...


class FeatureExtractor(Protocol):
    feature_names: tuple[str, ...]
    version: str

    def extract(self, source: SourceSample) -> dict[str, float]: ...


@dataclass(frozen=True)
class LearningTable:
    dataset: Dataset
    feature_names: tuple[str, ...]
    extractor_version: str
    x: np.ndarray
    targets: dict[str, np.ndarray]
    source_ids: tuple[str, ...]
    conditions: np.ndarray


def build_table(dataset: Dataset, extractor: FeatureExtractor) -> LearningTable:
    sources = {sample.source_id: sample for sample in dataset.sources}
    if len(sources) != len(dataset.sources) or not sources:
        raise ValueError("Source identities must be nonempty and unique")
    if not dataset.observations:
        raise ValueError("No observations available")
    # Extract once from original sources, never from encoded outputs or target columns.
    vectors = {}
    for identity, sample in sorted(sources.items()):
        features = extractor.extract(sample)
        if set(features) != set(extractor.feature_names):
            raise ValueError("Extractor returned an unexpected feature schema")
        vector = np.asarray([features[name] for name in extractor.feature_names], dtype=float)
        if not np.isfinite(vector).all():
            raise ValueError("Features must be finite")
        vectors[identity] = vector
    observations = sorted(dataset.observations, key=lambda o: (o.source_id, o.condition))
    keys = set(observations[0].targets)
    if not keys:
        raise ValueError("At least one target is required")
    seen = set()
    rows = []
    for observation in observations:
        if observation.source_id not in vectors or set(observation.targets) != keys:
            raise ValueError("Observation source or target schema mismatch")
        key = (observation.source_id, observation.condition)
        if key in seen:
            raise ValueError("Duplicate source/condition observation; select one run explicitly")
        seen.add(key)
        rows.append([*vectors[observation.source_id], observation.condition])
    x = np.asarray(rows, dtype=float)
    targets = {
        key: np.asarray([o.targets[key] for o in observations], dtype=float) for key in sorted(keys)
    }
    if not np.isfinite(x).all() or any(not np.isfinite(y).all() for y in targets.values()):
        raise ValueError("Training values must be finite")
    return LearningTable(
        dataset,
        (*extractor.feature_names, "condition"),
        extractor.version,
        x,
        targets,
        tuple(o.source_id for o in observations),
        np.asarray([o.condition for o in observations]),
    )


@dataclass(frozen=True)
class Partition:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    groups: tuple[str, ...]


def split_groups(
    table: LearningTable,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Partition:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Leave a nonempty training partition")
    sources = table.dataset.sources
    parent = {s.source_id: s.source_id for s in sources}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    # Connected components prevent leaks even across overlapping study/patient groups.
    owners = {}
    for sample in sorted(sources, key=lambda s: s.source_id):
        for kind, value in (
            ("hash", sample.content_hash),
            ("patient", sample.patient_group),
            ("study", sample.study_group),
        ):
            if value is None:
                continue
            key = (kind, value)
            if key in owners:
                a, b = find(sample.source_id), find(owners[key])
                parent[max(a, b)] = min(a, b)
            owners[key] = sample.source_id
    row_groups = tuple(find(identity) for identity in table.source_ids)
    groups = np.asarray(sorted(set(row_groups)))
    if len(groups) < 5:
        raise ValueError("At least five independent groups are required; more are recommended")
    groups = np.random.default_rng(seed).permutation(groups)
    n_test = max(1, int(np.ceil(len(groups) * test_fraction)))
    n_val = max(1, int(np.ceil(len(groups) * validation_fraction)))
    if n_test + n_val >= len(groups):
        raise ValueError("Too few independent groups for the requested fractions")
    test, val, train = groups[:n_test], groups[n_test : n_test + n_val], groups[n_test + n_val :]
    membership = np.asarray(row_groups)
    return Partition(
        np.flatnonzero(np.isin(membership, train)),
        np.flatnonzero(np.isin(membership, val)),
        np.flatnonzero(np.isin(membership, test)),
        row_groups,
    )


def validate_partition(table: LearningTable, split: Partition) -> None:
    parts = [split.train, split.validation, split.test]
    if any(not len(part) for part in parts):
        raise ValueError("Train, validation and test must all be nonempty")
    indices = [int(i) for part in parts for i in part]
    if sorted(indices) != list(range(len(table.x))) or len(split.groups) != len(table.x):
        raise ValueError("Partitions must cover each row exactly once")
    groups = [{split.groups[i] for i in part} for part in parts]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Group leakage between partitions")
