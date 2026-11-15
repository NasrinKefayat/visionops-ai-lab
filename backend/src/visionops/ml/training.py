"""Generic train/validation selection with pluggable estimators and metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from visionops.ml.data import LearningTable, Partition, validate_partition
from visionops.ml.evaluation import regression_metrics
from visionops.ml.inference import ModelBundle, Regressor

if TYPE_CHECKING:
    from visionops.ml.config import ForestSettings


def train_regressors(
    table: LearningTable,
    split: Partition,
    targets: tuple[str, ...],
    *,
    forest: ForestSettings,
    seed: int = 42,
    factories: dict[str, Callable[[], Regressor]] | None = None,
    metric_fn: Callable = regression_metrics,
    selection_metric: str = "mae",
    higher_is_better: bool = False,
    importance_scoring: str | Callable = "neg_mean_absolute_error",
) -> tuple[ModelBundle, dict]:
    validate_partition(table, split)
    if set(targets) & set(table.feature_names):
        raise ValueError("Targets cannot be model input columns")
    factories = factories or {
        "dummy": lambda: DummyRegressor(strategy="mean"),
        "random_forest": lambda: RandomForestRegressor(
            **forest.model_dump(), random_state=seed, n_jobs=1
        ),
    }
    selected, comparison, importance, choices = {}, {}, {}, {}
    for target in targets:
        comparison[target] = {}
        candidates = {}
        y = table.targets[target]
        for name, factory in factories.items():
            model = factory()
            model.fit(table.x[split.train], y[split.train])
            prediction = model.predict(table.x[split.validation])
            scores = metric_fn(
                y[split.validation], prediction, [split.groups[i] for i in split.validation]
            )
            comparison[target][name] = scores
            candidates[name] = model
        direction = -1 if higher_is_better else 1
        winner = min(
            comparison[target],
            key=lambda name: (direction * comparison[target][name][selection_metric], name),
        )
        selected[target] = candidates[winner]
        choices[target] = winner
        # Interpretation is on validation only; test outcomes never select the model.
        result = permutation_importance(
            selected[target],
            table.x[split.validation],
            y[split.validation],
            scoring=importance_scoring,
            random_state=seed,
            n_repeats=5,
            n_jobs=1,
        )
        importance[target] = {
            name: {"mean": float(mean), "std": float(std)}
            for name, mean, std in zip(
                table.feature_names, result.importances_mean, result.importances_std, strict=True
            )
        }
    metadata = {
        "seed": seed,
        "selected_models": choices,
        "extractor_version": table.extractor_version,
        "sensitive": table.dataset.restricted,
    }
    return ModelBundle(selected, table.feature_names, metadata), {
        "validation": comparison,
        "validation_permutation_importance": importance,
    }
