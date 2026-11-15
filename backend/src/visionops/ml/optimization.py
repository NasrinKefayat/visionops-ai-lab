"""Prediction-only constrained CRF selection; no measured targets or Oracle access."""

from dataclasses import dataclass

import numpy as np

from visionops.ml.inference import ModelBundle, predict_values


@dataclass(frozen=True)
class Recommendation:
    crf: int | None
    predicted_ssim: float | None
    predicted_bitrate_bps: float | None
    explanation: str


def select_crf(
    bundle: ModelBundle,
    source_features: dict[str, float],
    minimum_ssim: float,
    candidates: list[int],
) -> Recommendation:
    if not 0 < minimum_ssim <= 1 or not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Provide a valid SSIM threshold and unique candidate CRFs")
    allowed = bundle.metadata["candidates"]
    if any(c not in allowed or type(c) is not int for c in candidates):
        raise ValueError("CRF candidates must be inside the trained candidate grid")
    if set(source_features) != set(bundle.feature_names[:-1]):
        raise ValueError("Source features do not match the trained schema")
    x = np.asarray(
        [
            [*[source_features[name] for name in bundle.feature_names[:-1]], crf]
            for crf in candidates
        ],
        dtype=float,
    )
    predictions = predict_values(bundle, x)
    ssim, bitrate = predictions["ssim"], predictions["bitrate_bps"]
    if np.any((ssim < -1) | (ssim > 1)) or np.any(bitrate <= 0):
        raise ValueError("Model returned out-of-domain encoding predictions")
    feasible = [i for i in range(len(candidates)) if ssim[i] >= minimum_ssim]
    if not feasible:
        return Recommendation(
            None, None, None, "No candidate is predicted to satisfy minimum SSIM; abstain"
        )
    index = min(feasible, key=lambda i: (bitrate[i], candidates[i]))
    return Recommendation(
        candidates[index],
        float(ssim[index]),
        float(bitrate[index]),
        "Lowest predicted bitrate among candidates meeting predicted SSIM; measurement required",
    )
