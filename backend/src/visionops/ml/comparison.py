"""Retrospective measured-test-only policy comparison; Oracle is never inference input."""

import numpy as np

from visionops.ml.data import LearningTable, Partition
from visionops.ml.inference import ModelBundle
from visionops.ml.optimization import select_crf


def compare_policies(
    table: LearningTable,
    split: Partition,
    bundle: ModelBundle,
    *,
    minimum_ssim: float,
    candidates: list[int],
    fixed_crf: int,
) -> dict:
    records = []
    for source in sorted({table.source_ids[i] for i in split.test}):
        indices = [i for i in split.test if table.source_ids[i] == source]
        by_crf = {int(table.conditions[i]): i for i in indices}
        if set(by_crf) != set(candidates):
            raise ValueError("Test source must have a complete candidate grid")
        features = dict(zip(table.feature_names[:-1], table.x[indices[0], :-1], strict=True))
        recommendation = select_crf(bundle, features, minimum_ssim, candidates)
        # Only after prediction, look at measured held-out outcomes for evaluation.
        feasible = [i for i in indices if table.targets["ssim"][i] >= minimum_ssim]
        oracle = min(feasible, key=lambda i: table.targets["bitrate_bps"][i]) if feasible else None
        selections = {
            "fixed": by_crf[fixed_crf],
            "content_aware": by_crf.get(recommendation.crf),
            "oracle": oracle,
        }
        entry = {"source_id": source, "predicted_recommendation": recommendation.__dict__}
        for policy, index in selections.items():
            entry[policy] = (
                None
                if index is None
                else {
                    "crf": int(table.conditions[index]),
                    "measured_ssim": float(table.targets["ssim"][index]),
                    "measured_bitrate_bps": float(table.targets["bitrate_bps"][index]),
                }
            )
        records.append(entry)
    summary = {}
    for policy in ("fixed", "content_aware", "oracle"):
        chosen = [r for r in records if r[policy] is not None]
        reductions, distances = [], []
        for record in chosen:
            selected = record[policy]["measured_bitrate_bps"]
            reductions.append(100 * (1 - selected / record["fixed"]["measured_bitrate_bps"]))
            if record["oracle"] is not None:
                distances.append(
                    100
                    * abs(selected - record["oracle"]["measured_bitrate_bps"])
                    / record["oracle"]["measured_bitrate_bps"]
                )
        summary[policy] = {
            "test_sources": len(records),
            "selected_sources": len(chosen),
            "abstention_rate": 1 - len(chosen) / len(records),
            "violation_rate_among_selected": float(
                np.mean([r[policy]["measured_ssim"] < minimum_ssim for r in chosen])
            )
            if chosen
            else None,
            "mean_bitrate_reduction_percent_vs_fixed": float(np.mean(reductions))
            if reductions
            else None,
            "mean_absolute_distance_from_oracle_percent": float(np.mean(distances))
            if distances
            else None,
            "oracle_comparable_sources": len(distances),
        }
    return {"summary": summary, "per_source": records}
