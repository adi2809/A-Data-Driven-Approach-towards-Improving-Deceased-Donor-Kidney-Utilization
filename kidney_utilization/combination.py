from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class CombinedModelPredictions:
    decision: str
    discard_probability: float
    predicted_first_acceptance_rank: int | None
    row_probabilities: pd.DataFrame


def combine_model_predictions(
    frame: pd.DataFrame,
    discard_probability: float,
    discard_threshold: float = 0.5,
) -> CombinedModelPredictions:
    rows = frame.copy()
    discard_probability = min(max(float(discard_probability), 0.0), 1.0)
    decision = "discard" if discard_probability >= float(discard_threshold) else "localize"

    if rows.empty:
        rows["final_first_acceptance_probability"] = []
        return CombinedModelPredictions(
            decision=decision,
            discard_probability=discard_probability,
            predicted_first_acceptance_rank=None,
            row_probabilities=rows,
        )

    required_columns = {
        "offer_rank",
        "offerpred_score",
        "locationpred_segment_id",
        "locationpred_segment_probability",
    }
    missing_columns = sorted(required_columns.difference(rows.columns))
    if missing_columns:
        raise ValueError("missing inference columns: " + ", ".join(missing_columns))

    rows["final_first_acceptance_probability"] = 0.0
    if decision == "discard":
        return CombinedModelPredictions(
            decision=decision,
            discard_probability=discard_probability,
            predicted_first_acceptance_rank=None,
            row_probabilities=rows,
        )

    segment_mass = (
        rows.groupby("locationpred_segment_id", sort=True)["locationpred_segment_probability"]
        .first()
        .astype(float)
        .clip(lower=0.0)
    )
    if float(segment_mass.sum()) <= 1e-12:
        segment_mass[:] = 1.0 / len(segment_mass)
    else:
        segment_mass /= float(segment_mass.sum())

    for segment_id, segment_rows in rows.groupby("locationpred_segment_id", sort=False):
        scores = segment_rows["offerpred_score"].astype(float).clip(lower=0.0)
        if float(scores.sum()) <= 1e-12:
            weights = pd.Series(1.0 / len(segment_rows), index=segment_rows.index)
        else:
            weights = scores / float(scores.sum())
        rows.loc[segment_rows.index, "final_first_acceptance_probability"] = (
            weights * float(segment_mass.loc[segment_id])
        )

    predicted_row = rows.sort_values(
        ["final_first_acceptance_probability", "offer_rank"],
        ascending=[False, True],
    ).iloc[0]
    return CombinedModelPredictions(
        decision=decision,
        discard_probability=discard_probability,
        predicted_first_acceptance_rank=int(predicted_row["offer_rank"]),
        row_probabilities=rows,
    )
