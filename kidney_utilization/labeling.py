from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


KIDNEY_PLACEMENT_ORGANS = {"LKI", "RKI", "EKI"}
KIDNEY_PLACEMENT_DISPOSITION = 6

_OFFER_COLUMN_ALIASES: Mapping[str, str] = {
    "match_id": "match_id",
    "MATCH_ID": "match_id",
    "px_id": "px_id",
    "PX_ID": "px_id",
    "ptr_row_order": "ptr_row_order",
    "PTR_ROW_ORDER": "ptr_row_order",
    "ptr_sequence_num": "ptr_sequence_num",
    "PTR_SEQUENCE_NUM": "ptr_sequence_num",
    "offer_rank": "offer_rank",
    "OFFER_RANK": "offer_rank",
    "ptr_offer_acpt": "ptr_offer_acpt",
    "PTR_OFFER_ACPT": "ptr_offer_acpt",
    "match_year": "match_year",
    "MATCH_YEAR": "match_year",
}

_DISPOSITION_COLUMN_ALIASES: Mapping[str, str] = {
    "match_id": "match_id",
    "MATCH_ID": "match_id",
    "px_id": "px_id",
    "PX_ID": "px_id",
    "donor_organ": "donor_organ",
    "DON_ORG": "donor_organ",
    "don_org": "donor_organ",
    "don_disposition": "don_disposition",
    "DON_DISPOSITION": "don_disposition",
}


def _canonicalize_columns(frame: pd.DataFrame, aliases: Mapping[str, str]) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for source_name, target_name in aliases.items():
        if source_name in frame.columns:
            renamed[source_name] = target_name
    return frame.rename(columns=renamed).copy()


def derive_match_labels(
    offer_rows: pd.DataFrame,
    donor_disposition: pd.DataFrame,
) -> pd.DataFrame:
    offers = _canonicalize_columns(offer_rows, _OFFER_COLUMN_ALIASES)
    dispositions = _canonicalize_columns(donor_disposition, _DISPOSITION_COLUMN_ALIASES)

    required_offer_columns = {"match_id", "px_id", "ptr_row_order", "ptr_sequence_num"}
    missing_offer_columns = required_offer_columns.difference(offers.columns)
    if missing_offer_columns:
        raise ValueError(f"offer_rows is missing required columns: {sorted(missing_offer_columns)}")

    if "offer_rank" not in offers.columns:
        offers = offers.sort_values(["match_id", "ptr_sequence_num", "ptr_row_order"]).copy()
        offers["offer_rank"] = offers.groupby("match_id").cumcount() + 1

    if "ptr_offer_acpt" not in offers.columns:
        offers["ptr_offer_acpt"] = "N"

    if dispositions.empty:
        dispositions = pd.DataFrame(columns=["match_id", "px_id", "donor_organ", "don_disposition"])
    else:
        missing_disp_columns = {"match_id", "donor_organ", "don_disposition"}.difference(dispositions.columns)
        if missing_disp_columns:
            raise ValueError(
                "donor_disposition is missing required columns: "
                f"{sorted(missing_disp_columns)}"
            )

    results: list[dict[str, object]] = []
    dispositions = dispositions.copy()
    dispositions["don_disposition"] = pd.to_numeric(dispositions["don_disposition"], errors="coerce")

    for match_id, match_offers in offers.groupby("match_id", sort=True):
        match_offers = match_offers.sort_values(["offer_rank", "ptr_sequence_num", "ptr_row_order"])
        y_rows = match_offers.loc[match_offers["ptr_offer_acpt"] == "Y"].copy()
        run_len = int(len(match_offers))
        y_row_count = int(len(y_rows))
        has_observed_y = int(y_row_count > 0)

        first_y = y_rows.iloc[0] if has_observed_y else None
        match_disp = dispositions.loc[dispositions["match_id"] == match_id].copy()
        has_any_saf_link = int(not match_disp.empty)
        placed_mask = (
            (match_disp["don_disposition"] == KIDNEY_PLACEMENT_DISPOSITION)
            & match_disp["donor_organ"].isin(KIDNEY_PLACEMENT_ORGANS)
        )
        placed_any_kidney = int(placed_mask.any())
        placed_kidney_count = int(placed_mask.sum())

        audit_reason: str | None = None
        if has_observed_y and not placed_any_kidney:
            audit_reason = "missing_saf_link" if not has_any_saf_link else "non_kidney_only_saf_outcome"

        if has_observed_y and placed_any_kidney:
            run_state = "localizable_observed_y"
        elif not has_observed_y and placed_any_kidney:
            run_state = "censored_positive"
        elif has_observed_y and not placed_any_kidney:
            run_state = "audit_orphan_y"
        else:
            run_state = "none"

        results.append(
            {
                "match_id": match_id,
                "run_len": run_len,
                "y_row_count": y_row_count,
                "has_observed_y": has_observed_y,
                "placed_any_kidney": placed_any_kidney,
                "placed_kidney_count": placed_kidney_count,
                "has_any_saf_link": has_any_saf_link,
                "run_state": run_state,
                "audit_reason": audit_reason,
                "is_no_accept": int(run_state == "none"),
                "first_acceptance_ptr_row_order": None if first_y is None else int(first_y["ptr_row_order"]),
                "first_acceptance_sequence_num": None if first_y is None else int(first_y["ptr_sequence_num"]),
                "first_acceptance_offer_rank": None if first_y is None else int(first_y["offer_rank"]),
                "first_acceptance_px_id": None if first_y is None else int(first_y["px_id"]),
                "normalized_first_observed_y_rank": (
                    None if first_y is None or run_len == 0 else float(first_y["offer_rank"]) / float(run_len)
                ),
                "match_year": None if "match_year" not in match_offers.columns else int(match_offers["match_year"].iloc[0]),
            }
        )

    return pd.DataFrame(results).sort_values("match_id").reset_index(drop=True)
