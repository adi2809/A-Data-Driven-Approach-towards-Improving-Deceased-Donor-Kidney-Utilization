from __future__ import annotations


OFFERPRED_FEATURES = [
    "can_gender",
    "can_abo",
    "can_race_srtr",
    "can_on_dial",
    "can_age_at_listing",
    "can_dgn",
    "can_diab",
    "can_diab_ty",
    "can_prev_ki",
    "can_prev_tx",
    "can_prev_ki_tx_functn",
    "can_max_warm_tm",
    "can_most_recent_hgt_cm",
    "can_most_recent_wgt_kg",
    "canhx_cpra",
    "can_current_age_years",
    "can_is_adult",
    "don_gender",
    "don_abo",
    "don_age",
    "don_race_srtr",
    "don_hgt_cm",
    "don_wgt_kg",
    "don_cad_don_cod",
    "don_creat",
    "don_bun",
    "don_final_serum_creat",
    "don_peak_serum_creat",
    "don_hist_diab",
    "don_hist_cancer",
    "don_htn",
    "don_high_creat",
    "don_max_creat",
    "don_warm_isch_tm_mins",
    "dcd_ind",
    "tx_center_count_250nm",
    "kdri_rao",
    "kdri_med",
    "kdpi",
    "kdpi_bin",
    "distance_nm",
    "long_distance_flg",
    "high_kdpi_flg",
    "hcv_positive_flg",
    "hbc_positive_flg",
    "mm_a",
    "mm_b",
    "mm_dr",
    "mm_total",
    "match_day_of_week",
    "match_week_of_month",
    "match_month_of_year",
    "match_hour_of_day",
    "cand_prior_tx_count_30d",
    "cand_prior_tx_count_90d",
    "cand_prior_tx_count_150d",
    "cand_prior_tx_count_365d",
    "last_yn_offer_kdpi_bin",
    "cand_decline_count_30d",
    "cand_decline_count_90d",
    "cand_decline_count_150d",
    "cand_decline_count_365d",
    "cand_declined_kdpi_avg_30d",
    "cand_declined_kdpi_stddev_30d",
    "cand_declined_don_creat_avg_30d",
    "cand_declined_don_creat_stddev_30d",
    "cand_declined_mm_total_avg_30d",
    "cand_declined_mm_total_stddev_30d",
    "cand_declined_don_age_avg_30d",
    "cand_declined_don_age_stddev_30d",
    "cand_declined_dcd_frac_30d",
    "cand_declined_hcv_frac_30d",
    "time_since_last_offer_days",
    "center_yn_offer_count_30d",
    "center_positive_response_rate_30d",
    "center_rate_same_dcd_30d",
    "center_rate_same_high_kdpi_30d",
    "center_rate_same_hcv_pos_30d",
    "center_rate_same_long_distance_30d",
    "center_rate_same_mm_bucket_30d",
    "center_mean_accepted_sequence_30d",
    "center_mean_accepted_normalized_sequence_30d",
    "center_late_placement_rate_30d",
    "center_yn_offer_count_365d",
    "center_positive_response_rate_365d",
    "center_mean_accepted_sequence_365d",
    "center_mean_accepted_normalized_sequence_365d",
    "center_late_placement_rate_365d",
]

DISCARDPRED_SOURCE_COLUMNS = [
    "match_submit_dt",
    "match_year",
    "match_id",
    "offer_rank",
    "can_listing_ctr_cd",
    "canhx_cpra",
    "distance_nm",
    "mm_total",
    "dcd_ind",
    "hcv_positive_flg",
    "don_age",
    "kdpi",
    "donor_opo_success_rate_historical",
    "tx_center_count_250nm",
    "high_kdpi_flg",
    "opo_hist_dcd_frac_365d",
    "opo_hist_any_placed_frac_365d",
    "opo_hist_both_wasted_frac_365d",
    "opo_hist_kdpi_bin_placement_rate_365d",
    "opo_hist_mean_first_accept_declines_365d",
    "opo_hist_mean_run_len_365d",
]

DISCARDPRED_STATIC_FEATURES = [
    "run_len",
    "run_len_log",
    "frac_center_linked",
    "don_age",
    "kdpi",
    "dcd_ind",
    "high_kdpi_flg",
    "donor_opo_success_rate_historical",
    "tx_center_count_250nm",
    "opo_hist_dcd_frac_365d",
    "opo_hist_any_placed_frac_365d",
    "opo_hist_both_wasted_frac_365d",
    "opo_hist_kdpi_bin_placement_rate_365d",
    "opo_hist_mean_first_accept_declines_365d",
    "opo_hist_mean_run_len_365d",
    "count_distance_le_10",
    "count_distance_le_100",
    "count_distance_le_250",
    "count_mm_total_0",
    "count_mm_total_1_2",
    "count_mm_total_3_4",
    "count_mm_total_5_6",
    "count_cpra_ge_80",
    "count_cpra_ge_98",
    "count_dcd_offer_rows",
    "count_hcv_positive_offer_rows",
]

DISCARDPRED_SCORE_FEATURES = [
    "offerpred_score_max",
    "offerpred_score_mean",
    "offerpred_score_sum_first10",
    "offerpred_score_sum_first25",
    "offerpred_score_sum_first50",
    "offerpred_score_sum_tail",
    "offerpred_score_top10_max",
    "offerpred_score_top25_max",
    "offerpred_score_top50_max",
    "offerpred_score_head_minus_tail",
]

DISCARDPRED_RUN_FEATURES = DISCARDPRED_STATIC_FEATURES + DISCARDPRED_SCORE_FEATURES

LOCATIONPRED_STATIC_FEATURES = [
    "offer_rank",
    "normalized_offer_rank",
    "ptr_sequence_num",
    "ptr_tot_score",
    "ptr_stat_cd",
    "mm_total",
    "distance_nm",
    "canhx_cpra",
    "kdpi",
    "don_age",
    "dcd_ind",
    "high_kdpi_flg",
    "long_distance_flg",
    "cand_decline_count_365d",
    "time_since_last_offer_days",
    "center_positive_response_rate_365d",
    "center_mean_accepted_normalized_sequence_365d",
    "opo_hist_any_placed_frac_365d",
    "opo_hist_mean_first_accept_declines_365d",
]

LOCATIONPRED_SEGMENT_BOUNDS = (
    0.0,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
)

LOCATIONPRED_SEGMENT_FEATURES = [
    "locationpred_segment_id",
    "locationpred_segment_norm_lower",
    "locationpred_segment_norm_upper",
    "locationpred_segment_norm_mean",
    "locationpred_segment_offer_rank_min",
    "locationpred_segment_offer_rank_max",
    "locationpred_segment_offer_rank_mean",
    "locationpred_segment_row_count",
    "locationpred_segment_offerpred_sum",
    "locationpred_segment_offerpred_max",
    "locationpred_segment_offerpred_mean",
    "locationpred_segment_offerpred_mass",
    "locationpred_segment_cum_offerpred_mass_before",
    "locationpred_segment_offerpred_mass_after",
    "run_len",
    "kdpi_mean",
    "don_age_mean",
    "dcd_share",
    "distance_nm_mean",
    "mm_total_mean",
    "canhx_cpra_mean",
    "center_positive_response_rate_365d_mean",
    "center_mean_accepted_normalized_sequence_365d_mean",
    "opo_hist_any_placed_frac_365d_mean",
    "opo_hist_mean_first_accept_declines_365d_mean",
]

LOCATIONPRED_BASE_FEATURES = LOCATIONPRED_SEGMENT_FEATURES

OFFERPRED_NAME = "OfferPred"
DISCARDPRED_NAME = "DiscardPred"
LOCATIONPRED_NAME = "LocationPred"

MODEL_DISPLAY_NAMES = {
    "offerpred": OFFERPRED_NAME,
    "discardpred": DISCARDPRED_NAME,
    "locationpred": LOCATIONPRED_NAME,
}

PUBLIC_MODEL_FEATURE_SETS = {
    OFFERPRED_NAME: {
        "default": OFFERPRED_FEATURES,
    },
    DISCARDPRED_NAME: {
        "default": DISCARDPRED_RUN_FEATURES,
    },
    LOCATIONPRED_NAME: {
        "default": LOCATIONPRED_BASE_FEATURES,
    },
}


def get_discardpred_run_features() -> list[str]:
    return DISCARDPRED_RUN_FEATURES


def get_locationpred_base_features() -> list[str]:
    return LOCATIONPRED_BASE_FEATURES

BENCHMARK_METADATA_COLUMNS = [
    "match_id",
    "ptr_row_order",
    "offer_rank",
    "ptr_sequence_num",
    "match_submit_dt",
    "match_year",
    "split",
]

SOURCE_REQUIRED_COLUMNS = sorted(
    set(
        OFFERPRED_FEATURES
        + DISCARDPRED_SOURCE_COLUMNS
        + [column for column in LOCATIONPRED_STATIC_FEATURES if column != "normalized_offer_rank"]
        + [
            "ptr_row_order",
            "ptr_offer_acpt",
            "px_id",
            "ptr_org_placed",
            "match_id",
            "match_submit_dt",
            "match_year",
        ]
    )
)

ALL_NULL_FEATURE_PREFIXES = ("opo_center_pair_",)

OFFERPRED_BANNED_FEATURES = {
    "ptr_sequence_num",
    "offer_rank",
    "ptr_tot_score",
    "ptr_org_placed",
    "same_match_prior_decliner_count",
}

DISCARDPRED_BANNED_FEATURES = {
    "match_id",
    "donor_id",
    "px_id",
    "ptr_offer_acpt",
    "ptr_org_placed",
    "ptr_sequence_num",
    "offer_rank",
    "ptr_tot_score",
    "run_state",
    "audit_reason",
}

LOCATIONPRED_FEATURES = LOCATIONPRED_BASE_FEATURES
