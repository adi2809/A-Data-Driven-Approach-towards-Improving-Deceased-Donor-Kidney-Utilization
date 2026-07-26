from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from .config import BenchmarkConfig
from .feature_specs import (
    SOURCE_REQUIRED_COLUMNS,
    OFFERPRED_FEATURES,
    DISCARDPRED_SOURCE_COLUMNS,
    DISCARDPRED_RUN_FEATURES,
    LOCATIONPRED_STATIC_FEATURES,
)
from .utils import dump_json, duckdb_relation_columns, ensure_parent


@dataclass(slots=True)
class BenchmarkBuildArtifacts:
    benchmark_db: Path
    manifest_path: Path


def _sql_string(value: str) -> str:
    return value.replace("'", "''")


def _lowercase_projection(columns: list[str]) -> str:
    return ",\n    ".join(f'"{column}" AS {column.lower()}' for column in columns)


def _table_row_count(con: duckdb.DuckDBPyConnection, table_name: str) -> int:
    return int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def _create_source_views(con: duckdb.DuckDBPyConnection, config: BenchmarkConfig) -> None:
    feature_glob = _sql_string(config.feature_parquet_glob)
    disposition_glob = _sql_string(config.donor_disposition_glob)

    offer_relation = f"SELECT * FROM parquet_scan('{feature_glob}', union_by_name=true)"
    disposition_relation = f"SELECT * FROM parquet_scan('{disposition_glob}', union_by_name=true)"

    offer_columns = duckdb_relation_columns(con, offer_relation)
    disposition_columns = duckdb_relation_columns(con, disposition_relation)

    available_offer_columns = set(column.lower() for column in offer_columns)
    offer_projection = _lowercase_projection(offer_columns)
    if (
        "don_opo_success_rate_historical" in available_offer_columns
        and "donor_opo_success_rate_historical" not in available_offer_columns
    ):
        offer_projection = (
            f"{offer_projection},\n"
            "    don_opo_success_rate_historical AS donor_opo_success_rate_historical"
        )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW _offer_rows_source AS
        SELECT
            {offer_projection}
        FROM parquet_scan('{feature_glob}', union_by_name=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW _donor_disposition_source AS
        SELECT
            {_lowercase_projection(disposition_columns)}
        FROM parquet_scan('{disposition_glob}', union_by_name=true)
        """
    )

    available_offer_columns_with_aliases = set(available_offer_columns)
    if "don_opo_success_rate_historical" in available_offer_columns:
        available_offer_columns_with_aliases.add("donor_opo_success_rate_historical")
    missing_required_columns = sorted(set(SOURCE_REQUIRED_COLUMNS).difference(available_offer_columns_with_aliases))
    if missing_required_columns:
        raise ValueError(
            "feature parquet is missing required columns for the benchmark: "
            f"{missing_required_columns}"
        )


def build_benchmark(
    config: BenchmarkConfig,
    threads: int = 8,
    overwrite: bool = False,
    skip_locationpred_exports: bool = False,
) -> BenchmarkBuildArtifacts:
    benchmark_db = config.benchmark_db_path
    manifest_path = config.benchmark_manifest_path

    ensure_parent(benchmark_db)
    ensure_parent(manifest_path)
    if overwrite and benchmark_db.exists():
        benchmark_db.unlink()

    con = duckdb.connect(str(benchmark_db))
    con.execute(f"PRAGMA threads={int(threads)};")
    con.execute("CREATE SCHEMA IF NOT EXISTS benchmark;")
    _create_source_views(con, config)

    history_start = config.history_start.isoformat(sep=" ")
    supervised_start = config.supervised_start.isoformat(sep=" ")
    validation_start = config.validation_start.isoformat(sep=" ")
    test_start = config.test_start.isoformat(sep=" ")
    supervised_end = config.supervised_end.isoformat(sep=" ")

    split_case = f"""
        CASE
            WHEN match_submit_dt < TIMESTAMP '{supervised_start}' THEN 'history'
            WHEN match_submit_dt >= TIMESTAMP '{test_start}' THEN 'test'
            WHEN match_submit_dt >= TIMESTAMP '{validation_start}' THEN 'validation'
            ELSE 'train'
        END
    """
    timing_case = f"""
        CASE
            WHEN normalized_first_observed_y_rank IS NULL THEN NULL
            WHEN normalized_first_observed_y_rank <= {config.early_cutoff} THEN 'early'
            WHEN normalized_first_observed_y_rank <= {config.mid_cutoff} THEN 'mid'
            ELSE 'late'
        END
    """
    con.execute(
        f"""
        CREATE OR REPLACE TABLE benchmark.match_labels AS
        WITH run_offer_summary AS (
            SELECT
                match_id,
                MAX(match_submit_dt) AS match_submit_dt,
                MAX(match_year) AS match_year,
                COUNT(*) AS run_len,
                COUNT(*) FILTER (WHERE ptr_offer_acpt = 'Y') AS y_row_count,
                MAX(CASE WHEN ptr_offer_acpt = 'Y' THEN 1 ELSE 0 END) AS has_observed_y,
                ARG_MIN(ptr_row_order, offer_rank) FILTER (WHERE ptr_offer_acpt = 'Y') AS first_observed_y_ptr_row_order,
                ARG_MIN(ptr_sequence_num, offer_rank) FILTER (WHERE ptr_offer_acpt = 'Y') AS first_observed_y_sequence_num,
                ARG_MIN(px_id, offer_rank) FILTER (WHERE ptr_offer_acpt = 'Y') AS first_observed_y_px_id,
                MIN(offer_rank) FILTER (WHERE ptr_offer_acpt = 'Y') AS first_observed_y_rank
            FROM _offer_rows_source
            WHERE match_submit_dt BETWEEN TIMESTAMP '{history_start}' AND TIMESTAMP '{supervised_end}'
            GROUP BY 1
        ),
        run_outcomes AS (
            SELECT
                CAST(match_id AS BIGINT) AS match_id,
                1 AS has_any_saf_link,
                COUNT(*) FILTER (
                    WHERE don_disposition = 6 AND don_org IN ('LKI', 'RKI', 'EKI')
                ) AS placed_kidney_count,
                MAX(
                    CASE
                        WHEN don_disposition = 6 AND don_org IN ('LKI', 'RKI', 'EKI') THEN 1
                        ELSE 0
                    END
                ) AS placed_any_kidney
            FROM _donor_disposition_source
            WHERE match_id IS NOT NULL
            GROUP BY 1
        )
        SELECT
            r.match_id,
            r.match_submit_dt,
            r.match_year,
            {split_case} AS split,
            r.run_len,
            r.y_row_count,
            r.has_observed_y,
            r.first_observed_y_ptr_row_order,
            r.first_observed_y_sequence_num,
            r.first_observed_y_px_id,
            r.first_observed_y_rank,
            CASE
                WHEN r.first_observed_y_rank IS NULL OR r.run_len = 0 THEN NULL
                ELSE r.first_observed_y_rank * 1.0 / r.run_len
            END AS normalized_first_observed_y_rank,
            COALESCE(o.has_any_saf_link, 0) AS has_any_saf_link,
            COALESCE(o.placed_any_kidney, 0) AS placed_any_kidney,
            COALESCE(o.placed_kidney_count, 0) AS placed_kidney_count,
            CASE
                WHEN r.has_observed_y = 1 AND COALESCE(o.placed_any_kidney, 0) = 1 THEN 'localizable_observed_y'
                WHEN r.has_observed_y = 0 AND COALESCE(o.placed_any_kidney, 0) = 1 THEN 'censored_positive'
                WHEN r.has_observed_y = 1 AND COALESCE(o.placed_any_kidney, 0) = 0 THEN 'audit_orphan_y'
                ELSE 'none'
            END AS run_state,
            CASE
                WHEN r.has_observed_y = 1 AND COALESCE(o.placed_any_kidney, 0) = 0 AND COALESCE(o.has_any_saf_link, 0) = 0 THEN 'missing_saf_link'
                WHEN r.has_observed_y = 1 AND COALESCE(o.placed_any_kidney, 0) = 0 THEN 'non_kidney_only_saf_outcome'
                ELSE NULL
            END AS audit_reason
        FROM run_offer_summary AS r
        LEFT JOIN run_outcomes AS o
          ON r.match_id = o.match_id
        """
    )

    offerpred_projection = ",\n            ".join(OFFERPRED_FEATURES)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE benchmark.offerpred_rows AS
        WITH accepted_rows AS (
            SELECT
                match_id,
                offer_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY match_id
                    ORDER BY offer_rank, ptr_row_order
                ) AS acceptance_number
            FROM _offer_rows_source
            WHERE ptr_offer_acpt = 'Y'
        ),
        second_acceptance AS (
            SELECT match_id, offer_rank AS second_acceptance_rank
            FROM accepted_rows
            WHERE acceptance_number = 2
        )
        SELECT
            o.match_id,
            o.ptr_row_order,
            o.offer_rank,
            o.ptr_sequence_num,
            o.ptr_offer_acpt,
            o.match_submit_dt,
            o.match_year,
            l.split,
            l.run_state,
            CASE WHEN o.ptr_offer_acpt = 'Y' THEN 1 ELSE 0 END AS offerpred_target,
            {offerpred_projection}
        FROM _offer_rows_source AS o
        JOIN benchmark.match_labels AS l
          ON o.match_id = l.match_id
        LEFT JOIN second_acceptance AS s
          ON o.match_id = s.match_id
        WHERE l.split IN ('train', 'validation', 'test')
          AND l.run_state IN ('localizable_observed_y', 'none')
          AND o.ptr_offer_acpt IN ('Y', 'N')
          AND (s.second_acceptance_rank IS NULL OR o.offer_rank <= s.second_acceptance_rank)
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE benchmark.offerpred_scoring_rows AS
        WITH accepted_rows AS (
            SELECT
                match_id,
                offer_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY match_id
                    ORDER BY offer_rank, ptr_row_order
                ) AS acceptance_number
            FROM _offer_rows_source
            WHERE ptr_offer_acpt = 'Y'
        ),
        second_acceptance AS (
            SELECT match_id, offer_rank AS second_acceptance_rank
            FROM accepted_rows
            WHERE acceptance_number = 2
        )
        SELECT
            o.match_id,
            o.ptr_row_order,
            o.offer_rank,
            o.ptr_sequence_num,
            o.ptr_offer_acpt,
            o.match_submit_dt,
            o.match_year,
            l.split,
            l.run_state,
            CASE
                WHEN o.ptr_offer_acpt = 'Y' THEN 1
                WHEN o.ptr_offer_acpt = 'N' THEN 0
                ELSE NULL
            END AS offerpred_target,
            CASE
                WHEN l.run_state IN ('localizable_observed_y', 'none')
                  AND o.ptr_offer_acpt IN ('Y', 'N')
                  AND (s.second_acceptance_rank IS NULL OR o.offer_rank <= s.second_acceptance_rank)
                THEN 1
                ELSE 0
            END AS use_for_offerpred_loss,
            {offerpred_projection}
        FROM _offer_rows_source AS o
        JOIN benchmark.match_labels AS l
          ON o.match_id = l.match_id
        LEFT JOIN second_acceptance AS s
          ON o.match_id = s.match_id
        WHERE l.split IN ('train', 'validation', 'test')
          AND l.run_state IN ('localizable_observed_y', 'none', 'censored_positive')
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE benchmark.discardpred_runs AS
        WITH base AS (
            SELECT
                o.*,
                l.split,
                l.run_state,
                l.run_len,
                l.y_row_count,
                l.has_observed_y,
                l.first_observed_y_rank,
                l.normalized_first_observed_y_rank
            FROM _offer_rows_source AS o
            JOIN benchmark.match_labels AS l
              ON o.match_id = l.match_id
            WHERE l.split IN ('train', 'validation', 'test')
              AND l.run_state IN ('localizable_observed_y', 'none', 'censored_positive')
        ),
        aggregated AS (
            SELECT
                match_id,
                MAX(match_submit_dt) AS match_submit_dt,
                MAX(match_year) AS match_year,
                MAX(split) AS split,
                MAX(run_state) AS run_state,
                MAX(run_len) AS run_len,
                LOG(MAX(run_len) + 1) AS run_len_log,
                MAX(y_row_count) AS y_row_count,
                MAX(has_observed_y) AS has_observed_y,
                MAX(first_observed_y_rank) AS first_observed_y_rank,
                MAX(normalized_first_observed_y_rank) AS normalized_first_observed_y_rank,
                AVG(CASE WHEN can_listing_ctr_cd IS NOT NULL THEN 1.0 ELSE 0.0 END) AS frac_center_linked,
                MAX(don_age) AS don_age,
                MAX(kdpi) AS kdpi,
                MAX(dcd_ind) AS dcd_ind,
                MAX(high_kdpi_flg) AS high_kdpi_flg,
                MAX(donor_opo_success_rate_historical) AS donor_opo_success_rate_historical,
                MAX(tx_center_count_250nm) AS tx_center_count_250nm,
                MAX(opo_hist_dcd_frac_365d) AS opo_hist_dcd_frac_365d,
                MAX(opo_hist_any_placed_frac_365d) AS opo_hist_any_placed_frac_365d,
                MAX(opo_hist_both_wasted_frac_365d) AS opo_hist_both_wasted_frac_365d,
                MAX(opo_hist_kdpi_bin_placement_rate_365d) AS opo_hist_kdpi_bin_placement_rate_365d,
                MAX(opo_hist_mean_first_accept_declines_365d) AS opo_hist_mean_first_accept_declines_365d,
                MAX(opo_hist_mean_run_len_365d) AS opo_hist_mean_run_len_365d,
                COUNT(*) FILTER (WHERE distance_nm <= 10) AS count_distance_le_10,
                COUNT(*) FILTER (WHERE distance_nm <= 100) AS count_distance_le_100,
                COUNT(*) FILTER (WHERE distance_nm <= 250) AS count_distance_le_250,
                COUNT(*) FILTER (WHERE mm_total = 0) AS count_mm_total_0,
                COUNT(*) FILTER (WHERE mm_total BETWEEN 1 AND 2) AS count_mm_total_1_2,
                COUNT(*) FILTER (WHERE mm_total BETWEEN 3 AND 4) AS count_mm_total_3_4,
                COUNT(*) FILTER (WHERE mm_total BETWEEN 5 AND 6) AS count_mm_total_5_6,
                COUNT(*) FILTER (WHERE canhx_cpra >= 80) AS count_cpra_ge_80,
                COUNT(*) FILTER (WHERE canhx_cpra >= 98) AS count_cpra_ge_98,
                COUNT(*) FILTER (WHERE dcd_ind = 1) AS count_dcd_offer_rows,
                COUNT(*) FILTER (WHERE hcv_positive_flg = 1) AS count_hcv_positive_offer_rows
            FROM base
            GROUP BY 1
        )
        SELECT
            *,
            CASE WHEN run_state = 'none' THEN 1 ELSE 0 END AS discard_target,
            CASE
                WHEN run_state = 'none' THEN 'none'
                WHEN run_state = 'censored_positive' THEN 'censored_positive'
                ELSE 'localizable_observed_y'
            END AS route_target,
            {timing_case} AS timing_target
        FROM aggregated
        """
    )

    locationpred_projection = ",\n            ".join(LOCATIONPRED_STATIC_FEATURES)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE benchmark.locationpred_riskset AS
        WITH base AS (
            SELECT
                o.match_id,
                o.ptr_row_order,
                o.offer_rank,
                o.ptr_sequence_num,
                o.match_submit_dt,
                o.match_year,
                l.split,
                l.run_state,
                l.run_len,
                l.first_observed_y_rank,
                CASE
                    WHEN l.run_len = 0 THEN NULL
                    ELSE o.offer_rank * 1.0 / l.run_len
                END AS normalized_offer_rank,
                {locationpred_projection}
            FROM _offer_rows_source AS o
            JOIN benchmark.match_labels AS l
              ON o.match_id = l.match_id
            WHERE l.split IN ('train', 'validation', 'test')
              AND l.run_state = 'localizable_observed_y'
              AND o.offer_rank <= l.first_observed_y_rank
        ),
        labeled AS (
            SELECT
                *,
                CASE
                    WHEN run_state = 'localizable_observed_y' AND offer_rank = first_observed_y_rank THEN 1
                    ELSE 0
                END AS locationpred_target,
                CASE
                    WHEN normalized_offer_rank <= {config.early_cutoff} THEN 'early'
                    WHEN normalized_offer_rank <= {config.mid_cutoff} THEN 'mid'
                    ELSE 'late'
                END AS timing_bucket
            FROM base
        )
        SELECT
            *,
            COUNT(*) OVER (PARTITION BY match_id) AS risk_rows_in_match,
            1.0 / COUNT(*) OVER (PARTITION BY match_id) AS risk_row_weight
        FROM labeled
        """
    )

    locationpred_scoring_projection = ",\n                ".join(
        column for column in LOCATIONPRED_STATIC_FEATURES if column != "normalized_offer_rank"
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE benchmark.locationpred_scoring_rows AS
        WITH base AS (
            SELECT
                o.match_id,
                o.ptr_row_order,
                o.offer_rank,
                o.ptr_sequence_num,
                o.match_submit_dt,
                o.match_year,
                l.split,
                l.run_state,
                l.run_len,
                l.first_observed_y_rank,
                CASE
                    WHEN l.run_len = 0 THEN NULL
                    ELSE o.offer_rank * 1.0 / l.run_len
                END AS normalized_offer_rank,
                {locationpred_scoring_projection}
            FROM _offer_rows_source AS o
            JOIN benchmark.match_labels AS l
              ON o.match_id = l.match_id
            WHERE l.split IN ('train', 'validation', 'test')
              AND l.run_state IN ('localizable_observed_y', 'none', 'censored_positive')
        ),
        labeled AS (
            SELECT
                *,
                CASE
                    WHEN run_state = 'localizable_observed_y' AND offer_rank = first_observed_y_rank THEN 1
                    ELSE 0
                END AS locationpred_target,
                CASE
                    WHEN normalized_offer_rank <= {config.early_cutoff} THEN 'early'
                    WHEN normalized_offer_rank <= {config.mid_cutoff} THEN 'mid'
                    ELSE 'late'
                END AS timing_bucket
            FROM base
        )
        SELECT
            *,
            COUNT(*) OVER (PARTITION BY match_id) AS risk_rows_in_match,
            1.0 / COUNT(*) OVER (PARTITION BY match_id) AS risk_row_weight
        FROM labeled
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE benchmark.audit_runs AS
        SELECT *
        FROM benchmark.match_labels
        WHERE run_state = 'audit_orphan_y'
        """
    )

    table_counts = {
        "match_labels": _table_row_count(con, "benchmark.match_labels"),
        "offerpred_rows": _table_row_count(con, "benchmark.offerpred_rows"),
        "offerpred_scoring_rows": _table_row_count(con, "benchmark.offerpred_scoring_rows"),
        "discardpred_runs": _table_row_count(con, "benchmark.discardpred_runs"),
        "locationpred_riskset": _table_row_count(con, "benchmark.locationpred_riskset"),
        "locationpred_scoring_rows": _table_row_count(con, "benchmark.locationpred_scoring_rows"),
        "audit_runs": _table_row_count(con, "benchmark.audit_runs"),
    }
    run_state_counts = {
        row[0]: int(row[1])
        for row in con.execute(
            "SELECT run_state, COUNT(*) FROM benchmark.match_labels GROUP BY 1 ORDER BY 1"
        ).fetchall()
    }

    manifest = {
        "config": config.to_dict(),
        "benchmark_db": str(benchmark_db),
        "feature_parquet_glob": config.feature_parquet_glob,
        "donor_disposition_glob": config.donor_disposition_glob,
        "skip_locationpred_exports": bool(skip_locationpred_exports),
        "table_counts": table_counts,
        "run_state_counts": run_state_counts,
        "offerpred_features": OFFERPRED_FEATURES,
        "discardpred_source_columns": DISCARDPRED_SOURCE_COLUMNS,
        "discardpred_features": DISCARDPRED_RUN_FEATURES,
        "locationpred_static_features": LOCATIONPRED_STATIC_FEATURES,
        "timing_cutoffs": {
            "early": config.early_cutoff,
            "mid": config.mid_cutoff,
        },
        "absolute_timing_cutoffs": {
            "early_rank": int(config.absolute_early_rank_cutoff),
            "mid_rank": int(config.absolute_mid_rank_cutoff),
        },
    }
    dump_json(manifest_path, manifest)
    con.close()
    return BenchmarkBuildArtifacts(benchmark_db=benchmark_db, manifest_path=manifest_path)
