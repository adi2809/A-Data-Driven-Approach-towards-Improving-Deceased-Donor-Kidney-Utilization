#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import duckdb


HISTORY_WINDOWS: list[tuple[str, str]] = [
    ("30D", "INTERVAL 30 DAY"),
    ("90D", "INTERVAL 90 DAY"),
    ("150D", "INTERVAL 150 DAY"),
    ("365D", "INTERVAL 365 DAY"),
]

QUALIFYING_BYPASS_REFUSAL_CODES = "861, 862, 863"

REQUIRED_MATCH_OFFER_ENRICHED_COLUMNS = frozenset(
    {
        "MATCH_ID",
        "PTR_ROW_ORDER",
        "PTR_OFFER_ID",
        "OFFER_ROW_INSTANCE_ORDINAL",
        "MATCH_SUBMIT_DT",
        "source_year",
        "match_year",
        "DONOR_ID",
        "PX_ID",
        "PTR_SEQUENCE_NUM",
        "PTR_OFFER_ACPT",
        "CANHX_CPRA",
        "CAN_CURRENT_AGE_YEARS",
        "CAN_IS_ADULT",
        "MM_TOTAL",
        "MATCH_DAY_OF_WEEK",
        "MATCH_WEEK_OF_MONTH",
        "MATCH_MONTH_OF_YEAR",
        "MATCH_HOUR_OF_DAY",
        "DON_OPO_SUCCESS_RATE_HISTORICAL",
        "LONG_DISTANCE_FLG",
        "CAND_PRIOR_TX_COUNT_30D",
        "CAND_DECLINE_COUNT_30D",
        "LAST_YN_OFFER_KDPI_BIN",
        "SAME_MATCH_PRIOR_DECLINER_COUNT",
        "TIME_SINCE_LAST_OFFER_DAYS",
        "OPO_HIST_DCD_FRAC_30D",
        "OPO_HIST_KDPI_BIN_PLACEMENT_RATE_30D",
        "CENTER_YN_OFFER_COUNT_30D",
        "CENTER_POSITIVE_RESPONSE_RATE_30D",
        "CENTER_MEAN_ACCEPTED_SEQUENCE_30D",
        "OPO_CENTER_PAIR_YN_OFFER_COUNT_30D",
        "OPO_CENTER_PAIR_POSITIVE_RESPONSE_RATE_30D",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def recommended_duckdb_memory_limit() -> str:
    fallback_gib = 16
    try:
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return f"{fallback_gib}GiB"

    total_gib = (page_count * page_size) / (1024**3)
    recommended_gib = max(8, min(18, int(total_gib * 0.67)))
    return f"{recommended_gib}GiB"


def response_rate_sql(positive_expr: str, negative_expr: str) -> str:
    return (
        f"ROUND(({positive_expr}) * 1.0 / NULLIF(({positive_expr}) + ({negative_expr}), 0), 6)"
    )


class ProgressTracker:
    def __init__(self, total_steps: int, width: int = 28) -> None:
        self.total_steps = max(1, total_steps)
        self.width = width
        self.current_step = 0

    def advance(self, label: str) -> None:
        self.current_step += 1
        filled = min(self.width, int(self.width * self.current_step / self.total_steps))
        bar = "#" * filled + "-" * (self.width - filled)
        print(
            f"[progress] [{bar}] {self.current_step}/{self.total_steps} {label}",
            flush=True,
        )


_ACTIVE_PROGRESS_TRACKER: ProgressTracker | None = None


def total_build_steps(
    year_count: int,
    skip_opo_center_pair_mm_bucket: bool = False,
    skip_opo_center_pair_history: bool = False,
) -> int:
    fixed_steps = 22
    year_partitioned_tables = 22
    if skip_opo_center_pair_history:
        year_partitioned_tables -= 6
    elif skip_opo_center_pair_mm_bucket:
        year_partitioned_tables -= 1
    return fixed_steps + (year_partitioned_tables * year_count)


RESUME_FROM_CENTER_HISTORY = "analytics.listing_center_offer_history_exact"
RESUME_FROM_CENTER_HISTORY_COMPLETED_STEPS = 54
RESUME_FROM_PAIR_LONG_DISTANCE_HISTORY = "analytics.opo_center_pair_long_distance_history_exact"
RESUME_FROM_PAIR_LONG_DISTANCE_HISTORY_COMPLETED_STEPS = 176


def range_window_sql(partition_cols: list[str], order_col: str, interval_sql: str) -> str:
    return (
        f"PARTITION BY {', '.join(partition_cols)} "
        f"ORDER BY {order_col} "
        f"RANGE BETWEEN {interval_sql} PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING"
    )


def rows_window_sql(partition_cols: list[str], order_cols: list[str]) -> str:
    return (
        f"PARTITION BY {', '.join(partition_cols)} "
        f"ORDER BY {', '.join(order_cols)} "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
    )


def history_year_bounds(year: int) -> dict[str, str]:
    return {
        "year": year,
        "history_start_ts": f"TIMESTAMP '{year - 1}-01-01 00:00:00'",
        "year_start_ts": f"TIMESTAMP '{year}-01-01 00:00:00'",
        "year_end_ts": f"TIMESTAMP '{year + 1}-01-01 00:00:00'",
    }


def relation_columns(
    con: duckdb.DuckDBPyConnection,
    schema_name: str,
    relation_name: str,
) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema_name, relation_name],
        ).fetchall()
    }


def relation_exists(
    con: duckdb.DuckDBPyConnection,
    schema_name: str,
    relation_name: str,
) -> bool:
    return (
        con.execute(
            """
            SELECT COUNT(*)
            FROM duckdb_tables()
            WHERE schema_name = ?
              AND table_name = ?
            """,
            [schema_name, relation_name],
        ).fetchone()[0]
        > 0
        or con.execute(
            """
            SELECT COUNT(*)
            FROM duckdb_views()
            WHERE schema_name = ?
              AND view_name = ?
            """,
            [schema_name, relation_name],
        ).fetchone()[0]
        > 0
    )


def offer_feature_base_exact_row_counts(
    con: duckdb.DuckDBPyConnection,
) -> tuple[int, int]:
    match_run_count = int(con.execute("SELECT COUNT(*) FROM match_runs").fetchone()[0])
    enriched_base_count = int(
        con.execute("SELECT COUNT(*) FROM analytics.offer_feature_base_exact").fetchone()[0]
    )
    return match_run_count, enriched_base_count


def validate_offer_feature_base_exact_row_count(con: duckdb.DuckDBPyConnection) -> None:
    match_run_count, enriched_base_count = offer_feature_base_exact_row_counts(con)
    if match_run_count != enriched_base_count:
        raise RuntimeError(
            "analytics.offer_feature_base_exact row count does not match match_runs: "
            f"{enriched_base_count} != {match_run_count}"
        )


def validate_match_runs_lookup_row_count(con: duckdb.DuckDBPyConnection) -> None:
    match_run_count = int(con.execute("SELECT COUNT(*) FROM match_runs").fetchone()[0])
    lookup_count = int(con.execute("SELECT COUNT(*) FROM match_runs_lookup").fetchone()[0])
    if match_run_count != lookup_count:
        raise RuntimeError(
            "match_runs_lookup row count does not match match_runs: "
            f"{lookup_count} != {match_run_count}. Rebuild the match-run lakehouse."
        )


def validate_match_offer_enriched_schema(con: duckdb.DuckDBPyConnection) -> None:
    available_columns = relation_columns(con, "analytics", "match_offer_enriched")
    missing_columns = sorted(REQUIRED_MATCH_OFFER_ENRICHED_COLUMNS - available_columns)
    if missing_columns:
        raise RuntimeError(
            "analytics.match_offer_enriched is missing required columns. "
            f"Rebuild the linked query layer with the current builder. Missing: {missing_columns}"
        )


def build_match_offer_enriched_sql() -> str:
    return dedent(
        """
        SELECT
            b.* EXCLUDE(
                MATCH_RUN_ROWID,
                OFFER_SORT_TS,
                CAN_LISTING_CTR_ID,
                IS_POSITIVE_RESPONSE,
                IS_NEGATIVE_RESPONSE,
                IS_YN_RESPONSE
            ),
            match_opo.ENTIRE_NAME AS MATCH_OPO_NAME,
            match_opo.PRIMARY_STATE AS MATCH_OPO_STATE,
            listing_ctr.ENTIRE_NAME AS CAN_LISTING_CENTER_NAME,
            listing_ctr.PRIMARY_STATE AS CAN_LISTING_CENTER_STATE,
            ch.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            oh.* EXCLUDE(MATCH_ID),
            lch.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            pair_hist.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL)
        FROM analytics.offer_feature_base_exact AS b
        LEFT JOIN saf_link.center_code_dim AS match_opo
            ON b.MATCH_OPO_CTR_CD = match_opo.CTR_CD
        LEFT JOIN saf_link.center_dim AS listing_ctr
            ON b.CAN_LISTING_CTR_CD = listing_ctr.CTR_CD
           AND COALESCE(b.CAN_LISTING_CTR_TY, '') = COALESCE(listing_ctr.CTR_TY, '')
        LEFT JOIN analytics.candidate_history_exact AS ch
            ON b.MATCH_RUN_ROWID = ch.MATCH_RUN_ROWID
        LEFT JOIN analytics.opo_history_exact AS oh
            ON b.MATCH_ID = oh.MATCH_ID
        LEFT JOIN analytics.listing_center_history_exact AS lch
            ON b.MATCH_RUN_ROWID = lch.MATCH_RUN_ROWID
        LEFT JOIN analytics.opo_center_pair_history_exact AS pair_hist
            ON b.MATCH_RUN_ROWID = pair_hist.MATCH_RUN_ROWID
        """
    )


def build_match_offer_to_transplant_sql() -> str:
    return dedent(
        """
        SELECT
            o.*,
            b.DON_DISPOSITION,
            b.DON_REASON_CD,
            b.DON_TX_CTR_ID,
            b.DON_DISCARD_CD,
            b.DON_SHARE_TY,
            b.TRR_ID,
            b.REC_TX_DT,
            b.REC_CTR_CD,
            b.REC_CTR_TY,
            b.REC_OPO_ID,
            b.LATEST_TFL_PX_STAT,
            b.LATEST_TFL_PX_STAT_DT,
            b.LATEST_TFL_CREAT
        FROM analytics.match_offer_enriched AS o
        LEFT JOIN analytics.match_transplant_bridge AS b
            ON o.MATCH_ID = b.MATCH_ID
           AND o.DONOR_ID = b.DONOR_ID
           AND o.PX_ID = b.PX_ID
        """
    )


def build_candidate_tx_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    tx_filter = ""
    offer_filter = ""
    output_filter = "IS_OFFER_EVENT = 1"
    offer_match_year_expr = "CAST(NULL AS INTEGER)"
    if year is not None:
        tx_filter = f"""
              AND REC_TX_DT >= {history_start_ts}
              AND REC_TX_DT < {year_end_ts}
        """
        offer_filter = f"""
              AND match_year BETWEEN {year - 1} AND {year}
        """
        output_filter = f"IS_OFFER_EVENT = 1 AND match_year = {year}"
        offer_match_year_expr = "match_year"

    window_columns: list[str] = []
    window_aliases: list[str] = []
    for suffix, interval_sql in HISTORY_WINDOWS:
        frame = range_window_sql(["PX_ID"], "EVENT_TS", interval_sql)
        alias = f"CAND_PRIOR_TX_COUNT_{suffix}"
        window_aliases.append(alias)
        window_columns.append(f"COUNT(*) FILTER (WHERE IS_TX_EVENT = 1) OVER ({frame}) AS {alias}")

    return dedent(
        f"""
        WITH stream AS (
            SELECT
                CAST(PX_ID AS BIGINT) AS PX_ID,
                REC_TX_DT AS EVENT_TS,
                CAST(NULL AS INTEGER) AS match_year,
                CAST(NULL AS BIGINT) AS MATCH_RUN_ROWID,
                CAST(NULL AS INTEGER) AS MATCH_ID,
                CAST(NULL AS INTEGER) AS PTR_ROW_ORDER,
                CAST(NULL AS INTEGER) AS OFFER_ROW_INSTANCE_ORDINAL,
                0 AS IS_OFFER_EVENT,
                1 AS IS_TX_EVENT
            FROM saf_link.tx_ki_link
            WHERE PX_ID IS NOT NULL
              AND REC_TX_DT IS NOT NULL
              {tx_filter}

            UNION ALL

            SELECT
                PX_ID,
                MATCH_SUBMIT_DT AS EVENT_TS,
                {offer_match_year_expr} AS match_year,
                MATCH_RUN_ROWID,
                MATCH_ID,
                PTR_ROW_ORDER,
                OFFER_ROW_INSTANCE_ORDINAL,
                1 AS IS_OFFER_EVENT,
                0 AS IS_TX_EVENT
            FROM analytics.offer_feature_base_exact
            WHERE PX_ID IS NOT NULL
              AND MATCH_SUBMIT_DT IS NOT NULL
              {offer_filter}
        ),
        scored AS (
            SELECT
                match_year,
                MATCH_RUN_ROWID,
                MATCH_ID,
                PTR_ROW_ORDER,
                OFFER_ROW_INSTANCE_ORDINAL,
                IS_OFFER_EVENT,
                {", ".join(window_columns)}
            FROM stream
        )
        SELECT
            MATCH_RUN_ROWID,
            MATCH_ID,
            PTR_ROW_ORDER,
            OFFER_ROW_INSTANCE_ORDINAL,
            {", ".join(window_aliases)}
        FROM scored
        WHERE {output_filter}
        """
    )


def build_candidate_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    yn_rows_frame = rows_window_sql(["PX_ID"], ["OFFER_SORT_TS"])
    same_match_decliner_frame = rows_window_sql(
        ["MATCH_ID"],
        ["PTR_SEQUENCE_NUM", "PTR_ROW_ORDER", "OFFER_ROW_INSTANCE_ORDINAL"],
    )
    input_filter = ""
    output_filter = ""
    match_year_select = "CAST(NULL AS INTEGER) AS match_year,"
    if year is not None:
        input_filter = f"WHERE match_year BETWEEN {year - 1} AND {year}"
        output_filter = f"WHERE r.match_year = {year}"
        match_year_select = "match_year,"

    window_columns: list[str] = []
    for suffix, interval_sql in HISTORY_WINDOWS:
        frame = range_window_sql(["PX_ID"], "OFFER_SORT_TS", interval_sql)
        window_columns.extend(
            [
                f"COUNT(*) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINE_COUNT_{suffix}",
                f"AVG(KDPI) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_KDPI_AVG_{suffix}",
                f"STDDEV_SAMP(KDPI) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_KDPI_STDDEV_{suffix}",
                f"AVG(DON_CREAT) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_DON_CREAT_AVG_{suffix}",
                f"STDDEV_SAMP(DON_CREAT) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_DON_CREAT_STDDEV_{suffix}",
                f"AVG(MM_TOTAL) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_MM_TOTAL_AVG_{suffix}",
                f"STDDEV_SAMP(MM_TOTAL) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_MM_TOTAL_STDDEV_{suffix}",
                f"AVG(DON_AGE) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_DON_AGE_AVG_{suffix}",
                f"STDDEV_SAMP(DON_AGE) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_DON_AGE_STDDEV_{suffix}",
                f"AVG(CASE WHEN DCD_IND = 1 THEN 1.0 WHEN DCD_IND = 0 THEN 0.0 ELSE NULL END) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_DCD_FRAC_{suffix}",
                f"AVG(CASE WHEN HCV_POSITIVE_FLG = 1 THEN 1.0 WHEN HCV_POSITIVE_FLG = 0 THEN 0.0 ELSE NULL END) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({frame}) AS CAND_DECLINED_HCV_FRAC_{suffix}",
            ]
        )

    return dedent(
        f"""
        WITH filtered_input AS (
            SELECT
                MATCH_RUN_ROWID,
                MATCH_ID,
                PTR_ROW_ORDER,
                OFFER_ROW_INSTANCE_ORDINAL,
                {match_year_select}
                MATCH_SUBMIT_DT,
                OFFER_SORT_TS,
                PX_ID,
                PTR_SEQUENCE_NUM,
                IS_NEGATIVE_RESPONSE,
                IS_YN_RESPONSE,
                KDPI,
                KDPI_BIN,
                DON_CREAT,
                MM_TOTAL,
                DON_AGE,
                DCD_IND,
                HCV_POSITIVE_FLG
            FROM analytics.offer_feature_base_exact
            {input_filter}
        ),
        raw_history AS (
            SELECT
                MATCH_RUN_ROWID,
                MATCH_ID,
                PTR_ROW_ORDER,
                OFFER_ROW_INSTANCE_ORDINAL,
                match_year,
                MATCH_SUBMIT_DT,
                ARG_MAX(MATCH_SUBMIT_DT, OFFER_SORT_TS) FILTER (WHERE IS_YN_RESPONSE = 1) OVER ({yn_rows_frame}) AS LAST_YN_OFFER_MATCH_SUBMIT_DT,
                ARG_MAX(KDPI_BIN, OFFER_SORT_TS) FILTER (WHERE IS_YN_RESPONSE = 1) OVER ({yn_rows_frame}) AS LAST_YN_OFFER_KDPI_BIN,
                COUNT(DISTINCT PX_ID) FILTER (WHERE IS_NEGATIVE_RESPONSE = 1) OVER ({same_match_decliner_frame}) AS SAME_MATCH_PRIOR_DECLINER_COUNT,
                {", ".join(window_columns)}
            FROM filtered_input
        )
        SELECT
            r.MATCH_RUN_ROWID,
            r.MATCH_ID,
            r.PTR_ROW_ORDER,
            r.OFFER_ROW_INSTANCE_ORDINAL,
            tx.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            r.* EXCLUDE(
                MATCH_RUN_ROWID,
                MATCH_ID,
                PTR_ROW_ORDER,
                OFFER_ROW_INSTANCE_ORDINAL,
                match_year,
                MATCH_SUBMIT_DT,
                LAST_YN_OFFER_MATCH_SUBMIT_DT
            ),
            CASE
                WHEN r.LAST_YN_OFFER_MATCH_SUBMIT_DT IS NULL THEN NULL
                ELSE ROUND(DATE_DIFF('second', r.LAST_YN_OFFER_MATCH_SUBMIT_DT, r.MATCH_SUBMIT_DT) / 86400.0, 6)
            END AS TIME_SINCE_LAST_OFFER_DAYS
        FROM raw_history AS r
        LEFT JOIN analytics.candidate_tx_history_exact AS tx
            ON r.MATCH_RUN_ROWID = tx.MATCH_RUN_ROWID
        {output_filter}
        """
    )


def build_opo_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    input_filter = ""
    output_filter = ""
    if year is not None:
        input_filter = f"""
            WHERE MATCH_SUBMIT_DT >= {history_start_ts}
              AND MATCH_SUBMIT_DT < {year_end_ts}
        """
        output_filter = f"""
        WHERE MATCH_SUBMIT_DT >= {year_start_ts}
          AND MATCH_SUBMIT_DT < {year_end_ts}
        """

    window_columns: list[str] = []
    for suffix, interval_sql in HISTORY_WINDOWS:
        base_frame = range_window_sql(["MATCH_OPO_CTR_CD"], "MATCH_SORT_TS", interval_sql)
        kdpi_frame = range_window_sql(["MATCH_OPO_CTR_CD", "KDPI_BIN"], "MATCH_SORT_TS", interval_sql)
        window_columns.extend(
            [
                f"AVG(CASE WHEN DCD_IND = 1 THEN 1.0 WHEN DCD_IND = 0 THEN 0.0 ELSE NULL END) OVER ({base_frame}) AS OPO_HIST_DCD_FRAC_{suffix}",
                f"AVG(CASE WHEN AT_LEAST_ONE_PLACED_FLG = 1 THEN 1.0 WHEN AT_LEAST_ONE_PLACED_FLG = 0 THEN 0.0 ELSE NULL END) OVER ({base_frame}) AS OPO_HIST_ANY_PLACED_FRAC_{suffix}",
                f"AVG(CASE WHEN BOTH_PLACED_FLG = 1 THEN 1.0 WHEN BOTH_PLACED_FLG = 0 THEN 0.0 ELSE NULL END) OVER ({base_frame}) AS OPO_HIST_BOTH_PLACED_FRAC_{suffix}",
                f"AVG(CASE WHEN BOTH_WASTED_FLG = 1 THEN 1.0 WHEN BOTH_WASTED_FLG = 0 THEN 0.0 ELSE NULL END) OVER ({base_frame}) AS OPO_HIST_BOTH_WASTED_FRAC_{suffix}",
                f"AVG(CASE WHEN OUT_OF_SEQUENCE_RUN_FLG = 1 THEN 1.0 WHEN OUT_OF_SEQUENCE_RUN_FLG = 0 THEN 0.0 ELSE NULL END) OVER ({base_frame}) AS OPO_HIST_OUT_OF_SEQUENCE_FRAC_{suffix}",
                f"AVG(CASE WHEN AT_LEAST_ONE_PLACED_FLG = 1 THEN 1.0 WHEN AT_LEAST_ONE_PLACED_FLG = 0 THEN 0.0 ELSE NULL END) OVER ({kdpi_frame}) AS OPO_HIST_KDPI_BIN_PLACEMENT_RATE_{suffix}",
                f"AVG(FIRST_ACCEPT_DECLINE_COUNT) FILTER (WHERE AT_LEAST_ONE_PLACED_FLG = 1) OVER ({base_frame}) AS OPO_HIST_MEAN_FIRST_ACCEPT_DECLINES_{suffix}",
                f"AVG(RUN_LEN * 1.0) OVER ({base_frame}) AS OPO_HIST_MEAN_RUN_LEN_{suffix}",
            ]
        )

    return dedent(
        f"""
        WITH filtered_summary AS (
            SELECT *
            FROM analytics.match_run_summary_exact
            {input_filter}
        )
        SELECT
            MATCH_ID,
            {", ".join(window_columns)}
        FROM filtered_summary
        {output_filter}
        """
    )


def exact_rate_expr(positive_expr: str, negative_expr: str) -> str:
    total_expr = f"(({positive_expr}) + ({negative_expr}))"
    return (
        f"CASE WHEN {total_expr} = 0 THEN NULL "
        f"ELSE ROUND(({positive_expr}) * 1.0 / NULLIF({total_expr}, 0), 6) END"
    )


def build_exact_response_rate_history_pass_ctes(
    *,
    cte_prefix: str,
    input_cte: str,
    timestamp_column: str,
    partition_columns: list[str],
    row_filter: str = "",
    year: int | None = None,
    count_alias_template: str | None = None,
    rate_alias_template: str,
) -> tuple[list[str], str]:
    row_key_columns = [
        "MATCH_RUN_ROWID",
        "MATCH_ID",
        "PTR_ROW_ORDER",
        "OFFER_ROW_INSTANCE_ORDINAL",
        "match_year",
    ]
    rows_select_columns = row_key_columns + [
        timestamp_column,
        *partition_columns,
        "IS_POSITIVE_RESPONSE",
        "IS_NEGATIVE_RESPONSE",
    ]
    group_columns = partition_columns + [timestamp_column]
    partition_predicate = " AND ".join(f"cur.{col} = bound.{col}" for col in partition_columns)
    row_join_predicate = " AND ".join(f"rows.{col} = hist.{col}" for col in partition_columns)
    target_filter = f"WHERE rows.match_year = {year}" if year is not None else ""

    ctes = [
        dedent(
            f"""
            {cte_prefix}_rows AS (
                SELECT
                    {", ".join(rows_select_columns)}
                FROM {input_cte}
                {row_filter}
            )
            """
        ),
        dedent(
            f"""
            {cte_prefix}_events AS (
                SELECT
                    {", ".join(group_columns)},
                    SUM(IS_POSITIVE_RESPONSE) AS POS_AT_TS,
                    SUM(IS_NEGATIVE_RESPONSE) AS NEG_AT_TS
                FROM {cte_prefix}_rows
                GROUP BY {", ".join(group_columns)}
            )
            """
        ),
        dedent(
            f"""
            {cte_prefix}_prefix AS (
                SELECT
                    {", ".join(group_columns)},
                    COALESCE(
                        SUM(POS_AT_TS) OVER (
                            PARTITION BY {", ".join(partition_columns)}
                            ORDER BY {timestamp_column}
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        ),
                        0
                    ) AS POS_BEFORE_TS,
                    COALESCE(
                        SUM(NEG_AT_TS) OVER (
                            PARTITION BY {", ".join(partition_columns)}
                            ORDER BY {timestamp_column}
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        ),
                        0
                    ) AS NEG_BEFORE_TS,
                    SUM(POS_AT_TS) OVER (
                        PARTITION BY {", ".join(partition_columns)}
                        ORDER BY {timestamp_column}
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS POS_THROUGH_TS,
                    SUM(NEG_AT_TS) OVER (
                        PARTITION BY {", ".join(partition_columns)}
                        ORDER BY {timestamp_column}
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS NEG_THROUGH_TS
                FROM {cte_prefix}_events
            )
            """
        ),
    ]

    previous_cte = f"{cte_prefix}_prefix"
    for suffix, interval_sql in HISTORY_WINDOWS:
        cte_name = f"{cte_prefix}_{suffix.lower()}"
        ctes.append(
            dedent(
                f"""
                {cte_name} AS (
                    SELECT
                        cur.*,
                        cur.POS_BEFORE_TS - COALESCE(bound.POS_THROUGH_TS, 0) AS POS_{suffix},
                        cur.NEG_BEFORE_TS - COALESCE(bound.NEG_THROUGH_TS, 0) AS NEG_{suffix}
                    FROM {previous_cte} AS cur
                    ASOF LEFT JOIN {cte_prefix}_prefix AS bound
                      ON {partition_predicate}
                     AND cur.{timestamp_column} - {interval_sql} - INTERVAL 1 MICROSECOND >= bound.{timestamp_column}
                )
                """
            )
        )
        previous_cte = cte_name

    feature_columns: list[str] = []
    for suffix, _ in HISTORY_WINDOWS:
        if count_alias_template is not None:
            feature_columns.append(
                f"POS_{suffix} + NEG_{suffix} AS {count_alias_template.format(suffix=suffix)}"
            )
        feature_columns.append(
            f"{exact_rate_expr(f'POS_{suffix}', f'NEG_{suffix}')} "
            f"AS {rate_alias_template.format(suffix=suffix)}"
        )

    ctes.append(
        dedent(
            f"""
            {cte_prefix}_by_ts AS (
                SELECT
                    {", ".join(group_columns)},
                    {", ".join(feature_columns)}
                FROM {previous_cte}
            )
            """
        )
    )
    ctes.append(
        dedent(
            f"""
            {cte_prefix}_output AS (
                SELECT
                    rows.MATCH_RUN_ROWID,
                    rows.MATCH_ID,
                    rows.PTR_ROW_ORDER,
                    rows.OFFER_ROW_INSTANCE_ORDINAL,
                    hist.* EXCLUDE({", ".join(partition_columns + [timestamp_column])})
                FROM {cte_prefix}_rows AS rows
                LEFT JOIN {cte_prefix}_by_ts AS hist
                  ON {row_join_predicate}
                 AND rows.{timestamp_column} = hist.{timestamp_column}
                {target_filter}
            )
            """
        )
    )
    return ctes, f"{cte_prefix}_output"


def build_listing_center_offer_history_pass_sql(
    *,
    cte_prefix: str,
    partition_columns: list[str],
    rate_alias_template: str,
    row_filter: str = "",
    count_alias_template: str | None = None,
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    input_filter = ""
    match_year_expr = "CAST(NULL AS INTEGER) AS match_year"
    if year is not None:
        input_filter = f"AND match_year BETWEEN {year - 1} AND {year}"
        match_year_expr = "match_year"

    ctes = [
        dedent(
            f"""
            filtered_input AS (
                SELECT
                    MATCH_RUN_ROWID,
                    MATCH_ID,
                    PTR_ROW_ORDER,
                    OFFER_ROW_INSTANCE_ORDINAL,
                    {match_year_expr},
                    OFFER_SORT_TS,
                    CAN_LISTING_CTR_CD,
                    COALESCE(CAN_LISTING_CTR_TY, '') AS CAN_LISTING_CTR_TY_NORM,
                    DCD_IND,
                    HIGH_KDPI_FLG,
                    HCV_POSITIVE_FLG,
                    LONG_DISTANCE_FLG,
                    MM_TOTAL_BUCKET,
                    CAST(IS_POSITIVE_RESPONSE AS BIGINT) AS IS_POSITIVE_RESPONSE,
                    CAST(IS_NEGATIVE_RESPONSE AS BIGINT) AS IS_NEGATIVE_RESPONSE
                FROM analytics.offer_feature_base_exact
                WHERE CAN_LISTING_CTR_CD IS NOT NULL
                  {input_filter}
            )
            """
        )
    ]
    pass_ctes, output_cte = build_exact_response_rate_history_pass_ctes(
        cte_prefix=cte_prefix,
        input_cte="filtered_input",
        timestamp_column="OFFER_SORT_TS",
        partition_columns=partition_columns,
        row_filter=row_filter,
        year=year,
        count_alias_template=count_alias_template,
        rate_alias_template=rate_alias_template,
    )
    ctes.extend(pass_ctes)
    return dedent(
        f"""
        WITH
        {",\n".join(ctes)}
        SELECT * FROM {output_cte}
        """
    )


def build_listing_center_offer_base_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_listing_center_offer_history_pass_sql(
        cte_prefix="center_base",
        partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM"],
        count_alias_template="CENTER_YN_OFFER_COUNT_{suffix}",
        rate_alias_template="CENTER_POSITIVE_RESPONSE_RATE_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_listing_center_offer_dcd_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_listing_center_offer_history_pass_sql(
        cte_prefix="center_dcd",
        partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "DCD_IND"],
        row_filter="WHERE DCD_IND IS NOT NULL",
        rate_alias_template="CENTER_RATE_SAME_DCD_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_listing_center_offer_high_kdpi_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_listing_center_offer_history_pass_sql(
        cte_prefix="center_high_kdpi",
        partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "HIGH_KDPI_FLG"],
        row_filter="WHERE HIGH_KDPI_FLG IS NOT NULL",
        rate_alias_template="CENTER_RATE_SAME_HIGH_KDPI_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_listing_center_offer_hcv_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_listing_center_offer_history_pass_sql(
        cte_prefix="center_hcv",
        partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "HCV_POSITIVE_FLG"],
        row_filter="WHERE HCV_POSITIVE_FLG IS NOT NULL",
        rate_alias_template="CENTER_RATE_SAME_HCV_POS_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_listing_center_offer_long_distance_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_listing_center_offer_history_pass_sql(
        cte_prefix="center_long_distance",
        partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "LONG_DISTANCE_FLG"],
        row_filter="WHERE LONG_DISTANCE_FLG IS NOT NULL",
        rate_alias_template="CENTER_RATE_SAME_LONG_DISTANCE_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_listing_center_offer_mm_bucket_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_listing_center_offer_history_pass_sql(
        cte_prefix="center_mm_bucket",
        partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "MM_TOTAL_BUCKET"],
        row_filter="WHERE MM_TOTAL_BUCKET IS NOT NULL",
        rate_alias_template="CENTER_RATE_SAME_MM_BUCKET_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_opo_center_pair_history_pass_sql(
    *,
    cte_prefix: str,
    partition_columns: list[str],
    rate_alias_template: str,
    row_filter: str = "",
    count_alias_template: str | None = None,
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    input_filter = ""
    match_year_expr = "CAST(NULL AS INTEGER) AS match_year"
    if year is not None:
        input_filter = f"AND match_year BETWEEN {year - 1} AND {year}"
        match_year_expr = "match_year"

    ctes = [
        dedent(
            f"""
            filtered_input AS (
                SELECT
                    MATCH_RUN_ROWID,
                    MATCH_ID,
                    PTR_ROW_ORDER,
                    OFFER_ROW_INSTANCE_ORDINAL,
                    {match_year_expr},
                    OFFER_SORT_TS,
                    MATCH_OPO_CTR_CD,
                    CAN_LISTING_CTR_CD,
                    COALESCE(CAN_LISTING_CTR_TY, '') AS CAN_LISTING_CTR_TY_NORM,
                    DCD_IND,
                    HIGH_KDPI_FLG,
                    HCV_POSITIVE_FLG,
                    LONG_DISTANCE_FLG,
                    MM_TOTAL_BUCKET,
                    CAST(IS_POSITIVE_RESPONSE AS BIGINT) AS IS_POSITIVE_RESPONSE,
                    CAST(IS_NEGATIVE_RESPONSE AS BIGINT) AS IS_NEGATIVE_RESPONSE
                FROM analytics.offer_feature_base_exact
                WHERE MATCH_OPO_CTR_CD IS NOT NULL
                  AND CAN_LISTING_CTR_CD IS NOT NULL
                  {input_filter}
            )
            """
        )
    ]
    pass_ctes, output_cte = build_exact_response_rate_history_pass_ctes(
        cte_prefix=cte_prefix,
        input_cte="filtered_input",
        timestamp_column="OFFER_SORT_TS",
        partition_columns=partition_columns,
        row_filter=row_filter,
        year=year,
        count_alias_template=count_alias_template,
        rate_alias_template=rate_alias_template,
    )
    ctes.extend(pass_ctes)
    return dedent(
        f"""
        WITH
        {",\n".join(ctes)}
        SELECT * FROM {output_cte}
        """
    )


def build_opo_center_pair_base_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_opo_center_pair_history_pass_sql(
        cte_prefix="pair_base",
        partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM"],
        count_alias_template="OPO_CENTER_PAIR_YN_OFFER_COUNT_{suffix}",
        rate_alias_template="OPO_CENTER_PAIR_POSITIVE_RESPONSE_RATE_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_opo_center_pair_dcd_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_opo_center_pair_history_pass_sql(
        cte_prefix="pair_dcd",
        partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "DCD_IND"],
        row_filter="WHERE DCD_IND IS NOT NULL",
        rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_DCD_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_opo_center_pair_high_kdpi_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_opo_center_pair_history_pass_sql(
        cte_prefix="pair_high_kdpi",
        partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "HIGH_KDPI_FLG"],
        row_filter="WHERE HIGH_KDPI_FLG IS NOT NULL",
        rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_HIGH_KDPI_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_opo_center_pair_hcv_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_opo_center_pair_history_pass_sql(
        cte_prefix="pair_hcv",
        partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "HCV_POSITIVE_FLG"],
        row_filter="WHERE HCV_POSITIVE_FLG IS NOT NULL",
        rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_HCV_POS_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_opo_center_pair_long_distance_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_opo_center_pair_history_pass_sql(
        cte_prefix="pair_long_distance",
        partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "LONG_DISTANCE_FLG"],
        row_filter="WHERE LONG_DISTANCE_FLG IS NOT NULL",
        rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_LONG_DISTANCE_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_opo_center_pair_mm_bucket_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    return build_opo_center_pair_history_pass_sql(
        cte_prefix="pair_mm_bucket",
        partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "MM_TOTAL_BUCKET"],
        row_filter="WHERE MM_TOTAL_BUCKET IS NOT NULL",
        rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_MM_BUCKET_{suffix}",
        year=year,
        history_start_ts=history_start_ts,
        year_start_ts=year_start_ts,
        year_end_ts=year_end_ts,
    )


def build_listing_center_offer_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    input_filter = ""
    match_year_expr = "CAST(NULL AS INTEGER) AS match_year"
    if year is not None:
        input_filter = f"AND match_year BETWEEN {year - 1} AND {year}"
        match_year_expr = "match_year"

    ctes = [
        dedent(
            f"""
            filtered_input AS (
                SELECT
                    MATCH_RUN_ROWID,
                    MATCH_ID,
                    PTR_ROW_ORDER,
                    OFFER_ROW_INSTANCE_ORDINAL,
                    {match_year_expr},
                    OFFER_SORT_TS,
                    CAN_LISTING_CTR_CD,
                    COALESCE(CAN_LISTING_CTR_TY, '') AS CAN_LISTING_CTR_TY_NORM,
                    DCD_IND,
                    HIGH_KDPI_FLG,
                    HCV_POSITIVE_FLG,
                    LONG_DISTANCE_FLG,
                    MM_TOTAL_BUCKET,
                    CAST(IS_POSITIVE_RESPONSE AS BIGINT) AS IS_POSITIVE_RESPONSE,
                    CAST(IS_NEGATIVE_RESPONSE AS BIGINT) AS IS_NEGATIVE_RESPONSE
                FROM analytics.offer_feature_base_exact
                WHERE CAN_LISTING_CTR_CD IS NOT NULL
                  {input_filter}
            )
            """
        )
    ]

    for cte_sql, _ in [
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="center_base",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM"],
            year=year,
            count_alias_template="CENTER_YN_OFFER_COUNT_{suffix}",
            rate_alias_template="CENTER_POSITIVE_RESPONSE_RATE_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="center_dcd",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "DCD_IND"],
            row_filter="WHERE DCD_IND IS NOT NULL",
            year=year,
            rate_alias_template="CENTER_RATE_SAME_DCD_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="center_high_kdpi",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "HIGH_KDPI_FLG"],
            row_filter="WHERE HIGH_KDPI_FLG IS NOT NULL",
            year=year,
            rate_alias_template="CENTER_RATE_SAME_HIGH_KDPI_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="center_hcv",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "HCV_POSITIVE_FLG"],
            row_filter="WHERE HCV_POSITIVE_FLG IS NOT NULL",
            year=year,
            rate_alias_template="CENTER_RATE_SAME_HCV_POS_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="center_long_distance",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "LONG_DISTANCE_FLG"],
            row_filter="WHERE LONG_DISTANCE_FLG IS NOT NULL",
            year=year,
            rate_alias_template="CENTER_RATE_SAME_LONG_DISTANCE_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="center_mm_bucket",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "MM_TOTAL_BUCKET"],
            row_filter="WHERE MM_TOTAL_BUCKET IS NOT NULL",
            year=year,
            rate_alias_template="CENTER_RATE_SAME_MM_BUCKET_{suffix}",
        ),
    ]:
        ctes.extend(cte_sql)

    return dedent(
        f"""
        WITH
        {",\n".join(ctes)}
        SELECT
            base.MATCH_RUN_ROWID,
            base.MATCH_ID,
            base.PTR_ROW_ORDER,
            base.OFFER_ROW_INSTANCE_ORDINAL,
            base.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            dcd.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            high_kdpi.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            hcv.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            long_distance.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            mm_bucket.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL)
        FROM center_base_output AS base
        LEFT JOIN center_dcd_output AS dcd
          ON base.MATCH_RUN_ROWID = dcd.MATCH_RUN_ROWID
        LEFT JOIN center_high_kdpi_output AS high_kdpi
          ON base.MATCH_RUN_ROWID = high_kdpi.MATCH_RUN_ROWID
        LEFT JOIN center_hcv_output AS hcv
          ON base.MATCH_RUN_ROWID = hcv.MATCH_RUN_ROWID
        LEFT JOIN center_long_distance_output AS long_distance
          ON base.MATCH_RUN_ROWID = long_distance.MATCH_RUN_ROWID
        LEFT JOIN center_mm_bucket_output AS mm_bucket
          ON base.MATCH_RUN_ROWID = mm_bucket.MATCH_RUN_ROWID
        """
    )


def build_listing_center_acceptance_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    acceptance_filter = ""
    offer_filter = ""
    target_filter = ""
    offer_match_year_expr = "CAST(NULL AS INTEGER) AS match_year"
    if year is not None:
        acceptance_filter = f"""
              AND s.MATCH_SUBMIT_DT >= {history_start_ts}
              AND s.MATCH_SUBMIT_DT < {year_end_ts}
        """
        offer_filter = f"""
              AND match_year BETWEEN {year - 1} AND {year}
        """
        target_filter = f"WHERE offers.match_year = {year}"
        offer_match_year_expr = "match_year"

    prefix_metric_columns = [
        ("ACCEPTED_SEQUENCE_SUM", "ACCEPTED_SEQUENCE_SUM_AT_TS"),
        ("ACCEPTED_NORMALIZED_SEQUENCE_SUM", "ACCEPTED_NORMALIZED_SEQUENCE_SUM_AT_TS"),
        ("LATE_PLACEMENT_SUM", "LATE_PLACEMENT_SUM_AT_TS"),
        ("ACCEPTANCE_EVENT_COUNT", "ACCEPTANCE_EVENT_COUNT_AT_TS"),
    ]

    metric_window_columns: list[str] = []
    metric_through_columns: list[str] = []
    for metric_prefix, source_alias in prefix_metric_columns:
        metric_window_columns.extend(
            [
                (
                    f"COALESCE(SUM({source_alias}) OVER ("
                    "PARTITION BY CENTER_CD, CENTER_TY "
                    "ORDER BY EVENT_TS "
                    "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
                    f"), 0) AS {metric_prefix}_BEFORE_TS"
                ),
                (
                    f"SUM({source_alias}) OVER ("
                    "PARTITION BY CENTER_CD, CENTER_TY "
                    "ORDER BY EVENT_TS "
                    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                    f") AS {metric_prefix}_THROUGH_TS"
                ),
            ]
        )
        metric_through_columns.append(f"{metric_prefix}_THROUGH_TS")

    interval_ctes: list[str] = []
    previous_cte = "acceptance_prefix"
    for suffix, interval_sql in HISTORY_WINDOWS:
        interval_ctes.append(
            dedent(
                f"""
                acceptance_{suffix.lower()} AS (
                    SELECT
                        cur.*,
                        cur.ACCEPTED_SEQUENCE_SUM_BEFORE_TS - COALESCE(bound.ACCEPTED_SEQUENCE_SUM_THROUGH_TS, 0) AS ACCEPTED_SEQUENCE_SUM_{suffix},
                        cur.ACCEPTED_NORMALIZED_SEQUENCE_SUM_BEFORE_TS - COALESCE(bound.ACCEPTED_NORMALIZED_SEQUENCE_SUM_THROUGH_TS, 0) AS ACCEPTED_NORMALIZED_SEQUENCE_SUM_{suffix},
                        cur.LATE_PLACEMENT_SUM_BEFORE_TS - COALESCE(bound.LATE_PLACEMENT_SUM_THROUGH_TS, 0) AS LATE_PLACEMENT_SUM_{suffix},
                        cur.ACCEPTANCE_EVENT_COUNT_BEFORE_TS - COALESCE(bound.ACCEPTANCE_EVENT_COUNT_THROUGH_TS, 0) AS ACCEPTANCE_EVENT_COUNT_{suffix}
                    FROM {previous_cte} AS cur
                    ASOF LEFT JOIN acceptance_prefix AS bound
                      ON cur.CENTER_CD = bound.CENTER_CD
                     AND cur.CENTER_TY = bound.CENTER_TY
                     AND cur.EVENT_TS - {interval_sql} - INTERVAL 1 MICROSECOND >= bound.EVENT_TS
                )
                """
            )
        )
        previous_cte = f"acceptance_{suffix.lower()}"

    average_feature_columns: list[str] = []
    for suffix, _ in HISTORY_WINDOWS:
        average_feature_columns.extend(
            [
                (
                    "CASE WHEN ACCEPTANCE_EVENT_COUNT_{suffix} = 0 THEN NULL "
                    "ELSE ROUND(ACCEPTED_SEQUENCE_SUM_{suffix} * 1.0 / NULLIF(ACCEPTANCE_EVENT_COUNT_{suffix}, 0), 6) END "
                    "AS CENTER_MEAN_ACCEPTED_SEQUENCE_{suffix}"
                ).format(suffix=suffix),
                (
                    "CASE WHEN ACCEPTANCE_EVENT_COUNT_{suffix} = 0 THEN NULL "
                    "ELSE ROUND(ACCEPTED_NORMALIZED_SEQUENCE_SUM_{suffix} * 1.0 / NULLIF(ACCEPTANCE_EVENT_COUNT_{suffix}, 0), 6) END "
                    "AS CENTER_MEAN_ACCEPTED_NORMALIZED_SEQUENCE_{suffix}"
                ).format(suffix=suffix),
                (
                    "CASE WHEN ACCEPTANCE_EVENT_COUNT_{suffix} = 0 THEN NULL "
                    "ELSE ROUND(LATE_PLACEMENT_SUM_{suffix} * 1.0 / NULLIF(ACCEPTANCE_EVENT_COUNT_{suffix}, 0), 6) END "
                    "AS CENTER_LATE_PLACEMENT_RATE_{suffix}"
                ).format(suffix=suffix),
            ]
        )

    return dedent(
        f"""
        WITH acceptance_events AS (
            SELECT
                k.ACCEPTED_CENTER_CD AS CENTER_CD,
                COALESCE(k.ACCEPTED_CENTER_TY, '') AS CENTER_TY,
                s.MATCH_SUBMIT_DT AS EVENT_TS,
                k.ACCEPTED_SEQUENCE_NUM * 1.0 AS ACCEPTED_SEQUENCE_SUM,
                k.ACCEPTED_SEQUENCE_NUM * 1.0 / NULLIF(s.RUN_LEN, 0) AS ACCEPTED_NORMALIZED_SEQUENCE_SUM,
                CASE
                    WHEN k.ACCEPTED_SEQUENCE_NUM * 1.0 / NULLIF(s.RUN_LEN, 0) > 0.5 THEN 1.0
                    ELSE 0.0
                END AS LATE_PLACEMENT_SUM,
                1 AS ACCEPTANCE_EVENT_COUNT
            FROM analytics.kidney_outcome_exact AS k
            JOIN analytics.match_run_summary_exact AS s
              ON k.MATCH_ID = s.MATCH_ID
            WHERE k.ACCEPTED_CENTER_CD IS NOT NULL
              AND k.ACCEPTED_SEQUENCE_NUM IS NOT NULL
              {acceptance_filter}
        ),
        acceptance_events_by_ts AS (
            SELECT
                CENTER_CD,
                CENTER_TY,
                EVENT_TS,
                SUM(ACCEPTED_SEQUENCE_SUM) AS ACCEPTED_SEQUENCE_SUM_AT_TS,
                SUM(ACCEPTED_NORMALIZED_SEQUENCE_SUM) AS ACCEPTED_NORMALIZED_SEQUENCE_SUM_AT_TS,
                SUM(LATE_PLACEMENT_SUM) AS LATE_PLACEMENT_SUM_AT_TS,
                SUM(ACCEPTANCE_EVENT_COUNT) AS ACCEPTANCE_EVENT_COUNT_AT_TS
            FROM acceptance_events
            GROUP BY 1, 2, 3
        ),
        acceptance_prefix AS (
            SELECT
                CENTER_CD,
                CENTER_TY,
                EVENT_TS,
                {", ".join(metric_window_columns)}
            FROM acceptance_events_by_ts
        ),
        {",\n".join(interval_ctes)},
        acceptance_by_ts AS (
            SELECT
                CENTER_CD,
                CENTER_TY,
                EVENT_TS,
                {", ".join(average_feature_columns)}
            FROM {previous_cte}
        ),
        offers AS (
            SELECT
                MATCH_RUN_ROWID,
                MATCH_ID,
                PTR_ROW_ORDER,
                OFFER_ROW_INSTANCE_ORDINAL,
                {offer_match_year_expr},
                CAN_LISTING_CTR_CD AS CENTER_CD,
                COALESCE(CAN_LISTING_CTR_TY, '') AS CENTER_TY,
                MATCH_SUBMIT_DT AS EVENT_TS
            FROM analytics.offer_feature_base_exact
            WHERE CAN_LISTING_CTR_CD IS NOT NULL
              {offer_filter}
        )
        SELECT
            offers.MATCH_RUN_ROWID,
            offers.MATCH_ID,
            offers.PTR_ROW_ORDER,
            offers.OFFER_ROW_INSTANCE_ORDINAL,
            hist.* EXCLUDE(CENTER_CD, CENTER_TY, EVENT_TS)
        FROM offers
        LEFT JOIN acceptance_by_ts AS hist
          ON offers.CENTER_CD = hist.CENTER_CD
         AND offers.CENTER_TY = hist.CENTER_TY
         AND offers.EVENT_TS = hist.EVENT_TS
        {target_filter}
        """
    )


def build_opo_center_pair_history_sql(
    year: int | None = None,
    history_start_ts: str | None = None,
    year_start_ts: str | None = None,
    year_end_ts: str | None = None,
) -> str:
    input_filter = ""
    match_year_expr = "CAST(NULL AS INTEGER) AS match_year"
    if year is not None:
        input_filter = f"AND match_year BETWEEN {year - 1} AND {year}"
        match_year_expr = "match_year"

    ctes = [
        dedent(
            f"""
            filtered_input AS (
                SELECT
                    MATCH_RUN_ROWID,
                    MATCH_ID,
                    PTR_ROW_ORDER,
                    OFFER_ROW_INSTANCE_ORDINAL,
                    {match_year_expr},
                    OFFER_SORT_TS,
                    MATCH_OPO_CTR_CD,
                    CAN_LISTING_CTR_CD,
                    COALESCE(CAN_LISTING_CTR_TY, '') AS CAN_LISTING_CTR_TY_NORM,
                    DCD_IND,
                    HIGH_KDPI_FLG,
                    HCV_POSITIVE_FLG,
                    LONG_DISTANCE_FLG,
                    MM_TOTAL_BUCKET,
                    CAST(IS_POSITIVE_RESPONSE AS BIGINT) AS IS_POSITIVE_RESPONSE,
                    CAST(IS_NEGATIVE_RESPONSE AS BIGINT) AS IS_NEGATIVE_RESPONSE
                FROM analytics.offer_feature_base_exact
                WHERE MATCH_OPO_CTR_CD IS NOT NULL
                  AND CAN_LISTING_CTR_CD IS NOT NULL
                  {input_filter}
            )
            """
        )
    ]

    for cte_sql, _ in [
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="pair_base",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM"],
            year=year,
            count_alias_template="OPO_CENTER_PAIR_YN_OFFER_COUNT_{suffix}",
            rate_alias_template="OPO_CENTER_PAIR_POSITIVE_RESPONSE_RATE_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="pair_dcd",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "DCD_IND"],
            row_filter="WHERE DCD_IND IS NOT NULL",
            year=year,
            rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_DCD_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="pair_high_kdpi",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "HIGH_KDPI_FLG"],
            row_filter="WHERE HIGH_KDPI_FLG IS NOT NULL",
            year=year,
            rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_HIGH_KDPI_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="pair_hcv",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "HCV_POSITIVE_FLG"],
            row_filter="WHERE HCV_POSITIVE_FLG IS NOT NULL",
            year=year,
            rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_HCV_POS_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="pair_long_distance",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "LONG_DISTANCE_FLG"],
            row_filter="WHERE LONG_DISTANCE_FLG IS NOT NULL",
            year=year,
            rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_LONG_DISTANCE_{suffix}",
        ),
        build_exact_response_rate_history_pass_ctes(
            cte_prefix="pair_mm_bucket",
            input_cte="filtered_input",
            timestamp_column="OFFER_SORT_TS",
            partition_columns=["MATCH_OPO_CTR_CD", "CAN_LISTING_CTR_CD", "CAN_LISTING_CTR_TY_NORM", "MM_TOTAL_BUCKET"],
            row_filter="WHERE MM_TOTAL_BUCKET IS NOT NULL",
            year=year,
            rate_alias_template="OPO_CENTER_PAIR_RATE_SAME_MM_BUCKET_{suffix}",
        ),
    ]:
        ctes.extend(cte_sql)

    return dedent(
        f"""
        WITH
        {",\n".join(ctes)}
        SELECT
            base.MATCH_RUN_ROWID,
            base.MATCH_ID,
            base.PTR_ROW_ORDER,
            base.OFFER_ROW_INSTANCE_ORDINAL,
            base.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            dcd.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            high_kdpi.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            hcv.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            long_distance.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            mm_bucket.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL)
        FROM pair_base_output AS base
        LEFT JOIN pair_dcd_output AS dcd
          ON base.MATCH_RUN_ROWID = dcd.MATCH_RUN_ROWID
        LEFT JOIN pair_high_kdpi_output AS high_kdpi
          ON base.MATCH_RUN_ROWID = high_kdpi.MATCH_RUN_ROWID
        LEFT JOIN pair_hcv_output AS hcv
          ON base.MATCH_RUN_ROWID = hcv.MATCH_RUN_ROWID
        LEFT JOIN pair_long_distance_output AS long_distance
          ON base.MATCH_RUN_ROWID = long_distance.MATCH_RUN_ROWID
        LEFT JOIN pair_mm_bucket_output AS mm_bucket
          ON base.MATCH_RUN_ROWID = mm_bucket.MATCH_RUN_ROWID
        """
    )


def build_opo_center_pair_history_join_sql(
    skip_opo_center_pair_mm_bucket: bool = False,
) -> str:
    mm_bucket_select = ""
    mm_bucket_join = ""
    if not skip_opo_center_pair_mm_bucket:
        mm_bucket_select = (
            ",\n            mm_bucket.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL)"
        )
        mm_bucket_join = """
        LEFT JOIN analytics.opo_center_pair_mm_bucket_history_exact AS mm_bucket
          ON base.MATCH_RUN_ROWID = mm_bucket.MATCH_RUN_ROWID
        """

    return dedent(
        f"""
        SELECT
            base.*,
            dcd.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            high_kdpi.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            hcv.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            long_distance.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL){mm_bucket_select}
        FROM analytics.opo_center_pair_base_history_exact AS base
        LEFT JOIN analytics.opo_center_pair_dcd_history_exact AS dcd
          ON base.MATCH_RUN_ROWID = dcd.MATCH_RUN_ROWID
        LEFT JOIN analytics.opo_center_pair_high_kdpi_history_exact AS high_kdpi
          ON base.MATCH_RUN_ROWID = high_kdpi.MATCH_RUN_ROWID
        LEFT JOIN analytics.opo_center_pair_hcv_history_exact AS hcv
          ON base.MATCH_RUN_ROWID = hcv.MATCH_RUN_ROWID
        LEFT JOIN analytics.opo_center_pair_long_distance_history_exact AS long_distance
          ON base.MATCH_RUN_ROWID = long_distance.MATCH_RUN_ROWID{mm_bucket_join}
        """
    )


def opo_center_pair_history_feature_specs() -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for suffix, _ in HISTORY_WINDOWS:
        specs.extend(
            [
                (f"OPO_CENTER_PAIR_YN_OFFER_COUNT_{suffix}", "BIGINT"),
                (f"OPO_CENTER_PAIR_POSITIVE_RESPONSE_RATE_{suffix}", "DOUBLE"),
                (f"OPO_CENTER_PAIR_RATE_SAME_DCD_{suffix}", "DOUBLE"),
                (f"OPO_CENTER_PAIR_RATE_SAME_HIGH_KDPI_{suffix}", "DOUBLE"),
                (f"OPO_CENTER_PAIR_RATE_SAME_HCV_POS_{suffix}", "DOUBLE"),
                (f"OPO_CENTER_PAIR_RATE_SAME_LONG_DISTANCE_{suffix}", "DOUBLE"),
                (f"OPO_CENTER_PAIR_RATE_SAME_MM_BUCKET_{suffix}", "DOUBLE"),
            ]
        )
    return specs


def build_null_opo_center_pair_history_sql() -> str:
    null_columns = ",\n            ".join(
        f"CAST(NULL AS {column_type}) AS {column_name}"
        for column_name, column_type in opo_center_pair_history_feature_specs()
    )
    return dedent(
        f"""
        SELECT
            MATCH_RUN_ROWID,
            MATCH_ID,
            PTR_ROW_ORDER,
            OFFER_ROW_INSTANCE_ORDINAL,
            {null_columns}
        FROM analytics.offer_feature_base_exact
        """
    )


def drop_relation(con: duckdb.DuckDBPyConnection, name: str) -> None:
    for statement in [f"DROP VIEW IF EXISTS {name};", f"DROP TABLE IF EXISTS {name};"]:
        try:
            con.execute(statement)
        except duckdb.CatalogException as exc:
            if "trying to drop type View" in str(exc) or "trying to drop type Table" in str(exc):
                continue
            raise


def create_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    select_sql: str,
    indexes: list[tuple[str, str]] | None = None,
    progress: ProgressTracker | None = None,
) -> None:
    tracker = progress or _ACTIVE_PROGRESS_TRACKER
    print(f"[build] table={name}", flush=True)
    drop_relation(con, name)
    con.execute(f"CREATE TABLE {name} AS {select_sql}")
    if indexes:
        for index_name, columns in indexes:
            con.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {name}({columns});")
    con.execute(f"ANALYZE {name};")
    if tracker:
        tracker.advance(name)


def create_view(
    con: duckdb.DuckDBPyConnection,
    name: str,
    select_sql: str,
    progress: ProgressTracker | None = None,
) -> None:
    tracker = progress or _ACTIVE_PROGRESS_TRACKER
    print(f"[build] view={name}", flush=True)
    drop_relation(con, name)
    con.execute(f"CREATE VIEW {name} AS {select_sql}")
    if tracker:
        tracker.advance(name)


def create_year_partitioned_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    years: list[int],
    select_sql_template: str,
    indexes: list[tuple[str, str]] | None = None,
    progress: ProgressTracker | None = None,
) -> None:
    tracker = progress or _ACTIVE_PROGRESS_TRACKER
    print(f"[build] table={name}", flush=True)
    drop_relation(con, name)
    created = False
    for year in years:
        print(f"[build] table={name} match_year={year}", flush=True)
        select_sql = select_sql_template.format(year=year)
        if not created:
            con.execute(f"CREATE TABLE {name} AS {select_sql}")
            created = True
        else:
            con.execute(f"INSERT INTO {name} {select_sql}")
        if tracker:
            tracker.advance(f"{name} match_year={year}")
    if not created:
        raise ValueError(f"No years supplied while building {name}")
    if indexes:
        for index_name, columns in indexes:
            con.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {name}({columns});")
    con.execute(f"ANALYZE {name};")


def create_year_partitioned_table_from_builder(
    con: duckdb.DuckDBPyConnection,
    name: str,
    years: list[int],
    sql_builder,
    indexes: list[tuple[str, str]] | None = None,
    progress: ProgressTracker | None = None,
) -> None:
    tracker = progress or _ACTIVE_PROGRESS_TRACKER
    print(f"[build] table={name}", flush=True)
    drop_relation(con, name)
    created = False
    for year in years:
        print(f"[build] table={name} match_year={year}", flush=True)
        select_sql = sql_builder(**history_year_bounds(year))
        if not created:
            con.execute(f"CREATE TABLE {name} AS {select_sql}")
            created = True
        else:
            con.execute(f"INSERT INTO {name} {select_sql}")
        if tracker:
            tracker.advance(f"{name} match_year={year}")
    if not created:
        raise ValueError(f"No years supplied while building {name}")
    if indexes:
        for index_name, columns in indexes:
            con.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {name}({columns});")
    con.execute(f"ANALYZE {name};")


def write_manifest(
    con: duckdb.DuckDBPyConnection,
    manifest_path: Path,
    match_db: Path,
    saf_db: Path,
) -> None:
    manifest: dict[str, object] = {
        "built_at_utc": utc_now(),
        "match_db": str(match_db),
        "saf_db": str(saf_db),
        "schemas": {},
        "macros": [
            "candidate_history_enriched",
            "donor_history_enriched",
            "match_opo_history",
            "listing_center_history",
        ],
        "views": [
            "analytics.match_offer_enriched",
            "analytics.match_offer_to_transplant",
        ],
    }
    for schema_name in ["saf_link", "analytics"]:
        tables = [
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM duckdb_tables()
                WHERE schema_name = ?
                ORDER BY table_name;
                """,
                [schema_name],
            ).fetchall()
        ]
        manifest["schemas"][schema_name] = [
            {
                "table_name": table_name,
                "row_count": int(con.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}").fetchone()[0]),
            }
            for table_name in tables
        ]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def build_query_layer_tail(
    con: duckdb.DuckDBPyConnection,
    match_years: list[int],
    skip_opo_center_pair_mm_bucket: bool = False,
    skip_opo_center_pair_history: bool = False,
) -> None:
    create_year_partitioned_table_from_builder(
        con,
        "analytics.listing_center_offer_base_history_exact",
        match_years,
        build_listing_center_offer_base_history_sql,
        indexes=[("idx_analytics_listing_center_offer_base_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.listing_center_offer_dcd_history_exact",
        match_years,
        build_listing_center_offer_dcd_history_sql,
        indexes=[("idx_analytics_listing_center_offer_dcd_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.listing_center_offer_high_kdpi_history_exact",
        match_years,
        build_listing_center_offer_high_kdpi_history_sql,
        indexes=[("idx_analytics_listing_center_offer_high_kdpi_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.listing_center_offer_hcv_history_exact",
        match_years,
        build_listing_center_offer_hcv_history_sql,
        indexes=[("idx_analytics_listing_center_offer_hcv_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.listing_center_offer_long_distance_history_exact",
        match_years,
        build_listing_center_offer_long_distance_history_sql,
        indexes=[("idx_analytics_listing_center_offer_long_distance_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.listing_center_offer_mm_bucket_history_exact",
        match_years,
        build_listing_center_offer_mm_bucket_history_sql,
        indexes=[("idx_analytics_listing_center_offer_mm_bucket_history_exact_rowid", "MATCH_RUN_ROWID")],
    )

    create_table(
        con,
        "analytics.listing_center_offer_history_exact",
        """
        SELECT
            base.*,
            dcd.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            high_kdpi.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            hcv.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            long_distance.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL),
            mm_bucket.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL)
        FROM analytics.listing_center_offer_base_history_exact AS base
        LEFT JOIN analytics.listing_center_offer_dcd_history_exact AS dcd
          ON base.MATCH_RUN_ROWID = dcd.MATCH_RUN_ROWID
        LEFT JOIN analytics.listing_center_offer_high_kdpi_history_exact AS high_kdpi
          ON base.MATCH_RUN_ROWID = high_kdpi.MATCH_RUN_ROWID
        LEFT JOIN analytics.listing_center_offer_hcv_history_exact AS hcv
          ON base.MATCH_RUN_ROWID = hcv.MATCH_RUN_ROWID
        LEFT JOIN analytics.listing_center_offer_long_distance_history_exact AS long_distance
          ON base.MATCH_RUN_ROWID = long_distance.MATCH_RUN_ROWID
        LEFT JOIN analytics.listing_center_offer_mm_bucket_history_exact AS mm_bucket
          ON base.MATCH_RUN_ROWID = mm_bucket.MATCH_RUN_ROWID
        """,
        indexes=[
            (
                "idx_analytics_listing_center_offer_history_exact_offer",
                "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
            ),
            ("idx_analytics_listing_center_offer_history_exact_rowid", "MATCH_RUN_ROWID"),
        ],
    )

    create_year_partitioned_table_from_builder(
        con,
        "analytics.listing_center_acceptance_history_exact",
        match_years,
        build_listing_center_acceptance_history_sql,
        indexes=[
            (
                "idx_analytics_listing_center_acceptance_history_exact_offer",
                "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
            ),
            ("idx_analytics_listing_center_acceptance_history_exact_rowid", "MATCH_RUN_ROWID"),
        ],
    )

    create_table(
        con,
        "analytics.listing_center_history_exact",
        """
        SELECT
            offer.*,
            accept.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL)
        FROM analytics.listing_center_offer_history_exact AS offer
        LEFT JOIN analytics.listing_center_acceptance_history_exact AS accept
          ON offer.MATCH_RUN_ROWID = accept.MATCH_RUN_ROWID
        """,
        indexes=[
            (
                "idx_analytics_listing_center_history_exact_offer",
                "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
            ),
            ("idx_analytics_listing_center_history_exact_rowid", "MATCH_RUN_ROWID"),
        ],
    )

    if skip_opo_center_pair_history:
        build_query_layer_post_pair_long_distance(
            con,
            match_years,
            skip_opo_center_pair_mm_bucket=skip_opo_center_pair_mm_bucket,
            skip_opo_center_pair_history=skip_opo_center_pair_history,
        )
        return

    create_year_partitioned_table_from_builder(
        con,
        "analytics.opo_center_pair_base_history_exact",
        match_years,
        build_opo_center_pair_base_history_sql,
        indexes=[("idx_analytics_opo_center_pair_base_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.opo_center_pair_dcd_history_exact",
        match_years,
        build_opo_center_pair_dcd_history_sql,
        indexes=[("idx_analytics_opo_center_pair_dcd_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.opo_center_pair_high_kdpi_history_exact",
        match_years,
        build_opo_center_pair_high_kdpi_history_sql,
        indexes=[("idx_analytics_opo_center_pair_high_kdpi_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.opo_center_pair_hcv_history_exact",
        match_years,
        build_opo_center_pair_hcv_history_sql,
        indexes=[("idx_analytics_opo_center_pair_hcv_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    create_year_partitioned_table_from_builder(
        con,
        "analytics.opo_center_pair_long_distance_history_exact",
        match_years,
        build_opo_center_pair_long_distance_history_sql,
        indexes=[("idx_analytics_opo_center_pair_long_distance_history_exact_rowid", "MATCH_RUN_ROWID")],
    )
    build_query_layer_post_pair_long_distance(
        con,
        match_years,
        skip_opo_center_pair_mm_bucket=skip_opo_center_pair_mm_bucket,
    )


def build_query_layer_post_pair_long_distance(
    con: duckdb.DuckDBPyConnection,
    match_years: list[int],
    skip_opo_center_pair_mm_bucket: bool = False,
    skip_opo_center_pair_history: bool = False,
) -> None:
    if skip_opo_center_pair_history:
        for relation_name in [
            "analytics.opo_center_pair_base_history_exact",
            "analytics.opo_center_pair_dcd_history_exact",
            "analytics.opo_center_pair_high_kdpi_history_exact",
            "analytics.opo_center_pair_hcv_history_exact",
            "analytics.opo_center_pair_long_distance_history_exact",
            "analytics.opo_center_pair_mm_bucket_history_exact",
        ]:
            drop_relation(con, relation_name)
        create_view(
            con,
            "analytics.opo_center_pair_history_exact",
            build_null_opo_center_pair_history_sql(),
        )
    elif skip_opo_center_pair_mm_bucket:
        drop_relation(con, "analytics.opo_center_pair_mm_bucket_history_exact")
    else:
        create_year_partitioned_table_from_builder(
            con,
            "analytics.opo_center_pair_mm_bucket_history_exact",
            match_years,
            build_opo_center_pair_mm_bucket_history_sql,
            indexes=[("idx_analytics_opo_center_pair_mm_bucket_history_exact_rowid", "MATCH_RUN_ROWID")],
        )

    if not skip_opo_center_pair_history:
        create_table(
            con,
            "analytics.opo_center_pair_history_exact",
            build_opo_center_pair_history_join_sql(
                skip_opo_center_pair_mm_bucket=skip_opo_center_pair_mm_bucket
            ),
            indexes=[
                (
                    "idx_analytics_opo_center_pair_history_exact_offer",
                    "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
                ),
                ("idx_analytics_opo_center_pair_history_exact_rowid", "MATCH_RUN_ROWID"),
            ],
        )

    create_view(
        con,
        "analytics.match_offer_enriched",
        build_match_offer_enriched_sql(),
    )
    validate_match_offer_enriched_schema(con)

    create_view(
        con,
        "analytics.match_offer_to_transplant",
        build_match_offer_to_transplant_sql(),
    )

    create_year_partitioned_table(
        con,
        "analytics.candidate_year_stats",
        match_years,
        """
        SELECT
            l.match_year,
            l.PX_ID,
            c.PERS_ID,
            c.CAN_LISTING_CTR_CD,
            c.CAN_LISTING_CTR_TY,
            COUNT(*) AS offer_count,
            COUNT(DISTINCT l.MATCH_ID) AS distinct_match_count,
            COUNT(DISTINCT l.DONOR_ID) AS distinct_donor_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') AS accept_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Z') AS provisional_yes_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'N') AS decline_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'B') AS bypass_count,
            MIN(l.MATCH_SUBMIT_DT) AS first_match_submit_dt,
            MAX(l.MATCH_SUBMIT_DT) AS last_match_submit_dt,
            ROUND(AVG(l.PTR_SEQUENCE_NUM), 3) AS avg_sequence_num,
            ROUND(
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') * 1.0
                / NULLIF(COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT IN ('Y', 'N')), 0),
                6
            ) AS positive_response_rate
        FROM match_runs AS l
        LEFT JOIN saf_link.candidate_dim AS c
            ON l.PX_ID = c.PX_ID
        WHERE l.match_year = {year}
        GROUP BY 1, 2, 3, 4, 5
        """,
        indexes=[
            ("idx_analytics_candidate_year_stats_px_year", "PX_ID, match_year"),
            ("idx_analytics_candidate_year_stats_center", "CAN_LISTING_CTR_CD, CAN_LISTING_CTR_TY, match_year"),
        ],
    )

    create_year_partitioned_table(
        con,
        "analytics.donor_year_stats",
        match_years,
        """
        SELECT
            l.match_year,
            l.DONOR_ID,
            d.DON_OPO_CTR_ID,
            d.DCD_IND,
            d.TX_CENTER_COUNT_250NM,
            d.KDPI,
            d.KDPI_BIN,
            COUNT(*) AS offer_count,
            COUNT(DISTINCT l.MATCH_ID) AS distinct_match_count,
            COUNT(DISTINCT l.PX_ID) AS distinct_candidate_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') AS accept_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Z') AS provisional_yes_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'N') AS decline_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'B') AS bypass_count,
            MIN(l.MATCH_SUBMIT_DT) AS first_match_submit_dt,
            MAX(l.MATCH_SUBMIT_DT) AS last_match_submit_dt,
            ROUND(AVG(l.PTR_SEQUENCE_NUM), 3) AS avg_sequence_num,
            ROUND(
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') * 1.0
                / NULLIF(COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT IN ('Y', 'N')), 0),
                6
            ) AS positive_response_rate
        FROM match_runs AS l
        LEFT JOIN saf_link.donor_dim AS d
            ON l.DONOR_ID = d.DONOR_ID
        WHERE l.match_year = {year}
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """,
        indexes=[
            ("idx_analytics_donor_year_stats_donor_year", "DONOR_ID, match_year"),
            ("idx_analytics_donor_year_stats_kdpi", "KDPI_BIN, match_year"),
        ],
    )

    create_year_partitioned_table(
        con,
        "analytics.match_opo_year_stats",
        match_years,
        """
        SELECT
            l.match_year,
            l.MATCH_OPO_CTR_CD,
            c.ENTIRE_NAME AS MATCH_OPO_NAME,
            c.PRIMARY_STATE AS MATCH_OPO_STATE,
            COUNT(*) AS offer_count,
            COUNT(DISTINCT l.MATCH_ID) AS distinct_match_count,
            COUNT(DISTINCT l.DONOR_ID) AS distinct_donor_count,
            COUNT(DISTINCT l.PX_ID) AS distinct_candidate_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') AS accept_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Z') AS provisional_yes_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'N') AS decline_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'B') AS bypass_count,
            ROUND(
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') * 1.0
                / NULLIF(COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT IN ('Y', 'N')), 0),
                6
            ) AS positive_response_rate
        FROM match_runs AS l
        LEFT JOIN saf_link.center_code_dim AS c
            ON l.MATCH_OPO_CTR_CD = c.CTR_CD
        WHERE l.match_year = {year}
        GROUP BY 1, 2, 3, 4
        """,
        indexes=[("idx_analytics_match_opo_year_stats", "MATCH_OPO_CTR_CD, match_year")],
    )

    create_year_partitioned_table(
        con,
        "analytics.tx_center_match_year_stats",
        match_years,
        """
        SELECT
            l.match_year,
            c.CAN_LISTING_CTR_CD,
            c.CAN_LISTING_CTR_TY,
            ctr.ENTIRE_NAME AS CENTER_NAME,
            ctr.PRIMARY_STATE AS CENTER_STATE,
            COUNT(*) AS offer_count,
            COUNT(DISTINCT l.MATCH_ID) AS distinct_match_count,
            COUNT(DISTINCT l.DONOR_ID) AS distinct_donor_count,
            COUNT(DISTINCT l.PX_ID) AS distinct_candidate_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') AS accept_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Z') AS provisional_yes_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'N') AS decline_count,
            COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'B') AS bypass_count,
            ROUND(
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') * 1.0
                / NULLIF(COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT IN ('Y', 'N')), 0),
                6
            ) AS positive_response_rate
        FROM match_runs AS l
        LEFT JOIN saf_link.candidate_dim AS c
            ON l.PX_ID = c.PX_ID
        LEFT JOIN saf_link.center_dim AS ctr
            ON c.CAN_LISTING_CTR_CD = ctr.CTR_CD
           AND coalesce(c.CAN_LISTING_CTR_TY, '') = coalesce(ctr.CTR_TY, '')
        WHERE l.match_year = {year}
        GROUP BY 1, 2, 3, 4, 5
        """,
        indexes=[("idx_analytics_tx_center_match_year_stats", "CAN_LISTING_CTR_CD, CAN_LISTING_CTR_TY, match_year")],
    )

    create_table(
        con,
        "analytics.transplant_center_year_stats",
        """
        SELECT
            EXTRACT(YEAR FROM t.REC_TX_DT) AS tx_year,
            t.REC_CTR_CD,
            t.REC_CTR_TY,
            ctr.ENTIRE_NAME AS CENTER_NAME,
            ctr.PRIMARY_STATE AS CENTER_STATE,
            COUNT(*) AS transplant_count,
            COUNT(DISTINCT t.PX_ID) AS distinct_recipient_count,
            COUNT(DISTINCT t.DONOR_ID) AS distinct_donor_count,
            ROUND(AVG(d.KDPI), 3) AS avg_donor_kdpi,
            ROUND(AVG(f.TFL_CREAT), 3) AS avg_latest_followup_creat
        FROM saf_link.tx_ki_link AS t
        LEFT JOIN saf_link.donor_dim AS d
            ON t.DONOR_ID = d.DONOR_ID
        LEFT JOIN analytics.transplant_followup_latest AS f
            ON t.TRR_ID = f.TRR_ID
        LEFT JOIN saf_link.center_dim AS ctr
            ON t.REC_CTR_CD = ctr.CTR_CD
           AND coalesce(t.REC_CTR_TY, '') = coalesce(ctr.CTR_TY, '')
        GROUP BY 1, 2, 3, 4, 5
        """,
        indexes=[("idx_analytics_transplant_center_year_stats", "REC_CTR_CD, REC_CTR_TY, tx_year")],
    )

    create_table(
        con,
        "analytics.recovery_opo_year_stats",
        """
        SELECT
            EXTRACT(YEAR FROM d.DON_RECOV_DT) AS recov_year,
            d.DON_OPO_CTR_ID,
            ctr.CTR_CD AS OPO_CTR_CD,
            ctr.CTR_TY AS OPO_CTR_TY,
            ctr.ENTIRE_NAME AS OPO_NAME,
            ctr.PRIMARY_STATE AS OPO_STATE,
            MAX(d.TX_CENTER_COUNT_250NM) AS TX_CENTER_COUNT_250NM,
            COUNT(*) AS recovered_donor_count,
            ROUND(AVG(d.KDPI), 3) AS avg_kdpi,
            COUNT(*) FILTER (WHERE d.KDPI_BIN = '0-20') AS kdpi_0_20_count,
            COUNT(*) FILTER (WHERE d.KDPI_BIN = '20-40') AS kdpi_20_40_count,
            COUNT(*) FILTER (WHERE d.KDPI_BIN = '40-60') AS kdpi_40_60_count,
            COUNT(*) FILTER (WHERE d.KDPI_BIN = '60-80') AS kdpi_60_80_count,
            COUNT(*) FILTER (WHERE d.KDPI_BIN = '80-100') AS kdpi_80_100_count
        FROM saf_link.donor_dim AS d
        LEFT JOIN saf_link.center_dim AS ctr
            ON d.DON_OPO_CTR_ID = ctr.CTR_ID
        GROUP BY 1, 2, 3, 4, 5, 6
        """,
        indexes=[("idx_analytics_recovery_opo_year_stats", "DON_OPO_CTR_ID, recov_year")],
    )
    print("[build] macros", flush=True)
    con.execute(
        """
        CREATE OR REPLACE MACRO candidate_history_enriched(px_id_param) AS TABLE
        SELECT *
        FROM analytics.match_offer_to_transplant
        WHERE PX_ID = px_id_param
        ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE MACRO donor_history_enriched(donor_id_param) AS TABLE
        SELECT *
        FROM analytics.match_offer_to_transplant
        WHERE DONOR_ID = donor_id_param
        ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE MACRO match_opo_history(opo_code_param) AS TABLE
        SELECT *
        FROM analytics.match_offer_to_transplant
        WHERE MATCH_OPO_CTR_CD = opo_code_param
        ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE MACRO listing_center_history(center_cd_param) AS TABLE
        SELECT *
        FROM analytics.match_offer_to_transplant
        WHERE CAN_LISTING_CTR_CD = center_cd_param
        ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
        """
    )
def build_query_layer(
    match_db: Path,
    saf_db: Path,
    manifest_path: Path,
    threads: int,
    allow_offer_base_row_count_mismatch: bool = False,
    resume_from: str | None = None,
    skip_opo_center_pair_mm_bucket: bool = False,
    skip_opo_center_pair_history: bool = False,
) -> None:
    global _ACTIVE_PROGRESS_TRACKER
    if not match_db.exists():
        raise FileNotFoundError(f"Match-run DuckDB not found: {match_db}")
    if not saf_db.exists():
        raise FileNotFoundError(f"SAF DuckDB not found: {saf_db}")

    con = duckdb.connect(str(match_db))
    try:
        memory_limit = recommended_duckdb_memory_limit()
        con.execute(f"PRAGMA threads={max(1, threads)};")
        temp_dir = (match_db.parent / ".duckdb_tmp_match_saf_query_layer").resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_sql = str(temp_dir).replace("'", "''")
        con.execute(f"SET temp_directory='{temp_dir_sql}';")
        con.execute("SET max_temp_directory_size='480GiB';")
        con.execute("SET preserve_insertion_order=false;")
        con.execute(f"SET memory_limit='{memory_limit}';")
        print(
            f"[config] threads={max(1, threads)} memory_limit={memory_limit} temp_directory={temp_dir}",
            flush=True,
        )
        saf_db_sql = str(saf_db).replace("'", "''")
        con.execute(f"ATTACH '{saf_db_sql}' AS saf_src (READ_ONLY);")
        con.execute("CREATE SCHEMA IF NOT EXISTS saf_link;")
        con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
        match_years = [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT match_year FROM match_runs WHERE match_year IS NOT NULL ORDER BY match_year"
            ).fetchall()
        ]
        resume_from_center_history = False
        resume_from_pair_long_distance_history = False
        if resume_from is not None:
            normalized_resume_from = resume_from.strip().lower()
            if normalized_resume_from == RESUME_FROM_CENTER_HISTORY.lower():
                required_relations = [
                    ("saf_link", "center_dim"),
                    ("saf_link", "center_code_dim"),
                    ("saf_link", "candidate_dim"),
                    ("saf_link", "candidate_status_history"),
                    ("saf_link", "donor_dim"),
                    ("saf_link", "donor_disposition"),
                    ("saf_link", "opo_center_history"),
                    ("saf_link", "tx_ki_link"),
                    ("saf_link", "txf_ki_link"),
                    ("analytics", "transplant_followup_latest"),
                    ("analytics", "match_transplant_bridge"),
                    ("analytics", "donor_opo_success_exact"),
                    ("analytics", "offer_feature_base_exact"),
                    ("analytics", "kidney_outcome_exact"),
                    ("analytics", "match_run_summary_exact"),
                    ("analytics", "candidate_tx_history_exact"),
                    ("analytics", "candidate_history_exact"),
                    ("analytics", "opo_history_exact"),
                ]
                missing_relations = [
                    f"{schema_name}.{relation_name}"
                    for schema_name, relation_name in required_relations
                    if not relation_exists(con, schema_name, relation_name)
                ]
                if missing_relations:
                    raise RuntimeError(
                        "Cannot resume from analytics.listing_center_offer_history_exact because "
                        f"required upstream relations are missing: {missing_relations}"
                    )
                resume_from_center_history = True
                print(
                    "[resume] reusing completed query-layer state through analytics.opo_history_exact",
                    flush=True,
                )
            elif normalized_resume_from == RESUME_FROM_PAIR_LONG_DISTANCE_HISTORY.lower():
                required_relations = [
                    ("analytics", "listing_center_offer_history_exact"),
                    ("analytics", "listing_center_acceptance_history_exact"),
                    ("analytics", "listing_center_history_exact"),
                    ("analytics", "opo_center_pair_base_history_exact"),
                    ("analytics", "opo_center_pair_dcd_history_exact"),
                    ("analytics", "opo_center_pair_high_kdpi_history_exact"),
                    ("analytics", "opo_center_pair_hcv_history_exact"),
                    ("analytics", "opo_center_pair_long_distance_history_exact"),
                ]
                missing_relations = [
                    f"{schema_name}.{relation_name}"
                    for schema_name, relation_name in required_relations
                    if not relation_exists(con, schema_name, relation_name)
                ]
                if missing_relations:
                    raise RuntimeError(
                        "Cannot resume from analytics.opo_center_pair_long_distance_history_exact "
                        f"because required upstream relations are missing: {missing_relations}"
                    )
                resume_from_pair_long_distance_history = True
                print(
                    "[resume] reusing completed query-layer state through "
                    "analytics.opo_center_pair_long_distance_history_exact",
                    flush=True,
                )
            else:
                raise ValueError(
                    "Unsupported resume checkpoint. Supported values: "
                    f"{RESUME_FROM_CENTER_HISTORY}, {RESUME_FROM_PAIR_LONG_DISTANCE_HISTORY}"
                )

        total_steps = total_build_steps(
            len(match_years),
            skip_opo_center_pair_mm_bucket=skip_opo_center_pair_mm_bucket,
            skip_opo_center_pair_history=skip_opo_center_pair_history,
        )
        if resume_from_center_history:
            total_steps -= RESUME_FROM_CENTER_HISTORY_COMPLETED_STEPS
        if resume_from_pair_long_distance_history:
            total_steps -= RESUME_FROM_PAIR_LONG_DISTANCE_HISTORY_COMPLETED_STEPS
        _ACTIVE_PROGRESS_TRACKER = ProgressTracker(total_steps)

        if resume_from_pair_long_distance_history:
            build_query_layer_post_pair_long_distance(
                con,
                match_years,
                skip_opo_center_pair_mm_bucket=skip_opo_center_pair_mm_bucket,
                skip_opo_center_pair_history=skip_opo_center_pair_history,
            )
            con.execute("CHECKPOINT;")
            write_manifest(con, manifest_path=manifest_path, match_db=match_db, saf_db=saf_db)
            if _ACTIVE_PROGRESS_TRACKER:
                _ACTIVE_PROGRESS_TRACKER.advance("manifest")
            con.execute("DETACH saf_src;")
            return

        if resume_from_center_history:
            build_query_layer_tail(
                con,
                match_years,
                skip_opo_center_pair_mm_bucket=skip_opo_center_pair_mm_bucket,
                skip_opo_center_pair_history=skip_opo_center_pair_history,
            )
            con.execute("CHECKPOINT;")
            write_manifest(con, manifest_path=manifest_path, match_db=match_db, saf_db=saf_db)
            if _ACTIVE_PROGRESS_TRACKER:
                _ACTIVE_PROGRESS_TRACKER.advance("manifest")
            con.execute("DETACH saf_src;")
            return

        if not resume_from_center_history:
            create_table(
                con,
                "saf_link.center_dim",
                """
            SELECT
                CAST(CTR_ID AS BIGINT) AS CTR_ID,
                CTR_CD,
                CTR_TY,
                CAST(REGION AS INTEGER) AS REGION,
                ENTIRE_NAME,
                NAME_PART1,
                NAME_PART2,
                PRIMARY_CITY,
                PRIMARY_STATE,
                PRIMARY_ZIP,
                ZIP5,
                CAST(LATITUDE AS DOUBLE) AS LATITUDE,
                CAST(LONGITUDE AS DOUBLE) AS LONGITUDE,
                GEO_SOURCE,
                PROVIDER_NUM,
                PRIMARY_CTRY,
                CAST(OPTN_MBR AS INTEGER) AS OPTN_MBR,
                CAST(ESRD_REGION AS INTEGER) AS ESRD_REGION
            FROM saf_src.saf_data.institution_geo
            """,
            indexes=[
                ("idx_saf_link_center_dim_ctr_id", "CTR_ID"),
                ("idx_saf_link_center_dim_code_type", "CTR_CD, CTR_TY"),
            ],
            )

            create_table(
                con,
                "saf_link.center_code_dim",
                """
            SELECT * EXCLUDE(rn)
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY CTR_CD
                        ORDER BY
                            CASE
                                WHEN CTR_TY IN ('O', 'TX', 'TXC', 'HOSP', 'KI') THEN 0
                                ELSE 1
                            END,
                            ENTIRE_NAME
                    ) AS rn
                FROM saf_link.center_dim
            )
            WHERE rn = 1
            """,
            indexes=[("idx_saf_link_center_code_dim_ctr_cd", "CTR_CD")],
            )

        create_table(
            con,
            "saf_link.candidate_dim",
            """
            SELECT * EXCLUDE(rn)
            FROM (
                SELECT
                    CAST(PX_ID AS BIGINT) AS PX_ID,
                    CAST(PERS_ID AS BIGINT) AS PERS_ID,
                    WL_ORG,
                    CAN_GENDER,
                    CAN_ABO,
                    CAN_RACE_SRTR,
                    CAN_ETHNICITY_SRTR,
                    DON_TY,
                    CAN_SOURCE,
                    CAN_ON_DIAL,
                    CAN_LISTING_CTR_CD,
                    CAN_LISTING_CTR_TY,
                    CAN_LISTING_DT,
                    CAN_ACTIVATE_DT,
                    CAST(CAN_AGE_AT_LISTING AS DOUBLE) AS CAN_AGE_AT_LISTING,
                    CAST(CAN_DGN AS INTEGER) AS CAN_DGN,
                    CAST(CAN_DIAB AS INTEGER) AS CAN_DIAB,
                    CAST(CAN_DIAB_TY AS INTEGER) AS CAN_DIAB_TY,
                    CAST(CAN_PREV_KI AS INTEGER) AS CAN_PREV_KI,
                    CAST(CAN_PREV_TX AS INTEGER) AS CAN_PREV_TX,
                    CAN_PREV_KI_TX_FUNCTN,
                    CAST(CAN_MAX_WARM_TM AS DOUBLE) AS CAN_MAX_WARM_TM,
                    CAN_REM_DT,
                    CAST(CAN_REM_CD AS INTEGER) AS CAN_REM_CD,
                    CAN_DEATH_DT,
                    CAN_DIAL_DT,
                    CAST(CAN_MOST_RECENT_HGT_CM AS DOUBLE) AS CAN_MOST_RECENT_HGT_CM,
                    CAST(CAN_MOST_RECENT_WGT_KG AS DOUBLE) AS CAN_MOST_RECENT_WGT_KG,
                    ROW_NUMBER() OVER (
                        PARTITION BY CAST(PX_ID AS BIGINT)
                        ORDER BY
                            CAN_LISTING_DT DESC NULLS LAST,
                            CAN_ACTIVATE_DT DESC NULLS LAST,
                            CAST(PERS_ID AS BIGINT) DESC NULLS LAST
                    ) AS rn
                FROM saf_src.saf_data.cand_kipa
                WHERE PX_ID IS NOT NULL
            )
            WHERE rn = 1
            """,
            indexes=[
                ("idx_saf_link_candidate_dim_px_id", "PX_ID"),
                ("idx_saf_link_candidate_dim_pers_id", "PERS_ID"),
                ("idx_saf_link_candidate_dim_center", "CAN_LISTING_CTR_CD, CAN_LISTING_CTR_TY"),
            ],
        )

        create_table(
            con,
            "saf_link.candidate_status_history",
            """
            SELECT
                CAST(PX_ID AS BIGINT) AS PX_ID,
                WL_ORG,
                CAN_SOURCE,
                CAST(CAN_INIT_STAT AS INTEGER) AS CAN_INIT_STAT,
                CAN_LISTING_DT,
                CAN_REM_DT,
                CAST(CAN_REM_CD AS INTEGER) AS CAN_REM_CD,
                CAST(CAN_LAST_STAT AS INTEGER) AS CAN_LAST_STAT,
                CANHX_BEGIN_DT,
                CANHX_END_DT,
                CAST(CANHX_STAT_CD AS INTEGER) AS CANHX_STAT_CD,
                CAST(CANHX_REASON_STAT_INACT AS INTEGER) AS CANHX_REASON_STAT_INACT,
                CAST(CANHX_CPRA AS DOUBLE) AS CANHX_CPRA,
                CAN_INIT_ACT_STAT_DT,
                CAN_INIT_INACT_STAT_DT,
                CAN_LAST_ACT_STAT_DT,
                CAN_LAST_INACT_STAT_DT,
                CAST(CAN_INIT_ACT_STAT_CD AS INTEGER) AS CAN_INIT_ACT_STAT_CD,
                CANHX_BEGIN_DT_TM,
                CANHX_END_DT_TM
            FROM saf_src.saf_data.stathist_kipa
            WHERE PX_ID IS NOT NULL
            """,
            indexes=[
                ("idx_saf_link_candidate_status_history_px_id", "PX_ID"),
                ("idx_saf_link_candidate_status_history_begin_dt", "CANHX_BEGIN_DT"),
            ],
        )

        create_table(
            con,
            "saf_link.donor_dim",
            """
            SELECT
                CAST(DONOR_ID AS BIGINT) AS DONOR_ID,
                CAST(PERS_ID AS BIGINT) AS PERS_ID,
                DON_TY,
                CAST(DON_OPO_CTR_ID AS BIGINT) AS DON_OPO_CTR_ID,
                DON_RECOV_DT,
                DON_GENDER,
                DON_ABO,
                CAST(DON_AGE AS INTEGER) AS DON_AGE,
                DON_RACE_SRTR,
                DON_ETHNICITY_SRTR,
                CAST(DON_HGT_CM AS DOUBLE) AS DON_HGT_CM,
                CAST(DON_WGT_KG AS DOUBLE) AS DON_WGT_KG,
                CAST(DON_CAD_DON_COD AS INTEGER) AS DON_CAD_DON_COD,
                CAST(DON_CREAT AS DOUBLE) AS DON_CREAT,
                CAST(DON_BUN AS DOUBLE) AS DON_BUN,
                CAST(DON_FINAL_SERUM_CREAT AS DOUBLE) AS DON_FINAL_SERUM_CREAT,
                CAST(DON_PEAK_SERUM_CREAT AS DOUBLE) AS DON_PEAK_SERUM_CREAT,
                CAST(DON_HIST_DIAB AS INTEGER) AS DON_HIST_DIAB,
                CAST(DON_HIST_CANCER AS INTEGER) AS DON_HIST_CANCER,
                CAST(DON_HTN AS INTEGER) AS DON_HTN,
                CAST(DON_HIGH_CREAT AS INTEGER) AS DON_HIGH_CREAT,
                CAST(DON_MAX_CREAT AS DOUBLE) AS DON_MAX_CREAT,
                CAST(DON_WARM_ISCH_TM_MINS AS DOUBLE) AS DON_WARM_ISCH_TM_MINS,
                DON_NON_HR_BEAT,
                DON_PROTEIN_URINE,
                DON_ANTI_HBC,
                CAST(DCD_IND AS TINYINT) AS DCD_IND,
                DON_ANTI_HCV,
                DON_ANTI_HIV,
                DON_HCV_NAT,
                CAST(DON_HCV_STAT AS INTEGER) AS DON_HCV_STAT,
                CAST(TX_CENTER_COUNT_250NM AS INTEGER) AS TX_CENTER_COUNT_250NM,
                CAST(KDRI_RAO AS DOUBLE) AS KDRI_RAO,
                CAST(KDRI_MED AS DOUBLE) AS KDRI_MED,
                CAST(KDPI AS INTEGER) AS KDPI,
                KDPI_BIN
            FROM saf_src.saf_data.donor_deceased
            WHERE DONOR_ID IS NOT NULL
            """,
            indexes=[
                ("idx_saf_link_donor_dim_donor_id", "DONOR_ID"),
                ("idx_saf_link_donor_dim_opo_ctr_id", "DON_OPO_CTR_ID"),
                ("idx_saf_link_donor_dim_dcd_ind", "DCD_IND"),
                ("idx_saf_link_donor_dim_tx_center_count_250nm", "TX_CENTER_COUNT_250NM"),
                ("idx_saf_link_donor_dim_kdpi", "KDPI"),
                ("idx_saf_link_donor_dim_kdpi_bin", "KDPI_BIN"),
            ],
        )

        create_table(
            con,
            "saf_link.donor_disposition",
            """
            SELECT
                CAST(DONOR_ID AS BIGINT) AS DONOR_ID,
                CAST(MATCH_ID AS BIGINT) AS MATCH_ID,
                CAST(PX_ID AS BIGINT) AS PX_ID,
                DON_ORG,
                DON_RECOV_DT,
                CAST(DON_DISPOSITION AS INTEGER) AS DON_DISPOSITION,
                CAST(DON_REASON_CD AS INTEGER) AS DON_REASON_CD,
                CAST(DON_TX_CTR_ID AS BIGINT) AS DON_TX_CTR_ID,
                CAST(DON_DISCARD_CD AS INTEGER) AS DON_DISCARD_CD,
                CAST(DON_SHARE_TY AS INTEGER) AS DON_SHARE_TY,
                CAST(DON_STORAGE AS INTEGER) AS DON_STORAGE
            FROM saf_src.saf_data.donor_disposition
            WHERE DONOR_ID IS NOT NULL
            """,
            indexes=[
                ("idx_saf_link_donor_disposition_match_id", "MATCH_ID"),
                ("idx_saf_link_donor_disposition_donor_id", "DONOR_ID"),
                ("idx_saf_link_donor_disposition_px_id", "PX_ID"),
            ],
        )

        create_table(
            con,
            "saf_link.opo_center_history",
            """
            SELECT
                TXC_CTR_CD,
                TXC_CTR_TY,
                SERVED_OPO_CD,
                SERVED_OPO_TY,
                START_DT,
                END_DT
            FROM saf_src.saf_data.hist_opo_txc
            """,
            indexes=[
                ("idx_saf_link_opo_center_history_txc", "TXC_CTR_CD, TXC_CTR_TY"),
                ("idx_saf_link_opo_center_history_opo", "SERVED_OPO_CD, SERVED_OPO_TY"),
            ],
        )

        create_table(
            con,
            "saf_link.tx_ki_link",
            """
            SELECT
                CAST(TRR_ID AS BIGINT) AS TRR_ID,
                CAST(PX_ID AS BIGINT) AS PX_ID,
                CAST(PERS_ID AS BIGINT) AS PERS_ID,
                CAST(DONOR_ID AS BIGINT) AS DONOR_ID,
                REC_TX_DT,
                REC_CTR_CD,
                REC_CTR_TY,
                CAST(DON_OPO_CTR_ID AS BIGINT) AS DON_OPO_CTR_ID,
                CAST(REC_OPO_ID AS BIGINT) AS REC_OPO_ID,
                DON_TY,
                CAST(DON_AGE AS INTEGER) AS DON_AGE,
                CAST(DON_CREAT AS DOUBLE) AS DON_CREAT,
                CAST(DON_HIST_DIAB AS INTEGER) AS DON_HIST_DIAB,
                CAST(DON_HTN AS INTEGER) AS DON_HTN,
                DON_NON_HR_BEAT,
                CASE WHEN DON_NON_HR_BEAT = 'Y' THEN 1 ELSE 0 END AS DCD_IND
            FROM saf_src.saf_data.tx_ki
            WHERE TRR_ID IS NOT NULL
            """,
            indexes=[
                ("idx_saf_link_tx_ki_link_trr_id", "TRR_ID"),
                ("idx_saf_link_tx_ki_link_px_id", "PX_ID"),
                ("idx_saf_link_tx_ki_link_donor_id", "DONOR_ID"),
                ("idx_saf_link_tx_ki_link_center", "REC_CTR_CD, REC_CTR_TY"),
            ],
        )

        create_table(
            con,
            "saf_link.txf_ki_link",
            """
            SELECT
                CAST(TRR_FOL_ID AS BIGINT) AS TRR_FOL_ID,
                CAST(TRR_ID AS BIGINT) AS TRR_ID,
                CAST(PX_ID AS BIGINT) AS PX_ID,
                CAST(PERS_ID AS BIGINT) AS PERS_ID,
                REC_TX_DT,
                REC_CTR_CD,
                REC_CTR_TY,
                CAST(TFL_FOL_CD AS INTEGER) AS TFL_FOL_CD,
                TFL_PX_STAT,
                TFL_PX_STAT_DT,
                CAST(TFL_CREAT AS DOUBLE) AS TFL_CREAT
            FROM saf_src.saf_data.txf_ki
            WHERE TRR_ID IS NOT NULL
            """,
            indexes=[
                ("idx_saf_link_txf_ki_link_trr_id", "TRR_ID"),
                ("idx_saf_link_txf_ki_link_px_id", "PX_ID"),
            ],
        )

        create_table(
            con,
            "analytics.transplant_followup_latest",
            """
            SELECT * EXCLUDE(rn)
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY TRR_ID
                        ORDER BY TFL_PX_STAT_DT DESC NULLS LAST, TFL_FOL_CD DESC NULLS LAST
                    ) AS rn
                FROM saf_link.txf_ki_link
            )
            WHERE rn = 1
            """,
            indexes=[
                ("idx_analytics_transplant_followup_latest_trr_id", "TRR_ID"),
                ("idx_analytics_transplant_followup_latest_px_id", "PX_ID"),
            ],
        )

        create_table(
            con,
            "analytics.match_transplant_bridge",
            """
            SELECT
                dd.MATCH_ID,
                dd.DONOR_ID,
                dd.PX_ID,
                dd.DON_RECOV_DT,
                dd.DON_DISPOSITION,
                dd.DON_REASON_CD,
                dd.DON_TX_CTR_ID,
                dd.DON_DISCARD_CD,
                dd.DON_SHARE_TY,
                tx.TRR_ID,
                tx.PERS_ID,
                tx.REC_TX_DT,
                tx.REC_CTR_CD,
                tx.REC_CTR_TY,
                tx.REC_OPO_ID,
                d.TX_CENTER_COUNT_250NM,
                d.KDPI,
                d.KDPI_BIN,
                f.TFL_PX_STAT AS LATEST_TFL_PX_STAT,
                f.TFL_PX_STAT_DT AS LATEST_TFL_PX_STAT_DT,
                f.TFL_CREAT AS LATEST_TFL_CREAT
            FROM saf_link.donor_disposition AS dd
            LEFT JOIN saf_link.tx_ki_link AS tx
                ON dd.DONOR_ID = tx.DONOR_ID
               AND dd.PX_ID = tx.PX_ID
            LEFT JOIN saf_link.donor_dim AS d
                ON dd.DONOR_ID = d.DONOR_ID
            LEFT JOIN analytics.transplant_followup_latest AS f
                ON tx.TRR_ID = f.TRR_ID
            """,
            indexes=[
                ("idx_analytics_match_transplant_bridge_match", "MATCH_ID"),
                ("idx_analytics_match_transplant_bridge_offer_keys", "MATCH_ID, DONOR_ID, PX_ID"),
                ("idx_analytics_match_transplant_bridge_trr_id", "TRR_ID"),
            ],
        )

        create_table(
            con,
            "analytics.donor_opo_success_exact",
            """
            SELECT
                DONOR_ID,
                DON_OPO_CTR_ID,
                DON_RECOV_DT,
                PRIOR_DONOR_COUNT_SAME_OPO,
                PRIOR_SUCCESSFUL_DONOR_COUNT_SAME_OPO,
                ROUND(
                    PRIOR_SUCCESSFUL_DONOR_COUNT_SAME_OPO * 1.0
                    / NULLIF(PRIOR_DONOR_COUNT_SAME_OPO, 0),
                    6
                ) AS DON_OPO_SUCCESS_RATE_HISTORICAL
            FROM (
                SELECT
                    d.DONOR_ID,
                    d.DON_OPO_CTR_ID,
                    d.DON_RECOV_DT,
                    COUNT(*) OVER donor_window AS PRIOR_DONOR_COUNT_SAME_OPO,
                    SUM(
                        CASE WHEN placed.DONOR_ID IS NULL THEN 0 ELSE 1 END
                    ) OVER donor_window AS PRIOR_SUCCESSFUL_DONOR_COUNT_SAME_OPO
                FROM saf_link.donor_dim AS d
                LEFT JOIN (
                    SELECT DISTINCT DONOR_ID
                    FROM saf_link.donor_disposition
                    WHERE DON_DISPOSITION = 6
                      AND PX_ID IS NOT NULL
                ) AS placed
                  ON d.DONOR_ID = placed.DONOR_ID
                WINDOW donor_window AS (
                    PARTITION BY d.DON_OPO_CTR_ID
                    ORDER BY COALESCE(d.DON_RECOV_DT, TIMESTAMP '1900-01-01 00:00:00'), d.DONOR_ID
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                )
            )
            """,
            indexes=[
                ("idx_analytics_donor_opo_success_exact_donor", "DONOR_ID"),
                ("idx_analytics_donor_opo_success_exact_opo", "DON_OPO_CTR_ID"),
            ],
        )

        create_year_partitioned_table(
            con,
            "analytics.offer_feature_base_exact",
            match_years,
            """
            WITH base_rows AS (
                SELECT
                    hash(
                        m.match_year,
                        m.MATCH_SUBMIT_DT,
                        m.MATCH_ID,
                        m.PTR_ROW_ORDER,
                        m.PTR_SEQUENCE_NUM,
                        COALESCE(m.PTR_OFFER_ID, -1),
                        COALESCE(m.PX_ID, -1),
                        COALESCE(m.DONOR_ID, -1),
                        COALESCE(m.WLREG_AUDIT_ID, -1),
                        COALESCE(m.PTR_STAT_CD, -1),
                        COALESCE(m.PTR_TXC_REFUSAL_CD, -1),
                        COALESCE(m.PTR_CLASS_ALLOC_CAT, ''),
                        COALESCE(m.PTR_OFFER_ACPT, ''),
                        COALESCE(m.PTR_ORG_PLACED, '')
                    ) AS MATCH_RUN_ROWID,
                    m.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.MATCH_ID, m.PTR_ROW_ORDER
                        ORDER BY
                            COALESCE(m.PTR_OFFER_ID, -1),
                            m.PTR_SEQUENCE_NUM,
                            COALESCE(m.PX_ID, -1),
                            COALESCE(m.DONOR_ID, -1),
                            COALESCE(m.WLREG_AUDIT_ID, -1),
                            COALESCE(m.PTR_STAT_CD, -1),
                            COALESCE(m.PTR_TXC_REFUSAL_CD, -1),
                            COALESCE(m.PTR_CLASS_ALLOC_CAT, ''),
                            COALESCE(m.PTR_OFFER_ACPT, ''),
                            COALESCE(m.PTR_ORG_PLACED, '')
                    ) AS OFFER_ROW_INSTANCE_ORDINAL
                FROM match_runs AS m
                WHERE match_year = {year}
            )
            SELECT * EXCLUDE(status_rn)
            FROM (
                SELECT
                    m.MATCH_RUN_ROWID,
                    m.MATCH_ID,
                    m.PTR_ROW_ORDER,
                    m.PTR_OFFER_ID,
                    m.OFFER_ROW_INSTANCE_ORDINAL,
                    m.MATCH_SUBMIT_DT,
                    CASE
                        WHEN m.MATCH_SUBMIT_DT IS NULL THEN NULL
                        ELSE CAST(
                            m.MATCH_SUBMIT_DT
                            + (
                                (ROW_NUMBER() OVER (
                                    PARTITION BY m.MATCH_SUBMIT_DT
                                    ORDER BY
                                        m.MATCH_ID,
                                        m.PTR_SEQUENCE_NUM,
                                        m.PTR_ROW_ORDER,
                                        m.OFFER_ROW_INSTANCE_ORDINAL
                                ) - 1) * INTERVAL 1 MICROSECOND
                            )
                            AS TIMESTAMP_NS
                        )
                    END AS OFFER_SORT_TS,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.MATCH_ID
                        ORDER BY m.PTR_SEQUENCE_NUM, m.PTR_ROW_ORDER
                    ) AS OFFER_RANK,
                    m.source_year,
                    m.match_year,
                    m.MATCH_ORG,
                    m.DONOR_ID,
                    m.PX_ID,
                    m.WLREG_AUDIT_ID,
                    m.PTR_SEQUENCE_NUM,
                    m.MATCH_OPO_CTR_CD,
                    m.MATCH_OPO_CTR_TY,
                    m.PTR_CLASS_ALLOC_CAT,
                    m.PTR_TOT_SCORE,
                    m.PTR_OFFER_ACPT,
                    m.PTR_ORG_PLACED,
                    m.PTR_STAT_CD,
                    m.PTR_CHG_PROCESS_CD,
                    m.PTR_TXC_REFUSAL_CD,
                    m.PTR_HLA_GROUP_MATCH_FLG,
                    m.PTR_A1_MM,
                    m.PTR_A2_MM,
                    m.PTR_B1_MM,
                    m.PTR_B2_MM,
                    m.PTR_DR1_MM,
                    m.PTR_DR2_MM,
                    COALESCE(m.PTR_A1_MM, 0) + COALESCE(m.PTR_A2_MM, 0) AS MM_A,
                    COALESCE(m.PTR_B1_MM, 0) + COALESCE(m.PTR_B2_MM, 0) AS MM_B,
                    COALESCE(m.PTR_DR1_MM, 0) + COALESCE(m.PTR_DR2_MM, 0) AS MM_DR,
                    COALESCE(m.PTR_A1_MM, 0)
                        + COALESCE(m.PTR_A2_MM, 0)
                        + COALESCE(m.PTR_B1_MM, 0)
                        + COALESCE(m.PTR_B2_MM, 0)
                        + COALESCE(m.PTR_DR1_MM, 0)
                        + COALESCE(m.PTR_DR2_MM, 0) AS MM_TOTAL,
                    CASE
                        WHEN m.MATCH_SUBMIT_DT IS NULL THEN NULL
                        ELSE CAST(EXTRACT(ISODOW FROM m.MATCH_SUBMIT_DT) AS INTEGER)
                    END AS MATCH_DAY_OF_WEEK,
                    CASE
                        WHEN m.MATCH_SUBMIT_DT IS NULL THEN NULL
                        ELSE 1 + CAST(FLOOR((EXTRACT(DAY FROM m.MATCH_SUBMIT_DT) - 1) / 7) AS INTEGER)
                    END AS MATCH_WEEK_OF_MONTH,
                    CASE
                        WHEN m.MATCH_SUBMIT_DT IS NULL THEN NULL
                        ELSE CAST(EXTRACT(MONTH FROM m.MATCH_SUBMIT_DT) AS INTEGER)
                    END AS MATCH_MONTH_OF_YEAR,
                    CASE
                        WHEN m.MATCH_SUBMIT_DT IS NULL THEN NULL
                        ELSE CAST(EXTRACT(HOUR FROM m.MATCH_SUBMIT_DT) AS INTEGER)
                    END AS MATCH_HOUR_OF_DAY,
                    CASE WHEN m.PTR_OFFER_ACPT = 'Y' THEN 1 ELSE 0 END AS IS_POSITIVE_RESPONSE,
                    CASE WHEN m.PTR_OFFER_ACPT = 'N' THEN 1 ELSE 0 END AS IS_NEGATIVE_RESPONSE,
                    CASE WHEN m.PTR_OFFER_ACPT IN ('Y', 'N') THEN 1 ELSE 0 END AS IS_YN_RESPONSE,
                    c.PERS_ID AS CANDIDATE_PERS_ID,
                    c.WL_ORG,
                    c.CAN_GENDER,
                    c.CAN_ABO,
                    c.CAN_RACE_SRTR,
                    c.CAN_ETHNICITY_SRTR,
                    c.CAN_ON_DIAL,
                    c.CAN_LISTING_CTR_CD,
                    c.CAN_LISTING_CTR_TY,
                    ctr.CTR_ID AS CAN_LISTING_CTR_ID,
                    c.CAN_LISTING_DT,
                    c.CAN_ACTIVATE_DT,
                    c.CAN_AGE_AT_LISTING,
                    c.CAN_DGN,
                    c.CAN_DIAB,
                    c.CAN_DIAB_TY,
                    c.CAN_PREV_KI,
                    c.CAN_PREV_TX,
                    c.CAN_PREV_KI_TX_FUNCTN,
                    c.CAN_MAX_WARM_TM,
                    c.CAN_REM_DT,
                    c.CAN_REM_CD,
                    c.CAN_DEATH_DT,
                    c.CAN_DIAL_DT,
                    c.CAN_MOST_RECENT_HGT_CM,
                    c.CAN_MOST_RECENT_WGT_KG,
                    sh.CANHX_CPRA,
                    CASE
                        WHEN c.CAN_AGE_AT_LISTING IS NULL OR c.CAN_LISTING_DT IS NULL OR m.MATCH_SUBMIT_DT IS NULL THEN NULL
                        ELSE ROUND(
                            c.CAN_AGE_AT_LISTING
                            + GREATEST(DATE_DIFF('second', c.CAN_LISTING_DT, m.MATCH_SUBMIT_DT), 0) / 31556952.0,
                            6
                        )
                    END AS CAN_CURRENT_AGE_YEARS,
                    CASE
                        WHEN c.CAN_AGE_AT_LISTING IS NULL OR c.CAN_LISTING_DT IS NULL OR m.MATCH_SUBMIT_DT IS NULL THEN NULL
                        WHEN (
                            c.CAN_AGE_AT_LISTING
                            + GREATEST(DATE_DIFF('second', c.CAN_LISTING_DT, m.MATCH_SUBMIT_DT), 0) / 31556952.0
                        ) >= 18 THEN 1
                        ELSE 0
                    END AS CAN_IS_ADULT,
                    d.DON_OPO_CTR_ID,
                    d.DON_RECOV_DT,
                    d.DON_GENDER,
                    d.DON_ABO,
                    d.DON_AGE,
                    d.DON_RACE_SRTR,
                    d.DON_ETHNICITY_SRTR,
                    d.DON_HGT_CM,
                    d.DON_WGT_KG,
                    d.DON_CAD_DON_COD,
                    d.DON_CREAT,
                    d.DON_BUN,
                    d.DON_FINAL_SERUM_CREAT,
                    d.DON_PEAK_SERUM_CREAT,
                    d.DON_HIST_DIAB,
                    d.DON_HIST_CANCER,
                    d.DON_HTN,
                    d.DON_HIGH_CREAT,
                    d.DON_MAX_CREAT,
                    d.DON_WARM_ISCH_TM_MINS,
                    d.DON_NON_HR_BEAT,
                    d.DON_PROTEIN_URINE,
                    d.DON_ANTI_HBC,
                    d.DON_ANTI_HCV,
                    d.DON_ANTI_HIV,
                    d.DON_HCV_NAT,
                    d.DON_HCV_STAT,
                    d.DCD_IND,
                    d.TX_CENTER_COUNT_250NM,
                    d.KDRI_RAO,
                    d.KDRI_MED,
                    d.KDPI,
                    d.KDPI_BIN,
                    dos.DON_OPO_SUCCESS_RATE_HISTORICAL,
                    dist.DISTANCE_NM,
                    CASE
                        WHEN dist.DISTANCE_NM IS NULL THEN NULL
                        WHEN dist.DISTANCE_NM > 250 THEN 1
                        ELSE 0
                    END AS LONG_DISTANCE_FLG,
                    CASE
                        WHEN d.KDPI IS NULL THEN NULL
                        WHEN d.KDPI >= 85 THEN 1
                        ELSE 0
                    END AS HIGH_KDPI_FLG,
                    CASE
                        WHEN d.DON_ANTI_HCV IS NULL THEN NULL
                        WHEN d.DON_ANTI_HCV = 'Y' THEN 1
                        ELSE 0
                    END AS HCV_POSITIVE_FLG,
                    CASE
                        WHEN d.DON_ANTI_HBC IS NULL THEN NULL
                        WHEN d.DON_ANTI_HBC = 'Y' THEN 1
                        ELSE 0
                    END AS HBC_POSITIVE_FLG,
                    CASE
                        WHEN (
                            COALESCE(m.PTR_A1_MM, 0)
                            + COALESCE(m.PTR_A2_MM, 0)
                            + COALESCE(m.PTR_B1_MM, 0)
                            + COALESCE(m.PTR_B2_MM, 0)
                            + COALESCE(m.PTR_DR1_MM, 0)
                            + COALESCE(m.PTR_DR2_MM, 0)
                        ) BETWEEN 0 AND 6
                        THEN CAST(
                            COALESCE(m.PTR_A1_MM, 0)
                            + COALESCE(m.PTR_A2_MM, 0)
                            + COALESCE(m.PTR_B1_MM, 0)
                            + COALESCE(m.PTR_B2_MM, 0)
                            + COALESCE(m.PTR_DR1_MM, 0)
                            + COALESCE(m.PTR_DR2_MM, 0)
                            AS VARCHAR
                        )
                        ELSE NULL
                    END AS MM_TOTAL_BUCKET,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.MATCH_RUN_ROWID
                        ORDER BY COALESCE(sh.CANHX_BEGIN_DT_TM, sh.CANHX_BEGIN_DT, sh.CAN_LISTING_DT) DESC NULLS LAST
                    ) AS status_rn
                FROM base_rows AS m
                LEFT JOIN saf_link.candidate_dim AS c
                    ON m.PX_ID = c.PX_ID
                LEFT JOIN saf_link.center_dim AS ctr
                    ON c.CAN_LISTING_CTR_CD = ctr.CTR_CD
                   AND COALESCE(c.CAN_LISTING_CTR_TY, '') = COALESCE(ctr.CTR_TY, '')
                LEFT JOIN saf_link.candidate_status_history AS sh
                    ON m.PX_ID = sh.PX_ID
                   AND COALESCE(sh.CANHX_BEGIN_DT_TM, sh.CANHX_BEGIN_DT, sh.CAN_LISTING_DT) <= m.MATCH_SUBMIT_DT
                   AND COALESCE(sh.CANHX_END_DT_TM, sh.CANHX_END_DT, TIMESTAMP '2262-01-01 00:00:00') > m.MATCH_SUBMIT_DT
                LEFT JOIN saf_link.donor_dim AS d
                    ON m.DONOR_ID = d.DONOR_ID
                LEFT JOIN analytics.donor_opo_success_exact AS dos
                    ON m.DONOR_ID = dos.DONOR_ID
                LEFT JOIN saf_src.saf_data.opo_tx_center_distance AS dist
                    ON d.DON_OPO_CTR_ID = dist.DON_OPO_CTR_ID
                   AND ctr.CTR_ID = dist.TX_CTR_ID
            )
            WHERE status_rn = 1
            """,
            indexes=[
                (
                    "idx_analytics_offer_feature_base_exact_offer",
                    "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
                ),
                ("idx_analytics_offer_feature_base_exact_rowid", "MATCH_RUN_ROWID"),
            ],
        )
        match_run_count, enriched_base_count = offer_feature_base_exact_row_counts(con)
        if match_run_count != enriched_base_count:
            message = (
                "analytics.offer_feature_base_exact row count does not match match_runs: "
                f"{enriched_base_count} != {match_run_count}"
            )
            if allow_offer_base_row_count_mismatch:
                print(f"[warning] {message}", flush=True)
            else:
                raise RuntimeError(message)

        create_table(
            con,
            "analytics.kidney_outcome_exact",
            f"""
            WITH placed_kidneys AS (
                SELECT * EXCLUDE(rn)
                FROM (
                    SELECT
                        MATCH_ID,
                        DONOR_ID,
                        PX_ID,
                        DON_ORG,
                        ROW_NUMBER() OVER (
                            PARTITION BY MATCH_ID, DONOR_ID, PX_ID, DON_ORG
                            ORDER BY DON_RECOV_DT DESC NULLS LAST, DON_TX_CTR_ID DESC NULLS LAST
                        ) AS rn
                    FROM saf_link.donor_disposition
                    WHERE DON_DISPOSITION = 6
                      AND DON_ORG IN ('LKI', 'RKI')
                      AND MATCH_ID IS NOT NULL
                      AND DONOR_ID IS NOT NULL
                      AND PX_ID IS NOT NULL
                )
                WHERE rn = 1
            ),
            matched_offer AS (
                SELECT * EXCLUDE(rn)
                FROM (
                    SELECT
                        p.MATCH_ID,
                        p.DONOR_ID,
                        p.PX_ID,
                        p.DON_ORG,
                        o.PTR_ROW_ORDER AS ACCEPTED_PTR_ROW_ORDER,
                        o.OFFER_ROW_INSTANCE_ORDINAL AS ACCEPTED_OFFER_ROW_INSTANCE_ORDINAL,
                        o.PTR_SEQUENCE_NUM AS ACCEPTED_SEQUENCE_NUM,
                        o.OFFER_RANK AS ACCEPTED_OFFER_RANK,
                        o.CAN_LISTING_CTR_CD AS ACCEPTED_CENTER_CD,
                        o.CAN_LISTING_CTR_TY AS ACCEPTED_CENTER_TY,
                        ROW_NUMBER() OVER (
                            PARTITION BY p.MATCH_ID, p.DONOR_ID, p.PX_ID, p.DON_ORG
                            ORDER BY
                                CASE
                                    WHEN o.PTR_OFFER_ACPT = 'Y' AND (
                                        (p.DON_ORG = 'LKI' AND o.PTR_ORG_PLACED IN ('KI-L', 'KI-B'))
                                        OR (p.DON_ORG = 'RKI' AND o.PTR_ORG_PLACED IN ('KI-R', 'KI-B'))
                                    ) THEN 0
                                    WHEN o.PTR_OFFER_ACPT = 'Y' THEN 1
                                    ELSE 2
                                END,
                                o.PTR_SEQUENCE_NUM,
                                o.PTR_ROW_ORDER,
                                o.OFFER_ROW_INSTANCE_ORDINAL
                        ) AS rn
                    FROM placed_kidneys AS p
                    JOIN analytics.offer_feature_base_exact AS o
                      ON p.MATCH_ID = o.MATCH_ID
                     AND p.DONOR_ID = o.DONOR_ID
                     AND p.PX_ID = o.PX_ID
                )
                WHERE rn = 1
            )
            SELECT
                p.MATCH_ID,
                p.DONOR_ID,
                p.PX_ID,
                p.DON_ORG,
                m.ACCEPTED_PTR_ROW_ORDER,
                m.ACCEPTED_OFFER_ROW_INSTANCE_ORDINAL,
                m.ACCEPTED_SEQUENCE_NUM,
                m.ACCEPTED_OFFER_RANK,
                m.ACCEPTED_CENTER_CD,
                m.ACCEPTED_CENTER_TY,
                CASE
                    WHEN m.ACCEPTED_PTR_ROW_ORDER IS NULL THEN NULL
                    WHEN EXISTS (
                        SELECT 1
                        FROM match_runs AS prior_row
                        WHERE prior_row.MATCH_ID = p.MATCH_ID
                          AND (
                              prior_row.PTR_SEQUENCE_NUM < m.ACCEPTED_SEQUENCE_NUM
                              OR (
                                  prior_row.PTR_SEQUENCE_NUM = m.ACCEPTED_SEQUENCE_NUM
                                  AND prior_row.PTR_ROW_ORDER < m.ACCEPTED_PTR_ROW_ORDER
                              )
                          )
                          AND prior_row.PTR_OFFER_ACPT = 'B'
                          AND prior_row.PTR_TXC_REFUSAL_CD IN ({QUALIFYING_BYPASS_REFUSAL_CODES})
                    ) THEN 1
                    ELSE 0
                END AS OUT_OF_SEQUENCE_PLACEMENT_FLG
            FROM placed_kidneys AS p
            LEFT JOIN matched_offer AS m
              ON p.MATCH_ID = m.MATCH_ID
             AND p.DONOR_ID = m.DONOR_ID
             AND p.PX_ID = m.PX_ID
             AND p.DON_ORG = m.DON_ORG
            """,
            indexes=[
                ("idx_analytics_kidney_outcome_exact_match", "MATCH_ID"),
                ("idx_analytics_kidney_outcome_exact_offer", "MATCH_ID, ACCEPTED_PTR_ROW_ORDER"),
                ("idx_analytics_kidney_outcome_exact_center", "ACCEPTED_CENTER_CD, ACCEPTED_CENTER_TY"),
            ],
        )

        create_table(
            con,
            "analytics.match_run_summary_exact",
            """
            WITH run_base AS (
                SELECT
                    MATCH_ID,
                    MIN(MATCH_SUBMIT_DT) AS MATCH_SUBMIT_DT,
                    MIN(OFFER_SORT_TS) AS MATCH_SORT_TS,
                    MAX(MATCH_OPO_CTR_CD) AS MATCH_OPO_CTR_CD,
                    MAX(DONOR_ID) AS DONOR_ID,
                    MAX(DCD_IND) AS DCD_IND,
                    MAX(KDPI_BIN) AS KDPI_BIN,
                    COUNT(*) AS RUN_LEN
                FROM analytics.offer_feature_base_exact
                GROUP BY MATCH_ID
            ),
            kidney_counts AS (
                SELECT
                    MATCH_ID,
                    COUNT(*) AS PLACED_KIDNEY_COUNT,
                    COUNT(*) FILTER (WHERE OUT_OF_SEQUENCE_PLACEMENT_FLG = 1) AS OUT_OF_SEQUENCE_KIDNEY_COUNT,
                    ARG_MIN(ACCEPTED_PTR_ROW_ORDER, ACCEPTED_OFFER_RANK) AS FIRST_ACCEPTED_PTR_ROW_ORDER,
                    ARG_MIN(ACCEPTED_SEQUENCE_NUM, ACCEPTED_OFFER_RANK) AS FIRST_ACCEPTED_SEQUENCE_NUM,
                    MIN(ACCEPTED_OFFER_RANK) AS FIRST_ACCEPTED_OFFER_RANK,
                    AVG(ACCEPTED_SEQUENCE_NUM * 1.0) AS MEAN_ACCEPTED_SEQUENCE
                FROM analytics.kidney_outcome_exact
                GROUP BY MATCH_ID
            ),
            offer_decline_prefix AS (
                SELECT
                    MATCH_ID,
                    OFFER_RANK,
                    COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'N') OVER (
                        PARTITION BY MATCH_ID
                        ORDER BY OFFER_RANK
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ) AS PRIOR_DECLINE_COUNT
                FROM analytics.offer_feature_base_exact
            ),
            organ_status AS (
                SELECT
                    MATCH_ID,
                    COUNT(*) FILTER (WHERE DON_ORG IN ('LKI', 'RKI')) AS KIDNEY_ORG_COUNT,
                    COUNT(*) FILTER (WHERE DON_ORG IN ('LKI', 'RKI') AND DON_DISPOSITION = 6) AS PLACED_ORG_COUNT
                FROM saf_link.donor_disposition
                GROUP BY MATCH_ID
            ),
            first_accept_declines AS (
                SELECT
                    k.MATCH_ID,
                    o.PRIOR_DECLINE_COUNT AS FIRST_ACCEPT_DECLINE_COUNT
                FROM kidney_counts AS k
                LEFT JOIN offer_decline_prefix AS o
                  ON k.MATCH_ID = o.MATCH_ID
                 AND k.FIRST_ACCEPTED_OFFER_RANK = o.OFFER_RANK
            ),
            late_placement AS (
                SELECT
                    ko.MATCH_ID,
                    AVG(
                        CASE
                            WHEN ko.ACCEPTED_SEQUENCE_NUM * 1.0 / NULLIF(r.RUN_LEN, 0) > 0.5 THEN 1.0
                            ELSE 0.0
                        END
                    ) AS LATE_PLACEMENT_RATE
                FROM analytics.kidney_outcome_exact AS ko
                JOIN run_base AS r
                  ON ko.MATCH_ID = r.MATCH_ID
                GROUP BY ko.MATCH_ID
            )
            SELECT
                r.MATCH_ID,
                r.MATCH_SUBMIT_DT,
                r.MATCH_SORT_TS,
                r.MATCH_OPO_CTR_CD,
                r.DONOR_ID,
                r.DCD_IND,
                r.KDPI_BIN,
                r.RUN_LEN,
                COALESCE(o.KIDNEY_ORG_COUNT, 0) AS KIDNEY_ORG_COUNT,
                COALESCE(o.PLACED_ORG_COUNT, 0) AS PLACED_KIDNEY_COUNT,
                CASE WHEN COALESCE(o.PLACED_ORG_COUNT, 0) > 0 THEN 1 ELSE 0 END AS AT_LEAST_ONE_PLACED_FLG,
                CASE WHEN COALESCE(o.PLACED_ORG_COUNT, 0) = 2 THEN 1 ELSE 0 END AS BOTH_PLACED_FLG,
                CASE
                    WHEN COALESCE(o.KIDNEY_ORG_COUNT, 0) >= 2 AND COALESCE(o.PLACED_ORG_COUNT, 0) = 0 THEN 1
                    ELSE 0
                END AS BOTH_WASTED_FLG,
                CASE WHEN COALESCE(k.OUT_OF_SEQUENCE_KIDNEY_COUNT, 0) > 0 THEN 1 ELSE 0 END AS OUT_OF_SEQUENCE_RUN_FLG,
                k.FIRST_ACCEPTED_PTR_ROW_ORDER,
                k.FIRST_ACCEPTED_SEQUENCE_NUM,
                k.FIRST_ACCEPTED_OFFER_RANK,
                f.FIRST_ACCEPT_DECLINE_COUNT,
                k.MEAN_ACCEPTED_SEQUENCE,
                CASE
                    WHEN r.RUN_LEN = 0 THEN NULL
                    ELSE k.MEAN_ACCEPTED_SEQUENCE / r.RUN_LEN
                END AS MEAN_ACCEPTED_NORMALIZED_SEQUENCE,
                lp.LATE_PLACEMENT_RATE
            FROM run_base AS r
            LEFT JOIN organ_status AS o
              ON r.MATCH_ID = o.MATCH_ID
            LEFT JOIN kidney_counts AS k
              ON r.MATCH_ID = k.MATCH_ID
            LEFT JOIN first_accept_declines AS f
              ON r.MATCH_ID = f.MATCH_ID
            LEFT JOIN late_placement AS lp
              ON r.MATCH_ID = lp.MATCH_ID
            """,
            indexes=[
                ("idx_analytics_match_run_summary_exact_match", "MATCH_ID"),
                ("idx_analytics_match_run_summary_exact_opo", "MATCH_OPO_CTR_CD, MATCH_SORT_TS"),
            ],
        )

        create_year_partitioned_table_from_builder(
            con,
            "analytics.candidate_tx_history_exact",
            match_years,
            build_candidate_tx_history_sql,
            indexes=[
                (
                    "idx_analytics_candidate_tx_history_exact_offer",
                    "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
                ),
                ("idx_analytics_candidate_tx_history_exact_rowid", "MATCH_RUN_ROWID"),
            ],
        )

        create_year_partitioned_table_from_builder(
            con,
            "analytics.candidate_history_exact",
            match_years,
            build_candidate_history_sql,
            indexes=[
                (
                    "idx_analytics_candidate_history_exact_offer",
                    "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
                ),
                ("idx_analytics_candidate_history_exact_rowid", "MATCH_RUN_ROWID"),
            ],
        )

        create_year_partitioned_table_from_builder(
            con,
            "analytics.opo_history_exact",
            match_years,
            build_opo_history_sql,
            indexes=[("idx_analytics_opo_history_exact_match", "MATCH_ID")],
        )

        create_year_partitioned_table_from_builder(
            con,
            "analytics.listing_center_offer_history_exact",
            match_years,
            build_listing_center_offer_history_sql,
            indexes=[
                (
                    "idx_analytics_listing_center_offer_history_exact_offer",
                    "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
                ),
                ("idx_analytics_listing_center_offer_history_exact_rowid", "MATCH_RUN_ROWID"),
            ],
        )

        create_year_partitioned_table_from_builder(
            con,
            "analytics.listing_center_acceptance_history_exact",
            match_years,
            build_listing_center_acceptance_history_sql,
            indexes=[
                (
                    "idx_analytics_listing_center_acceptance_history_exact_offer",
                    "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
                ),
                ("idx_analytics_listing_center_acceptance_history_exact_rowid", "MATCH_RUN_ROWID"),
            ],
        )

        create_table(
            con,
            "analytics.listing_center_history_exact",
            """
            SELECT
                offer.*,
                accept.* EXCLUDE(MATCH_RUN_ROWID, MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL)
            FROM analytics.listing_center_offer_history_exact AS offer
            LEFT JOIN analytics.listing_center_acceptance_history_exact AS accept
              ON offer.MATCH_RUN_ROWID = accept.MATCH_RUN_ROWID
            """,
            indexes=[
                (
                    "idx_analytics_listing_center_history_exact_offer",
                    "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
                ),
                ("idx_analytics_listing_center_history_exact_rowid", "MATCH_RUN_ROWID"),
            ],
        )

        create_year_partitioned_table_from_builder(
            con,
            "analytics.opo_center_pair_history_exact",
            match_years,
            build_opo_center_pair_history_sql,
            indexes=[
                (
                    "idx_analytics_opo_center_pair_history_exact_offer",
                    "MATCH_ID, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL",
                ),
                ("idx_analytics_opo_center_pair_history_exact_rowid", "MATCH_RUN_ROWID"),
            ],
        )

        create_view(
            con,
            "analytics.match_offer_enriched",
            build_match_offer_enriched_sql(),
        )
        validate_match_offer_enriched_schema(con)

        create_view(
            con,
            "analytics.match_offer_to_transplant",
            build_match_offer_to_transplant_sql(),
        )

        create_year_partitioned_table(
            con,
            "analytics.candidate_year_stats",
            match_years,
            """
            SELECT
                l.match_year,
                l.PX_ID,
                c.PERS_ID,
                c.CAN_LISTING_CTR_CD,
                c.CAN_LISTING_CTR_TY,
                COUNT(*) AS offer_count,
                COUNT(DISTINCT l.MATCH_ID) AS distinct_match_count,
                COUNT(DISTINCT l.DONOR_ID) AS distinct_donor_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') AS accept_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Z') AS provisional_yes_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'N') AS decline_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'B') AS bypass_count,
                MIN(l.MATCH_SUBMIT_DT) AS first_match_submit_dt,
                MAX(l.MATCH_SUBMIT_DT) AS last_match_submit_dt,
                ROUND(AVG(l.PTR_SEQUENCE_NUM), 3) AS avg_sequence_num,
                ROUND(
                    COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') * 1.0
                    / NULLIF(COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT IN ('Y', 'N')), 0),
                    6
                ) AS positive_response_rate
            FROM match_runs AS l
            LEFT JOIN saf_link.candidate_dim AS c
                ON l.PX_ID = c.PX_ID
            WHERE l.match_year = {year}
            GROUP BY 1, 2, 3, 4, 5
            """,
            indexes=[
                ("idx_analytics_candidate_year_stats_px_year", "PX_ID, match_year"),
                ("idx_analytics_candidate_year_stats_center", "CAN_LISTING_CTR_CD, CAN_LISTING_CTR_TY, match_year"),
            ],
        )

        create_year_partitioned_table(
            con,
            "analytics.donor_year_stats",
            match_years,
            """
            SELECT
                l.match_year,
                l.DONOR_ID,
                d.DON_OPO_CTR_ID,
                d.DCD_IND,
                d.TX_CENTER_COUNT_250NM,
                d.KDPI,
                d.KDPI_BIN,
                COUNT(*) AS offer_count,
                COUNT(DISTINCT l.MATCH_ID) AS distinct_match_count,
                COUNT(DISTINCT l.PX_ID) AS distinct_candidate_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') AS accept_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Z') AS provisional_yes_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'N') AS decline_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'B') AS bypass_count,
                MIN(l.MATCH_SUBMIT_DT) AS first_match_submit_dt,
                MAX(l.MATCH_SUBMIT_DT) AS last_match_submit_dt,
                ROUND(AVG(l.PTR_SEQUENCE_NUM), 3) AS avg_sequence_num,
                ROUND(
                    COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') * 1.0
                    / NULLIF(COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT IN ('Y', 'N')), 0),
                    6
                ) AS positive_response_rate
            FROM match_runs AS l
            LEFT JOIN saf_link.donor_dim AS d
                ON l.DONOR_ID = d.DONOR_ID
            WHERE l.match_year = {year}
            GROUP BY 1, 2, 3, 4, 5, 6, 7
            """,
            indexes=[
                ("idx_analytics_donor_year_stats_donor_year", "DONOR_ID, match_year"),
                ("idx_analytics_donor_year_stats_kdpi", "KDPI_BIN, match_year"),
            ],
        )

        create_year_partitioned_table(
            con,
            "analytics.match_opo_year_stats",
            match_years,
            """
            SELECT
                l.match_year,
                l.MATCH_OPO_CTR_CD,
                c.ENTIRE_NAME AS MATCH_OPO_NAME,
                c.PRIMARY_STATE AS MATCH_OPO_STATE,
                COUNT(*) AS offer_count,
                COUNT(DISTINCT l.MATCH_ID) AS distinct_match_count,
                COUNT(DISTINCT l.DONOR_ID) AS distinct_donor_count,
                COUNT(DISTINCT l.PX_ID) AS distinct_candidate_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') AS accept_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Z') AS provisional_yes_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'N') AS decline_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'B') AS bypass_count,
                ROUND(
                    COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') * 1.0
                    / NULLIF(COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT IN ('Y', 'N')), 0),
                    6
                ) AS positive_response_rate
            FROM match_runs AS l
            LEFT JOIN saf_link.center_code_dim AS c
                ON l.MATCH_OPO_CTR_CD = c.CTR_CD
            WHERE l.match_year = {year}
            GROUP BY 1, 2, 3, 4
            """,
            indexes=[("idx_analytics_match_opo_year_stats", "MATCH_OPO_CTR_CD, match_year")],
        )

        create_year_partitioned_table(
            con,
            "analytics.tx_center_match_year_stats",
            match_years,
            """
            SELECT
                l.match_year,
                c.CAN_LISTING_CTR_CD,
                c.CAN_LISTING_CTR_TY,
                ctr.ENTIRE_NAME AS CENTER_NAME,
                ctr.PRIMARY_STATE AS CENTER_STATE,
                COUNT(*) AS offer_count,
                COUNT(DISTINCT l.MATCH_ID) AS distinct_match_count,
                COUNT(DISTINCT l.DONOR_ID) AS distinct_donor_count,
                COUNT(DISTINCT l.PX_ID) AS distinct_candidate_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') AS accept_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Z') AS provisional_yes_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'N') AS decline_count,
                COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'B') AS bypass_count,
                ROUND(
                    COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT = 'Y') * 1.0
                    / NULLIF(COUNT(*) FILTER (WHERE l.PTR_OFFER_ACPT IN ('Y', 'N')), 0),
                    6
                ) AS positive_response_rate
            FROM match_runs AS l
            LEFT JOIN saf_link.candidate_dim AS c
                ON l.PX_ID = c.PX_ID
            LEFT JOIN saf_link.center_dim AS ctr
                ON c.CAN_LISTING_CTR_CD = ctr.CTR_CD
               AND coalesce(c.CAN_LISTING_CTR_TY, '') = coalesce(ctr.CTR_TY, '')
            WHERE l.match_year = {year}
            GROUP BY 1, 2, 3, 4, 5
            """,
            indexes=[("idx_analytics_tx_center_match_year_stats", "CAN_LISTING_CTR_CD, CAN_LISTING_CTR_TY, match_year")],
        )

        create_table(
            con,
            "analytics.transplant_center_year_stats",
            """
            SELECT
                EXTRACT(YEAR FROM t.REC_TX_DT) AS tx_year,
                t.REC_CTR_CD,
                t.REC_CTR_TY,
                ctr.ENTIRE_NAME AS CENTER_NAME,
                ctr.PRIMARY_STATE AS CENTER_STATE,
                COUNT(*) AS transplant_count,
                COUNT(DISTINCT t.PX_ID) AS distinct_recipient_count,
                COUNT(DISTINCT t.DONOR_ID) AS distinct_donor_count,
                ROUND(AVG(d.KDPI), 3) AS avg_donor_kdpi,
                ROUND(AVG(f.TFL_CREAT), 3) AS avg_latest_followup_creat
            FROM saf_link.tx_ki_link AS t
            LEFT JOIN saf_link.donor_dim AS d
                ON t.DONOR_ID = d.DONOR_ID
            LEFT JOIN analytics.transplant_followup_latest AS f
                ON t.TRR_ID = f.TRR_ID
            LEFT JOIN saf_link.center_dim AS ctr
                ON t.REC_CTR_CD = ctr.CTR_CD
               AND coalesce(t.REC_CTR_TY, '') = coalesce(ctr.CTR_TY, '')
            GROUP BY 1, 2, 3, 4, 5
            """,
            indexes=[("idx_analytics_transplant_center_year_stats", "REC_CTR_CD, REC_CTR_TY, tx_year")],
        )

        create_table(
            con,
            "analytics.recovery_opo_year_stats",
            """
            SELECT
                EXTRACT(YEAR FROM d.DON_RECOV_DT) AS recov_year,
                d.DON_OPO_CTR_ID,
                ctr.CTR_CD AS OPO_CTR_CD,
                ctr.CTR_TY AS OPO_CTR_TY,
                ctr.ENTIRE_NAME AS OPO_NAME,
                ctr.PRIMARY_STATE AS OPO_STATE,
                MAX(d.TX_CENTER_COUNT_250NM) AS TX_CENTER_COUNT_250NM,
                COUNT(*) AS recovered_donor_count,
                ROUND(AVG(d.KDPI), 3) AS avg_kdpi,
                COUNT(*) FILTER (WHERE d.KDPI_BIN = '0-20') AS kdpi_0_20_count,
                COUNT(*) FILTER (WHERE d.KDPI_BIN = '20-40') AS kdpi_20_40_count,
                COUNT(*) FILTER (WHERE d.KDPI_BIN = '40-60') AS kdpi_40_60_count,
                COUNT(*) FILTER (WHERE d.KDPI_BIN = '60-80') AS kdpi_60_80_count,
                COUNT(*) FILTER (WHERE d.KDPI_BIN = '80-100') AS kdpi_80_100_count
            FROM saf_link.donor_dim AS d
            LEFT JOIN saf_link.center_dim AS ctr
                ON d.DON_OPO_CTR_ID = ctr.CTR_ID
            GROUP BY 1, 2, 3, 4, 5, 6
            """,
            indexes=[("idx_analytics_recovery_opo_year_stats", "DON_OPO_CTR_ID, recov_year")],
        )

        print("[build] macros", flush=True)
        con.execute(
            """
            CREATE OR REPLACE MACRO candidate_history_enriched(px_id_param) AS TABLE
            SELECT *
            FROM analytics.match_offer_to_transplant
            WHERE PX_ID = px_id_param
            ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE MACRO donor_history_enriched(donor_id_param) AS TABLE
            SELECT *
            FROM analytics.match_offer_to_transplant
            WHERE DONOR_ID = donor_id_param
            ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE MACRO match_opo_history(opo_code_param) AS TABLE
            SELECT *
            FROM analytics.match_offer_to_transplant
            WHERE MATCH_OPO_CTR_CD = opo_code_param
            ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE MACRO listing_center_history(center_cd_param) AS TABLE
            SELECT *
            FROM analytics.match_offer_to_transplant
            WHERE CAN_LISTING_CTR_CD = center_cd_param
            ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
            """
        )
        con.execute("CHECKPOINT;")
        write_manifest(con, manifest_path=manifest_path, match_db=match_db, saf_db=saf_db)
        if _ACTIVE_PROGRESS_TRACKER:
            _ACTIVE_PROGRESS_TRACKER.advance("manifest")
        con.execute("DETACH saf_src;")
    finally:
        _ACTIVE_PROGRESS_TRACKER = None
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a query-optimized SAF link layer and analytics schema inside the match-run DuckDB."
    )
    parser.add_argument(
        "--match-db",
        type=Path,
        default=Path("warehouse/match_runs/match_runs.duckdb"),
        help="Match-run DuckDB that will receive the linked SAF query layer.",
    )
    parser.add_argument(
        "--saf-db",
        type=Path,
        default=Path("warehouse/saf/saf.duckdb"),
        help="Source SAF DuckDB database.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("warehouse/match_runs/match_saf_query_layer_manifest.json"),
        help="JSON manifest describing the linked SAF/query-layer objects.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="DuckDB thread count.",
    )
    parser.add_argument(
        "--allow-offer-base-row-count-mismatch",
        action="store_true",
        help=(
            "Continue building even if analytics.offer_feature_base_exact does not exactly "
            "match match_runs row-for-row. Use only for training/export builds while diagnosing "
            "residual source-to-feature mismatches."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help=(
            "Optional checkpoint relation to resume from on an existing training DB. "
            f"Supported values: {RESUME_FROM_CENTER_HISTORY}, "
            f"{RESUME_FROM_PAIR_LONG_DISTANCE_HISTORY}"
        ),
    )
    parser.add_argument(
        "--skip-opo-center-pair-mm-bucket",
        action="store_true",
        help=(
            "Skip OPO-center-pair MM bucket history features and continue with the "
            "remaining pair-history join, downstream views, stats, and export."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_query_layer(
        match_db=args.match_db,
        saf_db=args.saf_db,
        manifest_path=args.manifest,
        threads=args.threads,
        allow_offer_base_row_count_mismatch=args.allow_offer_base_row_count_mismatch,
        resume_from=args.resume_from,
        skip_opo_center_pair_mm_bucket=args.skip_opo_center_pair_mm_bucket,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
