#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb


CANDIDATE_HISTORY_PATCH_FEATURES = [
    "LAST_YN_OFFER_KDPI_BIN",
    "CAND_DECLINE_COUNT_30D",
    "CAND_DECLINED_KDPI_AVG_30D",
    "CAND_DECLINED_KDPI_STDDEV_30D",
    "CAND_DECLINED_DON_CREAT_AVG_30D",
    "CAND_DECLINED_DON_CREAT_STDDEV_30D",
    "CAND_DECLINED_MM_TOTAL_AVG_30D",
    "CAND_DECLINED_MM_TOTAL_STDDEV_30D",
    "CAND_DECLINED_DON_AGE_AVG_30D",
    "CAND_DECLINED_DON_AGE_STDDEV_30D",
    "CAND_DECLINED_DCD_FRAC_30D",
    "CAND_DECLINED_HCV_FRAC_30D",
    "CAND_DECLINE_COUNT_90D",
    "CAND_DECLINED_KDPI_AVG_90D",
    "CAND_DECLINED_KDPI_STDDEV_90D",
    "CAND_DECLINED_DON_CREAT_AVG_90D",
    "CAND_DECLINED_DON_CREAT_STDDEV_90D",
    "CAND_DECLINED_MM_TOTAL_AVG_90D",
    "CAND_DECLINED_MM_TOTAL_STDDEV_90D",
    "CAND_DECLINED_DON_AGE_AVG_90D",
    "CAND_DECLINED_DON_AGE_STDDEV_90D",
    "CAND_DECLINED_DCD_FRAC_90D",
    "CAND_DECLINED_HCV_FRAC_90D",
    "CAND_DECLINE_COUNT_150D",
    "CAND_DECLINED_KDPI_AVG_150D",
    "CAND_DECLINED_KDPI_STDDEV_150D",
    "CAND_DECLINED_DON_CREAT_AVG_150D",
    "CAND_DECLINED_DON_CREAT_STDDEV_150D",
    "CAND_DECLINED_MM_TOTAL_AVG_150D",
    "CAND_DECLINED_MM_TOTAL_STDDEV_150D",
    "CAND_DECLINED_DON_AGE_AVG_150D",
    "CAND_DECLINED_DON_AGE_STDDEV_150D",
    "CAND_DECLINED_DCD_FRAC_150D",
    "CAND_DECLINED_HCV_FRAC_150D",
    "CAND_DECLINE_COUNT_365D",
    "CAND_DECLINED_KDPI_AVG_365D",
    "CAND_DECLINED_KDPI_STDDEV_365D",
    "CAND_DECLINED_DON_CREAT_AVG_365D",
    "CAND_DECLINED_DON_CREAT_STDDEV_365D",
    "CAND_DECLINED_MM_TOTAL_AVG_365D",
    "CAND_DECLINED_MM_TOTAL_STDDEV_365D",
    "CAND_DECLINED_DON_AGE_AVG_365D",
    "CAND_DECLINED_DON_AGE_STDDEV_365D",
    "CAND_DECLINED_DCD_FRAC_365D",
    "CAND_DECLINED_HCV_FRAC_365D",
    "TIME_SINCE_LAST_OFFER_DAYS",
]


CENTER_OFFER_HISTORY_PATCH_FEATURES = [
    "CENTER_YN_OFFER_COUNT_30D",
    "CENTER_POSITIVE_RESPONSE_RATE_30D",
    "CENTER_YN_OFFER_COUNT_90D",
    "CENTER_POSITIVE_RESPONSE_RATE_90D",
    "CENTER_YN_OFFER_COUNT_150D",
    "CENTER_POSITIVE_RESPONSE_RATE_150D",
    "CENTER_YN_OFFER_COUNT_365D",
    "CENTER_POSITIVE_RESPONSE_RATE_365D",
    "CENTER_RATE_SAME_DCD_30D",
    "CENTER_RATE_SAME_DCD_90D",
    "CENTER_RATE_SAME_DCD_150D",
    "CENTER_RATE_SAME_DCD_365D",
    "CENTER_RATE_SAME_HIGH_KDPI_30D",
    "CENTER_RATE_SAME_HIGH_KDPI_90D",
    "CENTER_RATE_SAME_HIGH_KDPI_150D",
    "CENTER_RATE_SAME_HIGH_KDPI_365D",
    "CENTER_RATE_SAME_HCV_POS_30D",
    "CENTER_RATE_SAME_HCV_POS_90D",
    "CENTER_RATE_SAME_HCV_POS_150D",
    "CENTER_RATE_SAME_HCV_POS_365D",
    "CENTER_RATE_SAME_LONG_DISTANCE_30D",
    "CENTER_RATE_SAME_LONG_DISTANCE_90D",
    "CENTER_RATE_SAME_LONG_DISTANCE_150D",
    "CENTER_RATE_SAME_LONG_DISTANCE_365D",
    "CENTER_RATE_SAME_MM_BUCKET_30D",
    "CENTER_RATE_SAME_MM_BUCKET_90D",
    "CENTER_RATE_SAME_MM_BUCKET_150D",
    "CENTER_RATE_SAME_MM_BUCKET_365D",
]


MATCH_LEVEL_PATCH_FEATURES = [
    "SAME_MATCH_PRIOR_DECLINER_COUNT",
]


ORDER_SQL = (
    "COALESCE(PTR_SEQUENCE_NUM, 0), "
    "COALESCE(PTR_ROW_ORDER, 0), "
    "COALESCE(OFFER_ROW_INSTANCE_ORDINAL, 0)"
)


@dataclass
class YearPatchStats:
    match_year: int
    source_row_count: int
    output_row_count: int
    files_written: int
    candidate_patch_groups: int
    center_patch_groups: int
    match_patch_groups: int
    candidate_groups_before: int
    candidate_groups_after: int
    center_groups_before: int
    center_groups_after: int
    started_at_utc: str
    finished_at_utc: str
    elapsed_seconds: float
    output_dir: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sql_quote(value: str) -> str:
    return value.replace("'", "''")


def _year_dir(root: Path, year: int) -> Path:
    return root / f"match_year={year}"


def _year_glob(root: Path, year: int) -> str:
    return str((_year_dir(root, year) / "*.parquet").resolve())


def _has_parquet_files(path: Path) -> bool:
    return path.exists() and any(path.glob("*.parquet"))


def _discover_years(source_root: Path, selected_years: set[int] | None) -> list[int]:
    available = sorted(
        int(path.name.split("=")[1])
        for path in source_root.glob("match_year=*")
        if path.is_dir() and path.name.split("=")[1].isdigit()
    )
    if selected_years is None:
        return available
    return [year for year in available if year in selected_years]


def _read_schema_columns_from_glob(con: duckdb.DuckDBPyConnection, parquet_glob: str) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{sql_quote(parquet_glob)}')"
    ).fetchall()
    return [row[0] for row in rows]


def _read_map_group_count(con: duckdb.DuckDBPyConnection, parquet_dir: Path) -> int:
    if not _has_parquet_files(parquet_dir):
        return 0
    parquet_glob = str((parquet_dir / "*.parquet").resolve())
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_quote(parquet_glob)}')"
        ).fetchone()[0]
    )


def _sentinel_validation_counts(
    con: duckdb.DuckDBPyConnection,
    parquet_path_or_glob: str,
    candidate_feature: str,
    center_feature: str,
) -> tuple[int, int, int, int]:
    row = con.execute(
        f"""
        WITH source AS (
            SELECT * FROM read_parquet('{sql_quote(parquet_path_or_glob)}')
        ),
        candidate_groups AS (
            SELECT
                MATCH_ID,
                PX_ID,
                COUNT(*) AS n_rows,
                MIN({candidate_feature}) AS min_value,
                MAX({candidate_feature}) AS max_value
            FROM source
            WHERE PX_ID IS NOT NULL
            GROUP BY 1, 2
            HAVING COUNT(*) >= 2
        ),
        center_groups AS (
            SELECT
                MATCH_ID,
                CAN_LISTING_CTR_CD,
                COALESCE(CAN_LISTING_CTR_TY, '') AS ctr_ty,
                COUNT(*) AS n_rows,
                MIN({center_feature}) AS min_value,
                MAX({center_feature}) AS max_value
            FROM source
            WHERE CAN_LISTING_CTR_CD IS NOT NULL
            GROUP BY 1, 2, 3
            HAVING COUNT(*) >= 2
        )
        SELECT
            (SELECT COUNT(*) FROM candidate_groups),
            (SELECT COUNT(*) FROM candidate_groups WHERE min_value IS DISTINCT FROM max_value),
            (SELECT COUNT(*) FROM center_groups),
            (SELECT COUNT(*) FROM center_groups WHERE min_value IS DISTINCT FROM max_value)
        """
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2]), int(row[3])


def _map_values_cte(
    family: str,
    map_glob: str,
    source_glob: str,
    value_columns: list[str],
) -> str:
    source_sql = sql_quote(source_glob)
    map_sql = sql_quote(map_glob)
    value_projection = ",\n                ".join(
        f"src_first.{column} AS {column}" for column in value_columns
    )
    if family == "candidate_first":
        return f"""
        candidate_values AS (
            SELECT
                maps.MATCH_ID,
                maps.PX_ID,
                {value_projection}
            FROM read_parquet('{map_sql}') AS maps
            JOIN read_parquet('{source_sql}') AS src_first
              ON src_first.MATCH_ID = maps.MATCH_ID
             AND COALESCE(src_first.PTR_SEQUENCE_NUM, 0) = maps.FIRST_PTR_SEQUENCE_NUM
             AND COALESCE(src_first.PTR_ROW_ORDER, 0) = maps.FIRST_PTR_ROW_ORDER
             AND COALESCE(src_first.OFFER_ROW_INSTANCE_ORDINAL, 0) = maps.FIRST_OFFER_ROW_INSTANCE_ORDINAL
        )
        """
    if family == "center_first":
        return f"""
        center_values AS (
            SELECT
                maps.MATCH_ID,
                maps.CAN_LISTING_CTR_CD,
                maps.CAN_LISTING_CTR_TY_NORM,
                {value_projection}
            FROM read_parquet('{map_sql}') AS maps
            JOIN read_parquet('{source_sql}') AS src_first
              ON src_first.MATCH_ID = maps.MATCH_ID
             AND COALESCE(src_first.PTR_SEQUENCE_NUM, 0) = maps.FIRST_PTR_SEQUENCE_NUM
             AND COALESCE(src_first.PTR_ROW_ORDER, 0) = maps.FIRST_PTR_ROW_ORDER
             AND COALESCE(src_first.OFFER_ROW_INSTANCE_ORDINAL, 0) = maps.FIRST_OFFER_ROW_INSTANCE_ORDINAL
        )
        """
    if family == "match_first":
        return f"""
        match_values AS (
            SELECT
                maps.MATCH_ID,
                {value_projection}
            FROM read_parquet('{map_sql}') AS maps
            JOIN read_parquet('{source_sql}') AS src_first
              ON src_first.MATCH_ID = maps.MATCH_ID
             AND COALESCE(src_first.PTR_SEQUENCE_NUM, 0) = maps.FIRST_PTR_SEQUENCE_NUM
             AND COALESCE(src_first.PTR_ROW_ORDER, 0) = maps.FIRST_PTR_ROW_ORDER
             AND COALESCE(src_first.OFFER_ROW_INSTANCE_ORDINAL, 0) = maps.FIRST_OFFER_ROW_INSTANCE_ORDINAL
        )
        """
    raise ValueError(f"unsupported family: {family}")


def _build_patched_select_sql(
    columns: list[str],
    source_glob: str,
    candidate_map_glob: str | None,
    center_map_glob: str | None,
    match_map_glob: str | None,
) -> str:
    ctes = [
        f"""
        src AS (
            SELECT *
            FROM read_parquet('{sql_quote(source_glob)}')
        )
        """.strip()
    ]
    joins: list[str] = []
    projection: list[str] = []

    candidate_enabled = candidate_map_glob is not None
    center_enabled = center_map_glob is not None
    match_enabled = match_map_glob is not None

    if candidate_enabled:
        ctes.append(
            _map_values_cte(
                "candidate_first",
                candidate_map_glob,
                source_glob,
                CANDIDATE_HISTORY_PATCH_FEATURES,
            ).strip()
        )
        joins.append(
            """
            LEFT JOIN candidate_values AS cand
              ON src.MATCH_ID = cand.MATCH_ID
             AND src.PX_ID = cand.PX_ID
            """.strip()
        )

    if center_enabled:
        ctes.append(
            _map_values_cte(
                "center_first",
                center_map_glob,
                source_glob,
                CENTER_OFFER_HISTORY_PATCH_FEATURES,
            ).strip()
        )
        joins.append(
            """
            LEFT JOIN center_values AS ctr
              ON src.MATCH_ID = ctr.MATCH_ID
             AND src.CAN_LISTING_CTR_CD = ctr.CAN_LISTING_CTR_CD
             AND COALESCE(src.CAN_LISTING_CTR_TY, '') = ctr.CAN_LISTING_CTR_TY_NORM
            """.strip()
        )

    if match_enabled:
        ctes.append(
            _map_values_cte(
                "match_first",
                match_map_glob,
                source_glob,
                MATCH_LEVEL_PATCH_FEATURES,
            ).strip()
        )
        joins.append(
            """
            LEFT JOIN match_values AS mfirst
              ON src.MATCH_ID = mfirst.MATCH_ID
            """.strip()
        )

    for column in columns:
        if candidate_enabled and column in CANDIDATE_HISTORY_PATCH_FEATURES:
            projection.append(
                f"CASE WHEN cand.MATCH_ID IS NULL THEN src.{column} ELSE cand.{column} END AS {column}"
            )
        elif center_enabled and column in CENTER_OFFER_HISTORY_PATCH_FEATURES:
            projection.append(
                f"CASE WHEN ctr.MATCH_ID IS NULL THEN src.{column} ELSE ctr.{column} END AS {column}"
            )
        elif match_enabled and column in MATCH_LEVEL_PATCH_FEATURES:
            projection.append(
                f"CASE WHEN mfirst.MATCH_ID IS NULL THEN src.{column} ELSE mfirst.{column} END AS {column}"
            )
        else:
            projection.append(f"src.{column}")

    cte_sql = ",\n        ".join(ctes)
    join_sql = "\n        ".join(joins)
    projection_sql = ",\n            ".join(projection)
    return f"""
        WITH
        {cte_sql}
        SELECT
            {projection_sql}
        FROM src
        {join_sql}
    """


def _patch_year(
    con: duckdb.DuckDBPyConnection,
    source_root: Path,
    patch_map_root: Path,
    output_root: Path,
    year: int,
    row_group_size: int,
    overwrite_output: bool,
) -> YearPatchStats:
    source_year_dir = _year_dir(source_root, year)
    source_glob = _year_glob(source_root, year)
    if not _has_parquet_files(source_year_dir):
        raise FileNotFoundError(f"missing source parquet for match_year={year}: {source_year_dir}")

    candidate_map_dir = patch_map_root / "candidate_first" / f"match_year={year}"
    center_map_dir = patch_map_root / "center_first" / f"match_year={year}"
    match_map_dir = patch_map_root / "match_first" / f"match_year={year}"

    candidate_map_glob = (
        str((candidate_map_dir / "*.parquet").resolve()) if _has_parquet_files(candidate_map_dir) else None
    )
    center_map_glob = (
        str((center_map_dir / "*.parquet").resolve()) if _has_parquet_files(center_map_dir) else None
    )
    match_map_glob = (
        str((match_map_dir / "*.parquet").resolve()) if _has_parquet_files(match_map_dir) else None
    )

    output_year_dir = _year_dir(output_root, year)
    temp_year_dir = output_root / f".match_year={year}.writing-{os.getpid()}-{int(time.time())}"
    if output_year_dir.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"output already exists for match_year={year}: {output_year_dir}; pass --overwrite-output to replace it"
            )
        shutil.rmtree(output_year_dir)
    shutil.rmtree(temp_year_dir, ignore_errors=True)
    temp_year_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started = time.perf_counter()

    source_row_count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_quote(source_glob)}')"
        ).fetchone()[0]
    )
    candidate_groups_before, candidate_drift_before, center_groups_before, center_drift_before = (
        _sentinel_validation_counts(
            con,
            source_glob,
            candidate_feature="TIME_SINCE_LAST_OFFER_DAYS",
            center_feature="CENTER_YN_OFFER_COUNT_30D",
        )
    )

    select_sql = _build_patched_select_sql(
        columns=_read_schema_columns_from_glob(con, source_glob),
        source_glob=source_glob,
        candidate_map_glob=candidate_map_glob,
        center_map_glob=center_map_glob,
        match_map_glob=match_map_glob,
    )
    try:
        con.execute(
            f"""
            COPY ({select_sql})
            TO '{sql_quote(str(temp_year_dir.resolve()))}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_size)}, PER_THREAD_OUTPUT TRUE);
            """
        )
        temp_glob = str((temp_year_dir / "*.parquet").resolve())
        output_row_count = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{sql_quote(temp_glob)}')"
            ).fetchone()[0]
        )
        if output_row_count != source_row_count:
            raise RuntimeError(
                f"row count mismatch for match_year={year}: output={output_row_count} source={source_row_count}"
            )

        _, candidate_drift_after, _, center_drift_after = _sentinel_validation_counts(
            con,
            temp_glob,
            candidate_feature="TIME_SINCE_LAST_OFFER_DAYS",
            center_feature="CENTER_YN_OFFER_COUNT_30D",
        )
        temp_year_dir.rename(output_year_dir)
    except Exception:
        shutil.rmtree(temp_year_dir, ignore_errors=True)
        raise

    return YearPatchStats(
        match_year=year,
        source_row_count=source_row_count,
        output_row_count=output_row_count,
        files_written=len(list(output_year_dir.glob("*.parquet"))),
        candidate_patch_groups=_read_map_group_count(con, candidate_map_dir),
        center_patch_groups=_read_map_group_count(con, center_map_dir),
        match_patch_groups=_read_map_group_count(con, match_map_dir),
        candidate_groups_before=candidate_drift_before,
        candidate_groups_after=candidate_drift_after,
        center_groups_before=center_drift_before,
        center_groups_after=center_drift_after,
        started_at_utc=started_at,
        finished_at_utc=utc_now(),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        output_dir=str(output_year_dir),
    )


def _write_manifest(
    manifest_path: Path,
    source_root: Path,
    patch_map_root: Path,
    output_root: Path,
    year_stats: list[YearPatchStats],
) -> None:
    payload = {
        "patched_at_utc": utc_now(),
        "source_root": str(source_root),
        "patch_map_root": str(patch_map_root),
        "output_root": str(output_root),
        "patch_method": "replace_same_match_candidate_center_and_match_history_features_with_first_row_values_using_patch_maps",
        "candidate_history_patch_features": CANDIDATE_HISTORY_PATCH_FEATURES,
        "center_offer_history_patch_features": CENTER_OFFER_HISTORY_PATCH_FEATURES,
        "match_level_patch_features": MATCH_LEVEL_PATCH_FEATURES,
        "ignored_feature_families": [
            "listing_center_acceptance_history",
            "opo_history",
            "opo_center_pair_history",
            "candidate_tx_history",
        ],
        "years": [asdict(item) for item in year_stats],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize corrected match_offer_features parquet using the existing same-match history patch maps."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("warehouse/match_offer_features/parquet"),
        help="Source match_offer_features parquet root.",
    )
    parser.add_argument(
        "--patch-map-root",
        type=Path,
        default=Path("warehouse/match_offer_features/history_patch_maps"),
        help="Root containing candidate_first/center_first/match_first patch-map parquet.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("warehouse/match_offer_features/parquet_same_match_history_fixed"),
        help="Output parquet root for the corrected feature export.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("warehouse/match_offer_features/same_match_history_fixed_manifest.json"),
        help="Output manifest for the corrected feature export.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        help="Optional subset of match years to patch.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace existing output year partitions if present.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=100_000,
        help="Row group size for rewritten parquet.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="DuckDB thread count.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_root.exists():
        raise FileNotFoundError(f"source root not found: {args.source_root}")
    if not args.patch_map_root.exists():
        raise FileNotFoundError(f"patch map root not found: {args.patch_map_root}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    base_temp_dir = (args.output_root.parent / ".duckdb_tmp_match_offer_feature_patches").resolve()
    run_temp_dir = base_temp_dir / f"run-{os.getpid()}-{int(time.time())}"
    try:
        con.execute(f"PRAGMA threads={max(1, int(args.threads))};")
        run_temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{sql_quote(str(run_temp_dir))}';")
        con.execute("SET max_temp_directory_size='480GiB';")
        con.execute("SET memory_limit='20GiB';")

        years = _discover_years(args.source_root, set(args.years) if args.years else None)
        stats: list[YearPatchStats] = []
        for year in years:
            print(f"[start] patch match_year={year}", flush=True)
            year_stats = _patch_year(
                con=con,
                source_root=args.source_root,
                patch_map_root=args.patch_map_root,
                output_root=args.output_root,
                year=year,
                row_group_size=args.row_group_size,
                overwrite_output=args.overwrite_output,
            )
            stats.append(year_stats)
            print(
                f"[done] match_year={year} rows={year_stats.output_row_count} "
                f"candidate_drift_before={year_stats.candidate_groups_before} "
                f"candidate_drift_after={year_stats.candidate_groups_after} "
                f"center_drift_before={year_stats.center_groups_before} "
                f"center_drift_after={year_stats.center_groups_after}",
                flush=True,
            )
    finally:
        con.close()
        shutil.rmtree(run_temp_dir, ignore_errors=True)

    _write_manifest(
        manifest_path=args.manifest,
        source_root=args.source_root,
        patch_map_root=args.patch_map_root,
        output_root=args.output_root,
        year_stats=stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
