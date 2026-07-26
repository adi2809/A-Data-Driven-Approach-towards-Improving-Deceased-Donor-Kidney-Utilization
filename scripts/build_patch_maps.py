#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patch_features import (
    CANDIDATE_HISTORY_PATCH_FEATURES,
    CENTER_OFFER_HISTORY_PATCH_FEATURES,
    MATCH_LEVEL_PATCH_FEATURES,
    sql_quote,
)


ORDER_SQL = (
    "COALESCE(PTR_SEQUENCE_NUM, 0), "
    "COALESCE(PTR_ROW_ORDER, 0), "
    "COALESCE(OFFER_ROW_INSTANCE_ORDINAL, 0)"
)


@dataclass
class YearPatchMapStats:
    match_year: int
    candidate_groups_written: int
    center_groups_written: int
    match_groups_written: int
    candidate_files_written: int
    center_files_written: int
    match_files_written: int
    started_at_utc: str
    finished_at_utc: str
    elapsed_seconds: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _year_parquet_glob(parquet_root: Path, year: int) -> str:
    return str((parquet_root / f"match_year={year}" / "*.parquet").resolve())


def _discover_years(parquet_root: Path, selected_years: set[int] | None) -> list[int]:
    available = sorted(
        int(path.name.split("=")[1])
        for path in parquet_root.glob("match_year=*")
        if path.is_dir() and path.name.split("=")[1].isdigit()
    )
    if selected_years is None:
        return available
    return [year for year in available if year in selected_years]


def _candidate_map_sql(source_glob: str) -> str:
    return f"""
        WITH src AS (
            SELECT *
            FROM read_parquet('{sql_quote(source_glob)}')
        ),
        candidate_first_keys AS (
            SELECT
                MATCH_ID,
                PX_ID,
                COUNT(*) AS GROUP_ROW_COUNT,
                first(COALESCE(PTR_SEQUENCE_NUM, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_SEQUENCE_NUM,
                first(COALESCE(PTR_ROW_ORDER, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_ROW_ORDER,
                first(COALESCE(OFFER_ROW_INSTANCE_ORDINAL, 0) ORDER BY {ORDER_SQL}) AS FIRST_OFFER_ROW_INSTANCE_ORDINAL
            FROM src
            WHERE PX_ID IS NOT NULL
            GROUP BY 1, 2
            HAVING COUNT(*) > 1
        )
        SELECT
            keys.MATCH_ID,
            keys.PX_ID,
            keys.GROUP_ROW_COUNT,
            keys.FIRST_PTR_SEQUENCE_NUM,
            keys.FIRST_PTR_ROW_ORDER,
            keys.FIRST_OFFER_ROW_INSTANCE_ORDINAL
        FROM candidate_first_keys AS keys
    """


def _center_map_sql(source_glob: str) -> str:
    return f"""
        WITH src AS (
            SELECT *
            FROM read_parquet('{sql_quote(source_glob)}')
        ),
        center_first_keys AS (
            SELECT
                MATCH_ID,
                CAN_LISTING_CTR_CD,
                COALESCE(CAN_LISTING_CTR_TY, '') AS CAN_LISTING_CTR_TY_NORM,
                COUNT(*) AS GROUP_ROW_COUNT,
                first(COALESCE(PTR_SEQUENCE_NUM, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_SEQUENCE_NUM,
                first(COALESCE(PTR_ROW_ORDER, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_ROW_ORDER,
                first(COALESCE(OFFER_ROW_INSTANCE_ORDINAL, 0) ORDER BY {ORDER_SQL}) AS FIRST_OFFER_ROW_INSTANCE_ORDINAL
            FROM src
            WHERE CAN_LISTING_CTR_CD IS NOT NULL
            GROUP BY 1, 2, 3
            HAVING COUNT(*) > 1
        )
        SELECT
            keys.MATCH_ID,
            keys.CAN_LISTING_CTR_CD,
            keys.CAN_LISTING_CTR_TY_NORM,
            keys.GROUP_ROW_COUNT,
            keys.FIRST_PTR_SEQUENCE_NUM,
            keys.FIRST_PTR_ROW_ORDER,
            keys.FIRST_OFFER_ROW_INSTANCE_ORDINAL
        FROM center_first_keys AS keys
    """


def _match_map_sql(source_glob: str) -> str:
    return f"""
        WITH src AS (
            SELECT *
            FROM read_parquet('{sql_quote(source_glob)}')
        ),
        match_first_keys AS (
            SELECT
                MATCH_ID,
                COUNT(*) AS GROUP_ROW_COUNT,
                first(COALESCE(PTR_SEQUENCE_NUM, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_SEQUENCE_NUM,
                first(COALESCE(PTR_ROW_ORDER, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_ROW_ORDER,
                first(COALESCE(OFFER_ROW_INSTANCE_ORDINAL, 0) ORDER BY {ORDER_SQL}) AS FIRST_OFFER_ROW_INSTANCE_ORDINAL
            FROM src
            GROUP BY 1
            HAVING COUNT(*) > 1
        )
        SELECT
            keys.MATCH_ID,
            keys.GROUP_ROW_COUNT,
            keys.FIRST_PTR_SEQUENCE_NUM,
            keys.FIRST_PTR_ROW_ORDER,
            keys.FIRST_OFFER_ROW_INSTANCE_ORDINAL
        FROM match_first_keys AS keys
    """


def _write_sql_to_dir(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    out_dir: Path,
    compression: str,
) -> tuple[int, int]:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_count = con.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0]
    if int(row_count) == 0:
        return 0, 0
    out_glob = str((out_dir / "*.parquet").resolve())
    con.execute(
        f"""
        COPY ({sql})
        TO '{sql_quote(str(out_dir.resolve()))}'
        (FORMAT PARQUET, COMPRESSION {compression}, PER_THREAD_OUTPUT TRUE);
        """
    )
    written_row_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_quote(out_glob)}')"
    ).fetchone()[0]
    file_count = len(list(out_dir.glob("*.parquet")))
    if int(row_count) != int(written_row_count):
        raise RuntimeError(f"row count mismatch while writing patch map to {out_dir}: {written_row_count} != {row_count}")
    return int(written_row_count), file_count


def _validate_sample_groups(
    con: duckdb.DuckDBPyConnection,
    source_glob: str,
    candidate_dir: Path,
    center_dir: Path,
) -> tuple[int, int]:
    candidate_ok = 0
    if list(candidate_dir.glob("*.parquet")):
        candidate_glob = str((candidate_dir / "*.parquet").resolve())
        candidate_ok = con.execute(
            f"""
            WITH src AS (
                SELECT *
                FROM read_parquet('{sql_quote(source_glob)}')
            ),
            src_first_keys AS (
                SELECT
                    MATCH_ID,
                    PX_ID,
                    first(COALESCE(PTR_SEQUENCE_NUM, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_SEQUENCE_NUM,
                    first(COALESCE(PTR_ROW_ORDER, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_ROW_ORDER,
                    first(COALESCE(OFFER_ROW_INSTANCE_ORDINAL, 0) ORDER BY {ORDER_SQL}) AS FIRST_OFFER_ROW_INSTANCE_ORDINAL
                FROM src
                WHERE PX_ID IS NOT NULL
                GROUP BY 1, 2
                HAVING COUNT(*) > 1
            ),
            sampled_keys AS (
                SELECT * FROM src_first_keys LIMIT 1000
            ),
            joined AS (
                SELECT
                    s.MATCH_ID,
                    s.PX_ID
                FROM sampled_keys AS s
                JOIN read_parquet('{sql_quote(candidate_glob)}') AS c
                  ON s.MATCH_ID = c.MATCH_ID
                 AND s.PX_ID = c.PX_ID
                WHERE s.FIRST_PTR_SEQUENCE_NUM = c.FIRST_PTR_SEQUENCE_NUM
                  AND s.FIRST_PTR_ROW_ORDER = c.FIRST_PTR_ROW_ORDER
                  AND s.FIRST_OFFER_ROW_INSTANCE_ORDINAL = c.FIRST_OFFER_ROW_INSTANCE_ORDINAL
            )
            SELECT COUNT(*) FROM joined
            """
        ).fetchone()[0]

    center_ok = 0
    if list(center_dir.glob("*.parquet")):
        center_glob = str((center_dir / "*.parquet").resolve())
        center_ok = con.execute(
            f"""
            WITH src AS (
                SELECT *
                FROM read_parquet('{sql_quote(source_glob)}')
            ),
            src_first_keys AS (
                SELECT
                    MATCH_ID,
                    CAN_LISTING_CTR_CD,
                    COALESCE(CAN_LISTING_CTR_TY, '') AS CAN_LISTING_CTR_TY_NORM,
                    first(COALESCE(PTR_SEQUENCE_NUM, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_SEQUENCE_NUM,
                    first(COALESCE(PTR_ROW_ORDER, 0) ORDER BY {ORDER_SQL}) AS FIRST_PTR_ROW_ORDER,
                    first(COALESCE(OFFER_ROW_INSTANCE_ORDINAL, 0) ORDER BY {ORDER_SQL}) AS FIRST_OFFER_ROW_INSTANCE_ORDINAL
                FROM src
                WHERE CAN_LISTING_CTR_CD IS NOT NULL
                GROUP BY 1, 2, 3
                HAVING COUNT(*) > 1
            ),
            sampled_keys AS (
                SELECT * FROM src_first_keys LIMIT 1000
            ),
            joined AS (
                SELECT
                    s.MATCH_ID,
                    s.CAN_LISTING_CTR_CD,
                    s.CAN_LISTING_CTR_TY_NORM
                FROM sampled_keys AS s
                JOIN read_parquet('{sql_quote(center_glob)}') AS c
                  ON s.MATCH_ID = c.MATCH_ID
                 AND s.CAN_LISTING_CTR_CD = c.CAN_LISTING_CTR_CD
                 AND s.CAN_LISTING_CTR_TY_NORM = c.CAN_LISTING_CTR_TY_NORM
                WHERE s.FIRST_PTR_SEQUENCE_NUM = c.FIRST_PTR_SEQUENCE_NUM
                  AND s.FIRST_PTR_ROW_ORDER = c.FIRST_PTR_ROW_ORDER
                  AND s.FIRST_OFFER_ROW_INSTANCE_ORDINAL = c.FIRST_OFFER_ROW_INSTANCE_ORDINAL
            )
            SELECT COUNT(*) FROM joined
            """
        ).fetchone()[0]
    return int(candidate_ok), int(center_ok)


def _write_manifest(
    manifest_path: Path,
    parquet_root: Path,
    output_root: Path,
    year_stats: list[YearPatchMapStats],
    compression: str,
) -> None:
    payload = {
        "built_at_utc": utc_now(),
        "source_parquet_root": str(parquet_root),
        "output_root": str(output_root),
        "build_kind": "same_match_history_patch_maps",
        "compression": compression,
        "candidate_history_patch_features": CANDIDATE_HISTORY_PATCH_FEATURES,
        "center_offer_history_patch_features": CENTER_OFFER_HISTORY_PATCH_FEATURES,
        "match_level_patch_features": MATCH_LEVEL_PATCH_FEATURES,
        "ignored_feature_families": [
            "listing_center_acceptance_history",
            "opo_history",
            "opo_center_pair_history",
            "candidate_tx_history",
        ],
        "join_keys": ["MATCH_ID", "PTR_ROW_ORDER", "OFFER_ROW_INSTANCE_ORDINAL"],
        "years": [asdict(item) for item in year_stats],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def _update_base_manifest(base_manifest_path: Path, patch_manifest_path: Path) -> None:
    if not base_manifest_path.exists():
        return
    payload = json.loads(base_manifest_path.read_text())
    payload["same_match_history_patch_maps"] = {
        "built_at_utc": utc_now(),
        "patch_manifest": str(patch_manifest_path),
        "candidate_history_patch_features": CANDIDATE_HISTORY_PATCH_FEATURES,
        "center_offer_history_patch_features": CENTER_OFFER_HISTORY_PATCH_FEATURES,
        "match_level_patch_features": MATCH_LEVEL_PATCH_FEATURES,
        "ignored_feature_families": [
            "listing_center_acceptance_history",
            "opo_history",
            "opo_center_pair_history",
            "candidate_tx_history",
        ],
    }
    base_manifest_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build same-match history patch maps for match_offer_features."
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=Path("warehouse/match_offer_features/parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("warehouse/match_offer_features/history_patch_maps"),
    )
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=Path("warehouse/match_offer_features/build_manifest.json"),
    )
    parser.add_argument(
        "--patch-manifest",
        type=Path,
        default=Path("warehouse/match_offer_features/history_patch_maps/manifest.json"),
    )
    parser.add_argument("--years", type=int, nargs="*")
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--compression", type=str, default="SNAPPY")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.parquet_root.exists():
        raise FileNotFoundError(f"parquet root not found: {args.parquet_root}")

    con = duckdb.connect()
    base_temp_dir = (args.output_root.parent / ".duckdb_tmp_match_offer_feature_patch_maps").resolve()
    run_temp_dir = base_temp_dir / f"run-{os.getpid()}-{int(time.time())}"
    try:
        con.execute(f"PRAGMA threads={max(1, int(args.threads))};")
        run_temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{sql_quote(str(run_temp_dir))}';")
        con.execute("SET max_temp_directory_size='480GiB';")
        con.execute("SET memory_limit='20GiB';")

        years = _discover_years(args.parquet_root, set(args.years) if args.years else None)
        stats: list[YearPatchMapStats] = []
        for year in years:
            started = time.perf_counter()
            started_at = utc_now()
            source_glob = _year_parquet_glob(args.parquet_root, year)

            candidate_dir = args.output_root / "candidate_first" / f"match_year={year}"
            center_dir = args.output_root / "center_first" / f"match_year={year}"
            match_dir = args.output_root / "match_first" / f"match_year={year}"
            if not args.overwrite:
                for directory in (candidate_dir, center_dir, match_dir):
                    if directory.exists():
                        raise FileExistsError(f"{directory} already exists. Use --overwrite to replace it.")

            print(f"[start] patch maps match_year={year}", flush=True)
            candidate_count, candidate_files = _write_sql_to_dir(
                con,
                _candidate_map_sql(source_glob),
                candidate_dir,
                args.compression,
            )
            center_count, center_files = _write_sql_to_dir(
                con,
                _center_map_sql(source_glob),
                center_dir,
                args.compression,
            )
            match_count, match_files = _write_sql_to_dir(
                con,
                _match_map_sql(source_glob),
                match_dir,
                args.compression,
            )
            candidate_ok, center_ok = _validate_sample_groups(
                con,
                source_glob,
                candidate_dir,
                center_dir,
            )
            if candidate_ok != min(candidate_count, 1000):
                raise RuntimeError(
                    f"candidate map validation failed for match_year={year}: {candidate_ok} != {min(candidate_count, 1000)}"
                )
            if center_ok != min(center_count, 1000):
                raise RuntimeError(
                    f"center map validation failed for match_year={year}: {center_ok} != {min(center_count, 1000)}"
                )

            year_stats = YearPatchMapStats(
                match_year=year,
                candidate_groups_written=candidate_count,
                center_groups_written=center_count,
                match_groups_written=match_count,
                candidate_files_written=candidate_files,
                center_files_written=center_files,
                match_files_written=match_files,
                started_at_utc=started_at,
                finished_at_utc=utc_now(),
                elapsed_seconds=round(time.perf_counter() - started, 2),
            )
            stats.append(year_stats)
            print(
                f"[done] match_year={year} candidate_groups={candidate_count} center_groups={center_count} "
                f"match_groups={match_count}",
                flush=True,
            )
    finally:
        con.close()
        shutil.rmtree(run_temp_dir, ignore_errors=True)

    _write_manifest(args.patch_manifest, args.parquet_root, args.output_root, stats, args.compression)
    _update_base_manifest(args.build_manifest, args.patch_manifest)
    print(f"[done] patch_manifest={args.patch_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
