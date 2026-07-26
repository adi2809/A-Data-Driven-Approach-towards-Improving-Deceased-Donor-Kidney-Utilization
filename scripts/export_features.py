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

from link_match_saf import (
    REQUIRED_MATCH_OFFER_ENRICHED_COLUMNS,
    validate_match_offer_enriched_schema,
)


SOURCE_VIEW = "analytics.match_offer_enriched"
LABEL_FILTER_SQL = "TRUE"
EXPORT_ORDER_SQL = (
    "MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER, OFFER_ROW_INSTANCE_ORDINAL"
)


@dataclass
class YearExportStats:
    match_year: int
    rows_exported: int
    files_written: int = 0
    min_match_submit_dt: str | None = None
    max_match_submit_dt: str | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    elapsed_seconds: float | None = None
    output_dir: str | None = None


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sql_quote(value: str) -> str:
    return value.replace("'", "''")


def export_schema(con: duckdb.DuckDBPyConnection) -> list[dict[str, str]]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM {SOURCE_VIEW} WHERE {LABEL_FILTER_SQL} LIMIT 0"
    ).fetchall()
    return [{"column_name": row[0], "column_type": row[1]} for row in rows]


def discover_years(
    con: duckdb.DuckDBPyConnection,
    selected_years: set[int] | None,
) -> list[int]:
    if selected_years is not None:
        return sorted(selected_years)

    available_years = [
        row[0]
        for row in con.execute(
            f"""
            SELECT DISTINCT match_year
            FROM {SOURCE_VIEW}
            WHERE match_year IS NOT NULL
              AND {LABEL_FILTER_SQL}
            ORDER BY match_year
            """
        ).fetchall()
    ]
    if selected_years is None:
        return available_years
    years = [year for year in available_years if year in selected_years]
    if not years:
        raise ValueError(f"No export years available from {SOURCE_VIEW} for {sorted(selected_years)}")
    return years


def validate_export_file(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
) -> tuple[int, str | None, str | None]:
    parquet_sql = sql_quote(str(parquet_path.resolve()))
    exported_row_count, min_submit_dt, max_submit_dt = con.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            MIN(MATCH_SUBMIT_DT) AS min_match_submit_dt,
            MAX(MATCH_SUBMIT_DT) AS max_match_submit_dt
        FROM read_parquet('{parquet_sql}')
        """
    ).fetchone()
    return (
        int(exported_row_count),
        min_submit_dt.isoformat() if min_submit_dt else None,
        max_submit_dt.isoformat() if max_submit_dt else None,
    )


def export_year(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
    year: int,
    row_group_size: int,
    overwrite: bool,
    ordered_export: bool,
    progress: ProgressTracker | None = None,
) -> YearExportStats:
    final_year_dir = output_dir / f"match_year={year}"
    temp_year_dir = output_dir / f"match_year={year}.tmp"

    if final_year_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{final_year_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(final_year_dir)

    if temp_year_dir.exists():
        shutil.rmtree(temp_year_dir)
    temp_year_dir.mkdir(parents=True, exist_ok=True)

    stats = YearExportStats(
        match_year=year,
        rows_exported=0,
        started_at_utc=utc_now(),
    )

    start = time.perf_counter()
    parquet_path = temp_year_dir / "part-00001.parquet"
    parquet_sql = sql_quote(str(parquet_path.resolve()))
    order_by_sql = f"\n                ORDER BY {EXPORT_ORDER_SQL}" if ordered_export else ""
    print(
        f"[start] match_year={year} ordered_export={str(ordered_export).lower()}",
        flush=True,
    )
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM {SOURCE_VIEW}
            WHERE match_year = {year}
              AND {LABEL_FILTER_SQL}
            {order_by_sql}
        ) TO '{parquet_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_size});
        """
    )
    row_count, min_submit_dt, max_submit_dt = validate_export_file(
        con,
        parquet_path=parquet_path,
    )
    stats.rows_exported = row_count
    stats.min_match_submit_dt = min_submit_dt
    stats.max_match_submit_dt = max_submit_dt
    stats.files_written = len(list(temp_year_dir.glob("*.parquet")))

    stats.elapsed_seconds = round(time.perf_counter() - start, 2)
    stats.finished_at_utc = utc_now()
    temp_year_dir.rename(final_year_dir)
    stats.output_dir = str(final_year_dir)
    if progress:
        progress.advance(f"export match_year={year}")
    return stats


def write_manifest(
    manifest_path: Path,
    match_db: Path,
    output_dir: Path,
    schema: list[dict[str, str]],
    year_stats: list[YearExportStats],
    ordered_export: bool,
) -> None:
    payload = {
        "built_at_utc": utc_now(),
        "match_db": str(match_db),
        "source_view": SOURCE_VIEW,
        "output_dir": str(output_dir),
        "response_code_filter": "all",
        "order_by": EXPORT_ORDER_SQL.split(", ") if ordered_export else None,
        "required_columns": sorted(REQUIRED_MATCH_OFFER_ENRICHED_COLUMNS),
        "schema": schema,
        "years": [asdict(item) for item in year_stats],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def build_match_offer_feature_exports(
    match_db: Path,
    output_dir: Path,
    manifest_path: Path,
    years: set[int] | None,
    overwrite: bool,
    row_group_size: int,
    threads: int,
    ordered_export: bool = False,
) -> list[YearExportStats]:
    if not match_db.exists():
        raise FileNotFoundError(f"Match-run DuckDB not found: {match_db}")

    con = duckdb.connect(str(match_db), read_only=True)
    base_temp_dir = (output_dir.parent / ".duckdb_tmp_match_offer_feature_exports").resolve()
    run_temp_dir = base_temp_dir / f"run-{os.getpid()}-{int(time.time())}"
    try:
        con.execute(f"PRAGMA threads={max(1, threads)};")
        run_temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{sql_quote(str(run_temp_dir))}';")
        con.execute("SET max_temp_directory_size='480GiB';")
        validate_match_offer_enriched_schema(con)

        schema = export_schema(con)
        export_years = discover_years(con, selected_years=years)
        progress = ProgressTracker(len(export_years) + 1)
        output_dir.mkdir(parents=True, exist_ok=True)
        year_stats = [
            export_year(
                con,
                output_dir=output_dir,
                year=year,
                row_group_size=row_group_size,
                overwrite=overwrite,
                ordered_export=ordered_export,
                progress=progress,
            )
            for year in export_years
        ]
        write_manifest(
            manifest_path=manifest_path,
            match_db=match_db,
            output_dir=output_dir,
            schema=schema,
            year_stats=year_stats,
            ordered_export=ordered_export,
        )
        progress.advance("manifest")
        return year_stats
    finally:
        try:
            con.close()
        finally:
            shutil.rmtree(run_temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export enriched kidney offer features as year-partitioned Parquet for all response-code rows."
    )
    parser.add_argument(
        "--match-db",
        type=Path,
        default=Path("warehouse/match_runs/match_runs.duckdb"),
        help="Match-run DuckDB containing analytics.match_offer_enriched.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("warehouse/match_offer_features/parquet"),
        help="Destination directory for match-year-partitioned enriched Parquet exports.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("warehouse/match_offer_features/build_manifest.json"),
        help="JSON manifest capturing export schema and per-year row counts.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        help="Optional subset of match years to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing year partitions.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=100_000,
        help="Parquet row group size for each exported year file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="DuckDB thread count.",
    )
    parser.add_argument(
        "--ordered-export",
        action="store_true",
        help="Force a deterministic ORDER BY during export. Disabled by default to avoid large sort bottlenecks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    year_stats = build_match_offer_feature_exports(
        match_db=args.match_db,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        years=set(args.years) if args.years else None,
        overwrite=args.overwrite,
        row_group_size=args.row_group_size,
        threads=args.threads,
        ordered_export=args.ordered_export,
    )
    for stats in year_stats:
        print(
            f"[done] match_year={stats.match_year} rows_exported={stats.rows_exported} files_written={stats.files_written}",
            flush=True,
        )
    print(f"[done] manifest={args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
