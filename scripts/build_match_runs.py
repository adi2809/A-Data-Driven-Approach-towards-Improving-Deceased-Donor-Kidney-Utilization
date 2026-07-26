#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat


FILENAME_PATTERN = re.compile(r"ptr_ki_(\d{4})0101_(\d{4})1231\.dta$", re.IGNORECASE)

PANDAS_INT_DTYPES = {
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
}

PANDAS_FLOAT_DTYPES = {
    "float": "Float32",
    "double": "Float64",
}


@dataclass
class SourceFile:
    year: int
    path: Path


@dataclass
class YearStats:
    source_year: int
    source_file: str
    rows_in_source: int
    parts_written: int = 0
    rows_written: int = 0
    rows_within_nominal_year: int = 0
    rows_prior_overlap: int = 0
    rows_next_overlap: int = 0
    rows_null_match_submit_dt: int = 0
    min_match_submit_dt: str | None = None
    max_match_submit_dt: str | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    elapsed_seconds: float | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_encoding_for_year(year: int) -> str | None:
    if year >= 2023:
        return "latin1"
    return None


def discover_source_files(source_dir: Path, years: set[int] | None) -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in sorted(source_dir.glob("*.dta")):
        match = FILENAME_PATTERN.fullmatch(path.name)
        if not match:
            continue
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        if start_year != end_year:
            raise ValueError(f"Unexpected file span in {path.name}")
        if years and start_year not in years:
            continue
        files.append(SourceFile(year=start_year, path=path))
    if not files:
        raise FileNotFoundError(f"No matching .dta files found in {source_dir}")
    return files


def build_column_groups(meta) -> tuple[set[str], dict[str, str], dict[str, str]]:
    date_cols: set[str] = set()
    int_cols: dict[str, str] = {}
    float_cols: dict[str, str] = {}

    for column, fmt in meta.original_variable_types.items():
        column = column.upper()
        if fmt.startswith("%t"):
            date_cols.add(column)

    for column, readstat_type in meta.readstat_variable_types.items():
        column = column.upper()
        if column in date_cols:
            continue
        if readstat_type in PANDAS_INT_DTYPES:
            int_cols[column] = PANDAS_INT_DTYPES[readstat_type]
        elif readstat_type in PANDAS_FLOAT_DTYPES:
            float_cols[column] = PANDAS_FLOAT_DTYPES[readstat_type]

    return date_cols, int_cols, float_cols


def normalize_chunk(
    chunk: pd.DataFrame,
    source_year: int,
    date_cols: set[str],
    int_cols: dict[str, str],
    float_cols: dict[str, str],
) -> pd.DataFrame:
    chunk.columns = [column.upper() for column in chunk.columns]

    for column in date_cols:
        if not pd.api.types.is_datetime64_any_dtype(chunk[column]):
            chunk[column] = pd.to_datetime(chunk[column], errors="coerce")

    for column, dtype in int_cols.items():
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype(dtype)

    for column, dtype in float_cols.items():
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype(dtype)

    match_submit = pd.to_datetime(chunk["MATCH_SUBMIT_DT"], errors="coerce")
    match_year = pd.Series(pd.NA, index=chunk.index, dtype="Int16")
    valid_match_submit = match_submit.notna()
    if valid_match_submit.any():
        match_year.loc[valid_match_submit] = match_submit.loc[valid_match_submit].dt.year.astype("Int16")

    chunk["match_year"] = match_year
    chunk["is_within_nominal_year"] = match_year.eq(source_year).astype("boolean")

    return chunk


def update_year_stats(stats: YearStats, chunk: pd.DataFrame) -> None:
    match_submit = pd.to_datetime(chunk["MATCH_SUBMIT_DT"], errors="coerce")
    match_year = chunk["match_year"]

    stats.rows_written += len(chunk)
    stats.rows_within_nominal_year += int(chunk["is_within_nominal_year"].fillna(False).sum())
    stats.rows_null_match_submit_dt += int(match_submit.isna().sum())
    stats.rows_prior_overlap += int(match_year.eq(stats.source_year - 1).fillna(False).sum())
    stats.rows_next_overlap += int(match_year.eq(stats.source_year + 1).fillna(False).sum())

    if match_submit.notna().any():
        chunk_min = match_submit.min()
        chunk_max = match_submit.max()
        if stats.min_match_submit_dt is None or chunk_min < pd.Timestamp(stats.min_match_submit_dt):
            stats.min_match_submit_dt = chunk_min.isoformat()
        if stats.max_match_submit_dt is None or chunk_max > pd.Timestamp(stats.max_match_submit_dt):
            stats.max_match_submit_dt = chunk_max.isoformat()


def convert_source_file(
    source: SourceFile,
    output_dir: Path,
    chunk_rows: int,
    row_group_size: int,
    overwrite: bool,
) -> YearStats:
    final_year_dir = output_dir / f"source_year={source.year}"
    temp_year_dir = output_dir / f"source_year={source.year}.tmp"

    if final_year_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{final_year_dir} already exists.")
        shutil.rmtree(final_year_dir)

    if temp_year_dir.exists():
        shutil.rmtree(temp_year_dir)
    temp_year_dir.mkdir(parents=True, exist_ok=True)

    source_encoding = source_encoding_for_year(source.year)
    _, meta = pyreadstat.read_dta(str(source.path), metadataonly=True, encoding=source_encoding)
    date_cols, int_cols, float_cols = build_column_groups(meta)

    stats = YearStats(
        source_year=source.year,
        source_file=str(source.path),
        rows_in_source=meta.number_rows,
        started_at_utc=utc_now(),
    )

    start = time.perf_counter()
    for part_number, (chunk, _) in enumerate(
        pyreadstat.read_file_in_chunks(
            pyreadstat.read_dta,
            str(source.path),
            chunksize=chunk_rows,
            apply_value_formats=False,
            encoding=source_encoding,
        ),
        start=1,
    ):
        chunk = normalize_chunk(chunk, source.year, date_cols, int_cols, float_cols)
        update_year_stats(stats, chunk)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        part_path = temp_year_dir / f"part-{part_number:05d}.parquet"
        pq.write_table(
            table,
            part_path,
            compression="zstd",
            use_dictionary=True,
            row_group_size=min(row_group_size, len(chunk)),
        )
        stats.parts_written = part_number
        if part_number % 10 == 0:
            print(
                (
                    f"[progress] source_year={source.year} parts={part_number} "
                    f"rows_written={stats.rows_written}"
                ),
                flush=True,
            )

    stats.elapsed_seconds = round(time.perf_counter() - start, 2)
    stats.finished_at_utc = utc_now()
    temp_year_dir.rename(final_year_dir)
    return stats


def write_manifest(manifest_path: Path, stats: list[YearStats]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "built_at_utc": utc_now(),
        "years": [asdict(item) for item in stats],
    }
    manifest_path.write_text(json.dumps(payload, indent=2))


def write_manifest_from_duckdb(
    manifest_path: Path,
    database_path: Path,
    output_dir: Path,
    source_files: dict[int, SourceFile],
) -> None:
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT
                source_year,
                raw_rows,
                canonical_rows,
                prior_overlap_rows,
                next_overlap_rows,
                null_match_year_rows,
                min_match_submit_dt,
                max_match_submit_dt
            FROM match_run_build_manifest
            ORDER BY source_year;
            """
        ).fetchall()
    finally:
        con.close()

    payload = {"built_at_utc": utc_now(), "years": []}
    for row in rows:
        year = int(row[0])
        year_dir = output_dir / f"source_year={year}"
        payload["years"].append(
            {
                "source_year": year,
                "source_file": str(source_files.get(year).path) if year in source_files else None,
                "rows_in_source": int(row[1]),
                "parts_written": len(list(year_dir.glob("*.parquet"))) if year_dir.exists() else 0,
                "rows_written": int(row[1]),
                "rows_within_nominal_year": int(row[2]),
                "rows_prior_overlap": int(row[3]),
                "rows_next_overlap": int(row[4]),
                "rows_null_match_submit_dt": int(row[5]),
                "min_match_submit_dt": row[6].isoformat() if row[6] is not None else None,
                "max_match_submit_dt": row[7].isoformat() if row[7] is not None else None,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def build_duckdb(
    database_path: Path,
    parquet_root: Path,
    years: Iterable[int],
    threads: int,
    build_lookup_table: bool,
) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_glob = str((parquet_root / "source_year=*" / "*.parquet").resolve()).replace("'", "''")
    years = sorted(set(years))

    con = duckdb.connect(str(database_path))
    try:
        con.execute(f"PRAGMA threads={max(1, threads)};")
        con.execute(
            f"""
            CREATE OR REPLACE VIEW match_runs_raw AS
            SELECT *
            FROM read_parquet('{parquet_glob}', hive_partitioning=1, union_by_name=true);
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW match_runs AS
            SELECT *
            FROM match_runs_raw
            WHERE coalesce(is_within_nominal_year, false);
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW match_runs_overlap AS
            SELECT *
            FROM match_runs_raw
            WHERE NOT coalesce(is_within_nominal_year, false);
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE match_run_build_manifest AS
            SELECT
                source_year,
                COUNT(*) AS raw_rows,
                COUNT(*) FILTER (WHERE coalesce(is_within_nominal_year, false)) AS canonical_rows,
                COUNT(*) FILTER (WHERE match_year = source_year - 1) AS prior_overlap_rows,
                COUNT(*) FILTER (WHERE match_year = source_year + 1) AS next_overlap_rows,
                COUNT(*) FILTER (WHERE match_year IS NULL) AS null_match_year_rows,
                MIN(MATCH_SUBMIT_DT) AS min_match_submit_dt,
                MAX(MATCH_SUBMIT_DT) AS max_match_submit_dt
            FROM match_runs_raw
            GROUP BY source_year
            ORDER BY source_year;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW match_run_daily_rates AS
            SELECT
                CAST(MATCH_SUBMIT_DT AS DATE) AS match_date,
                COUNT(*) AS row_count,
                COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'Y') AS accepted_rows,
                COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'Z') AS provisional_yes_rows,
                COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'N') AS declined_rows,
                COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'B') AS bypass_rows,
                ROUND(
                    COUNT(*) FILTER (WHERE PTR_OFFER_ACPT IN ('Y', 'Z')) * 1.0
                    / NULLIF(COUNT(*) FILTER (WHERE PTR_OFFER_ACPT IS NOT NULL), 0),
                    6
                ) AS positive_response_rate
            FROM match_runs
            GROUP BY 1
            ORDER BY 1;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW match_run_yearly_rates AS
            SELECT
                match_year,
                COUNT(*) AS row_count,
                COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'Y') AS accepted_rows,
                COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'Z') AS provisional_yes_rows,
                COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'N') AS declined_rows,
                COUNT(*) FILTER (WHERE PTR_OFFER_ACPT = 'B') AS bypass_rows,
                AVG(PTR_TOT_SCORE) AS avg_total_score,
                ROUND(
                    COUNT(*) FILTER (WHERE PTR_OFFER_ACPT IN ('Y', 'Z')) * 1.0
                    / NULLIF(COUNT(*) FILTER (WHERE PTR_OFFER_ACPT IS NOT NULL), 0),
                    6
                ) AS positive_response_rate
            FROM match_runs
            GROUP BY 1
            ORDER BY 1;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE MACRO lookup_match(match_id_param) AS TABLE
            SELECT *
            FROM match_runs
            WHERE MATCH_ID = match_id_param
            ORDER BY PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE MACRO px_history(px_id_param) AS TABLE
            SELECT *
            FROM match_runs
            WHERE PX_ID = px_id_param
            ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE MACRO donor_history(donor_id_param) AS TABLE
            SELECT *
            FROM match_runs
            WHERE DONOR_ID = donor_id_param
            ORDER BY MATCH_SUBMIT_DT, MATCH_ID, PTR_SEQUENCE_NUM, PTR_ROW_ORDER;
            """
        )
        for year in years:
            con.execute(
                f"""
                CREATE OR REPLACE VIEW match_runs_{year} AS
                SELECT *
                FROM match_runs
                WHERE match_year = {year};
                """
            )

        if build_lookup_table:
            con.execute("DROP TABLE IF EXISTS match_runs_lookup;")
            con.execute(
                """
                CREATE TABLE match_runs_lookup AS
                SELECT
                    MATCH_ID,
                    PTR_ROW_ORDER,
                    MATCH_SUBMIT_DT,
                    source_year,
                    match_year,
                    DONOR_ID,
                    PX_ID,
                    WLREG_AUDIT_ID,
                    PTR_SEQUENCE_NUM,
                    MATCH_OPO_CTR_CD,
                    PTR_CLASS_ALLOC_CAT,
                    PTR_OFFER_ACPT,
                    PTR_STAT_CD,
                    PTR_CHG_PROCESS_CD
                FROM match_runs;
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_match_runs_lookup_match_id ON match_runs_lookup(MATCH_ID);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_match_runs_lookup_px_id ON match_runs_lookup(PX_ID);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_match_runs_lookup_donor_id ON match_runs_lookup(DONOR_ID);")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_match_runs_lookup_wlreg_audit_id ON match_runs_lookup(WLREG_AUDIT_ID);"
            )

        con.execute("CHECKPOINT;")
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert annual PTR kidney match-run .dta files into year-partitioned Parquet and DuckDB."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/match runs"),
        help="Directory containing ptr_ki_YYYY0101_YYYY1231.dta files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("warehouse/match_runs/parquet"),
        help="Destination directory for year-partitioned Parquet.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("warehouse/match_runs/match_runs.duckdb"),
        help="DuckDB database file to create/update.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("warehouse/match_runs/build_manifest.json"),
        help="JSON manifest capturing per-year build stats.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=250_000,
        help="Rows to stream from each .dta chunk during conversion.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=100_000,
        help="Parquet row group size for each output part.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        help="Optional subset of source years to build.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="DuckDB thread count used when building the query layer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild already-converted source years and refresh the database.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip Parquet conversion and only rebuild the DuckDB layer from existing Parquet.",
    )
    parser.add_argument(
        "--skip-duckdb",
        action="store_true",
        help="Skip DuckDB database/view creation.",
    )
    parser.add_argument(
        "--skip-lookup-table",
        action="store_true",
        help="Do not materialize the narrow indexed lookup table inside DuckDB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_years = set(args.years) if args.years else None
    if args.skip_convert:
        parquet_years = sorted(
            int(path.name.split("=", 1)[1])
            for path in args.output_dir.glob("source_year=*")
            if path.is_dir() and path.name.split("=", 1)[1].isdigit()
        )
        if selected_years is not None:
            parquet_years = [year for year in parquet_years if year in selected_years]
        if not parquet_years:
            raise FileNotFoundError(f"No source_year=* parquet directories found in {args.output_dir}")
        source_files = []
        all_source_files = {}
    else:
        source_files = discover_source_files(args.source_dir, selected_years)
        all_source_files = {item.year: item for item in discover_source_files(args.source_dir, None)}

    built_stats: list[YearStats] = []
    if not args.skip_convert:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for source in source_files:
            final_year_dir = args.output_dir / f"source_year={source.year}"
            if final_year_dir.exists() and not args.overwrite:
                print(f"[skip] source_year={source.year} already exists at {final_year_dir}", flush=True)
                continue
            print(f"[convert] source_year={source.year} file={source.path}", flush=True)
            stats = convert_source_file(
                source=source,
                output_dir=args.output_dir,
                chunk_rows=args.chunk_rows,
                row_group_size=args.row_group_size,
                overwrite=args.overwrite,
            )
            built_stats.append(stats)
            print(
                (
                    f"[done] source_year={source.year} raw_rows={stats.rows_in_source} "
                    f"parts={stats.parts_written} canonical_rows={stats.rows_within_nominal_year} "
                    f"prior_overlap_rows={stats.rows_prior_overlap} next_overlap_rows={stats.rows_next_overlap} "
                    f"elapsed_seconds={stats.elapsed_seconds}"
                ),
                flush=True,
            )

        write_manifest(args.manifest, built_stats)

    if not args.skip_duckdb:
        years = parquet_years if args.skip_convert else [source.year for source in source_files]
        print(f"[duckdb] building query layer at {args.database}", flush=True)
        build_duckdb(
            database_path=args.database,
            parquet_root=args.output_dir,
            years=years,
            threads=args.threads,
            build_lookup_table=not args.skip_lookup_table,
        )
        write_manifest_from_duckdb(
            manifest_path=args.manifest,
            database_path=args.database,
            output_dir=args.output_dir,
            source_files=all_source_files,
        )
        print("[duckdb] done", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
