#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kidney_utilization.feature_specs import OFFERPRED_NAME
from kidney_utilization.plots import (
    plot_data_qa,
    plot_offerpred_diagnostics,
    plot_offerpred_feature_importance,
    plot_offerpred_topk,
    plot_offerpred_yearly_metrics,
)
from kidney_utilization.train import _load_benchmark_config, _train_offerpred_sampled_catboost
from kidney_utilization.utils import dump_json, ensure_parent, write_parquet


DEFAULT_BENCHMARK_MANIFEST = (
    REPO_ROOT / "warehouse/match_runs/kidney_utilization_benchmark_same_match_history_fixed_manifest.json"
)
DEFAULT_BENCHMARK_DB = REPO_ROOT / "warehouse/match_runs/kidney_utilization_benchmark_same_match_history_fixed.duckdb"
DEFAULT_WORKING_DB = REPO_ROOT / "warehouse/match_runs/offerpred_working.duckdb"
DEFAULT_SOURCE_PARQUET = REPO_ROOT / "warehouse/match_offer_features/parquet_same_match_history_fixed/match_year=*/*.parquet"


CANDIDATE_HISTORY_FIX_FEATURES = [
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
]


CENTER_HISTORY_FIX_FEATURES = [
    "center_yn_offer_count_30d",
    "center_positive_response_rate_30d",
    "center_rate_same_dcd_30d",
    "center_rate_same_high_kdpi_30d",
    "center_rate_same_hcv_pos_30d",
    "center_rate_same_long_distance_30d",
    "center_rate_same_mm_bucket_30d",
    "center_yn_offer_count_365d",
    "center_positive_response_rate_365d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OfferPred with same-match history leakage frozen to the first candidate and center row in each match."
    )
    parser.add_argument("--run-name", default="offerpred")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--match-limit-per-split", type=int)
    parser.add_argument("--benchmark-db", type=Path, default=DEFAULT_BENCHMARK_DB)
    parser.add_argument("--benchmark-manifest", type=Path, default=DEFAULT_BENCHMARK_MANIFEST)
    parser.add_argument("--working-db", type=Path, default=DEFAULT_WORKING_DB)
    parser.add_argument("--source-parquet-glob", type=Path, default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--artifact-root", type=Path)
    return parser.parse_args()


def _sql_string(value: str) -> str:
    return str(value).replace("'", "''")


def _build_source_keys_table(con: duckdb.DuckDBPyConnection, parquet_glob: Path) -> None:
    parquet_sql = _sql_string(str(parquet_glob))
    con.execute("DROP TABLE IF EXISTS source_row_keys")
    con.execute(
        f"""
        CREATE TABLE source_row_keys AS
        WITH deduped AS (
            SELECT
                MATCH_ID AS match_id,
                PTR_ROW_ORDER AS ptr_row_order,
                PX_ID AS px_id,
                CAN_LISTING_CTR_CD AS can_listing_ctr_cd,
                COALESCE(CAN_LISTING_CTR_TY, '') AS can_listing_ctr_ty_norm,
                ROW_NUMBER() OVER (
                    PARTITION BY MATCH_ID, PTR_ROW_ORDER
                    ORDER BY COALESCE(OFFER_ROW_INSTANCE_ORDINAL, 0), COALESCE(PX_ID, -1)
                ) AS rownum
            FROM read_parquet('{parquet_sql}')
        )
        SELECT
            match_id,
            ptr_row_order,
            px_id,
            can_listing_ctr_cd,
            can_listing_ctr_ty_norm
        FROM deduped
        WHERE rownum = 1
        """
    )


def _build_offerpred_corrected_view_sql(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
) -> str:
    columns = [row[0] for row in con.execute(f"DESCRIBE {source_table}").fetchall()]
    candidate_window_exprs = [
        (
            f"FIRST_VALUE(t.{feature}) OVER ("
            "PARTITION BY t.match_id, keys.px_id "
            "ORDER BY t.ptr_row_order, t.offer_rank, t.ptr_sequence_num "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f") AS fv_{feature}"
        )
        for feature in CANDIDATE_HISTORY_FIX_FEATURES
    ]
    center_window_exprs = [
        (
            f"FIRST_VALUE(t.{feature}) OVER ("
            "PARTITION BY t.match_id, keys.can_listing_ctr_cd, keys.can_listing_ctr_ty_norm "
            "ORDER BY t.ptr_row_order, t.offer_rank, t.ptr_sequence_num "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f") AS fv_{feature}"
        )
        for feature in CENTER_HISTORY_FIX_FEATURES
    ]
    joined_projection = ",\n                ".join(candidate_window_exprs + center_window_exprs)

    select_columns: list[str] = []
    for column in columns:
        if column in CANDIDATE_HISTORY_FIX_FEATURES:
            select_columns.append(
                f"CASE WHEN j._px_id IS NULL THEN j.{column} ELSE j.fv_{column} END AS {column}"
            )
        elif column in CENTER_HISTORY_FIX_FEATURES:
            select_columns.append(
                f"CASE WHEN j._center_cd IS NULL THEN j.{column} ELSE j.fv_{column} END AS {column}"
            )
        else:
            select_columns.append(f"j.{column}")

    return f"""
        WITH joined AS (
            SELECT
                t.*,
                keys.px_id AS _px_id,
                keys.can_listing_ctr_cd AS _center_cd,
                keys.can_listing_ctr_ty_norm AS _center_ty,
                {joined_projection}
            FROM {source_table} AS t
            LEFT JOIN source_row_keys AS keys
              ON t.match_id = keys.match_id
             AND t.ptr_row_order = keys.ptr_row_order
        )
        SELECT
            {", ".join(select_columns)}
        FROM joined AS j
    """


def _prepare_working_connection(
    working_db: Path,
    benchmark_db: Path,
    parquet_glob: Path,
    threads: int,
) -> duckdb.DuckDBPyConnection:
    ensure_parent(working_db)
    con = duckdb.connect(str(working_db))
    con.execute(f"PRAGMA threads={int(threads)};")
    con.execute("CREATE SCHEMA IF NOT EXISTS benchmark")
    con.execute(f"ATTACH '{_sql_string(str(benchmark_db))}' AS src (READ_ONLY)")
    con.execute("DROP VIEW IF EXISTS benchmark.match_labels")
    con.execute("CREATE VIEW benchmark.match_labels AS SELECT * FROM src.benchmark.match_labels")
    _build_source_keys_table(con, parquet_glob)
    for view_name, source_name in [
        ("benchmark.offerpred_rows", "src.benchmark.offerpred_rows"),
        ("benchmark.offerpred_scoring_rows", "src.benchmark.offerpred_scoring_rows"),
    ]:
        con.execute(f"DROP VIEW IF EXISTS {view_name}")
        con.execute(f"CREATE VIEW {view_name} AS {_build_offerpred_corrected_view_sql(con, source_name)}")
    return con


def main() -> None:
    args = parse_args()
    config, benchmark_manifest, config_source = _load_benchmark_config(None, args.benchmark_manifest)

    artifact_root = (args.artifact_root or config.artifact_root) / args.run_name
    plots_dir = artifact_root / "plots"
    intermediate_dir = artifact_root / "intermediate"
    artifact_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    con = _prepare_working_connection(
        working_db=args.working_db,
        benchmark_db=args.benchmark_db,
        parquet_glob=args.source_parquet_glob,
        threads=args.threads,
    )

    match_labels = con.execute("SELECT * FROM benchmark.match_labels ORDER BY match_year, match_id").fetch_df()
    plot_data_qa(match_labels, plots_dir / "data_qa.png")

    offerpred_state = _train_offerpred_sampled_catboost(
        con=con,
        config=config,
        match_limit_per_split=args.match_limit_per_split,
        intermediate_dir=intermediate_dir,
    )

    plot_offerpred_diagnostics(offerpred_state.eval_rows, plots_dir / "offerpred_diagnostics.png")
    plot_offerpred_topk(offerpred_state.eval_rows, plots_dir / "offerpred_topk.png")
    plot_offerpred_yearly_metrics(offerpred_state.yearly_metrics, plots_dir / "offerpred_yearly_metrics.png")
    plot_offerpred_feature_importance(offerpred_state.feature_importance, plots_dir / "offerpred_feature_importance.png")

    write_parquet(offerpred_state.eval_rows, artifact_root / "offerpred_eval_sample.parquet")
    write_parquet(offerpred_state.offerpred_aggregates, artifact_root / "offerpred_run_aggregates.parquet")
    write_parquet(offerpred_state.feature_importance, artifact_root / "offerpred_feature_importance.parquet")
    write_parquet(offerpred_state.yearly_metrics, artifact_root / "offerpred_yearly_metrics.parquet")

    plot_files = sorted(str(path.relative_to(artifact_root)) for path in plots_dir.glob("*.png"))
    run_manifest = {
        "model_name": OFFERPRED_NAME,
        "config_source": config_source,
        "config": config.to_dict(),
        "benchmark_db": str(args.benchmark_db),
        "benchmark_manifest": str(args.benchmark_manifest),
        "working_db": str(args.working_db),
        "source_parquet_glob": str(args.source_parquet_glob),
        "backend": offerpred_state.backend,
        "training_mode": offerpred_state.mode,
        "plot_files": plot_files,
        "offerpred_metrics": offerpred_state.metrics,
        "offerpred_yearly_metrics": offerpred_state.yearly_metrics.to_dict(orient="records"),
        "top_feature_importance": offerpred_state.feature_importance.head(25).to_dict(orient="records"),
        "benchmark_table_counts": benchmark_manifest.get("table_counts", {}),
        "offerpred_scored_parts_dir": None if offerpred_state.scored_parts_dir is None else str(offerpred_state.scored_parts_dir),
        "history_fix_mode": "freeze_to_first_candidate_or_center_row_within_match",
        "candidate_history_fix_features": CANDIDATE_HISTORY_FIX_FEATURES,
        "center_history_fix_features": CENTER_HISTORY_FIX_FEATURES,
    }
    dump_json(artifact_root / "run_manifest.json", run_manifest)
    con.close()
    print(f"[done] artifact_root={artifact_root}")


if __name__ == "__main__":
    main()
