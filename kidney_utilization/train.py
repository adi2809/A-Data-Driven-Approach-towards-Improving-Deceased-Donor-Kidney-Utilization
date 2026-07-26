from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .config import BenchmarkConfig
from .feature_specs import (
    DISCARDPRED_NAME,
    DISCARDPRED_SCORE_FEATURES,
    LOCATIONPRED_NAME,
    LOCATIONPRED_SEGMENT_BOUNDS,
    LOCATIONPRED_SEGMENT_FEATURES,
    OFFERPRED_NAME,
    OFFERPRED_FEATURES,
    get_discardpred_run_features,
)
from .modeling import fit_classifier, fit_native_catboost_classifier
from .plots import (
    plot_confusion,
    plot_pipeline_dashboard,
    plot_data_qa,
    plot_offerpred_diagnostics,
    plot_offerpred_feature_importance,
    plot_offerpred_topk,
    plot_offerpred_yearly_metrics,
    plot_discardpred_score_mass,
    plot_report_scorecard,
    plot_locationpred_localizer,
    plot_locationpred_error_analysis,
    plot_validation_sweep,
)
from .utils import dump_json, ensure_parent, load_json, write_parquet

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class TrainingArtifacts:
    artifact_root: Path
    manifest_path: Path


@dataclass(slots=True)
class ConsolidatedReportArtifacts:
    artifact_root: Path
    manifest_path: Path


@dataclass(slots=True)
class OfferPredState:
    model: Any
    backend: str
    metrics: dict[str, Any]
    eval_rows: pd.DataFrame
    yearly_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    offerpred_aggregates: pd.DataFrame
    scored_parts_dir: Path | None
    mode: str


@dataclass(slots=True)
class DiscardPredState:
    model: Any
    runs: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(slots=True)
class LocationPredState:
    model: Any
    backend: str
    validation_run_predictions: pd.DataFrame
    test_run_predictions: pd.DataFrame
    validation_final_eval: pd.DataFrame
    test_final_eval: pd.DataFrame
    locationpred_eval: pd.DataFrame
    mode: str


def _sql_string(value: str) -> str:
    return str(value).replace("'", "''")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_query_to_parquet(
    con: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    escaped = _sql_string(str(output_path))
    con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET)")


def _load_benchmark_config(
    config: BenchmarkConfig | None,
    benchmark_manifest_path: Path | None,
) -> tuple[BenchmarkConfig, dict[str, Any], str]:
    if config is not None:
        manifest: dict[str, Any] = {"config": config.to_dict()}
        if benchmark_manifest_path is not None and benchmark_manifest_path.exists():
            persisted = load_json(benchmark_manifest_path)
            for key, value in persisted.items():
                if key != "config":
                    manifest[key] = value
        return config, manifest, "explicit_config"

    if benchmark_manifest_path is None:
        raise ValueError("benchmark_manifest_path is required when config is None")

    manifest = load_json(benchmark_manifest_path)
    return BenchmarkConfig.from_dict(manifest["config"]), manifest, "benchmark_manifest"


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _discardpred_source_table(config: BenchmarkConfig) -> str:
    return "benchmark.discardpred_runs"


def _discardpred_feature_names(config: BenchmarkConfig) -> list[str]:
    return get_discardpred_run_features()


def _locationpred_source_table(config: BenchmarkConfig) -> str:
    return "benchmark.locationpred_riskset"


def _load_offerpred_state_from_artifact(offerpred_artifact_root: Path) -> OfferPredState:
    manifest_path = offerpred_artifact_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {OFFERPRED_NAME} artifact manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    parts_value = manifest.get("offerpred_scored_parts_dir")
    if parts_value is None:
        parts_dir = offerpred_artifact_root / "intermediate" / "offerpred_scored_rows"
    else:
        parts_dir = _resolve_repo_path(parts_value)
    if not parts_dir.exists():
        raise FileNotFoundError(f"missing {OFFERPRED_NAME} scored parts directory: {parts_dir}")
    offerpred_eval = _read_optional_parquet(offerpred_artifact_root / "offerpred_eval_sample.parquet")
    offerpred_yearly = _read_optional_parquet(offerpred_artifact_root / "offerpred_yearly_metrics.parquet")
    offerpred_importance = _read_optional_parquet(offerpred_artifact_root / "offerpred_feature_importance.parquet")
    offerpred_aggregates = _read_optional_parquet(offerpred_artifact_root / "offerpred_run_aggregates.parquet")
    return OfferPredState(
        model=None,
        backend=str(manifest.get("backend") or manifest.get("backends", {}).get("offerpred", "artifact_reuse")),
        metrics=manifest.get("offerpred_metrics", {}),
        eval_rows=offerpred_eval,
        yearly_metrics=offerpred_yearly,
        feature_importance=offerpred_importance,
        offerpred_aggregates=offerpred_aggregates,
        scored_parts_dir=parts_dir,
        mode="artifact_reuse",
    )


def _vectors_per_chunk(batch_row_count: int) -> int:
    return max(1, math.ceil(int(batch_row_count) / 2048))


def _iter_query_chunks(
    con: duckdb.DuckDBPyConnection,
    query: str,
    params: list[Any],
    batch_row_count: int,
):
    result = con.execute(query, params)
    vectors = _vectors_per_chunk(batch_row_count)
    while True:
        chunk = result.fetch_df_chunk(vectors)
        if chunk.empty:
            break
        yield chunk


def _iter_split_chunks(
    con: duckdb.DuckDBPyConnection,
    source_name: str,
    split: str,
    select_columns: list[str],
    batch_row_count: int,
    match_limit_per_split: int | None,
    extra_where: str | None = None,
    order_by: str | None = None,
):
    select_sql = ", ".join(f"t.{column}" for column in select_columns)
    where_clause = "t.split = ?"
    if extra_where:
        where_clause += f" AND ({extra_where})"
    order_clause = ""
    if order_by:
        order_clause = f" ORDER BY {order_by}"

    if match_limit_per_split is None:
        query = f"""
            SELECT {select_sql}
            FROM {source_name} AS t
            WHERE {where_clause}
            {order_clause}
        """
        params = [split]
    else:
        query = f"""
            WITH match_subset AS (
                SELECT match_id
                FROM {source_name}
                WHERE split = ?
                GROUP BY 1
                ORDER BY MIN(match_submit_dt), match_id
                LIMIT {int(match_limit_per_split)}
            )
            SELECT {select_sql}
            FROM {source_name} AS t
            JOIN match_subset AS s USING (match_id)
            WHERE {where_clause}
            {order_clause}
        """
        params = [split, split]

    yield from _iter_query_chunks(con, query, params, batch_row_count)


def _load_split_dataframe(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    splits: list[str],
    match_limit_per_split: int | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        order_by = "t.match_submit_dt, t.match_id"
        if "discardpred_runs" not in table_name:
            order_by += ", COALESCE(t.ptr_row_order, 0)"
        chunk_frames = list(
            _iter_split_chunks(
                con=con,
                source_name=table_name,
                split=split,
                select_columns=[row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()],
                batch_row_count=2_000_000,
                match_limit_per_split=match_limit_per_split,
                extra_where=None,
                order_by=order_by,
            )
        )
        if chunk_frames:
            frames.append(pd.concat(chunk_frames, ignore_index=True))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _binary_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    metrics: dict[str, float] = {
        "brier_score": float(brier_score_loss(y_true.astype(int), y_score.astype(float))),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def _localizer_metrics(eval_frame: pd.DataFrame) -> dict[str, float]:
    if eval_frame.empty:
        return {
            "delta_0_hit_rate": 0.0,
            "delta_1_hit_rate": 0.0,
            "delta_5_hit_rate": 0.0,
            "delta_10_hit_rate": 0.0,
            "delta_25_hit_rate": 0.0,
            "delta_50_hit_rate": 0.0,
        }
    pred_rank = pd.to_numeric(eval_frame["pred_rank"], errors="coerce")
    true_rank = pd.to_numeric(eval_frame["true_rank"], errors="coerce")
    errors = (pred_rank - true_rank).abs().astype(float).fillna(np.inf)
    return {
        "delta_0_hit_rate": float((errors <= 0).mean()),
        "delta_1_hit_rate": float((errors <= 1).mean()),
        "delta_5_hit_rate": float((errors <= 5).mean()),
        "delta_10_hit_rate": float((errors <= 10).mean()),
        "delta_25_hit_rate": float((errors <= 25).mean()),
        "delta_50_hit_rate": float((errors <= 50).mean()),
    }


def _empirical_rank_predictions(train_runs: pd.DataFrame, eval_runs: pd.DataFrame) -> pd.Series:
    localizable = train_runs.loc[train_runs["route_target"] == "localizable_observed_y"]
    if localizable.empty:
        return pd.Series([1] * len(eval_runs), index=eval_runs.index, dtype=int)
    rank_counts = localizable["first_observed_y_rank"].value_counts(normalize=True).sort_index()
    available_ranks = rank_counts.index.to_list()
    predictions = []
    for _, row in eval_runs.iterrows():
        feasible = [rank for rank in available_ranks if rank <= row["run_len"]]
        if feasible:
            best_rank = max(feasible, key=lambda rank: rank_counts.loc[rank])
            predictions.append(int(best_rank))
        else:
            predictions.append(1)
    return pd.Series(predictions, index=eval_runs.index, dtype=int)


def _query_binary_split_target_counts(
    con: duckdb.DuckDBPyConnection,
    source_name: str,
    split: str,
    target_column: str,
    extra_where: str | None = None,
) -> tuple[int, int]:
    where_tail = ""
    if extra_where:
        where_tail = f" AND ({extra_where})"
    row = con.execute(
        f"""
        SELECT
            COALESCE(SUM(CASE WHEN {target_column} = 1 THEN 1 ELSE 0 END), 0) AS positive_count,
            COALESCE(SUM(CASE WHEN {target_column} = 0 THEN 1 ELSE 0 END), 0) AS negative_count
        FROM {source_name}
        WHERE split = ?
          {where_tail}
        """,
        [split],
    ).fetchone()
    return int(row[0]), int(row[1])


def _sample_binary_training_frame(
    con: duckdb.DuckDBPyConnection,
    config: BenchmarkConfig,
    source_name: str,
    feature_names: list[str],
    target_column: str,
    chunk_rows: int,
    split: str,
    match_limit_per_split: int | None,
    negative_to_positive_ratio: int,
    extra_where: str | None = None,
    random_seed_offset: int = 0,
) -> pd.DataFrame:
    select_columns = feature_names + [target_column]
    positive_count, negative_count = _query_binary_split_target_counts(
        con=con,
        source_name=source_name,
        split=split,
        target_column=target_column,
        extra_where=extra_where,
    )
    if positive_count == 0:
        return pd.DataFrame(columns=select_columns)

    target_negative_count = positive_count * max(1, int(negative_to_positive_ratio))
    negative_keep_probability = (
        min(1.0, float(target_negative_count) / float(negative_count)) if negative_count > 0 else 0.0
    )
    rng = np.random.default_rng(config.random_seed + random_seed_offset + sum(ord(char) for char in split))
    sampled_frames: list[pd.DataFrame] = []
    sampled_positive_count = 0
    sampled_negative_count = 0

    for chunk in _iter_split_chunks(
        con=con,
        source_name=source_name,
        split=split,
        select_columns=select_columns,
        batch_row_count=chunk_rows,
        match_limit_per_split=match_limit_per_split,
        extra_where=extra_where,
        order_by=None,
    ):
        if chunk.empty:
            continue
        positive_frame = chunk.loc[chunk[target_column].astype(int) == 1].copy()
        negative_frame = chunk.loc[chunk[target_column].astype(int) == 0].copy()
        if not positive_frame.empty:
            sampled_frames.append(positive_frame)
            sampled_positive_count += int(len(positive_frame))
        if not negative_frame.empty and negative_keep_probability > 0.0:
            if negative_keep_probability < 1.0:
                keep_mask = rng.random(len(negative_frame)) < negative_keep_probability
                negative_frame = negative_frame.loc[keep_mask].copy()
            if not negative_frame.empty:
                sampled_frames.append(negative_frame)
                sampled_negative_count += int(len(negative_frame))

    if not sampled_frames:
        return pd.DataFrame(columns=select_columns)

    sampled_frame = pd.concat(sampled_frames, ignore_index=True)
    print(
        f"[binary_sample] source={source_name} split={split} positives={sampled_positive_count} negatives={sampled_negative_count} "
        f"target_negative_count={target_negative_count} negative_keep_probability={negative_keep_probability:.6f}",
        flush=True,
    )
    return sampled_frame


def _sample_offerpred_binary_training_frame(
    con: duckdb.DuckDBPyConnection,
    config: BenchmarkConfig,
    split: str,
    match_limit_per_split: int | None,
    negative_to_positive_ratio: int,
) -> pd.DataFrame:
    return _sample_binary_training_frame(
        con=con,
        config=config,
        source_name="benchmark.offerpred_scoring_rows",
        feature_names=OFFERPRED_FEATURES,
        target_column="offerpred_target",
        chunk_rows=config.offerpred_chunk_rows,
        split=split,
        match_limit_per_split=match_limit_per_split,
        negative_to_positive_ratio=negative_to_positive_ratio,
        extra_where="use_for_offerpred_loss = 1",
        random_seed_offset=0,
    )


def _binary_positive_probability_values(probability_frame: pd.DataFrame) -> np.ndarray:
    column_by_string = {str(column): column for column in probability_frame.columns}
    if "1" in column_by_string:
        return probability_frame[column_by_string["1"]].astype(float).to_numpy()
    if len(probability_frame.columns) == 1:
        only_column = probability_frame.columns[0]
        return (
            np.ones(len(probability_frame), dtype=float)
            if str(only_column) == "1"
            else np.zeros(len(probability_frame), dtype=float)
        )
    return probability_frame[probability_frame.columns[-1]].astype(float).to_numpy()


def _register_offerpred_scored_view(
    con: duckdb.DuckDBPyConnection,
    offerpred_state: OfferPredState,
) -> None:
    if offerpred_state.scored_parts_dir is None:
        raise ValueError("offerpred state has no scored parquet directory")
    _register_offerpred_scored_view_from_parts_dir(con, offerpred_state.scored_parts_dir)


def _register_offerpred_scored_view_from_parts_dir(
    con: duckdb.DuckDBPyConnection,
    scored_parts_dir: Path,
) -> None:
    parquet_glob = _sql_string(str(scored_parts_dir / "*.parquet"))
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW _offerpred_scored_rows AS
        SELECT *
        FROM parquet_scan('{parquet_glob}', union_by_name=true)
        """
    )


def _query_offerpred_aggregates(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            match_id,
            MAX(offerpred_score) AS offerpred_score_max,
            AVG(offerpred_score) AS offerpred_score_mean,
            SUM(CASE WHEN offer_rank <= 10 THEN offerpred_score ELSE 0.0 END) AS offerpred_score_sum_first10,
            SUM(CASE WHEN offer_rank <= 25 THEN offerpred_score ELSE 0.0 END) AS offerpred_score_sum_first25,
            SUM(CASE WHEN offer_rank <= 50 THEN offerpred_score ELSE 0.0 END) AS offerpred_score_sum_first50,
            SUM(CASE WHEN offer_rank > 50 THEN offerpred_score ELSE 0.0 END) AS offerpred_score_sum_tail,
            MAX(CASE WHEN offer_rank <= 10 THEN offerpred_score ELSE NULL END) AS offerpred_score_top10_max,
            MAX(CASE WHEN offer_rank <= 25 THEN offerpred_score ELSE NULL END) AS offerpred_score_top25_max,
            MAX(CASE WHEN offer_rank <= 50 THEN offerpred_score ELSE NULL END) AS offerpred_score_top50_max,
            COALESCE(AVG(CASE WHEN offer_rank <= 25 THEN offerpred_score ELSE NULL END), 0.0)
            - COALESCE(AVG(CASE WHEN offer_rank > 50 THEN offerpred_score ELSE NULL END), 0.0) AS offerpred_score_head_minus_tail
        FROM _offerpred_scored_rows
        GROUP BY 1
        """
    ).fetch_df()


def _query_offerpred_best_ranks(
    con: duckdb.DuckDBPyConnection,
    split: str,
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT match_id, offer_rank AS pred_rank
        FROM (
            SELECT
                match_id,
                offer_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY match_id
                    ORDER BY offerpred_score DESC, offer_rank ASC
                ) AS rn
            FROM _offerpred_scored_rows
            WHERE split = ?
              AND run_state = 'localizable_observed_y'
        ) AS ranked
        WHERE rn = 1
        """,
        [split],
    ).fetch_df()


def _sample_offerpred_eval_rows(
    con: duckdb.DuckDBPyConnection,
    source_name: str,
    splits: list[str],
    row_budget_per_state: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        for run_state in ["localizable_observed_y", "none"]:
            avg_run_len_row = con.execute(
                """
                SELECT COALESCE(AVG(run_len), 1.0)
                FROM benchmark.match_labels
                WHERE split = ? AND run_state = ?
                """,
                [split, run_state],
            ).fetchone()
            avg_run_len = float(avg_run_len_row[0]) if avg_run_len_row is not None else 1.0
            match_limit = max(1, int(row_budget_per_state / max(avg_run_len, 1.0)))
            frame = con.execute(
                f"""
                WITH sampled_matches AS (
                    SELECT match_id
                    FROM benchmark.match_labels
                    WHERE split = ?
                      AND run_state = ?
                    ORDER BY hash(match_id)
                    LIMIT {match_limit}
                )
                SELECT
                    s.match_id,
                    s.offer_rank,
                    s.match_year,
                    s.split,
                    s.offerpred_target,
                    s.offerpred_score
                FROM {source_name} AS s
                JOIN sampled_matches AS m USING (match_id)
                WHERE s.split = ?
                  AND s.use_for_offerpred_loss = 1
                ORDER BY s.match_submit_dt, s.match_id, s.offer_rank
                """,
                [split, run_state, split],
            ).fetch_df()
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["match_id", "offer_rank", "match_year", "split", "offerpred_target", "offerpred_score"])
    return pd.concat(frames, ignore_index=True)


def _compute_offerpred_yearly_metrics(eval_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if eval_rows.empty:
        return pd.DataFrame(columns=["split", "match_year", "roc_auc", "average_precision", "positive_rate", "row_count"])
    for (split, match_year), frame in eval_rows.groupby(["split", "match_year"], sort=True):
        y_true = frame["offerpred_target"].astype(int)
        y_score = frame["offerpred_score"].astype(float)
        metrics = _binary_metrics(y_true, y_score)
        rows.append(
            {
                "split": split,
                "match_year": int(match_year),
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "positive_rate": float(y_true.mean()) if not frame.empty else 0.0,
                "row_count": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def _extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    estimator = getattr(model, "estimator", None)
    importances = None
    if estimator is not None and hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    elif estimator is not None and hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif estimator is not None and hasattr(estimator, "get_feature_importance"):
        importances = np.asarray(estimator.get_feature_importance(), dtype=float)

    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances[: len(feature_names)],
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def _train_offerpred_sampled_catboost(
    con: duckdb.DuckDBPyConnection,
    config: BenchmarkConfig,
    match_limit_per_split: int | None,
    intermediate_dir: Path,
) -> OfferPredState:
    negative_ratio = max(1, int(config.offerpred_negative_to_positive_ratio))
    offerpred_train = _sample_offerpred_binary_training_frame(
        con=con,
        config=config,
        split="train",
        match_limit_per_split=match_limit_per_split,
        negative_to_positive_ratio=negative_ratio,
    )
    offerpred_validation_sample = _sample_offerpred_binary_training_frame(
        con=con,
        config=config,
        split="validation",
        match_limit_per_split=match_limit_per_split,
        negative_to_positive_ratio=negative_ratio,
    )

    offerpred_model = fit_native_catboost_classifier(
        train_frame=offerpred_train,
        feature_names=OFFERPRED_FEATURES,
        target_column="offerpred_target",
        validation_frame=offerpred_validation_sample,
        random_seed=config.random_seed,
        iterations=config.offerpred_catboost_iterations,
        depth=config.offerpred_catboost_depth,
        learning_rate=config.offerpred_catboost_learning_rate,
        l2_leaf_reg=config.offerpred_catboost_l2_leaf_reg,
        early_stopping_rounds=config.offerpred_catboost_early_stopping_rounds,
    )

    score_parts_dir = intermediate_dir / "offerpred_scored_rows"
    _reset_dir(score_parts_dir)
    metadata_columns = [
        "match_id",
        "ptr_row_order",
        "offer_rank",
        "ptr_sequence_num",
        "ptr_offer_acpt",
        "match_submit_dt",
        "match_year",
        "split",
        "run_state",
        "offerpred_target",
        "use_for_offerpred_loss",
    ]
    select_columns = metadata_columns + OFFERPRED_FEATURES
    for split in ["train", "validation", "test"]:
        part_index = 0
        for chunk in _iter_split_chunks(
            con=con,
            source_name="benchmark.offerpred_scoring_rows",
            split=split,
            select_columns=select_columns,
            batch_row_count=config.offerpred_chunk_rows,
            match_limit_per_split=match_limit_per_split,
            extra_where=None,
            order_by=None,
        ):
            probability_frame = offerpred_model.predict_proba(chunk)
            positive_column = "1" if "1" in probability_frame.columns else probability_frame.columns[-1]
            scored = chunk[metadata_columns].copy()
            scored["offerpred_score"] = probability_frame[positive_column].astype(float).to_numpy()
            write_parquet(scored, score_parts_dir / f"{split}_part_{part_index:05d}.parquet")
            part_index += 1

    _register_offerpred_scored_view(
        con,
        OfferPredState(
            model=offerpred_model,
            backend=offerpred_model.backend,
            metrics={},
            eval_rows=pd.DataFrame(),
            yearly_metrics=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            offerpred_aggregates=pd.DataFrame(),
            scored_parts_dir=score_parts_dir,
            mode="sampled_catboost",
        ),
    )
    offerpred_aggregates = _query_offerpred_aggregates(con)
    eval_rows = _sample_offerpred_eval_rows(
        con=con,
        source_name="_offerpred_scored_rows",
        splits=["validation", "test"],
        row_budget_per_state=config.evaluation_sample_rows_per_group,
    )
    offerpred_metrics = {
        split: _binary_metrics(
            eval_rows.loc[eval_rows["split"] == split, "offerpred_target"].astype(int),
            eval_rows.loc[eval_rows["split"] == split, "offerpred_score"].astype(float),
        )
        for split in ["validation", "test"]
    }
    offerpred_metrics["sample_rows"] = {split: int((eval_rows["split"] == split).sum()) for split in ["validation", "test"]}
    yearly_metrics = _compute_offerpred_yearly_metrics(eval_rows)
    feature_importance = _extract_feature_importance(offerpred_model, OFFERPRED_FEATURES)

    return OfferPredState(
        model=offerpred_model,
        backend=offerpred_model.backend,
        metrics=offerpred_metrics,
        eval_rows=eval_rows,
        yearly_metrics=yearly_metrics,
        feature_importance=feature_importance,
        offerpred_aggregates=offerpred_aggregates,
        scored_parts_dir=score_parts_dir,
        mode="sampled_catboost",
    )


def _prepare_discardpred_runs(
    con: duckdb.DuckDBPyConnection,
    offerpred_state: OfferPredState,
    config: BenchmarkConfig,
    match_limit_per_split: int | None,
    artifact_root: Path,
) -> pd.DataFrame:
    discardpred_feature_names = _discardpred_feature_names(config)
    discardpred_runs = _load_split_dataframe(
        con,
        _discardpred_source_table(config),
        ["train", "validation", "test"],
        match_limit_per_split,
    )
    if any(feature_name in DISCARDPRED_SCORE_FEATURES for feature_name in discardpred_feature_names):
        offerpred_aggregates = _query_offerpred_aggregates(con)
        if offerpred_aggregates.empty:
            offerpred_aggregates = offerpred_state.offerpred_aggregates
        discardpred_runs = discardpred_runs.merge(offerpred_aggregates, on="match_id", how="left")
    for feature_name in discardpred_feature_names:
        if feature_name not in discardpred_runs.columns:
            discardpred_runs[feature_name] = 0.0
    discardpred_runs[discardpred_feature_names] = discardpred_runs[discardpred_feature_names].fillna(0.0)
    discardpred_runs.attrs["artifact_root"] = artifact_root
    return discardpred_runs


def _plot_discardpred_confusion(
    discardpred_frame: pd.DataFrame,
    discard_threshold: float,
    title: str,
    output_path: Path,
) -> None:
    if discardpred_frame.empty:
        return
    y_true = discardpred_frame["discard_target"].astype(int).map({0: "placed", 1: "discard"})
    y_pred = (discardpred_frame["discard_probability"].astype(float) >= float(discard_threshold)).astype(int).map(
        {0: "placed", 1: "discard"}
    )
    plot_confusion(
        y_true,
        y_pred,
        ["placed", "discard"],
        title,
        output_path,
        subtitle=f"DiscardPred threshold = {float(discard_threshold):.2f}.",
        footnote="Cells show row-normalized percentages with raw counts in parentheses.",
    )


def _train_discardpred_models(
    discardpred_runs: pd.DataFrame,
    config: BenchmarkConfig,
    plots_dir: Path,
) -> DiscardPredState:
    discardpred_feature_names = _discardpred_feature_names(config)
    discardpred_train = discardpred_runs.loc[discardpred_runs["split"] == "train"].copy()
    discardpred_validation = discardpred_runs.loc[discardpred_runs["split"] == "validation"].copy()

    model = fit_classifier(
        discardpred_train,
        feature_names=discardpred_feature_names,
        target_column="discard_target",
        validation_frame=discardpred_validation,
        random_seed=config.random_seed,
    )
    probability_frame = model.predict_proba(discardpred_runs)
    discardpred_runs = discardpred_runs.reset_index(drop=True).copy()
    discardpred_runs["discard_probability"] = _binary_positive_probability_values(probability_frame)
    discardpred_runs["placed_probability"] = 1.0 - discardpred_runs["discard_probability"]

    discardpred_metrics = {
        split: _binary_metrics(
            discardpred_runs.loc[discardpred_runs["split"] == split, "discard_target"],
            discardpred_runs.loc[discardpred_runs["split"] == split, "discard_probability"],
        )
        for split in ("validation", "test")
    }

    _plot_discardpred_confusion(
        discardpred_runs.loc[discardpred_runs["split"] == "validation"].copy(),
        discard_threshold=config.discard_threshold,
        title="DiscardPred Confusion",
        output_path=plots_dir / "discardpred_route_confusion.png",
    )
    plot_discardpred_score_mass(discardpred_runs.loc[discardpred_runs["split"] == "validation"], plots_dir / "discardpred_score_mass.png")

    return DiscardPredState(
        model=model,
        runs=discardpred_runs,
        metrics=discardpred_metrics,
    )


def _create_locationpred_joined_view(con: duckdb.DuckDBPyConnection, source_name: str) -> None:
    print("[locationpred_join] creating non-materialized _locationpred_joined view", flush=True)
    con.execute("DROP VIEW IF EXISTS _locationpred_joined")
    con.execute("DROP TABLE IF EXISTS _locationpred_joined")
    con.execute(
        f"""
        CREATE TEMP VIEW _locationpred_joined AS
        SELECT
            r.*,
            COALESCE(s.offerpred_score, 0.0) AS offerpred_score,
            p.discard_probability,
            p.placed_probability
        FROM {source_name} AS r
        LEFT JOIN _offerpred_scored_rows AS s
         ON r.match_id = s.match_id
         AND r.ptr_row_order = s.ptr_row_order
        LEFT JOIN _discardpred_runs_pred AS p
          ON r.match_id = p.match_id
        """
    )
    print("[locationpred_join] ready _locationpred_joined", flush=True)


def _locationpred_normalized_rank_sql(alias: str) -> str:
    return (
        f"COALESCE(CAST({alias}.normalized_offer_rank AS DOUBLE), "
        f"CASE WHEN COALESCE({alias}.run_len, 0) = 0 THEN NULL "
        f"ELSE CAST({alias}.offer_rank AS DOUBLE) / CAST({alias}.run_len AS DOUBLE) END)"
    )


def _locationpred_segment_case_sql(normalized_rank_expression: str) -> str:
    clauses = [f"WHEN {normalized_rank_expression} IS NULL THEN {len(LOCATIONPRED_SEGMENT_BOUNDS) - 1}"]
    for segment_id, upper_bound in enumerate(LOCATIONPRED_SEGMENT_BOUNDS[1:-1], start=1):
        clauses.append(f"WHEN {normalized_rank_expression} <= {upper_bound} THEN {segment_id}")
    clauses.append(f"ELSE {len(LOCATIONPRED_SEGMENT_BOUNDS) - 1}")
    return "CASE " + " ".join(clauses) + " END"


def _locationpred_segment_bound_maps() -> tuple[dict[int, float], dict[int, float]]:
    lower = {segment_id: LOCATIONPRED_SEGMENT_BOUNDS[segment_id - 1] for segment_id in range(1, len(LOCATIONPRED_SEGMENT_BOUNDS))}
    upper = {segment_id: LOCATIONPRED_SEGMENT_BOUNDS[segment_id] for segment_id in range(1, len(LOCATIONPRED_SEGMENT_BOUNDS))}
    return lower, upper


def _create_locationpred_full_scoring_source_view(
    con: duckdb.DuckDBPyConnection,
    config: BenchmarkConfig,
    splits: tuple[str, ...] = ("validation", "test"),
) -> str:
    view_name = "_locationpred_full_scoring_source"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"DROP TABLE IF EXISTS {view_name}")
    split_clause = ", ".join(f"'{_sql_string(split)}'" for split in splits)

    con.execute(
        f"""
        CREATE TEMP VIEW {view_name} AS
        SELECT *
        FROM benchmark.locationpred_scoring_rows
        WHERE split IN ({split_clause})
        """
    )
    return view_name


def _fetch_locationpred_segment_frame_from_joined_view(
    con: duckdb.DuckDBPyConnection,
    splits: tuple[str, ...],
    localizable_only: bool,
) -> pd.DataFrame:
    split_clause = ", ".join(f"'{_sql_string(split)}'" for split in splits)
    norm_expr = _locationpred_normalized_rank_sql("j")
    segment_case = _locationpred_segment_case_sql(norm_expr)
    run_state_clause = (
        "j.run_state = 'localizable_observed_y'"
        if localizable_only
        else "j.run_state IN ('localizable_observed_y', 'none', 'censored_positive')"
    )
    query = f"""
        WITH row_base AS (
            SELECT
                j.match_id,
                j.split,
                j.run_state,
                j.run_len,
                j.first_observed_y_rank,
                j.offer_rank,
                {norm_expr} AS normalized_offer_rank,
                {segment_case} AS locationpred_segment_id,
                COALESCE(j.offerpred_score, 0.0) AS offerpred_score,
                CAST(j.kdpi AS DOUBLE) AS kdpi,
                CAST(j.don_age AS DOUBLE) AS don_age,
                CAST(j.dcd_ind AS DOUBLE) AS dcd_ind,
                CAST(j.distance_nm AS DOUBLE) AS distance_nm,
                CAST(j.mm_total AS DOUBLE) AS mm_total,
                CAST(j.canhx_cpra AS DOUBLE) AS canhx_cpra,
                CAST(j.center_positive_response_rate_365d AS DOUBLE) AS center_positive_response_rate_365d,
                CAST(j.center_mean_accepted_normalized_sequence_365d AS DOUBLE)
                    AS center_mean_accepted_normalized_sequence_365d,
                CAST(j.opo_hist_any_placed_frac_365d AS DOUBLE) AS opo_hist_any_placed_frac_365d,
                CAST(j.opo_hist_mean_first_accept_declines_365d AS DOUBLE)
                    AS opo_hist_mean_first_accept_declines_365d
            FROM _locationpred_joined AS j
            WHERE j.split IN ({split_clause})
              AND {run_state_clause}
        ),
        segment_base AS (
            SELECT
                match_id,
                split,
                run_state,
                run_len,
                first_observed_y_rank,
                locationpred_segment_id,
                MIN(normalized_offer_rank) AS locationpred_segment_norm_min,
                MAX(normalized_offer_rank) AS locationpred_segment_norm_max,
                AVG(normalized_offer_rank) AS locationpred_segment_norm_mean,
                MIN(offer_rank) AS locationpred_segment_offer_rank_min,
                MAX(offer_rank) AS locationpred_segment_offer_rank_max,
                AVG(offer_rank) AS locationpred_segment_offer_rank_mean,
                COUNT(*) AS locationpred_segment_row_count,
                SUM(offerpred_score) AS locationpred_segment_offerpred_sum,
                MAX(offerpred_score) AS locationpred_segment_offerpred_max,
                AVG(offerpred_score) AS locationpred_segment_offerpred_mean,
                AVG(kdpi) AS kdpi_mean,
                AVG(don_age) AS don_age_mean,
                AVG(dcd_ind) AS dcd_share,
                AVG(distance_nm) AS distance_nm_mean,
                AVG(mm_total) AS mm_total_mean,
                AVG(canhx_cpra) AS canhx_cpra_mean,
                AVG(center_positive_response_rate_365d) AS center_positive_response_rate_365d_mean,
                AVG(center_mean_accepted_normalized_sequence_365d)
                    AS center_mean_accepted_normalized_sequence_365d_mean,
                AVG(opo_hist_any_placed_frac_365d) AS opo_hist_any_placed_frac_365d_mean,
                AVG(opo_hist_mean_first_accept_declines_365d) AS opo_hist_mean_first_accept_declines_365d_mean
            FROM row_base
            GROUP BY 1, 2, 3, 4, 5, 6
        ),
        labeled AS (
            SELECT
                *,
                CASE
                    WHEN run_state = 'localizable_observed_y'
                         AND first_observed_y_rank BETWEEN locationpred_segment_offer_rank_min AND locationpred_segment_offer_rank_max
                    THEN 1
                    ELSE 0
                END AS locationpred_segment_target
            FROM segment_base
        )
        SELECT
            *,
            MAX(CASE WHEN locationpred_segment_target = 1 THEN locationpred_segment_id END)
                OVER (PARTITION BY match_id) AS locationpred_true_segment_id
        FROM labeled
        ORDER BY split, match_id, locationpred_segment_id
    """
    return con.execute(query).fetch_df()


def _prepare_locationpred_segment_features(
    frame: pd.DataFrame,
    riskset_only: bool,
) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        for feature_name in LOCATIONPRED_SEGMENT_FEATURES:
            if feature_name not in prepared.columns:
                prepared[feature_name] = pd.Series(dtype="float64")
        prepared["locationpred_segment_sample_weight"] = pd.Series(dtype="float64")
        return prepared

    prepared = prepared.sort_values(["split", "match_id", "locationpred_segment_id"]).reset_index(drop=True)
    lower_map, upper_map = _locationpred_segment_bound_maps()
    prepared["locationpred_segment_norm_lower"] = prepared["locationpred_segment_id"].map(lower_map).astype(float)
    prepared["locationpred_segment_norm_upper"] = prepared["locationpred_segment_id"].map(upper_map).astype(float)

    group_keys = ["split", "match_id"]
    total_offerpred = prepared.groupby(group_keys)["locationpred_segment_offerpred_sum"].transform("sum")
    total_rows = prepared.groupby(group_keys)["locationpred_segment_row_count"].transform("sum").astype(float)
    fallback_mass = (
        prepared["locationpred_segment_row_count"].astype(float) / total_rows.clip(lower=1.0)
    ).fillna(0.0)
    prepared["locationpred_segment_offerpred_mass"] = np.where(
        total_offerpred.astype(float).abs() > 1e-12,
        prepared["locationpred_segment_offerpred_sum"].astype(float) / total_offerpred.astype(float),
        fallback_mass,
    )
    cumulative_mass = prepared.groupby(group_keys)["locationpred_segment_offerpred_mass"].cumsum()
    prepared["locationpred_segment_cum_offerpred_mass_before"] = (
        cumulative_mass - prepared["locationpred_segment_offerpred_mass"]
    ).clip(lower=0.0)
    prepared["locationpred_segment_offerpred_mass_after"] = (
        1.0 - prepared["locationpred_segment_cum_offerpred_mass_before"] - prepared["locationpred_segment_offerpred_mass"]
    ).clip(lower=0.0)

    if riskset_only:
        true_segment = pd.to_numeric(prepared["locationpred_true_segment_id"], errors="coerce")
        prepared = prepared.loc[
            true_segment.notna() & (prepared["locationpred_segment_id"].astype(float) <= true_segment)
        ].copy()

    risk_counts = prepared.groupby(group_keys)["locationpred_segment_id"].transform("count").astype(float)
    prepared["locationpred_segment_sample_weight"] = 1.0 / risk_counts.clip(lower=1.0)
    return prepared.reset_index(drop=True)


def _score_locationpred_segment_frame(
    frame: pd.DataFrame,
    feature_names: list[str],
    model: Any,
) -> pd.DataFrame:
    scored = frame.copy()
    probability_frame = model.predict_proba(scored[feature_names])
    scored["locationpred_segment_hazard_probability"] = _binary_positive_probability_values(
        probability_frame
    )
    return scored


def _segment_hazard_to_probability(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("locationpred_segment_id").copy()
    hazards = pd.to_numeric(ordered["locationpred_segment_hazard_probability"], errors="coerce").fillna(0.0).astype(float)
    hazards = np.clip(hazards.to_numpy(), 0.0, 1.0 - 1e-6)
    survival = 1.0
    segment_mass: list[float] = []
    for hazard in hazards:
        mass = survival * float(hazard)
        segment_mass.append(mass)
        survival *= (1.0 - float(hazard))
    total_mass = float(sum(segment_mass))
    if total_mass <= 1e-12:
        segment_mass = [1.0 / len(ordered)] * len(ordered)
        total_mass = 1.0
    ordered["locationpred_segment_probability"] = np.asarray(segment_mass, dtype=float) / total_mass
    ordered["locationpred_segment_no_event_probability"] = survival
    return ordered


def _create_final_row_predictions_segment_view(
    con: duckdb.DuckDBPyConnection,
    config: BenchmarkConfig,
) -> None:
    discard_threshold = float(config.discard_threshold)
    norm_expr = _locationpred_normalized_rank_sql("r")
    segment_case = _locationpred_segment_case_sql(norm_expr)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW _final_row_predictions AS
        WITH base AS (
            SELECT
                r.match_id,
                r.offer_rank,
                r.ptr_row_order,
                r.match_submit_dt,
                r.split,
                r.run_state,
                r.first_observed_y_rank,
                r.timing_bucket,
                r.locationpred_target,
                COALESCE(r.offerpred_score, 0.0) AS offerpred_score,
                {segment_case} AS locationpred_segment_id,
                p.discard_target,
                p.route_target,
                p.timing_target,
                p.run_len,
                p.discard_probability,
                p.placed_probability,
                sp.locationpred_segment_hazard_probability,
                sp.locationpred_segment_probability
            FROM _locationpred_joined AS r
            JOIN _discardpred_runs_pred AS p USING (match_id)
            LEFT JOIN _locationpred_segment_predictions AS sp
              ON r.match_id = sp.match_id
             AND {segment_case} = sp.locationpred_segment_id
        ),
        weighted AS (
            SELECT
                *,
                COUNT(*) OVER (PARTITION BY match_id, locationpred_segment_id) AS locationpred_segment_row_count,
                SUM(COALESCE(offerpred_score, 0.0)) OVER (PARTITION BY match_id, locationpred_segment_id)
                    AS locationpred_segment_offerpred_row_sum,
                CASE
                    WHEN discard_probability >= {discard_threshold} THEN 'discard'
                    ELSE 'localize'
                END AS decision
            FROM base
        )
        SELECT
            *,
            CASE
                WHEN decision <> 'localize' THEN 0.0
                WHEN COALESCE(locationpred_segment_probability, 0.0) <= 0.0 THEN 0.0
                WHEN locationpred_segment_offerpred_row_sum > 1e-12
                THEN locationpred_segment_probability * (COALESCE(offerpred_score, 0.0) / locationpred_segment_offerpred_row_sum)
                ELSE locationpred_segment_probability / locationpred_segment_row_count
            END AS locationpred_first_event_probability,
            CASE
                WHEN decision <> 'localize' THEN 0.0
                WHEN COALESCE(locationpred_segment_probability, 0.0) <= 0.0 THEN 0.0
                WHEN locationpred_segment_offerpred_row_sum > 1e-12
                THEN locationpred_segment_probability
                     * (COALESCE(offerpred_score, 0.0) / locationpred_segment_offerpred_row_sum)
                ELSE locationpred_segment_probability / locationpred_segment_row_count
            END AS final_row_probability,
            COALESCE(locationpred_segment_hazard_probability, 0.0) AS locationpred_hazard_probability
        FROM weighted
        """
    )


def _fetch_locationpred_eval_from_final_row_view(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            best.match_id,
            best.offer_rank AS pred_rank,
            best.first_observed_y_rank AS true_rank,
            best.timing_bucket AS pred_timing,
            p.timing_target AS true_timing
        FROM (
            SELECT *
            FROM (
                SELECT
                    match_id,
                    offer_rank,
                    first_observed_y_rank,
                    timing_bucket,
                    ROW_NUMBER() OVER (
                        PARTITION BY match_id
                        ORDER BY final_row_probability DESC, offer_rank ASC
                    ) AS rn
                FROM _final_row_predictions
                WHERE split = 'validation'
                  AND run_state = 'localizable_observed_y'
                  AND decision = 'localize'
            ) AS ranked
            WHERE rn = 1
        ) AS best
        JOIN _discardpred_runs_pred AS p USING (match_id)
        WHERE p.split = 'validation'
        ORDER BY best.match_id
        """
    ).fetch_df()


def _fetch_run_predictions_from_view(
    con: duckdb.DuckDBPyConnection,
    split: str,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    discard_threshold = float(config.discard_threshold)
    return con.execute(
        f"""
        WITH best_pred AS (
            SELECT match_id, offer_rank AS predicted_rank
            FROM (
                SELECT
                    match_id,
                    offer_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY match_id
                        ORDER BY final_row_probability DESC, offer_rank ASC
                    ) AS rn
                FROM _final_row_predictions
                WHERE split = ?
                  AND decision = 'localize'
            ) AS ranked
            WHERE rn = 1
        )
        SELECT
            p.match_id,
            p.split,
            p.discard_target,
            p.route_target,
            p.timing_target,
            p.first_observed_y_rank,
            p.run_len,
            CASE
                WHEN p.discard_probability >= {discard_threshold} THEN 'discard'
                ELSE 'localize'
            END AS decision,
            b.predicted_rank,
            p.discard_probability,
            p.placed_probability
        FROM _discardpred_runs_pred AS p
        LEFT JOIN best_pred AS b USING (match_id)
        WHERE p.split = ?
        ORDER BY p.match_submit_dt, p.match_id
        """,
        [split, split],
    ).fetch_df()


def _train_locationpred_segment_hazard_catboost(
    con: duckdb.DuckDBPyConnection,
    config: BenchmarkConfig,
    match_limit_per_split: int | None,
    discardpred_runs: pd.DataFrame,
    final_artifact_root: Path,
) -> LocationPredState:
    del match_limit_per_split
    con.register("_discardpred_runs_pred", discardpred_runs)

    training_source_name = _locationpred_source_table(config)
    _create_locationpred_joined_view(con, training_source_name)
    training_segments = _fetch_locationpred_segment_frame_from_joined_view(
        con=con,
        splits=("train", "validation"),
        localizable_only=True,
    )
    training_segments = _prepare_locationpred_segment_features(training_segments, riskset_only=True)

    train_frame = training_segments.loc[training_segments["split"] == "train"].copy()
    validation_frame = training_segments.loc[training_segments["split"] == "validation"].copy()

    model = fit_native_catboost_classifier(
        train_frame=train_frame,
        feature_names=LOCATIONPRED_SEGMENT_FEATURES,
        target_column="locationpred_segment_target",
        validation_frame=validation_frame,
        sample_weight_column="locationpred_segment_sample_weight",
        random_seed=config.random_seed,
        iterations=config.locationpred_catboost_iterations,
        depth=config.locationpred_catboost_depth,
        learning_rate=config.locationpred_catboost_learning_rate,
        l2_leaf_reg=config.locationpred_catboost_l2_leaf_reg,
        early_stopping_rounds=config.locationpred_catboost_early_stopping_rounds,
    )

    write_parquet(
        _extract_feature_importance(model, LOCATIONPRED_SEGMENT_FEATURES),
        final_artifact_root / "locationpred_feature_importance.parquet",
    )

    scoring_source_name = _create_locationpred_full_scoring_source_view(con, config, splits=("validation", "test"))
    _create_locationpred_joined_view(con, scoring_source_name)
    scoring_segments = _fetch_locationpred_segment_frame_from_joined_view(
        con=con,
        splits=("validation", "test"),
        localizable_only=False,
    )
    scoring_segments = _prepare_locationpred_segment_features(scoring_segments, riskset_only=False)
    scoring_segments = _score_locationpred_segment_frame(
        frame=scoring_segments,
        feature_names=LOCATIONPRED_SEGMENT_FEATURES,
        model=model,
    )
    scoring_segments = pd.concat(
        [_segment_hazard_to_probability(group) for _, group in scoring_segments.groupby(["split", "match_id"], sort=False)],
        ignore_index=True,
    )
    write_parquet(scoring_segments, final_artifact_root / "locationpred_segment_predictions.parquet")

    con.register("_locationpred_segment_predictions_frame", scoring_segments)
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW _locationpred_segment_predictions AS
        SELECT * FROM _locationpred_segment_predictions_frame
        """
    )
    _create_final_row_predictions_segment_view(con, config)

    for split in ("validation", "test"):
        row_predictions_path = final_artifact_root / f"{split}_row_predictions.parquet"
        row_predictions_full_run_path = final_artifact_root / f"{split}_row_predictions_full_run.parquet"
        _copy_query_to_parquet(
            con,
            f"SELECT * FROM _final_row_predictions WHERE split = '{split}' ORDER BY match_submit_dt, match_id, offer_rank",
            row_predictions_path,
        )
        shutil.copyfile(row_predictions_path, row_predictions_full_run_path)

    validation_run_predictions = _fetch_run_predictions_from_view(con, "validation", config)
    test_run_predictions = _fetch_run_predictions_from_view(con, "test", config)
    write_parquet(validation_run_predictions, final_artifact_root / "validation_run_predictions_full_run.parquet")
    write_parquet(test_run_predictions, final_artifact_root / "test_run_predictions_full_run.parquet")

    locationpred_eval = _fetch_locationpred_eval_from_final_row_view(con)
    validation_final_eval = validation_run_predictions.loc[
        validation_run_predictions["route_target"] == "localizable_observed_y",
        ["match_id", "predicted_rank", "first_observed_y_rank"],
    ].rename(columns={"predicted_rank": "pred_rank", "first_observed_y_rank": "true_rank"})
    test_final_eval = test_run_predictions.loc[
        test_run_predictions["route_target"] == "localizable_observed_y",
        ["match_id", "predicted_rank", "first_observed_y_rank"],
    ].rename(columns={"predicted_rank": "pred_rank", "first_observed_y_rank": "true_rank"})

    return LocationPredState(
        model=model,
        backend="segment_hazard_catboost",
        validation_run_predictions=validation_run_predictions,
        test_run_predictions=test_run_predictions,
        validation_final_eval=validation_final_eval,
        test_final_eval=test_final_eval,
        locationpred_eval=locationpred_eval,
        mode="segment_hazard",
    )


def train_offerpred_benchmark(
    config: BenchmarkConfig | None,
    run_name: str = "offerpred",
    thread_count: int = 8,
    match_limit_per_split: int | None = None,
    benchmark_db: Path | None = None,
    benchmark_manifest_path: Path | None = None,
    artifact_root: Path | None = None,
) -> TrainingArtifacts:
    config, benchmark_manifest, config_source = _load_benchmark_config(config, benchmark_manifest_path)
    benchmark_db = benchmark_db or Path(benchmark_manifest.get("benchmark_db", str(config.benchmark_db_path)))

    final_artifact_root = (artifact_root or config.artifact_root) / run_name
    plots_dir = final_artifact_root / "plots"
    intermediate_dir = final_artifact_root / "intermediate"
    final_artifact_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(benchmark_db), read_only=True)
    con.execute(f"PRAGMA threads={int(thread_count)};")

    match_labels = con.execute("SELECT * FROM benchmark.match_labels ORDER BY match_year, match_id").fetch_df()
    plot_data_qa(match_labels, plots_dir / "data_qa.png")

    offerpred_state = _train_offerpred_sampled_catboost(
        con=con,
        config=config,
        match_limit_per_split=match_limit_per_split,
        intermediate_dir=intermediate_dir,
    )

    _register_offerpred_scored_view(con, offerpred_state)
    plot_offerpred_diagnostics(offerpred_state.eval_rows, plots_dir / "offerpred_diagnostics.png")
    plot_offerpred_topk(offerpred_state.eval_rows, plots_dir / "offerpred_topk.png")
    plot_offerpred_yearly_metrics(offerpred_state.yearly_metrics, plots_dir / "offerpred_yearly_metrics.png")
    plot_offerpred_feature_importance(offerpred_state.feature_importance, plots_dir / "offerpred_feature_importance.png")

    write_parquet(offerpred_state.eval_rows, final_artifact_root / "offerpred_eval_sample.parquet")
    write_parquet(offerpred_state.offerpred_aggregates, final_artifact_root / "offerpred_run_aggregates.parquet")
    write_parquet(offerpred_state.feature_importance, final_artifact_root / "offerpred_feature_importance.parquet")
    write_parquet(offerpred_state.yearly_metrics, final_artifact_root / "offerpred_yearly_metrics.parquet")

    plot_files = sorted(str(path.relative_to(final_artifact_root)) for path in plots_dir.glob("*.png"))
    run_manifest = {
        "model_name": OFFERPRED_NAME,
        "config_source": config_source,
        "config": config.to_dict(),
        "benchmark_db": str(benchmark_db),
        "benchmark_manifest": None if benchmark_manifest_path is None else str(benchmark_manifest_path),
        "backend": offerpred_state.backend,
        "training_mode": offerpred_state.mode,
        "plot_files": plot_files,
        "offerpred_metrics": offerpred_state.metrics,
        "offerpred_yearly_metrics": offerpred_state.yearly_metrics.to_dict(orient="records"),
        "top_feature_importance": offerpred_state.feature_importance.head(25).to_dict(orient="records"),
        "benchmark_table_counts": benchmark_manifest.get("table_counts", {}),
        "offerpred_scored_parts_dir": None if offerpred_state.scored_parts_dir is None else str(offerpred_state.scored_parts_dir),
    }
    dump_json(final_artifact_root / "run_manifest.json", run_manifest)
    con.close()
    return TrainingArtifacts(
        artifact_root=final_artifact_root,
        manifest_path=final_artifact_root / "run_manifest.json",
    )


def train_discardpred_benchmark(
    config: BenchmarkConfig | None,
    run_name: str = "discardpred",
    thread_count: int = 8,
    match_limit_per_split: int | None = None,
    benchmark_db: Path | None = None,
    benchmark_manifest_path: Path | None = None,
    artifact_root: Path | None = None,
    offerpred_artifact_root: Path | None = None,
) -> TrainingArtifacts:
    config, benchmark_manifest, config_source = _load_benchmark_config(config, benchmark_manifest_path)
    benchmark_db = benchmark_db or Path(benchmark_manifest.get("benchmark_db", str(config.benchmark_db_path)))

    final_artifact_root = (artifact_root or config.artifact_root) / run_name
    plots_dir = final_artifact_root / "plots"
    intermediate_dir = final_artifact_root / "intermediate"
    final_artifact_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(benchmark_db), read_only=True)
    con.execute(f"PRAGMA threads={int(thread_count)};")

    match_labels = con.execute("SELECT * FROM benchmark.match_labels ORDER BY match_year, match_id").fetch_df()
    plot_data_qa(match_labels, plots_dir / "data_qa.png")

    if offerpred_artifact_root is not None:
        offerpred_state = _load_offerpred_state_from_artifact(Path(offerpred_artifact_root))
    else:
        offerpred_state = _train_offerpred_sampled_catboost(
            con=con,
            config=config,
            match_limit_per_split=match_limit_per_split,
            intermediate_dir=intermediate_dir,
        )

    _register_offerpred_scored_view(con, offerpred_state)
    plot_offerpred_diagnostics(offerpred_state.eval_rows, plots_dir / "offerpred_diagnostics.png")
    plot_offerpred_topk(offerpred_state.eval_rows, plots_dir / "offerpred_topk.png")
    plot_offerpred_yearly_metrics(offerpred_state.yearly_metrics, plots_dir / "offerpred_yearly_metrics.png")
    plot_offerpred_feature_importance(offerpred_state.feature_importance, plots_dir / "offerpred_feature_importance.png")

    discardpred_runs = _prepare_discardpred_runs(
        con=con,
        offerpred_state=offerpred_state,
        config=config,
        match_limit_per_split=match_limit_per_split,
        artifact_root=final_artifact_root,
    )
    discardpred_state = _train_discardpred_models(
        discardpred_runs=discardpred_runs,
        config=config,
        plots_dir=plots_dir,
    )

    write_parquet(offerpred_state.eval_rows, final_artifact_root / "offerpred_eval_sample.parquet")
    write_parquet(offerpred_state.offerpred_aggregates, final_artifact_root / "offerpred_run_aggregates.parquet")
    write_parquet(offerpred_state.feature_importance, final_artifact_root / "offerpred_feature_importance.parquet")
    write_parquet(offerpred_state.yearly_metrics, final_artifact_root / "offerpred_yearly_metrics.parquet")
    write_parquet(discardpred_state.runs, final_artifact_root / "discardpred_scored_runs.parquet")
    write_parquet(
        discardpred_state.runs[
            [
                "match_id",
                "split",
                "discard_target",
                "route_target",
                "timing_target",
                "discard_probability",
                "placed_probability",
            ]
        ],
        final_artifact_root / "discardpred_run_predictions.parquet",
    )

    plot_files = sorted(str(path.relative_to(final_artifact_root)) for path in plots_dir.glob("*.png"))
    run_manifest = {
        "model_name": DISCARDPRED_NAME,
        "config_source": config_source,
        "config": config.to_dict(),
        "benchmark_db": str(benchmark_db),
        "benchmark_manifest": None if benchmark_manifest_path is None else str(benchmark_manifest_path),
        "source_offerpred_artifact_root": None if offerpred_artifact_root is None else str(offerpred_artifact_root),
        "backends": {
            "offerpred": offerpred_state.backend,
            "discardpred": discardpred_state.model.backend,
        },
        "training_modes": {
            "offerpred": offerpred_state.mode,
        },
        "plot_files": plot_files,
        "offerpred_metrics": offerpred_state.metrics,
        "discardpred_metrics": discardpred_state.metrics,
        "benchmark_table_counts": benchmark_manifest.get("table_counts", {}),
        "offerpred_scored_parts_dir": None if offerpred_state.scored_parts_dir is None else str(offerpred_state.scored_parts_dir),
    }
    dump_json(final_artifact_root / "run_manifest.json", run_manifest)
    con.close()
    return TrainingArtifacts(
        artifact_root=final_artifact_root,
        manifest_path=final_artifact_root / "run_manifest.json",
    )


def train_locationpred_benchmark(
    config: BenchmarkConfig | None,
    run_name: str = "locationpred",
    thread_count: int = 8,
    match_limit_per_split: int | None = None,
    benchmark_db: Path | None = None,
    benchmark_manifest_path: Path | None = None,
    artifact_root: Path | None = None,
    offerpred_artifact_root: Path | None = None,
    discardpred_artifact_root: Path | None = None,
    offerpred_scored_parts_dir: Path | None = None,
    discardpred_predictions_path: Path | None = None,
) -> TrainingArtifacts:
    if offerpred_scored_parts_dir is None and offerpred_artifact_root is not None:
        offerpred_scored_parts_dir = Path(offerpred_artifact_root) / "intermediate" / "offerpred_scored_rows"
    if discardpred_predictions_path is None and discardpred_artifact_root is not None:
        discardpred_predictions_path = Path(discardpred_artifact_root) / "discardpred_scored_runs.parquet"
    if offerpred_scored_parts_dir is None:
        raise ValueError("offerpred_scored_parts_dir is required for LocationPred training")
    if discardpred_predictions_path is None:
        raise ValueError("discardpred_predictions_path is required for LocationPred training")

    config, benchmark_manifest, config_source = _load_benchmark_config(config, benchmark_manifest_path)
    benchmark_db = benchmark_db or Path(benchmark_manifest.get("benchmark_db", str(config.benchmark_db_path)))
    offerpred_scored_parts_dir = Path(offerpred_scored_parts_dir)
    discardpred_predictions_path = Path(discardpred_predictions_path)
    offerpred_backend = "artifact_reuse"
    if offerpred_artifact_root is not None:
        offerpred_manifest_path = Path(offerpred_artifact_root) / "run_manifest.json"
        if offerpred_manifest_path.exists():
            offerpred_manifest = load_json(offerpred_manifest_path)
            offerpred_backend = str(
                offerpred_manifest.get("backend")
                or offerpred_manifest.get("backends", {}).get("offerpred")
                or "artifact_reuse"
            )

    final_artifact_root = (artifact_root or config.artifact_root) / run_name
    plots_dir = final_artifact_root / "plots"
    final_artifact_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(benchmark_db), read_only=True)
    con.execute(f"PRAGMA threads={int(thread_count)};")

    match_labels = con.execute("SELECT * FROM benchmark.match_labels ORDER BY match_year, match_id").fetch_df()
    plot_data_qa(match_labels, plots_dir / "data_qa.png")

    _register_offerpred_scored_view_from_parts_dir(con, offerpred_scored_parts_dir)
    discardpred_runs = con.execute(
        "SELECT * FROM read_parquet(?) ORDER BY match_submit_dt, match_id",
        [str(discardpred_predictions_path)],
    ).fetch_df()
    required_discardpred_columns = {
        "match_id",
        "split",
        "match_submit_dt",
        "discard_target",
        "route_target",
        "timing_target",
        "first_observed_y_rank",
        "run_len",
        "discard_probability",
        "placed_probability",
    }
    missing_columns = sorted(required_discardpred_columns.difference(discardpred_runs.columns))
    if missing_columns:
        raise ValueError(
            "discardpred_predictions_path is missing required columns for LocationPred training: "
            + ", ".join(missing_columns)
    )

    discardpred_train = discardpred_runs.loc[discardpred_runs["split"] == "train"].copy()
    locationpred_state = _train_locationpred_segment_hazard_catboost(
        con=con,
        config=config,
        match_limit_per_split=match_limit_per_split,
        discardpred_runs=discardpred_runs,
        final_artifact_root=final_artifact_root,
    )

    plot_locationpred_localizer(
        locationpred_state.locationpred_eval[["match_id", "pred_rank", "true_rank"]],
        locationpred_state.locationpred_eval[["true_timing", "pred_timing"]],
        plots_dir / "locationpred_localizer.png",
    )

    validation_localizable = locationpred_state.validation_run_predictions.loc[
        locationpred_state.validation_run_predictions["route_target"] == "localizable_observed_y"
    ].copy()

    offerpred_best_validation = _query_offerpred_best_ranks(con, "validation")
    offerpred_validation_best = validation_localizable[["match_id"]].merge(
        offerpred_best_validation,
        on="match_id",
        how="left",
    )["pred_rank"].fillna(1).astype(int)

    empirical_validation = _empirical_rank_predictions(discardpred_train, validation_localizable)
    earliest_validation = pd.Series([1] * len(validation_localizable), index=validation_localizable.index, dtype=int)

    validation_baselines = {
        "final_pipeline": _localizer_metrics(locationpred_state.validation_final_eval),
        "empirical_rank": _localizer_metrics(
            pd.DataFrame({"pred_rank": empirical_validation, "true_rank": validation_localizable["first_observed_y_rank"]})
        ),
        "earliest_offer": _localizer_metrics(
            pd.DataFrame({"pred_rank": earliest_validation, "true_rank": validation_localizable["first_observed_y_rank"]})
        ),
        "offerpred_only": _localizer_metrics(
            pd.DataFrame({"pred_rank": offerpred_validation_best, "true_rank": validation_localizable["first_observed_y_rank"]})
        ),
    }

    coverage_rows = pd.DataFrame(
        [
            {
                "model": model_name,
                "coverage": (
                    float((locationpred_state.validation_run_predictions["decision"] == "localize").mean())
                    if model_name == "final_pipeline"
                    else 1.0
                ),
                **metrics,
            }
            for model_name, metrics in validation_baselines.items()
        ]
    )
    plot_validation_sweep(
        [{"model": row["model"], **{key: row[key] for key in row.index if key != "model"}} for _, row in coverage_rows.iterrows()],
        plots_dir / "validation_sweep.png",
    )
    plot_pipeline_dashboard(coverage_rows, plots_dir / "whole_pipeline_dashboard.png")

    write_parquet(locationpred_state.validation_run_predictions, final_artifact_root / "validation_run_predictions.parquet")
    write_parquet(locationpred_state.test_run_predictions, final_artifact_root / "test_run_predictions.parquet")

    plot_files = sorted(str(path.relative_to(final_artifact_root)) for path in plots_dir.glob("*.png"))
    run_manifest = {
        "model_name": LOCATIONPRED_NAME,
        "config_source": config_source,
        "config": config.to_dict(),
        "benchmark_db": str(benchmark_db),
        "benchmark_manifest": None if benchmark_manifest_path is None else str(benchmark_manifest_path),
        "source_offerpred_artifact_root": None if offerpred_artifact_root is None else str(offerpred_artifact_root),
        "source_discardpred_artifact_root": None if discardpred_artifact_root is None else str(discardpred_artifact_root),
        "source_offerpred_scored_parts_dir": str(offerpred_scored_parts_dir),
        "source_discardpred_predictions_path": str(discardpred_predictions_path),
        "backends": {
            "offerpred": offerpred_backend,
            "locationpred": locationpred_state.backend,
        },
        "training_modes": {
            "offerpred": "artifact_reuse",
            "locationpred": locationpred_state.mode,
        },
        "plot_files": plot_files,
        "validation_metrics": validation_baselines,
        "test_localizer_metrics": _localizer_metrics(locationpred_state.test_final_eval),
        "timing_cutoffs": {
            "early": config.early_cutoff,
            "mid": config.mid_cutoff,
        },
        "best_validation_metrics": validation_baselines["final_pipeline"],
        "benchmark_table_counts": benchmark_manifest.get("table_counts", {}),
    }
    dump_json(final_artifact_root / "run_manifest.json", run_manifest)
    con.close()
    return TrainingArtifacts(
        artifact_root=final_artifact_root,
        manifest_path=final_artifact_root / "run_manifest.json",
    )


def train_benchmark(
    config: BenchmarkConfig | None,
    run_name: str = "default_run",
    max_trials: int = 1,
    thread_count: int = 8,
    match_limit_per_split: int | None = None,
    benchmark_db: Path | None = None,
    benchmark_manifest_path: Path | None = None,
    artifact_root: Path | None = None,
) -> TrainingArtifacts:
    del max_trials

    config, benchmark_manifest, config_source = _load_benchmark_config(config, benchmark_manifest_path)
    benchmark_db = benchmark_db or Path(benchmark_manifest.get("benchmark_db", str(config.benchmark_db_path)))

    final_artifact_root = (artifact_root or config.artifact_root) / run_name
    plots_dir = final_artifact_root / "plots"
    intermediate_dir = final_artifact_root / "intermediate"
    final_artifact_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(benchmark_db), read_only=True)
    con.execute(f"PRAGMA threads={int(thread_count)};")

    match_labels = con.execute("SELECT * FROM benchmark.match_labels ORDER BY match_year, match_id").fetch_df()
    plot_data_qa(match_labels, plots_dir / "data_qa.png")

    offerpred_state = _train_offerpred_sampled_catboost(
        con=con,
        config=config,
        match_limit_per_split=match_limit_per_split,
        intermediate_dir=intermediate_dir,
    )

    _register_offerpred_scored_view(con, offerpred_state)
    plot_offerpred_diagnostics(offerpred_state.eval_rows, plots_dir / "offerpred_diagnostics.png")
    plot_offerpred_topk(offerpred_state.eval_rows, plots_dir / "offerpred_topk.png")
    discardpred_runs = _prepare_discardpred_runs(
        con=con,
        offerpred_state=offerpred_state,
        config=config,
        match_limit_per_split=match_limit_per_split,
        artifact_root=final_artifact_root,
    )
    discardpred_state = _train_discardpred_models(
        discardpred_runs=discardpred_runs,
        config=config,
        plots_dir=plots_dir,
    )
    discardpred_runs = discardpred_state.runs
    discardpred_metrics = discardpred_state.metrics
    discardpred_model = discardpred_state.model
    discardpred_train = discardpred_runs.loc[discardpred_runs["split"] == "train"].copy()

    locationpred_state = _train_locationpred_segment_hazard_catboost(
        con=con,
        config=config,
        match_limit_per_split=match_limit_per_split,
        discardpred_runs=discardpred_runs,
        final_artifact_root=final_artifact_root,
    )

    plot_locationpred_localizer(
        locationpred_state.locationpred_eval[["match_id", "pred_rank", "true_rank"]],
        locationpred_state.locationpred_eval[["true_timing", "pred_timing"]],
        plots_dir / "locationpred_localizer.png",
    )

    validation_localizable = locationpred_state.validation_run_predictions.loc[
        locationpred_state.validation_run_predictions["route_target"] == "localizable_observed_y"
    ].copy()

    offerpred_best_validation = _query_offerpred_best_ranks(con, "validation")
    offerpred_validation_best = validation_localizable[["match_id"]].merge(
        offerpred_best_validation,
        on="match_id",
        how="left",
    )["pred_rank"].fillna(1).astype(int)

    empirical_validation = _empirical_rank_predictions(discardpred_train, validation_localizable)
    earliest_validation = pd.Series([1] * len(validation_localizable), index=validation_localizable.index, dtype=int)

    validation_baselines = {
        "final_pipeline": _localizer_metrics(locationpred_state.validation_final_eval),
        "empirical_rank": _localizer_metrics(
            pd.DataFrame({"pred_rank": empirical_validation, "true_rank": validation_localizable["first_observed_y_rank"]})
        ),
        "earliest_offer": _localizer_metrics(
            pd.DataFrame({"pred_rank": earliest_validation, "true_rank": validation_localizable["first_observed_y_rank"]})
        ),
        "offerpred_only": _localizer_metrics(
            pd.DataFrame({"pred_rank": offerpred_validation_best, "true_rank": validation_localizable["first_observed_y_rank"]})
        ),
    }

    coverage_rows = pd.DataFrame(
        [
            {
                "model": model_name,
                "coverage": (
                    float((locationpred_state.validation_run_predictions["decision"] == "localize").mean())
                    if model_name == "final_pipeline"
                    else 1.0
                ),
                **metrics,
            }
            for model_name, metrics in validation_baselines.items()
        ]
    )
    plot_validation_sweep(
        [{"model": row["model"], **{key: row[key] for key in row.index if key != "model"}} for _, row in coverage_rows.iterrows()],
        plots_dir / "validation_sweep.png",
    )
    plot_pipeline_dashboard(coverage_rows, plots_dir / "whole_pipeline_dashboard.png")

    write_parquet(offerpred_state.eval_rows, final_artifact_root / "offerpred_row_scores.parquet")
    write_parquet(locationpred_state.validation_run_predictions, final_artifact_root / "validation_run_predictions.parquet")
    write_parquet(locationpred_state.test_run_predictions, final_artifact_root / "test_run_predictions.parquet")

    plot_files = sorted(str(path.relative_to(final_artifact_root)) for path in plots_dir.glob("*.png"))
    run_manifest = {
        "model_name": "kidney_utilization_pipeline",
        "config_source": config_source,
        "config": config.to_dict(),
        "benchmark_db": str(benchmark_db),
        "benchmark_manifest": None if benchmark_manifest_path is None else str(benchmark_manifest_path),
        "backends": {
            "offerpred": offerpred_state.backend,
            "discardpred": discardpred_model.backend,
            "locationpred": locationpred_state.backend,
        },
        "training_modes": {
            "offerpred": offerpred_state.mode,
            "locationpred": locationpred_state.mode,
        },
        "plot_files": plot_files,
        "offerpred_metrics": offerpred_state.metrics,
        "discardpred_metrics": discardpred_metrics,
        "validation_metrics": validation_baselines,
        "test_localizer_metrics": _localizer_metrics(locationpred_state.test_final_eval),
        "timing_cutoffs": {
            "early": config.early_cutoff,
            "mid": config.mid_cutoff,
        },
        "best_validation_metrics": validation_baselines["final_pipeline"],
        "benchmark_table_counts": benchmark_manifest.get("table_counts", {}),
        "offerpred_scored_parts_dir": None if offerpred_state.scored_parts_dir is None else str(offerpred_state.scored_parts_dir),
    }
    dump_json(final_artifact_root / "run_manifest.json", run_manifest)
    con.close()
    return TrainingArtifacts(
        artifact_root=final_artifact_root,
        manifest_path=final_artifact_root / "run_manifest.json",
    )


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _append_metric_row(
    rows: list[dict[str, Any]],
    model: str,
    split: str,
    metric: str,
    value: Any,
) -> None:
    if value is None:
        return
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return
    if math.isnan(numeric_value):
        return
    rows.append(
        {
            "model": model,
            "split": split,
            "metric": metric,
            "value": numeric_value,
        }
    )


def _locationpred_decision_shares(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "decision" not in frame.columns:
        return {}
    shares = frame["decision"].value_counts(normalize=True).to_dict()
    return {
        f"share_{label}": float(shares.get(label, 0.0))
        for label in ["localize", "discard"]
    }


def build_consolidated_report(
    offerpred_artifact_root: Path,
    discardpred_artifact_root: Path,
    locationpred_artifact_root: Path,
    report_root: Path | None = None,
    report_name: str = "consolidated_best_artifacts_report",
) -> ConsolidatedReportArtifacts:
    offerpred_artifact_root = Path(offerpred_artifact_root)
    discardpred_artifact_root = Path(discardpred_artifact_root)
    locationpred_artifact_root = Path(locationpred_artifact_root)
    report_root = Path(report_root) if report_root is not None else locationpred_artifact_root.parent / report_name
    plots_dir = report_root / "plots"
    report_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    offerpred_manifest = load_json(offerpred_artifact_root / "run_manifest.json")
    discardpred_manifest = load_json(discardpred_artifact_root / "run_manifest.json")
    locationpred_manifest = load_json(locationpred_artifact_root / "run_manifest.json")

    offerpred_eval = _read_optional_parquet(offerpred_artifact_root / "offerpred_eval_sample.parquet")
    offerpred_yearly = _read_optional_parquet(offerpred_artifact_root / "offerpred_yearly_metrics.parquet")
    offerpred_importance = _read_optional_parquet(offerpred_artifact_root / "offerpred_feature_importance.parquet")
    discardpred_runs = _read_optional_parquet(discardpred_artifact_root / "discardpred_scored_runs.parquet")
    locationpred_validation = _read_optional_parquet(locationpred_artifact_root / "validation_run_predictions.parquet")
    locationpred_test = _read_optional_parquet(locationpred_artifact_root / "test_run_predictions.parquet")

    summary_rows: list[dict[str, Any]] = []
    offerpred_metrics = offerpred_manifest.get("offerpred_metrics", {})
    for split in ["validation", "test"]:
        for metric_name, metric_value in offerpred_metrics.get(split, {}).items():
            _append_metric_row(summary_rows, OFFERPRED_NAME, split, metric_name, metric_value)

    discardpred_metrics = discardpred_manifest.get("discardpred_metrics", {})
    for split in ["validation", "test"]:
        for metric_name, metric_value in discardpred_metrics.get(split, {}).items():
            _append_metric_row(summary_rows, DISCARDPRED_NAME, split, metric_name, metric_value)

    locationpred_validation_metrics = locationpred_manifest.get("validation_metrics", {}).get("final_pipeline", {})
    for metric_name, metric_value in locationpred_validation_metrics.items():
        _append_metric_row(summary_rows, LOCATIONPRED_NAME, "validation", metric_name, metric_value)
    for metric_name, metric_value in locationpred_manifest.get("test_localizer_metrics", {}).items():
        _append_metric_row(summary_rows, LOCATIONPRED_NAME, "test", metric_name, metric_value)

    scorecard_frame = pd.DataFrame(summary_rows)
    plot_report_scorecard(scorecard_frame, plots_dir / "scorecard.png")

    if not offerpred_eval.empty:
        plot_offerpred_diagnostics(offerpred_eval, plots_dir / "offerpred_diagnostics.png")
        plot_offerpred_topk(offerpred_eval, plots_dir / "offerpred_topk.png")
    if not offerpred_yearly.empty:
        plot_offerpred_yearly_metrics(offerpred_yearly, plots_dir / "offerpred_yearly_metrics.png")
    if not offerpred_importance.empty:
        plot_offerpred_feature_importance(offerpred_importance, plots_dir / "offerpred_feature_importance.png")

    if not discardpred_runs.empty:
        validation_discardpred = discardpred_runs.loc[discardpred_runs["split"] == "validation"].copy()
        test_discardpred = discardpred_runs.loc[discardpred_runs["split"] == "test"].copy()
        if not validation_discardpred.empty:
            _plot_discardpred_confusion(
                validation_discardpred,
                discard_threshold=0.5,
                title="DiscardPred Confusion (Validation)",
                output_path=plots_dir / "discardpred_route_confusion_validation.png",
            )
            plot_discardpred_score_mass(
                validation_discardpred,
                plots_dir / "discardpred_score_mass_validation.png",
            )
        if not test_discardpred.empty:
            _plot_discardpred_confusion(
                test_discardpred,
                discard_threshold=0.5,
                title="DiscardPred Confusion (Test)",
                output_path=plots_dir / "discardpred_route_confusion_test.png",
            )

    if not locationpred_validation.empty or not locationpred_test.empty:
        plot_locationpred_error_analysis(
            locationpred_validation,
            locationpred_test,
            plots_dir / "locationpred_error_analysis.png",
        )

    validation_summary = {
        "offerpred": offerpred_metrics.get("validation", {}),
        "discardpred": discardpred_metrics.get("validation", {}),
        "locationpred": locationpred_manifest.get("validation_metrics", {}).get("final_pipeline", {}),
        "locationpred_baselines": locationpred_manifest.get("validation_metrics", {}),
        "locationpred_test": locationpred_manifest.get("test_localizer_metrics", {}),
        "decision_shares_validation": _locationpred_decision_shares(locationpred_validation),
        "decision_shares_test": _locationpred_decision_shares(locationpred_test),
    }

    markdown_lines = [
        "# OfferPred / DiscardPred / LocationPred Report",
        "",
        "## Source Artifacts",
        f"- {OFFERPRED_NAME}: `{offerpred_artifact_root}`",
        f"- {DISCARDPRED_NAME}: `{discardpred_artifact_root}`",
        f"- {LOCATIONPRED_NAME}: `{locationpred_artifact_root}`",
        "",
        "## Key Metrics",
        scorecard_frame.sort_values(["model", "split", "metric"]).to_string(index=False) if not scorecard_frame.empty else "No metrics available.",
        "",
        "## Validation Summary",
        json.dumps(validation_summary, indent=2, sort_keys=True),
    ]
    (report_root / "summary.md").write_text("\n".join(markdown_lines))

    report_manifest = {
        "source_artifacts": {
            "offerpred": str(offerpred_artifact_root),
            "discardpred": str(discardpred_artifact_root),
            "locationpred": str(locationpred_artifact_root),
        },
        "summary_rows": scorecard_frame.to_dict(orient="records"),
        "validation_summary": validation_summary,
        "plot_files": sorted(str(path.relative_to(report_root)) for path in plots_dir.glob("*.png")),
    }
    dump_json(report_root / "report_manifest.json", report_manifest)
    return ConsolidatedReportArtifacts(artifact_root=report_root, manifest_path=report_root / "report_manifest.json")
