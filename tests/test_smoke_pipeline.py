from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from kidney_utilization.build import build_benchmark
from kidney_utilization.combination import combine_model_predictions
from kidney_utilization.config import BenchmarkConfig
from kidney_utilization.feature_specs import SOURCE_REQUIRED_COLUMNS
from kidney_utilization.train import (
    build_consolidated_report,
    train_benchmark,
    train_offerpred_benchmark,
    train_discardpred_benchmark,
    train_locationpred_benchmark,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STRING_COLUMNS = {
    "ptr_offer_acpt",
    "can_gender",
    "can_abo",
    "can_race_srtr",
    "can_on_dial",
    "can_prev_ki_tx_functn",
    "don_gender",
    "don_abo",
    "don_race_srtr",
    "kdpi_bin",
    "last_yn_offer_kdpi_bin",
}


def _base_offer_row() -> dict[str, object]:
    row: dict[str, object] = {}
    for column in SOURCE_REQUIRED_COLUMNS:
        row[column] = "UNK" if column in STRING_COLUMNS else 0.0
    row["can_listing_ctr_cd"] = "CTR1"
    row["ptr_tot_score"] = 0.0
    row["match_submit_dt"] = pd.Timestamp("2022-01-01 12:00:00")
    row["match_year"] = 2022
    row["match_id"] = 0
    row["ptr_row_order"] = 0
    row["offer_rank"] = 0
    row["ptr_sequence_num"] = 0
    row["px_id"] = 0
    row["ptr_offer_acpt"] = "N"
    row["can_gender"] = "F"
    row["can_abo"] = "O"
    row["can_race_srtr"] = "W"
    row["can_on_dial"] = "Y"
    row["can_prev_ki_tx_functn"] = "N"
    row["don_gender"] = "M"
    row["don_abo"] = "O"
    row["don_race_srtr"] = "W"
    row["kdpi_bin"] = "20-40"
    row["last_yn_offer_kdpi_bin"] = "20-40"
    return row


def _offer_rows_for_match(match_id: int, year: int, kind: str) -> list[dict[str, object]]:
    first_y_rank = {
        "early": 1,
        "mid": 3,
        "late": 8,
        "audit_missing": 2,
        "audit_nonkidney": 2,
    }.get(kind)

    rows: list[dict[str, object]] = []
    for offer_rank in range(1, 11):
        row = _base_offer_row()
        row["match_id"] = match_id
        row["match_year"] = year
        row["match_submit_dt"] = pd.Timestamp(f"{year}-01-{(match_id % 20) + 1:02d} 12:00:00")
        row["ptr_row_order"] = match_id * 100 + offer_rank
        row["offer_rank"] = offer_rank
        row["ptr_sequence_num"] = offer_rank
        row["px_id"] = match_id * 1000 + offer_rank
        row["can_listing_ctr_cd"] = f"CTR{(match_id % 3) + 1}"

        if kind == "early":
            row["don_age"] = 25
            row["kdpi"] = 20
            row["opo_hist_any_placed_frac_365d"] = 0.9
            row["opo_hist_mean_first_accept_declines_365d"] = 1.0
        elif kind == "mid":
            row["don_age"] = 40
            row["kdpi"] = 45
            row["opo_hist_any_placed_frac_365d"] = 0.7
            row["opo_hist_mean_first_accept_declines_365d"] = 3.0
        elif kind == "late":
            row["don_age"] = 60
            row["kdpi"] = 75
            row["opo_hist_any_placed_frac_365d"] = 0.5
            row["opo_hist_mean_first_accept_declines_365d"] = 7.0
        elif kind == "none":
            row["don_age"] = 70
            row["kdpi"] = 90
            row["opo_hist_any_placed_frac_365d"] = 0.1
            row["opo_hist_mean_first_accept_declines_365d"] = 10.0
        elif kind == "censored":
            row["don_age"] = 55
            row["kdpi"] = 65
            row["dcd_ind"] = 1
            row["opo_hist_any_placed_frac_365d"] = 0.3
            row["opo_hist_mean_first_accept_declines_365d"] = 8.0
        else:
            row["don_age"] = 35
            row["kdpi"] = 35
            row["opo_hist_any_placed_frac_365d"] = 0.6
            row["opo_hist_mean_first_accept_declines_365d"] = 2.0

        row["donor_opo_success_rate_historical"] = row["opo_hist_any_placed_frac_365d"]
        row["tx_center_count_250nm"] = 6
        row["dcd_ind"] = int(kind == "censored")
        row["high_kdpi_flg"] = int(row["kdpi"] >= 85)

        is_positive_row = first_y_rank is not None and offer_rank == first_y_rank and kind in {
            "early",
            "mid",
            "late",
            "audit_missing",
            "audit_nonkidney",
        }
        if offer_rank == 9:
            row["ptr_offer_acpt"] = "B"
        elif offer_rank == 10:
            row["ptr_offer_acpt"] = "Z"
        else:
            row["ptr_offer_acpt"] = "Y" if is_positive_row else "N"
        row["mm_total"] = 0 if is_positive_row else 5
        row["distance_nm"] = 15 if is_positive_row else 400
        row["canhx_cpra"] = 98 if is_positive_row else 45
        row["cand_decline_count_365d"] = 1 if is_positive_row else 8
        row["center_positive_response_rate_365d"] = 0.85 if is_positive_row else 0.15
        row["center_mean_accepted_normalized_sequence_365d"] = 0.12 if is_positive_row else 0.75
        row["time_since_last_offer_days"] = 2 if is_positive_row else 30
        row["match_day_of_week"] = 2
        row["match_week_of_month"] = 1
        row["match_month_of_year"] = 1
        row["match_hour_of_day"] = 12
        row["mm_a"] = 0 if is_positive_row else 2
        row["mm_b"] = 0 if is_positive_row else 2
        row["mm_dr"] = 0 if is_positive_row else 1
        row["long_distance_flg"] = int(row["distance_nm"] > 250)
        row["hcv_positive_flg"] = 0
        row["hbc_positive_flg"] = 0
        row["cand_prior_tx_count_30d"] = 0
        row["cand_prior_tx_count_90d"] = 0
        row["cand_prior_tx_count_150d"] = 0
        row["cand_prior_tx_count_365d"] = 0
        row["cand_decline_count_30d"] = 0 if is_positive_row else 2
        row["cand_decline_count_90d"] = 0 if is_positive_row else 4
        row["cand_decline_count_150d"] = 0 if is_positive_row else 6
        row["cand_declined_kdpi_avg_30d"] = 25 if is_positive_row else 70
        row["cand_declined_kdpi_stddev_30d"] = 2 if is_positive_row else 5
        row["cand_declined_don_creat_avg_30d"] = 1.0 if is_positive_row else 2.2
        row["cand_declined_don_creat_stddev_30d"] = 0.1 if is_positive_row else 0.2
        row["cand_declined_mm_total_avg_30d"] = 1 if is_positive_row else 5
        row["cand_declined_mm_total_stddev_30d"] = 0.1 if is_positive_row else 0.5
        row["cand_declined_don_age_avg_30d"] = 30 if is_positive_row else 65
        row["cand_declined_don_age_stddev_30d"] = 1 if is_positive_row else 4
        row["cand_declined_dcd_frac_30d"] = 0 if is_positive_row else 0.5
        row["cand_declined_hcv_frac_30d"] = 0
        row["center_yn_offer_count_30d"] = 40
        row["center_positive_response_rate_30d"] = 0.8 if is_positive_row else 0.2
        row["center_rate_same_dcd_30d"] = 0.5
        row["center_rate_same_high_kdpi_30d"] = 0.2
        row["center_rate_same_hcv_pos_30d"] = 0.1
        row["center_rate_same_long_distance_30d"] = 0.1
        row["center_rate_same_mm_bucket_30d"] = 0.9 if is_positive_row else 0.2
        row["center_mean_accepted_sequence_30d"] = 2
        row["center_mean_accepted_normalized_sequence_30d"] = 0.2
        row["center_late_placement_rate_30d"] = 0.1
        row["center_yn_offer_count_365d"] = 120
        row["center_mean_accepted_sequence_365d"] = 3
        row["center_late_placement_rate_365d"] = 0.2
        row["don_creat"] = 1.0 if is_positive_row else 2.0
        row["don_bun"] = 15 if is_positive_row else 35
        row["don_final_serum_creat"] = row["don_creat"]
        row["don_peak_serum_creat"] = row["don_creat"] + 0.2
        row["don_hist_diab"] = 0
        row["don_hist_cancer"] = 0
        row["don_htn"] = 0
        row["don_high_creat"] = int(row["don_creat"] > 1.8)
        row["don_max_creat"] = row["don_creat"] + 0.3
        row["don_warm_isch_tm_mins"] = 20
        row["kdri_rao"] = 1.0
        row["kdri_med"] = 1.0
        row["can_age_at_listing"] = 40
        row["can_dgn"] = 101
        row["can_diab"] = 0
        row["can_diab_ty"] = 0
        row["can_prev_ki"] = 0
        row["can_prev_tx"] = 0
        row["can_max_warm_tm"] = 30
        row["can_most_recent_hgt_cm"] = 170
        row["can_most_recent_wgt_kg"] = 70
        row["can_current_age_years"] = 42
        row["can_is_adult"] = 1
        row["don_hgt_cm"] = 175
        row["don_wgt_kg"] = 75
        row["don_cad_don_cod"] = 10
        rows.append(row)
    return rows


def _disposition_rows_for_match(match_id: int, kind: str) -> list[dict[str, object]]:
    if kind in {"early", "mid", "late"}:
        first_rank = {"early": 1, "mid": 3, "late": 8}[kind]
        return [
            {
                "match_id": match_id,
                "px_id": match_id * 1000 + first_rank,
                "don_org": "LKI",
                "don_disposition": 6.0,
                "donor_id": float(match_id * 10 + 1),
            }
        ]
    if kind == "censored":
        return [
            {
                "match_id": match_id,
                "px_id": match_id * 1000 + 999,
                "don_org": "EKI",
                "don_disposition": 6.0,
                "donor_id": float(match_id * 10 + 1),
            }
        ]
    if kind == "audit_nonkidney":
        return [
            {
                "match_id": match_id,
                "px_id": match_id * 1000 + 2,
                "don_org": "LI",
                "don_disposition": 6.0,
                "donor_id": float(match_id * 10 + 1),
            }
        ]
    return []


def _write_parquet(dataframe: pd.DataFrame, output_path: Path, temp_name: str) -> None:
    con = duckdb.connect()
    con.register(temp_name, dataframe)
    escaped_output_path = str(output_path).replace("'", "''")
    con.execute(f"COPY {temp_name} TO '{escaped_output_path}' (FORMAT PARQUET)")
    con.unregister(temp_name)
    con.close()


def _build_synthetic_artifacts(temp_path: Path) -> tuple[Path, Path, Path]:
    feature_dir = temp_path / "match_offer_features"
    disposition_dir = temp_path / "donor_disposition"
    feature_dir.mkdir(parents=True, exist_ok=True)
    disposition_dir.mkdir(parents=True, exist_ok=True)

    kinds = ["early", "mid", "late", "none", "censored", "audit_missing", "audit_nonkidney"]
    all_offer_rows: list[dict[str, object]] = []
    all_disposition_rows: list[dict[str, object]] = []
    match_id = 100
    for year in [2022, 2023, 2024]:
        for kind in kinds:
            all_offer_rows.extend(_offer_rows_for_match(match_id, year, kind))
            all_disposition_rows.extend(_disposition_rows_for_match(match_id, kind))
            match_id += 1

    offer_frame = pd.DataFrame(all_offer_rows)
    disposition_frame = pd.DataFrame(all_disposition_rows)
    feature_path = feature_dir / "part-00001.parquet"
    disposition_path = disposition_dir / "part-00001.parquet"
    _write_parquet(offer_frame, feature_path, "offer_frame")
    _write_parquet(disposition_frame, disposition_path, "disposition_frame")

    benchmark_db = temp_path / "kidney_utilization_benchmark.duckdb"
    benchmark_manifest = temp_path / "kidney_utilization_benchmark_manifest.json"
    artifact_root = temp_path / "artifacts"

    config = BenchmarkConfig(
        history_start=pd.Timestamp("2022-01-01").to_pydatetime(),
        supervised_start=pd.Timestamp("2022-01-01").to_pydatetime(),
        supervised_end=pd.Timestamp("2024-12-31T23:59:59").to_pydatetime(),
        benchmark_db_path=benchmark_db,
        benchmark_manifest_path=benchmark_manifest,
        feature_parquet_glob=str(feature_path),
        donor_disposition_glob=str(disposition_path),
        artifact_root=artifact_root,
        offerpred_chunk_rows=25,
        evaluation_sample_rows_per_group=10,
        offerpred_negative_to_positive_ratio=5,
        offerpred_catboost_iterations=30,
        offerpred_catboost_depth=4,
        offerpred_catboost_learning_rate=0.1,
        offerpred_catboost_early_stopping_rounds=5,
        locationpred_catboost_iterations=30,
        locationpred_catboost_depth=4,
        locationpred_catboost_learning_rate=0.1,
        locationpred_catboost_early_stopping_rounds=5,
    )

    build_benchmark(config=config, threads=1, overwrite=True)
    offerpred_artifacts = train_offerpred_benchmark(
        config=None,
        run_name="offerpred",
        thread_count=1,
        benchmark_db=benchmark_db,
        benchmark_manifest_path=benchmark_manifest,
        artifact_root=artifact_root,
    )
    discardpred_artifacts = train_discardpred_benchmark(
        config=None,
        run_name="discardpred",
        thread_count=1,
        benchmark_db=benchmark_db,
        benchmark_manifest_path=benchmark_manifest,
        artifact_root=artifact_root,
        offerpred_artifact_root=offerpred_artifacts.artifact_root,
    )
    locationpred_artifacts = train_locationpred_benchmark(
        config=None,
        run_name="locationpred",
        thread_count=1,
        benchmark_db=benchmark_db,
        benchmark_manifest_path=benchmark_manifest,
        artifact_root=artifact_root,
        offerpred_artifact_root=offerpred_artifacts.artifact_root,
        discardpred_artifact_root=discardpred_artifacts.artifact_root,
    )
    return offerpred_artifacts.artifact_root, discardpred_artifacts.artifact_root, locationpred_artifacts.artifact_root


class SmokePipelineTest(unittest.TestCase):
    def test_synthetic_pipeline_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            feature_dir = temp_path / "match_offer_features"
            disposition_dir = temp_path / "donor_disposition"
            feature_dir.mkdir(parents=True, exist_ok=True)
            disposition_dir.mkdir(parents=True, exist_ok=True)

            kinds = ["early", "mid", "late", "none", "censored", "audit_missing", "audit_nonkidney"]
            all_offer_rows: list[dict[str, object]] = []
            all_disposition_rows: list[dict[str, object]] = []
            match_id = 100
            for year in [2022, 2023, 2024]:
                for kind in kinds:
                    all_offer_rows.extend(_offer_rows_for_match(match_id, year, kind))
                    all_disposition_rows.extend(_disposition_rows_for_match(match_id, kind))
                    match_id += 1

            offer_frame = pd.DataFrame(all_offer_rows)
            disposition_frame = pd.DataFrame(all_disposition_rows)
            feature_path = feature_dir / "part-00001.parquet"
            disposition_path = disposition_dir / "part-00001.parquet"
            _write_parquet(offer_frame, feature_path, "offer_frame")
            _write_parquet(disposition_frame, disposition_path, "disposition_frame")

            benchmark_db = temp_path / "kidney_utilization_benchmark.duckdb"
            benchmark_manifest = temp_path / "kidney_utilization_benchmark_manifest.json"
            artifact_root = temp_path / "artifacts"

            config = BenchmarkConfig(
                history_start=pd.Timestamp("2022-01-01").to_pydatetime(),
                supervised_start=pd.Timestamp("2022-01-01").to_pydatetime(),
                supervised_end=pd.Timestamp("2024-12-31T23:59:59").to_pydatetime(),
                benchmark_db_path=benchmark_db,
                benchmark_manifest_path=benchmark_manifest,
                feature_parquet_glob=str(feature_path),
                donor_disposition_glob=str(disposition_path),
                artifact_root=artifact_root,
                offerpred_chunk_rows=25,
                evaluation_sample_rows_per_group=10,
                offerpred_negative_to_positive_ratio=5,
                offerpred_catboost_iterations=30,
                offerpred_catboost_depth=4,
                offerpred_catboost_learning_rate=0.1,
                offerpred_catboost_early_stopping_rounds=5,
                locationpred_catboost_iterations=30,
                locationpred_catboost_depth=4,
                locationpred_catboost_learning_rate=0.1,
                locationpred_catboost_early_stopping_rounds=5,
            )

            build_artifacts = build_benchmark(config=config, threads=1, overwrite=True)
            self.assertTrue(build_artifacts.benchmark_db.exists())
            self.assertTrue(build_artifacts.manifest_path.exists())

            benchmark_build_manifest = json.loads(build_artifacts.manifest_path.read_text())
            self.assertEqual(benchmark_build_manifest["table_counts"]["match_labels"], 21)
            self.assertEqual(benchmark_build_manifest["run_state_counts"]["audit_orphan_y"], 6)
            self.assertEqual(benchmark_build_manifest["table_counts"]["audit_runs"], 6)

            con = duckdb.connect(str(benchmark_db), read_only=True)
            run_states = dict(con.execute("SELECT run_state, COUNT(*) FROM benchmark.match_labels GROUP BY 1").fetchall())
            self.assertEqual(run_states["localizable_observed_y"], 9)
            self.assertEqual(run_states["none"], 3)
            self.assertEqual(run_states["censored_positive"], 3)
            self.assertEqual(run_states["audit_orphan_y"], 6)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM benchmark.offerpred_rows").fetchone()[0], 96)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM benchmark.offerpred_scoring_rows").fetchone()[0], 150)
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM benchmark.offerpred_scoring_rows WHERE ptr_offer_acpt IN ('B', 'Z')"
                ).fetchone()[0],
                30,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM benchmark.offerpred_scoring_rows WHERE use_for_offerpred_loss = 1"
                ).fetchone()[0],
                96,
            )
            self.assertEqual(
                con.execute(
                    "SELECT COUNT(*) FROM benchmark.offerpred_scoring_rows WHERE ptr_offer_acpt IN ('B', 'Z') AND use_for_offerpred_loss = 1"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                con.execute("SELECT MIN(run_len), MAX(run_len) FROM benchmark.discardpred_runs").fetchone(),
                (10, 10),
            )
            self.assertEqual(con.execute("SELECT COUNT(*) FROM benchmark.discardpred_runs").fetchone()[0], 15)
            self.assertEqual(
                con.execute(
                    "SELECT SUM(discard_target), COUNT(*) - SUM(discard_target) FROM benchmark.discardpred_runs"
                ).fetchone(),
                (3, 12),
            )
            self.assertEqual(con.execute("SELECT COUNT(*) FROM benchmark.locationpred_riskset").fetchone()[0], 36)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM benchmark.locationpred_scoring_rows").fetchone()[0], 150)
            test_scoring_coverage = con.execute(
                """
                SELECT
                    s.match_id,
                    COUNT(*) AS scored_rows,
                    MAX(l.run_len) AS run_len,
                    COUNT(*) FILTER (WHERE s.ptr_offer_acpt IN ('B', 'Z')) AS bz_rows
                FROM benchmark.offerpred_scoring_rows AS s
                JOIN benchmark.match_labels AS l USING (match_id)
                WHERE s.split = 'test'
                GROUP BY 1
                ORDER BY 1
                """
            ).fetchall()
            self.assertTrue(test_scoring_coverage)
            self.assertTrue(all(scored_rows == run_len for _, scored_rows, run_len, _ in test_scoring_coverage))
            self.assertTrue(all(bz_rows == 2 for _, _, _, bz_rows in test_scoring_coverage))
            con.close()

            artifacts = train_benchmark(
                config=None,
                run_name="synthetic_smoke",
                max_trials=1,
                thread_count=1,
                benchmark_db=benchmark_db,
                benchmark_manifest_path=benchmark_manifest,
                artifact_root=artifact_root,
            )

            self.assertTrue((artifacts.artifact_root / "run_manifest.json").exists())
            self.assertTrue((artifacts.artifact_root / "plots" / "validation_sweep.png").exists())
            self.assertTrue((artifacts.artifact_root / "plots" / "data_qa.png").exists())
            self.assertTrue((artifacts.artifact_root / "plots" / "offerpred_diagnostics.png").exists())
            self.assertTrue((artifacts.artifact_root / "plots" / "discardpred_route_confusion.png").exists())
            self.assertTrue((artifacts.artifact_root / "plots" / "locationpred_localizer.png").exists())
            self.assertTrue((artifacts.artifact_root / "test_row_predictions.parquet").exists())

            run_manifest = json.loads((artifacts.artifact_root / "run_manifest.json").read_text())
            self.assertEqual(run_manifest["config_source"], "benchmark_manifest")
            self.assertEqual(run_manifest["config"]["discard_threshold"], 0.5)
            self.assertEqual(run_manifest["backends"]["offerpred"], "catboost_native")
            self.assertEqual(run_manifest["backends"]["locationpred"], "segment_hazard_catboost")
            self.assertEqual(run_manifest["training_modes"]["locationpred"], "segment_hazard")
            self.assertIn("plots/validation_sweep.png".split("/", 1)[1], [Path(p).name for p in run_manifest["plot_files"]])
            self.assertIsNotNone(run_manifest["best_validation_metrics"])

            discardpred_artifacts = train_discardpred_benchmark(
                config=None,
                run_name="synthetic_discardpred_smoke",
                thread_count=1,
                benchmark_db=benchmark_db,
                benchmark_manifest_path=benchmark_manifest,
                artifact_root=artifact_root,
            )
            self.assertTrue((discardpred_artifacts.artifact_root / "run_manifest.json").exists())
            self.assertTrue((discardpred_artifacts.artifact_root / "plots" / "discardpred_route_confusion.png").exists())
            self.assertTrue((discardpred_artifacts.artifact_root / "discardpred_run_predictions.parquet").exists())

            discardpred_manifest = json.loads((discardpred_artifacts.artifact_root / "run_manifest.json").read_text())
            self.assertEqual(discardpred_manifest["backends"]["offerpred"], "catboost_native")
            self.assertEqual(discardpred_manifest["config"]["discard_threshold"], 0.5)
            self.assertIn("validation", discardpred_manifest["discardpred_metrics"])
            self.assertIn("test", discardpred_manifest["discardpred_metrics"])
            self.assertIn("roc_auc", discardpred_manifest["discardpred_metrics"]["test"])
            self.assertIn("average_precision", discardpred_manifest["discardpred_metrics"]["test"])
            self.assertIn("brier_score", discardpred_manifest["discardpred_metrics"]["test"])

            locationpred_artifacts = train_locationpred_benchmark(
                config=None,
                run_name="synthetic_locationpred_smoke",
                thread_count=1,
                benchmark_db=benchmark_db,
                benchmark_manifest_path=benchmark_manifest,
                artifact_root=artifact_root,
                offerpred_artifact_root=artifacts.artifact_root,
                discardpred_artifact_root=discardpred_artifacts.artifact_root,
            )
            self.assertTrue((locationpred_artifacts.artifact_root / "run_manifest.json").exists())
            self.assertTrue((locationpred_artifacts.artifact_root / "plots" / "locationpred_localizer.png").exists())
            self.assertTrue((locationpred_artifacts.artifact_root / "validation_run_predictions.parquet").exists())

            locationpred_manifest = json.loads((locationpred_artifacts.artifact_root / "run_manifest.json").read_text())
            self.assertEqual(locationpred_manifest["backends"]["offerpred"], "catboost_native")
            self.assertEqual(locationpred_manifest["backends"]["locationpred"], "segment_hazard_catboost")
            self.assertEqual(locationpred_manifest["training_modes"]["offerpred"], "artifact_reuse")
            self.assertEqual(locationpred_manifest["training_modes"]["locationpred"], "segment_hazard")
            self.assertIn("final_pipeline", locationpred_manifest["validation_metrics"])

            validation_predictions = duckdb.sql(
                f"SELECT * FROM read_parquet('{(locationpred_artifacts.artifact_root / 'validation_row_predictions.parquet').as_posix()}')"
            ).df()
            self.assertIn("locationpred_hazard_probability", validation_predictions.columns)
            self.assertIn("discard_probability", validation_predictions.columns)
            self.assertEqual(set(validation_predictions["decision"]).difference({"discard", "localize"}), set())

            full_run_counts = validation_predictions.groupby("match_id").agg(
                scored_rows=("offer_rank", "size"),
                run_len=("run_len", "first"),
            )
            self.assertTrue((full_run_counts["scored_rows"] == full_run_counts["run_len"]).all())

            localize_sums = (
                validation_predictions.loc[validation_predictions["decision"] == "localize"]
                .groupby("match_id", as_index=False)
                .agg(final_mass=("final_row_probability", "sum"))
            )
            self.assertTrue(((localize_sums["final_mass"] - 1.0).abs() < 1e-9).all())
            discard_sums = (
                validation_predictions.loc[validation_predictions["decision"] == "discard"]
                .groupby("match_id", as_index=False)
                .agg(final_mass=("final_row_probability", "sum"))
            )
            self.assertTrue((discard_sums["final_mass"].abs() < 1e-12).all())

            validation_runs = pd.read_parquet(locationpred_artifacts.artifact_root / "validation_run_predictions.parquet")
            expected_decisions = validation_runs["discard_probability"].ge(0.5).map(
                {True: "discard", False: "localize"}
            )
            self.assertTrue((validation_runs["decision"] == expected_decisions).all())
            self.assertTrue(validation_runs.loc[validation_runs["decision"] == "discard", "predicted_rank"].isna().all())
            self.assertTrue(validation_runs.loc[validation_runs["decision"] == "localize", "predicted_rank"].notna().all())

    def test_full_run_combination_keeps_all_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "offer_rank": [1, 2, 3, 4],
                "ptr_offer_acpt": ["N", "B", "Z", "N"],
                "offerpred_score": [0.1, 0.3, 0.2, 0.4],
                "locationpred_segment_id": [1, 1, 2, 2],
                "locationpred_segment_probability": [0.25, 0.25, 0.75, 0.75],
            }
        )

        placed = combine_model_predictions(frame, discard_probability=0.49)
        self.assertEqual(placed.decision, "localize")
        self.assertEqual(len(placed.row_probabilities), len(frame))
        self.assertEqual(placed.row_probabilities["ptr_offer_acpt"].tolist(), ["N", "B", "Z", "N"])
        self.assertAlmostEqual(
            float(placed.row_probabilities["final_first_acceptance_probability"].sum()),
            1.0,
        )
        self.assertEqual(placed.predicted_first_acceptance_rank, 4)

        discarded = combine_model_predictions(frame, discard_probability=0.5)
        self.assertEqual(discarded.decision, "discard")
        self.assertEqual(len(discarded.row_probabilities), len(frame))
        self.assertAlmostEqual(
            float(discarded.row_probabilities["final_first_acceptance_probability"].sum()),
            0.0,
        )
        self.assertIsNone(discarded.predicted_first_acceptance_rank)

    def test_consolidated_report_builds_from_best_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            offerpred_root, discardpred_root, locationpred_root = _build_synthetic_artifacts(Path(tempdir) / "synthetic")
            report_root = Path(tempdir) / "consolidated_report"
            artifacts = build_consolidated_report(
                offerpred_artifact_root=offerpred_root,
                discardpred_artifact_root=discardpred_root,
                locationpred_artifact_root=locationpred_root,
                report_root=report_root,
            )

            self.assertTrue((artifacts.artifact_root / "report_manifest.json").exists())
            self.assertTrue((artifacts.artifact_root / "summary.md").exists())
            self.assertTrue((artifacts.artifact_root / "plots" / "scorecard.png").exists())
            self.assertTrue((artifacts.artifact_root / "plots" / "locationpred_error_analysis.png").exists())

            manifest = json.loads((artifacts.artifact_root / "report_manifest.json").read_text())
            self.assertIn("offerpred", manifest["source_artifacts"])
            self.assertIn("discardpred", manifest["source_artifacts"])
            self.assertIn("locationpred", manifest["source_artifacts"])

    def test_submission_audits_build_from_synthetic_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            offerpred_root, discardpred_root, locationpred_root = _build_synthetic_artifacts(Path(tempdir) / "synthetic")
            audit_root = Path(tempdir) / "audits"

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "audit_features.py"),
                    "--output-dir",
                    str(audit_root),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "audit_hyperparams.py"),
                    "--output-dir",
                    str(audit_root),
                    "--offerpred-manifest",
                    str(offerpred_root / "run_manifest.json"),
                    "--discardpred-manifest",
                    str(discardpred_root / "run_manifest.json"),
                    "--locationpred-manifest",
                    str(locationpred_root / "run_manifest.json"),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            feature_audit = json.loads((audit_root / "feature_audit.json").read_text())
            hyperparameter_audit = json.loads((audit_root / "hyperparameter_audit.json").read_text())

            self.assertIn("OfferPred", feature_audit["models"])
            self.assertIn("OfferPred__DiscardPred", feature_audit["pairwise_overlap"])
            self.assertIn("OfferPred", hyperparameter_audit["artifacts"])
            self.assertEqual(hyperparameter_audit["artifacts"]["OfferPred"]["model_name"], "OfferPred")
