#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kidney_utilization.config import BenchmarkConfig
from kidney_utilization.train import train_locationpred_benchmark


DEFAULT_BENCHMARK_DB = REPO_ROOT / "warehouse/match_runs/kidney_utilization_benchmark_same_match_history_fixed.duckdb"
DEFAULT_BENCHMARK_MANIFEST = REPO_ROOT / "warehouse/match_runs/kidney_utilization_benchmark_same_match_history_fixed_manifest.json"
DEFAULT_OFFERPRED_SCORED_PARTS_DIR = (
    REPO_ROOT
    / "warehouse/match_runs/artifacts/kidney_utilization/offerpred/intermediate/offerpred_scored_rows"
)
DEFAULT_DISCARDPRED_PREDICTIONS_PATH = (
    REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/discardpred/discardpred_scored_runs.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LocationPred from the corrected benchmark plus OfferPred and DiscardPred outputs.")
    parser.add_argument("--run-name", default="locationpred")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--match-limit-per-split", type=int)
    parser.add_argument("--benchmark-db", type=Path, default=DEFAULT_BENCHMARK_DB)
    parser.add_argument("--benchmark-manifest", type=Path, default=DEFAULT_BENCHMARK_MANIFEST)
    parser.add_argument("--offerpred-scored-parts-dir", type=Path, default=DEFAULT_OFFERPRED_SCORED_PARTS_DIR)
    parser.add_argument("--discardpred-predictions-path", type=Path, default=DEFAULT_DISCARDPRED_PREDICTIONS_PATH)
    parser.add_argument("--locationpred-catboost-iterations", type=int, default=600)
    parser.add_argument("--locationpred-catboost-depth", type=int, default=8)
    parser.add_argument("--locationpred-catboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--locationpred-catboost-early-stopping-rounds", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.benchmark_manifest.read_text())
    config = BenchmarkConfig.from_dict(manifest["config"])
    config = replace(
        config,
        benchmark_db_path=args.benchmark_db,
        benchmark_manifest_path=args.benchmark_manifest,
        locationpred_catboost_iterations=int(args.locationpred_catboost_iterations),
        locationpred_catboost_depth=int(args.locationpred_catboost_depth),
        locationpred_catboost_learning_rate=float(args.locationpred_catboost_learning_rate),
        locationpred_catboost_early_stopping_rounds=int(args.locationpred_catboost_early_stopping_rounds),
    )
    artifacts = train_locationpred_benchmark(
        config=config,
        run_name=args.run_name,
        thread_count=args.threads,
        match_limit_per_split=args.match_limit_per_split,
        benchmark_db=args.benchmark_db,
        benchmark_manifest_path=args.benchmark_manifest,
        offerpred_scored_parts_dir=args.offerpred_scored_parts_dir,
        discardpred_predictions_path=args.discardpred_predictions_path,
    )
    print(f"[done] artifact_root={artifacts.artifact_root}")
    print(f"[done] manifest={artifacts.manifest_path}")


if __name__ == "__main__":
    main()
