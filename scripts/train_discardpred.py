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
from kidney_utilization.train import train_discardpred_benchmark


DEFAULT_BENCHMARK_DB = REPO_ROOT / "warehouse/match_runs/kidney_utilization_benchmark_same_match_history_fixed.duckdb"
DEFAULT_BENCHMARK_MANIFEST = REPO_ROOT / "warehouse/match_runs/kidney_utilization_benchmark_same_match_history_fixed_manifest.json"
DEFAULT_OFFERPRED_ARTIFACT_ROOT = (
    REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/offerpred"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DiscardPred from the corrected benchmark and OfferPred outputs.")
    parser.add_argument("--run-name", default="discardpred")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--match-limit-per-split", type=int)
    parser.add_argument("--benchmark-db", type=Path, default=DEFAULT_BENCHMARK_DB)
    parser.add_argument("--benchmark-manifest", type=Path, default=DEFAULT_BENCHMARK_MANIFEST)
    parser.add_argument("--offerpred-artifact-root", type=Path, default=DEFAULT_OFFERPRED_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.benchmark_manifest.read_text())
    config = BenchmarkConfig.from_dict(manifest["config"])
    config = replace(
        config,
        benchmark_db_path=args.benchmark_db,
        benchmark_manifest_path=args.benchmark_manifest,
    )
    artifacts = train_discardpred_benchmark(
        config=config,
        run_name=args.run_name,
        thread_count=args.threads,
        match_limit_per_split=args.match_limit_per_split,
        benchmark_db=args.benchmark_db,
        benchmark_manifest_path=args.benchmark_manifest,
        offerpred_artifact_root=args.offerpred_artifact_root,
    )
    print(f"[done] artifact_root={artifacts.artifact_root}")
    print(f"[done] manifest={artifacts.manifest_path}")


if __name__ == "__main__":
    main()
