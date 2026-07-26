#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kidney_utilization.build import build_benchmark
from kidney_utilization.config import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the leakage-safe kidney utilization benchmark warehouse.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-locationpred-exports", action="store_true")
    parser.add_argument("--history-start", type=datetime.fromisoformat, default=datetime.fromisoformat("2015-01-01T00:00:00"))
    parser.add_argument("--supervised-start", type=datetime.fromisoformat, default=datetime.fromisoformat("2021-05-01T00:00:00"))
    parser.add_argument("--validation-start", type=datetime.fromisoformat, default=datetime.fromisoformat("2023-01-01T00:00:00"))
    parser.add_argument("--test-start", type=datetime.fromisoformat, default=datetime.fromisoformat("2024-01-01T00:00:00"))
    parser.add_argument("--supervised-end", type=datetime.fromisoformat, default=datetime.fromisoformat("2024-12-31T23:59:59"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        history_start=args.history_start,
        supervised_start=args.supervised_start,
        validation_start=args.validation_start,
        test_start=args.test_start,
        supervised_end=args.supervised_end,
    )
    artifacts = build_benchmark(
        config=config,
        threads=args.threads,
        overwrite=args.overwrite,
        skip_locationpred_exports=args.skip_locationpred_exports,
    )
    print(f"[done] benchmark_db={artifacts.benchmark_db}")
    print(f"[done] manifest={artifacts.manifest_path}")


if __name__ == "__main__":
    main()
