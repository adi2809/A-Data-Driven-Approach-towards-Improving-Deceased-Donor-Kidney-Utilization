#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kidney_utilization.train import build_consolidated_report


DEFAULT_OFFERPRED_ARTIFACT_ROOT = (
    REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/offerpred"
)
DEFAULT_DISCARDPRED_ARTIFACT_ROOT = (
    REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/discardpred"
)
DEFAULT_LOCATIONPRED_ARTIFACT_ROOT = (
    REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/locationpred"
)
DEFAULT_REPORT_ROOT = REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated OfferPred / DiscardPred / LocationPred report.")
    parser.add_argument("--offerpred-artifact-root", type=Path, default=DEFAULT_OFFERPRED_ARTIFACT_ROOT)
    parser.add_argument("--discardpred-artifact-root", type=Path, default=DEFAULT_DISCARDPRED_ARTIFACT_ROOT)
    parser.add_argument("--locationpred-artifact-root", type=Path, default=DEFAULT_LOCATIONPRED_ARTIFACT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--report-name", default="submission_report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_consolidated_report(
        offerpred_artifact_root=args.offerpred_artifact_root,
        discardpred_artifact_root=args.discardpred_artifact_root,
        locationpred_artifact_root=args.locationpred_artifact_root,
        report_root=args.report_root,
        report_name=args.report_name,
    )
    print(f"[done] artifact_root={artifacts.artifact_root}")
    print(f"[done] manifest={artifacts.manifest_path}")


if __name__ == "__main__":
    main()
