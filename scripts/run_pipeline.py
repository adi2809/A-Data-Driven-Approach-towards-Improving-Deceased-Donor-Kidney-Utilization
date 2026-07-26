#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deceased-donor kidney utilization pipeline from raw match-run and SAF files."
    )
    parser.add_argument("--match-source-dir", type=Path, required=True)
    parser.add_argument("--saf-source-dir", type=Path, required=True)
    parser.add_argument("--kdpi-do-file", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--years", type=int, nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_step(label: str, script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    print(f"[step] {label}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    year_args = [str(year) for year in args.years] if args.years else []
    overwrite_flag = ["--overwrite"] if args.overwrite else []

    run_step(
        "build_match_runs",
        SCRIPT_DIR / "build_match_runs.py",
        [
            "--source-dir",
            str(args.match_source_dir),
            "--threads",
            str(args.threads),
            *([] if not year_args else ["--years", *year_args]),
            *overwrite_flag,
        ],
    )
    run_step(
        "build_saf",
        SCRIPT_DIR / "build_saf.py",
        [
            "--source-dir",
            str(args.saf_source_dir),
            "--kdpi-do-file",
            str(args.kdpi_do_file),
            "--threads",
            str(args.threads),
            *overwrite_flag,
        ],
    )
    run_step(
        "link_match_saf",
        SCRIPT_DIR / "link_match_saf.py",
        [
            "--threads",
            str(args.threads),
        ],
    )
    run_step(
        "export_features",
        SCRIPT_DIR / "export_features.py",
        [
            "--threads",
            str(args.threads),
            *([] if not year_args else ["--years", *year_args]),
            *overwrite_flag,
        ],
    )
    run_step(
        "build_patch_maps",
        SCRIPT_DIR / "build_patch_maps.py",
        [
            "--threads",
            str(args.threads),
            *([] if not year_args else ["--years", *year_args]),
            *overwrite_flag,
        ],
    )
    run_step(
        "patch_features",
        SCRIPT_DIR / "patch_features.py",
        [
            "--threads",
            str(args.threads),
            *([] if not year_args else ["--years", *year_args]),
            *(["--overwrite-output"] if args.overwrite else []),
        ],
    )
    run_step(
        "build_dataset",
        SCRIPT_DIR / "build_dataset.py",
        [
            "--threads",
            str(args.threads),
            "--overwrite",
        ],
    )
    run_step(
        "train_offerpred",
        SCRIPT_DIR / "train_offerpred.py",
        [
            "--threads",
            str(args.threads),
            "--run-name",
            "offerpred",
        ],
    )
    run_step(
        "train_discardpred",
        SCRIPT_DIR / "train_discardpred.py",
        [
            "--threads",
            str(args.threads),
            "--run-name",
            "discardpred",
        ],
    )
    run_step(
        "train_locationpred",
        SCRIPT_DIR / "train_locationpred.py",
        [
            "--threads",
            str(args.threads),
            "--run-name",
            "locationpred",
        ],
    )
    run_step(
        "build_report",
        SCRIPT_DIR / "build_report.py",
        [
            "--report-name",
            "submission_report",
        ],
    )
    run_step(
        "audit_features",
        SCRIPT_DIR / "audit_features.py",
        [],
    )
    run_step(
        "audit_hyperparams",
        SCRIPT_DIR / "audit_hyperparams.py",
        [
            "--offerpred-manifest",
            str(REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/offerpred/run_manifest.json"),
            "--discardpred-manifest",
            str(REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/discardpred/run_manifest.json"),
            "--locationpred-manifest",
            str(REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/locationpred/run_manifest.json"),
        ],
    )

if __name__ == "__main__":
    main()
