#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kidney_utilization.config import BenchmarkConfig
from kidney_utilization.feature_specs import DISCARDPRED_NAME, LOCATIONPRED_NAME, OFFERPRED_NAME


GLOBAL_KEYS = [
    "history_start",
    "supervised_start",
    "validation_start",
    "test_start",
    "supervised_end",
    "benchmark_db_path",
    "benchmark_manifest_path",
    "feature_parquet_glob",
    "donor_disposition_glob",
    "artifact_root",
    "evaluation_sample_rows_per_group",
    "random_seed",
]

OFFERPRED_KEYS = [
    "offerpred_chunk_rows",
    "offerpred_negative_to_positive_ratio",
    "offerpred_catboost_iterations",
    "offerpred_catboost_depth",
    "offerpred_catboost_learning_rate",
    "offerpred_catboost_l2_leaf_reg",
    "offerpred_catboost_early_stopping_rounds",
]

DISCARDPRED_KEYS = [
    "discard_threshold",
]

LOCATIONPRED_KEYS = [
    "locationpred_catboost_iterations",
    "locationpred_catboost_depth",
    "locationpred_catboost_learning_rate",
    "locationpred_catboost_l2_leaf_reg",
    "locationpred_catboost_early_stopping_rounds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write grouped hyperparameter audits for OfferPred, DiscardPred, and LocationPred.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/audits",
    )
    parser.add_argument(
        "--offerpred-manifest",
        type=Path,
        default=REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/offerpred/run_manifest.json",
    )
    parser.add_argument(
        "--discardpred-manifest",
        type=Path,
        default=REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/discardpred/run_manifest.json",
    )
    parser.add_argument(
        "--locationpred-manifest",
        type=Path,
        default=REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/locationpred/run_manifest.json",
    )
    return parser.parse_args()


def group_payload(config_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        "global": {key: config_payload[key] for key in GLOBAL_KEYS},
        OFFERPRED_NAME: {key: config_payload[key] for key in OFFERPRED_KEYS},
        DISCARDPRED_NAME: {key: config_payload[key] for key in DISCARDPRED_KEYS},
        LOCATIONPRED_NAME: {key: config_payload[key] for key in LOCATIONPRED_KEYS},
    }


def read_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def manifest_summary(manifest: dict[str, object] | None) -> dict[str, object] | None:
    if manifest is None:
        return None
    default_config = BenchmarkConfig().to_dict()
    config = manifest.get("config", {})
    grouped_config = group_payload(config) if config else None
    changed_keys = {
        key: config[key]
        for key in sorted(config)
        if key in default_config and config[key] != default_config[key]
    }
    return {
        "model_name": manifest.get("model_name"),
        "backend_summary": manifest.get("backends"),
        "training_modes": manifest.get("training_modes"),
        "grouped_config": grouped_config,
        "non_default_config": changed_keys,
    }


def markdown_group(title: str, payload: dict[str, object]) -> list[str]:
    lines = [f"## {title}"]
    for key, value in payload.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    default_config = BenchmarkConfig().to_dict()
    audit_payload: dict[str, object] = {
        "default_config": group_payload(default_config),
        "artifacts": {
            OFFERPRED_NAME: manifest_summary(read_manifest(args.offerpred_manifest)),
            DISCARDPRED_NAME: manifest_summary(read_manifest(args.discardpred_manifest)),
            LOCATIONPRED_NAME: manifest_summary(read_manifest(args.locationpred_manifest)),
        },
    }

    json_path = args.output_dir / "hyperparameter_audit.json"
    json_path.write_text(json.dumps(audit_payload, indent=2))

    markdown_lines = [
        "# Hyperparameter Audit",
        "",
        "This audit groups benchmark and training settings into OfferPred, DiscardPred, and LocationPred sections.",
        "",
    ]
    for group_name, payload in audit_payload["default_config"].items():
        markdown_lines.extend(markdown_group(group_name, payload))
    markdown_lines.append("## Artifact Manifests")
    for model_name, payload in audit_payload["artifacts"].items():
        if payload is None:
            markdown_lines.append(f"- {model_name}: missing")
            continue
        markdown_lines.append(
            f"- {model_name}: present; non-default keys = {', '.join(payload['non_default_config']) if payload['non_default_config'] else 'none'}"
        )
    markdown_lines.append("")
    (args.output_dir / "hyperparameter_audit.md").write_text("\n".join(markdown_lines))


if __name__ == "__main__":
    main()
