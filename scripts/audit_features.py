#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kidney_utilization.feature_specs import DISCARDPRED_NAME, LOCATIONPRED_NAME, OFFERPRED_NAME, PUBLIC_MODEL_FEATURE_SETS


SNAKE_CASE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write feature-name audits for OfferPred, DiscardPred, and LocationPred.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "warehouse/match_runs/artifacts/kidney_utilization/audits",
    )
    return parser.parse_args()


def analyze_feature_list(feature_names: list[str]) -> dict[str, object]:
    counts = Counter(feature_names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    non_snake_case = sorted(name for name in feature_names if not SNAKE_CASE.fullmatch(name))
    longest = sorted(feature_names, key=lambda name: (-len(name), name))[:10]
    prefix_counts = Counter(name.split("_", 1)[0] for name in feature_names)
    return {
        "count": len(feature_names),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "non_snake_case_count": len(non_snake_case),
        "non_snake_case": non_snake_case,
        "longest_feature_names": longest,
        "prefix_counts": dict(sorted(prefix_counts.items(), key=lambda item: (-item[1], item[0]))[:10]),
        "features": feature_names,
    }


def overlap_summary(left: list[str], right: list[str]) -> dict[str, object]:
    overlap = sorted(set(left).intersection(right))
    return {
        "count": len(overlap),
        "features": overlap,
    }


def markdown_section(model_name: str, variant_name: str, payload: dict[str, object]) -> list[str]:
    return [
        f"## {model_name} ({variant_name})",
        f"- Feature count: {payload['count']}",
        f"- Duplicate names: {payload['duplicate_count']}",
        f"- Non-snake-case names: {payload['non_snake_case_count']}",
        f"- Longest feature names: {', '.join(payload['longest_feature_names']) if payload['longest_feature_names'] else 'None'}",
        "",
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_audit: dict[str, object] = {"models": {}, "pairwise_overlap": {}, "variant_overlap": {}}

    default_sets = {
        OFFERPRED_NAME: PUBLIC_MODEL_FEATURE_SETS[OFFERPRED_NAME]["default"],
        DISCARDPRED_NAME: PUBLIC_MODEL_FEATURE_SETS[DISCARDPRED_NAME]["default"],
        LOCATIONPRED_NAME: PUBLIC_MODEL_FEATURE_SETS[LOCATIONPRED_NAME]["default"],
    }

    for model_name, variants in PUBLIC_MODEL_FEATURE_SETS.items():
        feature_audit["models"][model_name] = {
            variant_name: analyze_feature_list(feature_names)
            for variant_name, feature_names in variants.items()
        }
        variant_names = sorted(variants)
        for index, left_variant in enumerate(variant_names):
            for right_variant in variant_names[index + 1 :]:
                feature_audit["variant_overlap"][f"{model_name}__{left_variant}__{right_variant}"] = overlap_summary(
                    variants[left_variant],
                    variants[right_variant],
                )

    pair_keys = [
        (OFFERPRED_NAME, DISCARDPRED_NAME),
        (OFFERPRED_NAME, LOCATIONPRED_NAME),
        (DISCARDPRED_NAME, LOCATIONPRED_NAME),
    ]
    for left_name, right_name in pair_keys:
        feature_audit["pairwise_overlap"][f"{left_name}__{right_name}"] = overlap_summary(
            default_sets[left_name],
            default_sets[right_name],
        )

    json_path = args.output_dir / "feature_audit.json"
    json_path.write_text(json.dumps(feature_audit, indent=2))

    markdown_lines = [
        "# Feature Audit",
        "",
        "This audit summarizes feature inventories, duplicate-name checks, naming checks, and default-set overlap.",
        "",
    ]
    for model_name, variants in feature_audit["models"].items():
        for variant_name, payload in variants.items():
            markdown_lines.extend(markdown_section(model_name, variant_name, payload))
            markdown_lines.append(
                f"- Top prefixes: {', '.join(f'{name}={count}' for name, count in payload['prefix_counts'].items()) if payload['prefix_counts'] else 'None'}"
            )
            markdown_lines.append("")
    markdown_lines.append("## Pairwise Overlap")
    for pair_name, payload in feature_audit["pairwise_overlap"].items():
        markdown_lines.append(f"- {pair_name}: {payload['count']} overlapping features")
    markdown_lines.append("")
    markdown_lines.append("## Variant Overlap")
    for pair_name, payload in feature_audit["variant_overlap"].items():
        markdown_lines.append(f"- {pair_name}: {payload['count']} overlapping features")
    markdown_lines.append("")
    (args.output_dir / "feature_audit.md").write_text("\n".join(markdown_lines))


if __name__ == "__main__":
    main()
