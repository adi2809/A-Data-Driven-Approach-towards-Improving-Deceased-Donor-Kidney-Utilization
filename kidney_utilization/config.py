from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BenchmarkConfig:
    history_start: datetime = datetime.fromisoformat("2015-01-01T00:00:00")
    supervised_start: datetime = datetime.fromisoformat("2021-05-01T00:00:00")
    validation_start: datetime = datetime.fromisoformat("2023-01-01T00:00:00")
    test_start: datetime = datetime.fromisoformat("2024-01-01T00:00:00")
    supervised_end: datetime = datetime.fromisoformat("2024-12-31T23:59:59")

    benchmark_db_path: Path = Path("warehouse/match_runs/kidney_utilization_benchmark_same_match_history_fixed.duckdb")
    benchmark_manifest_path: Path = Path("warehouse/match_runs/kidney_utilization_benchmark_same_match_history_fixed_manifest.json")
    feature_parquet_glob: str = "warehouse/match_offer_features/parquet_same_match_history_fixed/match_year=*/*.parquet"
    donor_disposition_glob: str = "warehouse/saf/parquet/donor_disposition/*.parquet"
    artifact_root: Path = Path("warehouse/match_runs/artifacts/kidney_utilization")

    discard_threshold: float = 0.5
    early_cutoff: float = 0.1
    mid_cutoff: float = 0.5
    absolute_early_rank_cutoff: int = 10
    absolute_mid_rank_cutoff: int = 50

    offerpred_chunk_rows: int = 250_000
    evaluation_sample_rows_per_group: int = 100_000
    offerpred_negative_to_positive_ratio: int = 50
    offerpred_catboost_iterations: int = 600
    offerpred_catboost_depth: int = 8
    offerpred_catboost_learning_rate: float = 0.05
    offerpred_catboost_l2_leaf_reg: float = 3.0
    offerpred_catboost_early_stopping_rounds: int = 50
    locationpred_catboost_iterations: int = 600
    locationpred_catboost_depth: int = 8
    locationpred_catboost_learning_rate: float = 0.05
    locationpred_catboost_l2_leaf_reg: float = 3.0
    locationpred_catboost_early_stopping_rounds: int = 50

    random_seed: int = 42

    def split_for_timestamp(self, match_submit_dt: datetime) -> str:
        if match_submit_dt < self.supervised_start:
            return "history"
        if match_submit_dt >= self.test_start:
            return "test"
        if match_submit_dt >= self.validation_start:
            return "validation"
        return "train"

    def split_for_year(self, match_year: int) -> str:
        return self.split_for_timestamp(datetime.fromisoformat(f"{match_year}-01-01T00:00:00"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_start": self.history_start.isoformat(),
            "supervised_start": self.supervised_start.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "test_start": self.test_start.isoformat(),
            "supervised_end": self.supervised_end.isoformat(),
            "benchmark_db_path": str(self.benchmark_db_path),
            "benchmark_manifest_path": str(self.benchmark_manifest_path),
            "feature_parquet_glob": self.feature_parquet_glob,
            "donor_disposition_glob": self.donor_disposition_glob,
            "artifact_root": str(self.artifact_root),
            "discard_threshold": self.discard_threshold,
            "early_cutoff": self.early_cutoff,
            "mid_cutoff": self.mid_cutoff,
            "absolute_early_rank_cutoff": self.absolute_early_rank_cutoff,
            "absolute_mid_rank_cutoff": self.absolute_mid_rank_cutoff,
            "offerpred_chunk_rows": self.offerpred_chunk_rows,
            "evaluation_sample_rows_per_group": self.evaluation_sample_rows_per_group,
            "offerpred_negative_to_positive_ratio": self.offerpred_negative_to_positive_ratio,
            "offerpred_catboost_iterations": self.offerpred_catboost_iterations,
            "offerpred_catboost_depth": self.offerpred_catboost_depth,
            "offerpred_catboost_learning_rate": self.offerpred_catboost_learning_rate,
            "offerpred_catboost_l2_leaf_reg": self.offerpred_catboost_l2_leaf_reg,
            "offerpred_catboost_early_stopping_rounds": self.offerpred_catboost_early_stopping_rounds,
            "locationpred_catboost_iterations": self.locationpred_catboost_iterations,
            "locationpred_catboost_depth": self.locationpred_catboost_depth,
            "locationpred_catboost_learning_rate": self.locationpred_catboost_learning_rate,
            "locationpred_catboost_l2_leaf_reg": self.locationpred_catboost_l2_leaf_reg,
            "locationpred_catboost_early_stopping_rounds": self.locationpred_catboost_early_stopping_rounds,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkConfig":
        return cls(
            history_start=datetime.fromisoformat(payload["history_start"]),
            supervised_start=datetime.fromisoformat(payload["supervised_start"]),
            validation_start=datetime.fromisoformat(payload.get("validation_start", "2023-01-01T00:00:00")),
            test_start=datetime.fromisoformat(payload.get("test_start", "2024-01-01T00:00:00")),
            supervised_end=datetime.fromisoformat(payload["supervised_end"]),
            benchmark_db_path=Path(payload["benchmark_db_path"]),
            benchmark_manifest_path=Path(payload["benchmark_manifest_path"]),
            feature_parquet_glob=payload["feature_parquet_glob"],
            donor_disposition_glob=payload["donor_disposition_glob"],
            artifact_root=Path(payload["artifact_root"]),
            discard_threshold=float(payload.get("discard_threshold", 0.5)),
            early_cutoff=float(payload.get("early_cutoff", 0.1)),
            mid_cutoff=float(payload.get("mid_cutoff", 0.5)),
            absolute_early_rank_cutoff=int(payload.get("absolute_early_rank_cutoff", 10)),
            absolute_mid_rank_cutoff=int(payload.get("absolute_mid_rank_cutoff", 50)),
            offerpred_chunk_rows=int(payload.get("offerpred_chunk_rows", 250_000)),
            evaluation_sample_rows_per_group=int(payload.get("evaluation_sample_rows_per_group", 100_000)),
            offerpred_negative_to_positive_ratio=int(payload.get("offerpred_negative_to_positive_ratio", 50)),
            offerpred_catboost_iterations=int(payload.get("offerpred_catboost_iterations", 600)),
            offerpred_catboost_depth=int(payload.get("offerpred_catboost_depth", 8)),
            offerpred_catboost_learning_rate=float(payload.get("offerpred_catboost_learning_rate", 0.05)),
            offerpred_catboost_l2_leaf_reg=float(payload.get("offerpred_catboost_l2_leaf_reg", 3.0)),
            offerpred_catboost_early_stopping_rounds=int(payload.get("offerpred_catboost_early_stopping_rounds", 50)),
            locationpred_catboost_iterations=int(payload.get("locationpred_catboost_iterations", 600)),
            locationpred_catboost_depth=int(payload.get("locationpred_catboost_depth", 8)),
            locationpred_catboost_learning_rate=float(payload.get("locationpred_catboost_learning_rate", 0.05)),
            locationpred_catboost_l2_leaf_reg=float(payload.get("locationpred_catboost_l2_leaf_reg", 3.0)),
            locationpred_catboost_early_stopping_rounds=int(payload.get("locationpred_catboost_early_stopping_rounds", 50)),
            random_seed=int(payload.get("random_seed", 42)),
        )
