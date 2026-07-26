from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class TabularEncoder:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self.feature_kinds: dict[str, str] = {}
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_fill_values: dict[str, float] = {}
        self.numeric_sums: dict[str, float] = {}
        self.numeric_counts: dict[str, int] = {}

    @staticmethod
    def _datetime_to_float(series: pd.Series) -> pd.Series:
        converted = pd.to_datetime(series, errors="coerce")
        numeric = converted.astype("datetime64[ns]").view("int64").astype("float64")
        return pd.Series(
            np.where(converted.notna(), numeric, np.nan),
            index=series.index,
            dtype="float64",
        )

    @staticmethod
    def _normalize_category_series(series: pd.Series) -> pd.Series:
        object_series = series.astype("object")
        return object_series.where(pd.notna(object_series), "__MISSING__")

    def partial_fit(self, frame: pd.DataFrame) -> "TabularEncoder":
        for feature_name in self.feature_names:
            series = frame[feature_name] if feature_name in frame.columns else pd.Series([np.nan] * len(frame))
            if feature_name not in self.feature_kinds:
                if pd.api.types.is_datetime64_any_dtype(series):
                    self.feature_kinds[feature_name] = "datetime"
                    self.numeric_sums[feature_name] = 0.0
                    self.numeric_counts[feature_name] = 0
                elif pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                    self.feature_kinds[feature_name] = "numeric"
                    self.numeric_sums[feature_name] = 0.0
                    self.numeric_counts[feature_name] = 0
                else:
                    self.feature_kinds[feature_name] = "category"
                    self.category_maps[feature_name] = {}

            kind = self.feature_kinds[feature_name]
            if kind == "datetime":
                numeric_series = self._datetime_to_float(series)
                valid = numeric_series.dropna()
                if not valid.empty:
                    self.numeric_sums[feature_name] += float(valid.sum())
                    self.numeric_counts[feature_name] += int(valid.shape[0])
            elif kind == "numeric":
                numeric_series = pd.to_numeric(series, errors="coerce").astype("float64")
                valid = numeric_series.dropna()
                if not valid.empty:
                    self.numeric_sums[feature_name] += float(valid.sum())
                    self.numeric_counts[feature_name] += int(valid.shape[0])
            else:
                normalized = self._normalize_category_series(series)
                mapping = self.category_maps[feature_name]
                for value in pd.unique(normalized):
                    if value not in mapping:
                        mapping[value] = len(mapping)

            if kind in {"datetime", "numeric"}:
                count = self.numeric_counts.get(feature_name, 0)
                self.numeric_fill_values[feature_name] = (
                    self.numeric_sums.get(feature_name, 0.0) / count if count > 0 else 0.0
                )
        return self

    def fit(self, frame: pd.DataFrame) -> "TabularEncoder":
        self.partial_fit(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded: dict[str, pd.Series] = {}
        index = frame.index
        for feature_name in self.feature_names:
            kind = self.feature_kinds[feature_name]
            if feature_name in frame.columns:
                series = frame[feature_name]
            else:
                series = pd.Series([np.nan] * len(frame), index=index)

            if kind == "datetime":
                encoded[feature_name] = self._datetime_to_float(series).fillna(self.numeric_fill_values[feature_name])
            elif kind == "numeric":
                encoded[feature_name] = pd.to_numeric(series, errors="coerce").astype("float64").fillna(
                    self.numeric_fill_values[feature_name]
                )
            else:
                mapping = self.category_maps[feature_name]
                normalized = self._normalize_category_series(series)
                encoded[feature_name] = normalized.map(mapping).fillna(-1).astype("float64")

        return pd.DataFrame(encoded, index=index)


class NativeCatBoostFrameEncoder:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self.categorical_features: list[str] = []
        self.numeric_features: list[str] = []
        self.numeric_fill_values: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> "NativeCatBoostFrameEncoder":
        self.categorical_features = []
        self.numeric_features = []
        self.numeric_fill_values = {}
        for feature_name in self.feature_names:
            series = frame[feature_name] if feature_name in frame.columns else pd.Series([np.nan] * len(frame))
            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                self.categorical_features.append(feature_name)
                continue
            self.numeric_features.append(feature_name)
            numeric_series = pd.to_numeric(series, errors="coerce").astype("float64")
            valid = numeric_series.dropna()
            self.numeric_fill_values[feature_name] = float(valid.mean()) if not valid.empty else 0.0
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = pd.DataFrame(index=frame.index)
        for feature_name in self.feature_names:
            if feature_name in frame.columns:
                series = frame[feature_name]
            else:
                series = pd.Series([np.nan] * len(frame), index=frame.index)

            if feature_name in self.categorical_features:
                prepared[feature_name] = TabularEncoder._normalize_category_series(series).astype(str)
            else:
                prepared[feature_name] = pd.to_numeric(series, errors="coerce").astype("float64").fillna(
                    self.numeric_fill_values.get(feature_name, 0.0)
                )
        return prepared


class ConstantProbabilityModel:
    def __init__(self, class_labels: list[Any], constant_label: Any) -> None:
        self.class_labels = list(class_labels)
        self.constant_label = constant_label

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        row_count = len(features)
        probabilities = np.zeros((row_count, len(self.class_labels)), dtype=float)
        constant_index = self.class_labels.index(self.constant_label)
        probabilities[:, constant_index] = 1.0
        return probabilities


@dataclass(slots=True)
class FittedTabularModel:
    feature_names: list[str]
    class_labels: list[Any]
    encoder: TabularEncoder
    estimator: Any
    backend: str

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = self.encoder.transform(frame[self.feature_names])
        raw = self.estimator.predict_proba(encoded)
        columns = [str(label) for label in self.class_labels]
        return pd.DataFrame(raw, columns=columns, index=frame.index)


@dataclass(slots=True)
class FittedRawFrameModel:
    feature_names: list[str]
    class_labels: list[Any]
    encoder: NativeCatBoostFrameEncoder
    estimator: Any
    backend: str

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self.encoder.transform(frame[self.feature_names])
        raw = self.estimator.predict_proba(prepared)
        columns = [str(label) for label in self.class_labels]
        return pd.DataFrame(raw, columns=columns, index=frame.index)


def _fit_with_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame | None,
    y_validation: pd.Series | None,
    class_count: int,
    random_seed: int,
    train_weight: pd.Series | None = None,
    validation_weight: pd.Series | None = None,
) -> tuple[Any, str]:
    from catboost import CatBoostClassifier

    disable_file_backed = os.environ.get("KIDNEY_FIRST_HIT_DISABLE_FILE_BACKED_CATBOOST") == "1"
    params = {
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.08,
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": not disable_file_backed,
    }
    if class_count == 2:
        params["loss_function"] = "Logloss"
        params["eval_metric"] = "AUC"
    else:
        params["loss_function"] = "MultiClass"
        params["eval_metric"] = "MultiClass"
    model = CatBoostClassifier(**params)
    eval_set = None
    fit_kwargs: dict[str, Any] = {}
    if X_validation is not None and y_validation is not None and len(X_validation) > 0:
        eval_set = (X_validation, y_validation)
    if train_weight is not None:
        fit_kwargs["sample_weight"] = train_weight
    if validation_weight is not None and eval_set is not None:
        fit_kwargs["sample_weight_eval_set"] = [validation_weight]
    model.fit(X_train, y_train, eval_set=eval_set, use_best_model=False, **fit_kwargs)
    return model, "catboost"


def _fit_with_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_count: int,
    random_seed: int,
    train_weight: pd.Series | None = None,
) -> tuple[Any, str]:
    from xgboost import XGBClassifier

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_seed,
        "tree_method": "hist",
        "nthread": 1,
        "verbosity": 0,
    }
    if class_count == 2:
        params["objective"] = "binary:logistic"
        model = XGBClassifier(**params)
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = class_count
        model = XGBClassifier(**params)
    fit_kwargs: dict[str, Any] = {}
    if train_weight is not None:
        fit_kwargs["sample_weight"] = train_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model, "xgboost"


def _fit_with_sklearn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_count: int,
    random_seed: int,
    train_weight: pd.Series | None = None,
) -> tuple[Any, str]:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_seed,
        class_weight="balanced_subsample" if class_count == 2 else None,
        n_jobs=1,
    )
    fit_kwargs: dict[str, Any] = {}
    if train_weight is not None:
        fit_kwargs["sample_weight"] = train_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model, "random_forest"


def fit_classifier(
    train_frame: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    validation_frame: pd.DataFrame | None = None,
    sample_weight_column: str | None = None,
    random_seed: int = 42,
) -> FittedTabularModel:
    class_labels = sorted(train_frame[target_column].dropna().unique().tolist())
    if len(class_labels) == 0:
        raise ValueError(f"training target {target_column!r} has no non-null values")

    encoder = TabularEncoder(feature_names).fit(train_frame[feature_names])
    X_train = encoder.transform(train_frame[feature_names])
    y_train = train_frame[target_column]
    train_weight = None
    if sample_weight_column is not None and sample_weight_column in train_frame.columns:
        train_weight = pd.to_numeric(train_frame[sample_weight_column], errors="coerce").astype("float64").fillna(0.0)

    if len(class_labels) == 1:
        estimator = ConstantProbabilityModel(class_labels=class_labels, constant_label=class_labels[0])
        return FittedTabularModel(
            feature_names=feature_names,
            class_labels=class_labels,
            encoder=encoder,
            estimator=estimator,
            backend="constant",
        )

    X_validation = None
    y_validation = None
    validation_weight = None
    if validation_frame is not None and len(validation_frame) > 0:
        X_validation = encoder.transform(validation_frame[feature_names])
        y_validation = validation_frame[target_column]
        if sample_weight_column is not None and sample_weight_column in validation_frame.columns:
            validation_weight = (
                pd.to_numeric(validation_frame[sample_weight_column], errors="coerce").astype("float64").fillna(0.0)
            )

    backend_error_messages: list[str] = []
    for backend_name, fitter in [
        ("catboost", _fit_with_catboost),
        ("xgboost", _fit_with_xgboost),
        ("sklearn", _fit_with_sklearn),
    ]:
        try:
            if backend_name == "catboost":
                estimator, backend = fitter(
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    len(class_labels),
                    random_seed,
                    train_weight=train_weight,
                    validation_weight=validation_weight,
                )
            elif backend_name == "xgboost":
                estimator, backend = fitter(X_train, y_train, len(class_labels), random_seed, train_weight=train_weight)
            else:
                estimator, backend = fitter(X_train, y_train, len(class_labels), random_seed, train_weight=train_weight)
            return FittedTabularModel(
                feature_names=feature_names,
                class_labels=class_labels,
                encoder=encoder,
                estimator=estimator,
                backend=backend,
            )
        except Exception as exc:  # pragma: no cover - exercised only when optional backends fail.
            backend_error_messages.append(f"{backend_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError("no classifier backend succeeded: " + " | ".join(backend_error_messages))


def fit_native_catboost_classifier(
    train_frame: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    validation_frame: pd.DataFrame | None = None,
    sample_weight_column: str | None = None,
    random_seed: int = 42,
    iterations: int = 600,
    depth: int = 8,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 3.0,
    early_stopping_rounds: int = 50,
) -> FittedRawFrameModel | FittedTabularModel:
    from catboost import CatBoostClassifier

    class_labels = sorted(train_frame[target_column].dropna().unique().tolist())
    if len(class_labels) == 0:
        raise ValueError(f"training target {target_column!r} has no non-null values")

    if len(class_labels) == 1:
        estimator = ConstantProbabilityModel(class_labels=class_labels, constant_label=class_labels[0])
        encoder = NativeCatBoostFrameEncoder(feature_names).fit(train_frame[feature_names])
        return FittedRawFrameModel(
            feature_names=feature_names,
            class_labels=class_labels,
            encoder=encoder,
            estimator=estimator,
            backend="constant",
        )

    encoder = NativeCatBoostFrameEncoder(feature_names).fit(train_frame[feature_names])
    X_train = encoder.transform(train_frame[feature_names])
    y_train = train_frame[target_column].astype(int)
    train_weight = None
    if sample_weight_column is not None and sample_weight_column in train_frame.columns:
        train_weight = pd.to_numeric(train_frame[sample_weight_column], errors="coerce").astype("float64").fillna(0.0)

    X_validation = None
    y_validation = None
    eval_set = None
    if validation_frame is not None and len(validation_frame) > 0:
        X_validation = encoder.transform(validation_frame[feature_names])
        y_validation = validation_frame[target_column].astype(int)
        eval_set = (X_validation, y_validation)

    disable_file_backed = os.environ.get("KIDNEY_FIRST_HIT_DISABLE_FILE_BACKED_CATBOOST") == "1"
    model = CatBoostClassifier(
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(learning_rate),
        l2_leaf_reg=float(l2_leaf_reg),
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        custom_metric=["AUC", "PRAUC:type=Classic"],
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=not disable_file_backed,
    )
    fit_kwargs: dict[str, Any] = {
        "X": X_train,
        "y": y_train,
        "cat_features": encoder.categorical_features,
        "use_best_model": eval_set is not None,
    }
    if train_weight is not None:
        fit_kwargs["sample_weight"] = train_weight
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    model.fit(**fit_kwargs)
    return FittedRawFrameModel(
        feature_names=feature_names,
        class_labels=class_labels,
        encoder=encoder,
        estimator=model,
        backend="catboost_native",
    )

