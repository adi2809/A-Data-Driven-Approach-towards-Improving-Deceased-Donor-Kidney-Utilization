#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, time as dt_time, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pgeocode
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat


PANDAS_INT_DTYPES = {
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
}

PANDAS_FLOAT_DTYPES = {
    "float": "Float32",
    "double": "Float64",
}

INDEX_CANDIDATE_COLUMNS = [
    "PX_ID",
    "PERS_ID",
    "TRR_ID",
    "TRR_FOL_ID",
    "TX_ID",
    "DONOR_ID",
    "MATCH_ID",
    "MALIG_ID",
    "REC_CTR_CD",
    "CAN_LISTING_CTR_CD",
    "CTR_CD",
    "TXC_CTR_CD",
]

NAUTICAL_CIRCLE_RADIUS_NM = 250.0
EARTH_RADIUS_NM = 3440.065


@dataclass
class SourceFile:
    table_name: str
    path: Path
    source_type: str = "sas7bdat"


@dataclass
class TableStats:
    table_name: str
    source_file: str
    rows_in_source: int
    columns_in_source: int
    parts_written: int = 0
    rows_written: int = 0
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class KDPIReference:
    hyp_scale: dict[int, float]
    diab_scale: dict[int, float]
    kdri_scale: dict[int, float]
    kdpi_upper_bounds: dict[int, np.ndarray]


@dataclass(frozen=True)
class OPOCenterGeoContext:
    institution_geo: pd.DataFrame
    opo_tx_center_distance: pd.DataFrame
    opo_tx_center_count_250nm: pd.DataFrame
    donor_opo_count_250nm: dict[int, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_source_files(source_dir: Path, tables: set[str] | None) -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in sorted(source_dir.glob("*.sas7bdat")):
        table_name = path.stem.lower()
        if tables and table_name not in tables:
            continue
        files.append(SourceFile(table_name=table_name, path=path, source_type="sas7bdat"))
    for path in sorted(source_dir.glob("*.sas7bcat")):
        table_name = path.stem.lower()
        if tables and table_name not in tables:
            continue
        files.append(SourceFile(table_name=table_name, path=path, source_type="sas7bcat"))
    files.sort(key=lambda item: item.table_name)
    if not files:
        raise FileNotFoundError(f"No .sas7bdat or .sas7bcat files found in {source_dir}")
    return files


def classify_columns(meta) -> tuple[set[str], set[str], dict[str, str], dict[str, str]]:
    date_cols: set[str] = set()
    time_cols: set[str] = set()
    int_cols: dict[str, str] = {}
    float_cols: dict[str, str] = {}

    for column, fmt in meta.original_variable_types.items():
        column = column.upper()
        fmt = (fmt or "").upper()
        if fmt.startswith("TIME"):
            time_cols.add(column)
        elif "DATE" in fmt or "MMDDYY" in fmt or "YYMMDD" in fmt:
            date_cols.add(column)

    for column, readstat_type in meta.readstat_variable_types.items():
        column = column.upper()
        if column in date_cols or column in time_cols:
            continue
        if readstat_type in PANDAS_INT_DTYPES:
            int_cols[column] = PANDAS_INT_DTYPES[readstat_type]
        elif readstat_type in PANDAS_FLOAT_DTYPES:
            float_cols[column] = PANDAS_FLOAT_DTYPES[readstat_type]

    return date_cols, time_cols, int_cols, float_cols


def format_time_value(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, dt_time):
        return value.strftime("%H:%M:%S")
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S")
    return str(value)


def format_catalog_value_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return format(float(value), "g")
    return str(value)


def build_catalog_frame(meta) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for format_name, mapping in meta.value_labels.items():
        non_missing_keys = [key for key in mapping.keys() if not pd.isna(key)]
        value_type = "string" if any(isinstance(key, str) for key in non_missing_keys) else "numeric"
        for value_order, (raw_value, label) in enumerate(mapping.items(), start=1):
            is_missing_value = bool(pd.isna(raw_value))
            raw_value_text = format_catalog_value_text(raw_value)
            raw_value_num = pd.NA
            raw_value_string = pd.NA
            if value_type == "numeric" and not is_missing_value:
                raw_value_num = float(raw_value)
            if value_type == "string" and not is_missing_value:
                raw_value_string = str(raw_value)
            label_text = pd.NA if pd.isna(label) else str(label)
            label_trimmed = pd.NA if pd.isna(label_text) else label_text.strip()
            rows.append(
                {
                    "FORMAT_NAME": format_name,
                    "VALUE_ORDER": value_order,
                    "VALUE_TYPE": value_type,
                    "RAW_VALUE_TEXT": raw_value_text,
                    "RAW_VALUE_NUM": raw_value_num,
                    "RAW_VALUE_STRING": raw_value_string,
                    "IS_MISSING_VALUE": is_missing_value,
                    "LABEL": label_text,
                    "LABEL_TRIMMED": label_trimmed,
                }
            )

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return pd.DataFrame(
            {
                "FORMAT_NAME": pd.Series(dtype="string"),
                "VALUE_ORDER": pd.Series(dtype="Int32"),
                "VALUE_TYPE": pd.Series(dtype="string"),
                "RAW_VALUE_TEXT": pd.Series(dtype="string"),
                "RAW_VALUE_NUM": pd.Series(dtype="Float64"),
                "RAW_VALUE_STRING": pd.Series(dtype="string"),
                "IS_MISSING_VALUE": pd.Series(dtype="boolean"),
                "LABEL": pd.Series(dtype="string"),
                "LABEL_TRIMMED": pd.Series(dtype="string"),
            }
        )

    frame["FORMAT_NAME"] = frame["FORMAT_NAME"].astype("string")
    frame["VALUE_ORDER"] = pd.array(frame["VALUE_ORDER"], dtype="Int32")
    frame["VALUE_TYPE"] = frame["VALUE_TYPE"].astype("string")
    frame["RAW_VALUE_TEXT"] = frame["RAW_VALUE_TEXT"].astype("string")
    frame["RAW_VALUE_NUM"] = pd.to_numeric(frame["RAW_VALUE_NUM"], errors="coerce").astype("Float64")
    frame["RAW_VALUE_STRING"] = frame["RAW_VALUE_STRING"].astype("string")
    frame["IS_MISSING_VALUE"] = frame["IS_MISSING_VALUE"].astype("boolean")
    frame["LABEL"] = frame["LABEL"].astype("string")
    frame["LABEL_TRIMMED"] = frame["LABEL_TRIMMED"].astype("string")
    return frame


def normalize_chunk(
    chunk: pd.DataFrame,
    date_cols: set[str],
    time_cols: set[str],
    int_cols: dict[str, str],
    float_cols: dict[str, str],
) -> pd.DataFrame:
    chunk.columns = [column.upper() for column in chunk.columns]

    for column in date_cols:
        chunk[column] = pd.to_datetime(chunk[column], errors="coerce")

    for column in time_cols:
        chunk[column] = chunk[column].map(format_time_value).astype("string")

    for column, dtype in int_cols.items():
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype(dtype)

    for column, dtype in float_cols.items():
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype(dtype)

    return chunk


def default_kdpi_do_file() -> Path:
    return Path.home() / "Downloads/kdpi_mapping.do"


def load_kdpi_reference(do_file: Path) -> KDPIReference:
    if not do_file.exists():
        raise FileNotFoundError(f"KDPI mapping do-file not found: {do_file}")

    text = do_file.read_text()
    scale_pattern = re.compile(
        r"(?:gen|replace)\s+(hyp_scale|diab_scale|kdri_scale)\s*=\s*([-0-9.]+)\s+if\s+recov_yr\s*==\s*(\d{4})",
        flags=re.IGNORECASE,
    )
    kdpi_pattern = re.compile(
        r"replace\s+KDPI(?P<yy>\d{2})\s*=\s*(?P<kdpi>\d{1,3})\s+if\s+\(KDRI_20(?P<yy2>\d{2})_med\s*>"
        r"\s*[-0-9.]+\s*&\s*KDRI_20\d{2}_med\s*<=\s*(?P<upper>[-0-9.]+)\)",
        flags=re.IGNORECASE,
    )

    scales: dict[str, dict[int, float]] = {"hyp_scale": {}, "diab_scale": {}, "kdri_scale": {}}
    for name, value, year in scale_pattern.findall(text):
        scales[name.lower()][int(year)] = float(value)

    kdpi_cutoffs: dict[int, dict[int, float]] = {}
    for match in kdpi_pattern.finditer(text):
        year_suffix = int(match.group("yy"))
        if year_suffix != int(match.group("yy2")):
            raise ValueError(f"Mismatched KDPI year suffix in {do_file}: {match.group(0)}")
        year = 2000 + year_suffix
        kdpi_value = int(match.group("kdpi"))
        kdpi_cutoffs.setdefault(year, {})[kdpi_value] = float(match.group("upper"))

    expected_years = set(range(2012, 2026))
    for scale_name, mapping in scales.items():
        missing_years = sorted(expected_years - set(mapping))
        if missing_years:
            raise ValueError(f"Missing {scale_name} years in {do_file}: {missing_years}")

    kdpi_upper_bounds: dict[int, np.ndarray] = {}
    for year in expected_years:
        cutoff_map = kdpi_cutoffs.get(year)
        if not cutoff_map:
            raise ValueError(f"Missing KDPI cutoffs for {year} in {do_file}")
        missing_kdpi = sorted(set(range(101)) - set(cutoff_map))
        if missing_kdpi:
            raise ValueError(f"Missing KDPI cutoff values for {year} in {do_file}: {missing_kdpi[:10]}")
        kdpi_upper_bounds[year] = np.array([cutoff_map[kdpi] for kdpi in range(101)], dtype=np.float64)

    return KDPIReference(
        hyp_scale=scales["hyp_scale"],
        diab_scale=scales["diab_scale"],
        kdri_scale=scales["kdri_scale"],
        kdpi_upper_bounds=kdpi_upper_bounds,
    )


def normalize_city_key(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).upper().strip()
    if not text:
        return None
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\bST\b", "SAINT", text)
    text = re.sub(r"\bFT\b", "FORT", text)
    text = re.sub(r"\bMT\b", "MOUNT", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_zip5(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"(\d{5})", str(value))
    if not match:
        return None
    zip5 = match.group(1)
    if zip5 == "99999":
        return None
    return zip5


def load_institution_frame(source_path: Path) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"Institution source file not found: {source_path}")
    frame, meta = pyreadstat.read_sas7bdat(str(source_path))
    date_cols, time_cols, int_cols, float_cols = classify_columns(meta)
    return normalize_chunk(frame, date_cols, time_cols, int_cols, float_cols)


def build_opo_tx_center_geo_context_from_df(institution: pd.DataFrame) -> OPOCenterGeoContext:
    institution = institution.copy()
    institution.columns = [column.upper() for column in institution.columns]
    required_columns = {
        "CTR_ID",
        "CTR_CD",
        "CTR_TY",
        "REGION",
        "ENTIRE_NAME",
        "NAME_PART1",
        "NAME_PART2",
        "PRIMARY_CITY",
        "PRIMARY_STATE",
        "PRIMARY_ZIP",
        "PROVIDER_NUM",
        "PRIMARY_CTRY",
        "OPTN_MBR",
        "ESRD_REGION",
    }
    missing_columns = sorted(required_columns - set(institution.columns))
    if missing_columns:
        raise ValueError(f"Institution frame missing required columns: {missing_columns}")

    geo = institution[list(required_columns)].copy()
    geo["CTR_ID"] = pd.to_numeric(geo["CTR_ID"], errors="coerce").astype("Int64")
    geo["ZIP5"] = geo["PRIMARY_ZIP"].map(extract_zip5).astype("string")
    geo["CITY_KEY"] = geo["PRIMARY_CITY"].map(normalize_city_key).astype("string")
    geo["STATE_KEY"] = (
        geo["PRIMARY_STATE"].astype("string").str.upper().str.strip().replace({"": pd.NA, "NAN": pd.NA})
    )

    postal = pgeocode.Nominatim("us")._data[["postal_code", "place_name", "state_code", "latitude", "longitude"]].copy()
    postal = postal.dropna(subset=["latitude", "longitude"])
    postal["POSTAL_CODE"] = postal["postal_code"].astype(str).str.extract(r"(\d{5})", expand=False)
    postal_zip = (
        postal.dropna(subset=["POSTAL_CODE"])
        .drop_duplicates("POSTAL_CODE")[["POSTAL_CODE", "latitude", "longitude"]]
        .rename(columns={"latitude": "ZIP_LATITUDE", "longitude": "ZIP_LONGITUDE"})
    )
    postal["CITY_KEY"] = postal["place_name"].map(normalize_city_key).astype("string")
    postal["STATE_KEY"] = postal["state_code"].astype("string").str.upper().str.strip()
    postal_city = (
        postal.dropna(subset=["CITY_KEY", "STATE_KEY"])
        .groupby(["CITY_KEY", "STATE_KEY"], as_index=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "POSTAL_CITY_LATITUDE", "longitude": "POSTAL_CITY_LONGITUDE"})
    )

    geo = geo.merge(postal_zip, left_on="ZIP5", right_on="POSTAL_CODE", how="left")
    geo["LATITUDE"] = pd.to_numeric(geo["ZIP_LATITUDE"], errors="coerce").astype("Float64")
    geo["LONGITUDE"] = pd.to_numeric(geo["ZIP_LONGITUDE"], errors="coerce").astype("Float64")
    geo["GEO_SOURCE"] = pd.Series(pd.NA, index=geo.index, dtype="string")
    zip_mask = geo["LATITUDE"].notna() & geo["LONGITUDE"].notna()
    geo.loc[zip_mask, "GEO_SOURCE"] = "zip"

    institution_city = (
        geo.loc[zip_mask]
        .dropna(subset=["CITY_KEY", "STATE_KEY"])
        .groupby(["CITY_KEY", "STATE_KEY"], as_index=False)[["LATITUDE", "LONGITUDE"]]
        .mean()
        .rename(columns={"LATITUDE": "INST_CITY_LATITUDE", "LONGITUDE": "INST_CITY_LONGITUDE"})
    )
    geo = geo.merge(institution_city, on=["CITY_KEY", "STATE_KEY"], how="left")
    city_mask = geo["LATITUDE"].isna() & geo["INST_CITY_LATITUDE"].notna() & geo["INST_CITY_LONGITUDE"].notna()
    geo.loc[city_mask, "LATITUDE"] = geo.loc[city_mask, "INST_CITY_LATITUDE"]
    geo.loc[city_mask, "LONGITUDE"] = geo.loc[city_mask, "INST_CITY_LONGITUDE"]
    geo.loc[city_mask, "GEO_SOURCE"] = "city_state_institution"

    geo = geo.merge(postal_city, on=["CITY_KEY", "STATE_KEY"], how="left")
    postal_city_mask = (
        geo["LATITUDE"].isna()
        & geo["POSTAL_CITY_LATITUDE"].notna()
        & geo["POSTAL_CITY_LONGITUDE"].notna()
    )
    geo.loc[postal_city_mask, "LATITUDE"] = geo.loc[postal_city_mask, "POSTAL_CITY_LATITUDE"]
    geo.loc[postal_city_mask, "LONGITUDE"] = geo.loc[postal_city_mask, "POSTAL_CITY_LONGITUDE"]
    geo.loc[postal_city_mask, "GEO_SOURCE"] = "city_state_postal"

    institution_geo = geo[
        [
            "CTR_ID",
            "CTR_CD",
            "CTR_TY",
            "REGION",
            "ENTIRE_NAME",
            "NAME_PART1",
            "NAME_PART2",
            "PRIMARY_CITY",
            "PRIMARY_STATE",
            "PRIMARY_ZIP",
            "PROVIDER_NUM",
            "PRIMARY_CTRY",
            "OPTN_MBR",
            "ESRD_REGION",
            "ZIP5",
            "LATITUDE",
            "LONGITUDE",
            "GEO_SOURCE",
        ]
    ].copy()

    opo_geo = (
        institution_geo.loc[
            institution_geo["CTR_TY"].isin(["OP1", "IO1"])
            & institution_geo["CTR_ID"].notna()
            & institution_geo["LATITUDE"].notna()
            & institution_geo["LONGITUDE"].notna()
        ]
        .drop_duplicates(subset=["CTR_ID"])
        .rename(
            columns={
                "CTR_ID": "DON_OPO_CTR_ID",
                "CTR_CD": "DON_OPO_CTR_CD",
                "GEO_SOURCE": "OPO_GEO_SOURCE",
            }
        )
        .reset_index(drop=True)
    )
    tx_geo = (
        institution_geo.loc[
            institution_geo["CTR_TY"].eq("TX1")
            & institution_geo["CTR_ID"].notna()
            & institution_geo["LATITUDE"].notna()
            & institution_geo["LONGITUDE"].notna()
        ]
        .drop_duplicates(subset=["CTR_ID"])
        .rename(
            columns={
                "CTR_ID": "TX_CTR_ID",
                "CTR_CD": "TX_CTR_CD",
                "GEO_SOURCE": "TX_CENTER_GEO_SOURCE",
            }
        )
        .reset_index(drop=True)
    )

    if opo_geo.empty or tx_geo.empty:
        opo_tx_center_distance = pd.DataFrame(
            {
                "DON_OPO_CTR_ID": pd.Series(dtype="Int64"),
                "DON_OPO_CTR_CD": pd.Series(dtype="string"),
                "TX_CTR_ID": pd.Series(dtype="Int64"),
                "TX_CTR_CD": pd.Series(dtype="string"),
                "DISTANCE_NM": pd.Series(dtype="Float64"),
                "WITHIN_250_NM": pd.Series(dtype="Int8"),
                "OPO_GEO_SOURCE": pd.Series(dtype="string"),
                "TX_CENTER_GEO_SOURCE": pd.Series(dtype="string"),
            }
        )
        opo_tx_center_count_250nm = opo_geo[["DON_OPO_CTR_ID", "DON_OPO_CTR_CD", "OPO_GEO_SOURCE"]].copy()
        opo_tx_center_count_250nm["TX_CENTER_COUNT_250NM"] = pd.Series(dtype="Int16")
        opo_tx_center_count_250nm["GEOCODED_TX_CENTER_COUNT"] = pd.Series(dtype="Int16")
    else:
        opo_lat = np.radians(opo_geo["LATITUDE"].astype(float).to_numpy())[:, None]
        opo_lon = np.radians(opo_geo["LONGITUDE"].astype(float).to_numpy())[:, None]
        tx_lat = np.radians(tx_geo["LATITUDE"].astype(float).to_numpy())[None, :]
        tx_lon = np.radians(tx_geo["LONGITUDE"].astype(float).to_numpy())[None, :]

        delta_lat = tx_lat - opo_lat
        delta_lon = tx_lon - opo_lon
        a = np.sin(delta_lat / 2.0) ** 2 + np.cos(opo_lat) * np.cos(tx_lat) * np.sin(delta_lon / 2.0) ** 2
        distance_nm = 2.0 * EARTH_RADIUS_NM * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        within_250_nm = distance_nm <= NAUTICAL_CIRCLE_RADIUS_NM

        opo_repeat = len(tx_geo)
        tx_repeat = len(opo_geo)
        opo_tx_center_distance = pd.DataFrame(
            {
                "DON_OPO_CTR_ID": pd.array(
                    np.repeat(opo_geo["DON_OPO_CTR_ID"].astype(np.int64).to_numpy(), opo_repeat), dtype="Int64"
                ),
                "DON_OPO_CTR_CD": pd.Series(
                    np.repeat(opo_geo["DON_OPO_CTR_CD"].astype(str).to_numpy(), opo_repeat), dtype="string"
                ),
                "TX_CTR_ID": pd.array(np.tile(tx_geo["TX_CTR_ID"].astype(np.int64).to_numpy(), tx_repeat), dtype="Int64"),
                "TX_CTR_CD": pd.Series(np.tile(tx_geo["TX_CTR_CD"].astype(str).to_numpy(), tx_repeat), dtype="string"),
                "DISTANCE_NM": pd.Series(distance_nm.reshape(-1), dtype="Float64"),
                "WITHIN_250_NM": pd.Series(within_250_nm.reshape(-1).astype(np.int8), dtype="Int8"),
                "OPO_GEO_SOURCE": pd.Series(
                    np.repeat(opo_geo["OPO_GEO_SOURCE"].astype(str).to_numpy(), opo_repeat), dtype="string"
                ),
                "TX_CENTER_GEO_SOURCE": pd.Series(
                    np.tile(tx_geo["TX_CENTER_GEO_SOURCE"].astype(str).to_numpy(), tx_repeat), dtype="string"
                ),
            }
        )
        opo_tx_center_count_250nm = opo_geo[["DON_OPO_CTR_ID", "DON_OPO_CTR_CD", "OPO_GEO_SOURCE"]].copy()
        opo_tx_center_count_250nm["TX_CENTER_COUNT_250NM"] = pd.Series(
            within_250_nm.sum(axis=1), dtype="Int16"
        )
        opo_tx_center_count_250nm["GEOCODED_TX_CENTER_COUNT"] = pd.Series(
            np.repeat(len(tx_geo), len(opo_geo)), dtype="Int16"
        )

    donor_opo_count_250nm = {
        int(don_opo_ctr_id): int(count)
        for don_opo_ctr_id, count in zip(
            opo_tx_center_count_250nm["DON_OPO_CTR_ID"].dropna(),
            opo_tx_center_count_250nm["TX_CENTER_COUNT_250NM"].dropna(),
        )
    }

    return OPOCenterGeoContext(
        institution_geo=institution_geo.sort_values(["CTR_TY", "CTR_CD"], na_position="last").reset_index(drop=True),
        opo_tx_center_distance=opo_tx_center_distance.sort_values(
            ["DON_OPO_CTR_ID", "TX_CTR_ID"], na_position="last"
        ).reset_index(drop=True),
        opo_tx_center_count_250nm=opo_tx_center_count_250nm.sort_values(
            ["DON_OPO_CTR_ID"], na_position="last"
        ).reset_index(drop=True),
        donor_opo_count_250nm=donor_opo_count_250nm,
    )


def add_donor_deceased_enrichment(
    chunk: pd.DataFrame,
    kdpi_reference: KDPIReference | None,
    donor_opo_count_250nm: dict[int, int] | None,
) -> pd.DataFrame:
    dcd = chunk["DON_NON_HR_BEAT"].astype("string").str.upper()
    dcd_ind = pd.Series(np.where(dcd.eq("Y").fillna(False), 1, 0), index=chunk.index, dtype="Int8")
    if kdpi_reference is None:
        chunk["DCD_IND"] = dcd_ind
        if donor_opo_count_250nm is not None:
            donor_opo_ctr_id = pd.to_numeric(chunk["DON_OPO_CTR_ID"], errors="coerce").astype("Int64")
            tx_center_count_250nm = donor_opo_ctr_id.map(donor_opo_count_250nm)
            chunk["TX_CENTER_COUNT_250NM"] = pd.array(tx_center_count_250nm, dtype="Int16")
        return chunk

    recov_year = pd.to_datetime(chunk["DON_RECOV_DT"], errors="coerce").dt.year
    recov_year_int = pd.Series(pd.array(recov_year, dtype="Int16"), index=chunk.index)

    def scale_series(mapping: dict[int, float]) -> pd.Series:
        return pd.to_numeric(recov_year.map(mapping), errors="coerce")

    hyp_scale = scale_series(kdpi_reference.hyp_scale)
    diab_scale = scale_series(kdpi_reference.diab_scale)
    kdri_scale = scale_series(kdpi_reference.kdri_scale)

    age = pd.to_numeric(chunk["DON_AGE"], errors="coerce")
    height = pd.to_numeric(chunk["DON_HGT_CM"], errors="coerce")
    weight = pd.to_numeric(chunk["DON_WGT_KG"], errors="coerce")
    htn = pd.to_numeric(chunk["DON_HTN"], errors="coerce")
    diab = pd.to_numeric(chunk["DON_HIST_DIAB"], errors="coerce")
    cause_of_death = pd.to_numeric(chunk["DON_CAD_DON_COD"], errors="coerce")
    donor_creat = pd.to_numeric(chunk["DON_CREAT"], errors="coerce")
    donor_final_creat = pd.to_numeric(chunk["DON_FINAL_SERUM_CREAT"], errors="coerce")
    donor_peak_creat = pd.to_numeric(chunk["DON_PEAK_SERUM_CREAT"], errors="coerce")

    creatinine = donor_creat.fillna(donor_final_creat).fillna(donor_peak_creat).clip(upper=8)
    creat_missing = donor_creat.isna() & donor_final_creat.isna() & donor_peak_creat.isna()

    race = chunk["DON_RACE_SRTR"].astype("string").str.upper()
    anti_hcv = chunk["DON_ANTI_HCV"].astype("string").str.upper()
    hcv_nat = chunk["DON_HCV_NAT"].astype("string").str.upper()
    hcv_status = pd.to_numeric(chunk["DON_HCV_STAT"], errors="coerce")

    old_year_mask = recov_year_int.between(2012, 2022, inclusive="both")
    new_year_mask = recov_year_int.between(2023, 2025, inclusive="both")

    def init_term(default=0.0) -> pd.Series:
        return pd.Series(default, index=chunk.index, dtype="Float64")

    old_x_age = init_term()
    old_x_age = old_x_age.mask(age > 50, 0.0107 * (age - 50))
    old_x_age = old_x_age.mask(age < 18, -0.0194 * (age - 18))
    old_xbeta_age = (old_x_age + 0.0128 * (age - 40)).where(age.notna())
    old_xbeta_height = (-0.0464 * ((height - 170) / 10)).where(height.notna())
    old_xbeta_weight = init_term().mask(weight < 80, -0.0199 * ((weight - 80) / 5)).where(weight.notna())

    old_xbeta_race = init_term()
    old_xbeta_race = old_xbeta_race.mask(race == "BLACK", 0.1790)
    old_xbeta_race = old_xbeta_race.mask(race.eq(""), pd.NA)

    old_xbeta_hyp = init_term()
    old_xbeta_hyp = old_xbeta_hyp.mask(htn == 1, 0.1260)
    old_xbeta_hyp = old_xbeta_hyp.mask(htn.isna(), 0.1260 * hyp_scale)

    old_xbeta_diab = pd.Series(pd.NA, index=chunk.index, dtype="Float64")
    old_xbeta_diab = old_xbeta_diab.mask(diab == 1, 0.0)
    old_xbeta_diab = old_xbeta_diab.mask(diab.isin([2, 3, 4, 5]), 0.1300)
    old_xbeta_diab = old_xbeta_diab.mask(diab.isna() | (diab == 998), 0.1300 * diab_scale)

    old_xbeta_cvd = init_term()
    old_xbeta_cvd = old_xbeta_cvd.mask(cause_of_death == 2, 0.0881)
    old_xbeta_cvd = old_xbeta_cvd.mask(cause_of_death.isna(), pd.NA)

    old_xbeta_creat = pd.Series(pd.NA, index=chunk.index, dtype="Float64")
    old_xbeta_creat = old_xbeta_creat.mask(creatinine > 1.5, 0.2200 * (creatinine - 1) - 0.2090 * (creatinine - 1.5))
    old_xbeta_creat = old_xbeta_creat.mask(creatinine.le(1.5), 0.2200 * (creatinine - 1))
    old_xbeta_creat = old_xbeta_creat.mask(creat_missing, 0.0)

    old_xbeta_hcv = init_term()
    hcv_positive = anti_hcv.eq("P") | hcv_nat.eq("P") | (hcv_status == 1)
    old_xbeta_hcv = old_xbeta_hcv.mask(hcv_positive, 0.24)

    old_xbeta_dcd = init_term()
    old_xbeta_dcd = old_xbeta_dcd.mask(dcd == "Y", 0.1330)
    old_xbeta_dcd = old_xbeta_dcd.mask(dcd.eq(""), pd.NA)

    new_x_age = init_term()
    new_x_age = new_x_age.mask(age > 50, 0.0067 * (age - 50))
    new_x_age = new_x_age.mask(age < 18, 0.0113 * (age - 18))
    new_xbeta_age = (new_x_age + 0.0092 * (age - 40)).where(age.notna())
    new_xbeta_height = (-0.0557 * ((height - 170) / 10)).where(height.notna())
    new_xbeta_weight = init_term().mask(weight < 80, -0.0333 * ((weight - 80) / 5)).where(weight.notna())

    new_xbeta_hyp = init_term()
    new_xbeta_hyp = new_xbeta_hyp.mask(htn == 1, 0.1106)
    new_xbeta_hyp = new_xbeta_hyp.mask(htn.isna(), 0.1106 * hyp_scale)

    new_xbeta_diab = pd.Series(pd.NA, index=chunk.index, dtype="Float64")
    new_xbeta_diab = new_xbeta_diab.mask(diab == 1, 0.0)
    new_xbeta_diab = new_xbeta_diab.mask(diab.isin([2, 3, 4, 5]), 0.2577)
    new_xbeta_diab = new_xbeta_diab.mask(diab.isna() | (diab == 998), 0.2577 * diab_scale)

    new_xbeta_cvd = init_term()
    new_xbeta_cvd = new_xbeta_cvd.mask(cause_of_death == 2, 0.0743)
    new_xbeta_cvd = new_xbeta_cvd.mask(cause_of_death.isna(), pd.NA)

    new_xbeta_creat = pd.Series(pd.NA, index=chunk.index, dtype="Float64")
    new_xbeta_creat = new_xbeta_creat.mask(creatinine > 1.5, 0.2128 * (creatinine - 1) - 0.2199 * (creatinine - 1.5))
    new_xbeta_creat = new_xbeta_creat.mask(creatinine.le(1.5), 0.2128 * (creatinine - 1))
    new_xbeta_creat = new_xbeta_creat.mask(creat_missing, 0.0)

    new_xbeta_dcd = init_term()
    new_xbeta_dcd = new_xbeta_dcd.mask(dcd == "Y", 0.1966)
    new_xbeta_dcd = new_xbeta_dcd.mask(dcd.eq(""), pd.NA)

    old_sum = pd.concat(
        [
            old_xbeta_age,
            old_xbeta_height,
            old_xbeta_weight,
            old_xbeta_race,
            old_xbeta_hyp,
            old_xbeta_diab,
            old_xbeta_cvd,
            old_xbeta_creat,
            old_xbeta_hcv,
            old_xbeta_dcd,
        ],
        axis=1,
    ).sum(axis=1, min_count=10)
    new_sum = pd.concat(
        [
            new_xbeta_age,
            new_xbeta_height,
            new_xbeta_weight,
            new_xbeta_hyp,
            new_xbeta_diab,
            new_xbeta_cvd,
            new_xbeta_creat,
            new_xbeta_dcd,
        ],
        axis=1,
    ).sum(axis=1, min_count=8)

    kdri_rao = pd.Series(pd.NA, index=chunk.index, dtype="Float64")
    kdri_rao.loc[old_year_mask] = np.exp(old_sum.loc[old_year_mask].astype(np.float64))
    kdri_rao.loc[new_year_mask] = np.exp(new_sum.loc[new_year_mask].astype(np.float64))

    kdri_med = (kdri_rao / kdri_scale).astype("Float64")

    kdpi = pd.Series(pd.NA, index=chunk.index, dtype="Int16")
    kdri_med_values = pd.to_numeric(kdri_med, errors="coerce")
    for year, upper_bounds in kdpi_reference.kdpi_upper_bounds.items():
        mask = recov_year_int.eq(year) & kdri_med_values.notna() & kdri_med_values.gt(0)
        if not mask.any():
            continue
        positions = np.searchsorted(upper_bounds, kdri_med_values.loc[mask].to_numpy(dtype=np.float64), side="left")
        kdpi.loc[mask] = pd.array(positions, dtype="Int16")

    kdpi_bin = pd.Series(pd.NA, index=chunk.index, dtype="string")
    kdpi_numeric = pd.to_numeric(kdpi.astype("Float64"), errors="coerce")
    kdpi_bin = kdpi_bin.mask(kdpi_numeric.between(0, 20, inclusive="both").fillna(False), "0-20")
    kdpi_bin = kdpi_bin.mask((kdpi_numeric.gt(20) & kdpi_numeric.le(40)).fillna(False), "20-40")
    kdpi_bin = kdpi_bin.mask((kdpi_numeric.gt(40) & kdpi_numeric.le(60)).fillna(False), "40-60")
    kdpi_bin = kdpi_bin.mask((kdpi_numeric.gt(60) & kdpi_numeric.le(80)).fillna(False), "60-80")
    kdpi_bin = kdpi_bin.mask((kdpi_numeric.gt(80) & kdpi_numeric.le(100)).fillna(False), "80-100")

    chunk["DCD_IND"] = dcd_ind
    chunk["KDRI_RAO"] = kdri_rao
    chunk["KDRI_MED"] = kdri_med
    chunk["KDPI"] = kdpi
    chunk["KDPI_BIN"] = kdpi_bin
    if donor_opo_count_250nm is not None:
        donor_opo_ctr_id = pd.to_numeric(chunk["DON_OPO_CTR_ID"], errors="coerce").astype("Int64")
        tx_center_count_250nm = donor_opo_ctr_id.map(donor_opo_count_250nm)
        chunk["TX_CENTER_COUNT_250NM"] = pd.array(tx_center_count_250nm, dtype="Int16")
    return chunk


def convert_source_file(
    source: SourceFile,
    output_dir: Path,
    chunk_rows: int,
    row_group_size: int,
    overwrite: bool,
    kdpi_reference: KDPIReference | None = None,
    donor_opo_count_250nm: dict[int, int] | None = None,
) -> TableStats:
    final_table_dir = output_dir / source.table_name
    temp_table_dir = output_dir / f"{source.table_name}.tmp"

    if final_table_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{final_table_dir} already exists.")
        shutil.rmtree(final_table_dir)

    if temp_table_dir.exists():
        shutil.rmtree(temp_table_dir)
    temp_table_dir.mkdir(parents=True, exist_ok=True)

    _, meta = pyreadstat.read_sas7bdat(str(source.path), metadataonly=True)
    date_cols, time_cols, int_cols, float_cols = classify_columns(meta)

    stats = TableStats(
        table_name=source.table_name,
        source_file=str(source.path),
        rows_in_source=meta.number_rows,
        columns_in_source=len(meta.column_names),
        started_at_utc=utc_now(),
    )

    start = time.perf_counter()
    for part_number, (chunk, _) in enumerate(
        pyreadstat.read_file_in_chunks(
            pyreadstat.read_sas7bdat,
            str(source.path),
            chunksize=chunk_rows,
        ),
        start=1,
    ):
        chunk = normalize_chunk(chunk, date_cols, time_cols, int_cols, float_cols)
        if source.table_name == "donor_deceased":
            chunk = add_donor_deceased_enrichment(
                chunk,
                kdpi_reference=kdpi_reference,
                donor_opo_count_250nm=donor_opo_count_250nm,
            )
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        part_path = temp_table_dir / f"part-{part_number:05d}.parquet"
        pq.write_table(
            table,
            part_path,
            compression="zstd",
            use_dictionary=True,
            row_group_size=min(row_group_size, len(chunk)),
        )
        stats.parts_written = part_number
        stats.rows_written += len(chunk)
        if part_number % 10 == 0:
            print(
                f"[progress] table={source.table_name} parts={part_number} rows_written={stats.rows_written}",
                flush=True,
            )

    stats.elapsed_seconds = round(time.perf_counter() - start, 2)
    stats.finished_at_utc = utc_now()
    temp_table_dir.rename(final_table_dir)
    return stats


def convert_catalog_file(
    source: SourceFile,
    output_dir: Path,
    overwrite: bool,
) -> TableStats:
    final_table_dir = output_dir / source.table_name
    temp_table_dir = output_dir / f"{source.table_name}.tmp"

    if final_table_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{final_table_dir} already exists.")
        shutil.rmtree(final_table_dir)

    if temp_table_dir.exists():
        shutil.rmtree(temp_table_dir)
    temp_table_dir.mkdir(parents=True, exist_ok=True)

    _, meta = pyreadstat.read_sas7bcat(str(source.path))
    frame = build_catalog_frame(meta)

    stats = TableStats(
        table_name=source.table_name,
        source_file=str(source.path),
        rows_in_source=len(frame),
        columns_in_source=len(frame.columns),
        started_at_utc=utc_now(),
    )

    start = time.perf_counter()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    part_path = temp_table_dir / "part-00001.parquet"
    pq.write_table(table, part_path, compression="zstd", use_dictionary=True)
    stats.parts_written = 1
    stats.rows_written = len(frame)
    stats.elapsed_seconds = round(time.perf_counter() - start, 2)
    stats.finished_at_utc = utc_now()
    temp_table_dir.rename(final_table_dir)
    return stats


def write_manifest(manifest_path: Path, stats: list[TableStats]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "built_at_utc": utc_now(),
        "tables": [asdict(item) for item in stats],
    }
    manifest_path.write_text(json.dumps(payload, indent=2))


def refresh_manifest_from_duckdb(
    manifest_path: Path,
    database_path: Path,
    output_dir: Path,
    source_files: dict[str, SourceFile],
) -> None:
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT table_name, row_count, column_count
            FROM saf_data.table_manifest
            ORDER BY table_name;
            """
        ).fetchall()
    finally:
        con.close()

    payload = {"built_at_utc": utc_now(), "tables": []}
    for table_name, row_count, column_count in rows:
        table_dir = output_dir / table_name
        payload["tables"].append(
            {
                "table_name": table_name,
                "source_file": str(source_files.get(table_name).path) if table_name in source_files else None,
                "rows_in_source": int(row_count),
                "columns_in_source": int(column_count),
                "parts_written": len(list(table_dir.glob("*.parquet"))) if table_dir.exists() else 0,
                "rows_written": int(row_count),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def build_duckdb(
    database_path: Path,
    parquet_root: Path,
    source_files: list[SourceFile],
    threads: int,
) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path))
    try:
        con.execute(f"PRAGMA threads={max(1, threads)};")
        con.execute("CREATE SCHEMA IF NOT EXISTS saf_data;")
        con.execute("CREATE SCHEMA IF NOT EXISTS saf_raw;")

        for source in source_files:
            parquet_glob = str((parquet_root / source.table_name / "*.parquet").resolve()).replace("'", "''")
            raw_view = f"saf_raw.{source.table_name}"
            table_name = f"saf_data.{source.table_name}"

            con.execute(
                f"""
                CREATE OR REPLACE VIEW {raw_view} AS
                SELECT *
                FROM read_parquet('{parquet_glob}', union_by_name=true);
                """
            )
            con.execute(f"DROP TABLE IF EXISTS {table_name};")
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {raw_view};")
            con.execute(f"ANALYZE {table_name};")

            columns = {
                row[0]
                for row in con.execute(
                    """
                    SELECT column_name
                    FROM duckdb_columns()
                    WHERE schema_name = 'saf_data' AND table_name = ?
                    """,
                    [source.table_name],
                ).fetchall()
            }
            for column in INDEX_CANDIDATE_COLUMNS:
                if column in columns:
                    con.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{source.table_name}_{column.lower()}
                            ON {table_name}({column});
                            """
                        )
            if source.table_name == "donor_deceased":
                for column in ["DCD_IND", "KDPI", "KDPI_BIN", "DON_OPO_CTR_ID", "TX_CENTER_COUNT_250NM"]:
                    if column in columns:
                        con.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS idx_{source.table_name}_{column.lower()}
                            ON {table_name}({column});
                            """
                        )
            if source.source_type == "sas7bcat":
                for column in ["FORMAT_NAME", "RAW_VALUE_TEXT", "RAW_VALUE_NUM", "RAW_VALUE_STRING", "VALUE_TYPE"]:
                    if column in columns:
                        con.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS idx_{source.table_name}_{column.lower()}
                            ON {table_name}({column});
                            """
                        )

        all_tables = {
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM duckdb_tables()
                WHERE schema_name = 'saf_data'
                """
            ).fetchall()
        }

        if "institution" in all_tables:
            institution_frame = con.execute(
                """
                SELECT
                    CTR_ID,
                    CTR_CD,
                    CTR_TY,
                    REGION,
                    ENTIRE_NAME,
                    NAME_PART1,
                    NAME_PART2,
                    PRIMARY_CITY,
                    PRIMARY_STATE,
                    PRIMARY_ZIP,
                    PROVIDER_NUM,
                    PRIMARY_CTRY,
                    OPTN_MBR,
                    ESRD_REGION
                FROM saf_data.institution
                """
            ).fetchdf()
            geo_context = build_opo_tx_center_geo_context_from_df(institution_frame)

            for table_name, frame, indexes in [
                (
                    "saf_data.institution_geo",
                    geo_context.institution_geo,
                    [
                        ("idx_institution_geo_ctr_id", "CTR_ID"),
                        ("idx_institution_geo_code_type", "CTR_CD, CTR_TY"),
                        ("idx_institution_geo_geo_source", "GEO_SOURCE"),
                    ],
                ),
                (
                    "saf_data.opo_tx_center_distance",
                    geo_context.opo_tx_center_distance,
                    [
                        ("idx_opo_tx_center_distance_opo", "DON_OPO_CTR_ID"),
                        ("idx_opo_tx_center_distance_tx", "TX_CTR_ID"),
                        ("idx_opo_tx_center_distance_within", "WITHIN_250_NM"),
                    ],
                ),
                (
                    "saf_data.opo_tx_center_count_250nm",
                    geo_context.opo_tx_center_count_250nm,
                    [
                        ("idx_opo_tx_center_count_250nm_opo", "DON_OPO_CTR_ID"),
                        ("idx_opo_tx_center_count_250nm_opo_cd", "DON_OPO_CTR_CD"),
                    ],
                ),
            ]:
                temp_view_name = table_name.replace(".", "_") + "_frame"
                con.register(temp_view_name, frame)
                con.execute(f"DROP TABLE IF EXISTS {table_name};")
                con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {temp_view_name};")
                con.unregister(temp_view_name)
                con.execute(f"ANALYZE {table_name};")
                for index_name, index_expr in indexes:
                    con.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({index_expr});")

            if "donor_deceased" in all_tables:
                con.execute("ALTER TABLE saf_data.donor_deceased ADD COLUMN IF NOT EXISTS TX_CENTER_COUNT_250NM INTEGER;")
                con.execute("UPDATE saf_data.donor_deceased SET TX_CENTER_COUNT_250NM = NULL;")
                con.execute(
                    """
                    UPDATE saf_data.donor_deceased AS d
                    SET TX_CENTER_COUNT_250NM = c.TX_CENTER_COUNT_250NM
                    FROM saf_data.opo_tx_center_count_250nm AS c
                    WHERE CAST(d.DON_OPO_CTR_ID AS BIGINT) = c.DON_OPO_CTR_ID;
                    """
                )
                con.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_donor_deceased_tx_center_count_250nm
                    ON saf_data.donor_deceased(TX_CENTER_COUNT_250NM);
                    """
                )
                con.execute("ANALYZE saf_data.donor_deceased;")

        con.execute("DROP TABLE IF EXISTS saf_data.table_manifest;")
        con.execute(
            """
            CREATE TABLE saf_data.table_manifest (
                table_name VARCHAR,
                row_count BIGINT,
                column_count BIGINT
            );
            """
        )
        manifest_rows = []
        all_tables = [
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM duckdb_tables()
                WHERE schema_name = 'saf_data' AND table_name <> 'table_manifest'
                ORDER BY table_name;
                """
            ).fetchall()
        ]
        for table in all_tables:
            row_count = con.execute(f"SELECT COUNT(*) FROM saf_data.{table}").fetchone()[0]
            column_count = con.execute(
                """
                SELECT COUNT(*)
                FROM duckdb_columns()
                WHERE schema_name = 'saf_data' AND table_name = ?
                """,
                [table],
            ).fetchone()[0]
            manifest_rows.append((table, row_count, column_count))
        for row in manifest_rows:
            con.execute("INSERT INTO saf_data.table_manifest VALUES (?, ?, ?)", row)

        con.execute("CHECKPOINT;")
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SAF .sas7bdat tables and .sas7bcat catalogs into Parquet plus a DuckDB SAF schema."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/saf files"),
        help="Directory containing SAF .sas7bdat tables and optional .sas7bcat catalogs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("warehouse/saf/parquet"),
        help="Destination directory for Parquet output.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("warehouse/saf/saf.duckdb"),
        help="DuckDB database file to create/update.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("warehouse/saf/build_manifest.json"),
        help="JSON manifest for the SAF build.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=100_000,
        help="Rows to stream per chunk while converting to Parquet.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=50_000,
        help="Parquet row-group size per output part.",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        help="Optional subset of source stems to build, for example tx_ki txf_ki donor_disposition formats.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="DuckDB thread count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild existing Parquet table directories.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip Parquet conversion and only rebuild DuckDB from existing Parquet.",
    )
    parser.add_argument(
        "--skip-duckdb",
        action="store_true",
        help="Skip DuckDB schema creation.",
    )
    parser.add_argument(
        "--kdpi-do-file",
        type=Path,
        default=default_kdpi_do_file(),
        help="Stata do-file containing the KDPI/KDRI mapping used to enrich donor_deceased.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_tables = {name.lower() for name in args.tables} if args.tables else None
    source_files = discover_source_files(args.source_dir, selected_tables)
    all_source_files = {item.table_name: item for item in discover_source_files(args.source_dir, None)}

    built_stats: list[TableStats] = []
    kdpi_reference = None
    donor_opo_count_250nm = None
    if not args.skip_convert and any(source.table_name == "donor_deceased" for source in source_files):
        kdpi_reference = load_kdpi_reference(args.kdpi_do_file)
        institution_source = all_source_files.get("institution")
        if institution_source is not None:
            donor_opo_count_250nm = build_opo_tx_center_geo_context_from_df(
                load_institution_frame(institution_source.path)
            ).donor_opo_count_250nm
        else:
            print(
                "[warn] institution source file not found; donor_deceased Parquet will omit TX_CENTER_COUNT_250NM until DuckDB refresh.",
                flush=True,
            )
    if not args.skip_convert:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for source in source_files:
            final_table_dir = args.output_dir / source.table_name
            if final_table_dir.exists() and not args.overwrite:
                print(f"[skip] table={source.table_name} already exists at {final_table_dir}", flush=True)
                continue
            print(f"[convert] table={source.table_name} file={source.path}", flush=True)
            if source.source_type == "sas7bcat":
                stats = convert_catalog_file(
                    source=source,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                )
            else:
                stats = convert_source_file(
                    source=source,
                    output_dir=args.output_dir,
                    chunk_rows=args.chunk_rows,
                    row_group_size=args.row_group_size,
                    overwrite=args.overwrite,
                    kdpi_reference=kdpi_reference,
                    donor_opo_count_250nm=donor_opo_count_250nm,
                )
            built_stats.append(stats)
            print(
                f"[done] table={source.table_name} rows={stats.rows_in_source} parts={stats.parts_written} elapsed_seconds={stats.elapsed_seconds}",
                flush=True,
            )
        write_manifest(args.manifest, built_stats)

    if not args.skip_duckdb:
        print(f"[duckdb] building SAF schema at {args.database}", flush=True)
        build_duckdb(
            database_path=args.database,
            parquet_root=args.output_dir,
            source_files=source_files,
            threads=args.threads,
        )
        refresh_manifest_from_duckdb(
            manifest_path=args.manifest,
            database_path=args.database,
            output_dir=args.output_dir,
            source_files=all_source_files,
        )
        print("[duckdb] done", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
