from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def duckdb_table_exists(
    con: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> bool:
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
        LIMIT 1
        """,
        [schema_name, table_name],
    ).fetchone()
    return row is not None


def duckdb_relation_columns(
    con: duckdb.DuckDBPyConnection,
    relation_sql: str,
) -> list[str]:
    rows = con.execute(f"DESCRIBE {relation_sql}").fetchall()
    return [row[0] for row in rows]


def write_parquet(dataframe: pd.DataFrame, output_path: Path, table_name: str = "frame") -> None:
    ensure_parent(output_path)
    con = duckdb.connect()
    con.register(table_name, dataframe)
    escaped_output_path = str(output_path).replace("'", "''")
    con.execute(f"COPY {table_name} TO '{escaped_output_path}' (FORMAT PARQUET)")
    con.unregister(table_name)
    con.close()
