"""Warehouse access. Same code runs on Postgres (Docker) and SQLite (quick demo).

Layers are table prefixes so the SQL is identical on both engines:
raw_*   untouched copy of each source
clean_* typed, deduplicated, adjusted
mart_*  joined analysis features, signals, evaluation
dq_*    data quality results
ops_*   pipeline run log
"""
import json
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from . import config

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(config.DATABASE_URL, future=True)
    return _engine


def dialect() -> str:
    return engine().dialect.name


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def table_exists(name: str) -> bool:
    return inspect(engine()).has_table(name)


def write_df(df: pd.DataFrame, table: str, if_exists: str = "replace") -> int:
    """Write a dataframe with SQL INSERTs (pandas to_sql)."""
    df.to_sql(table, engine(), if_exists=if_exists, index=False, chunksize=1000, method="multi")
    return len(df)


def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def execute(sql: str, params: dict | None = None) -> None:
    with engine().begin() as conn:
        conn.execute(text(sql), params or {})


def run_sql_file(path) -> list[str]:
    """Run a .sql file. Statements are separated by a line containing only ';'."""
    body = open(path, encoding="utf-8").read()
    statements = [s.strip() for s in body.split("\n;\n") if s.strip() and not _only_comments(s)]
    with engine().begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt.rstrip(";")))
    return statements


def _only_comments(block: str) -> bool:
    return all(not ln.strip() or ln.strip().startswith("--") for ln in block.splitlines())


def log_run(step: str, status: str, rows: int = 0, detail: dict | str | None = None,
            started: str | None = None) -> None:
    rec = pd.DataFrame([{
        "step": step,
        "status": status,
        "rows_out": int(rows),
        "detail": json.dumps(detail) if isinstance(detail, (dict, list)) else (detail or ""),
        "started_at": started or now_utc(),
        "finished_at": now_utc(),
    }])
    write_df(rec, "ops_pipeline_runs", if_exists="append")
