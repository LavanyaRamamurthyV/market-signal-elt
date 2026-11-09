"""Step 3 - Load raw: land an untouched copy of each landing file in the warehouse.

Every value is stored as text, exactly as received. Three lineage columns are added:
_loaded_at, _source (live or sample) and _batch_id. Postgres loads use COPY; SQLite uses
batched INSERTs. Raw tables are append-only; the three most recent batches are kept.
"""
import csv
import io
import json

import pandas as pd

from . import config, db

KEEP_BATCHES = 3


def latest_landing(source: str):
    files = sorted((config.LANDING_DIR / source).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no landing file for {source}; run extract first")
    return files[-1]


def _copy_postgres(df: pd.DataFrame, table: str) -> None:
    if not db.table_exists(table):
        df.head(0).to_sql(table, db.engine(), index=False)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
    buf.seek(0)
    raw = db.engine().raw_connection()
    try:
        cols = ",".join(f'"{c}"' for c in df.columns)
        raw.cursor().copy_expert(f'COPY "{table}" ({cols}) FROM STDIN WITH CSV', buf)
        raw.commit()
    finally:
        raw.close()


def load_raw(sources: list[str] | None = None, batch_id: str | None = None) -> dict:
    summary = {}
    for src in sources or config.SOURCES:
        path = (config.LANDING_DIR / src / f"{batch_id}.json") if batch_id else latest_landing(src)
        payload = json.loads(path.read_text())
        df = pd.DataFrame(payload["records"]).drop(columns=["data_origin"], errors="ignore")  # kept in _source
        df = df.astype(object).where(df.notna(), None)
        df = df.map(lambda v: None if v is None else str(v))  # untouched: everything as text
        df["_loaded_at"] = db.now_utc()
        df["_source"] = payload["origin"]
        df["_batch_id"] = payload["batch_id"]
        table = f"raw_{src}"
        if db.table_exists(table):
            existing = set(db.read_sql(f'SELECT * FROM "{table}" LIMIT 0').columns)
            if set(df.columns) != existing:  # schema drift between sample and live: rebuild raw
                db.execute(f'DROP TABLE "{table}"')
            else:
                db.execute(f'DELETE FROM "{table}" WHERE _batch_id = :b', {"b": payload["batch_id"]})
        if db.dialect() == "postgresql":
            _copy_postgres(df, table)
        else:
            db.write_df(df, table, if_exists="append")
        batches = db.read_sql(f'SELECT DISTINCT _batch_id FROM "{table}" ORDER BY _batch_id DESC')["_batch_id"].tolist()
        for old in batches[KEEP_BATCHES:]:
            db.execute(f'DELETE FROM "{table}" WHERE _batch_id = :b', {"b": old})
        summary[src] = {"table": table, "rows": len(df), "origin": payload["origin"], "batch_id": payload["batch_id"]}
    return summary
