"""Pipeline steps and command line.

The same step functions are called by the Airflow DAGs (dags/market_signal_dags.py) and by
this CLI, so the demo runs with or without Airflow.

    python -m msp.pipeline run                 # all steps, steps 2-10 of the roadmap
    python -m msp.pipeline run --sample        # force the bundled sample (no network)
    python -m msp.pipeline step build_mart     # one step
    python -m msp.pipeline inject-bad-data     # plant bad rows so the quality gate fails (demo)
"""
import argparse
import json
import logging
import sys
import time

from . import config, db

log = logging.getLogger("msp")


def _timed(step: str, fn, *args, **kwargs):
    started = db.now_utc()
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        db.log_run(step, "success", _rows(out), {**(out if isinstance(out, dict) else {}),
                                                 "seconds": round(time.time() - t0, 1)}, started)
        log.info("%-16s ok   %.1fs  %s", step, time.time() - t0, json.dumps(out, default=str)[:300])
        return out
    except Exception as e:
        db.log_run(step, "failed", 0, {"error": str(e)[:2000], "seconds": round(time.time() - t0, 1)}, started)
        log.error("%-16s FAILED %s", step, str(e)[:500])
        raise


def _rows(out) -> int:
    if not isinstance(out, dict):
        return 0
    for k in ("rows", "rows_out", "unique_stories", "evidence_rows", "assets", "signals_judged", "checks"):
        if k in out:
            return int(out[k])
    return sum(int(v.get("rows") or v.get("rows_out") or v.get("records") or 0)
               for v in out.get("sources", out).values() if isinstance(v, dict))


# --------------------------------------------------------------------------- steps
def step_extract(sources=None):
    from .extract import extract

    return _timed("extract", extract, sources)


def step_load_raw(sources=None):
    from .load_raw import load_raw

    return _timed("load_raw", load_raw, sources)


def step_clean(sources=None):
    from .clean import clean

    return _timed("clean", clean, sources)


def step_quality_gate(tables=None):
    from .quality import quality_gate

    return _timed("quality_gate", quality_gate, tables)


def step_enrich():
    from .enrich import enrich

    return _timed("enrich", enrich)


def build_mart():
    stmts = db.run_sql_file(config.SQL_DIR / "mart_features.sql")
    n = int(db.read_sql("SELECT COUNT(*) AS n FROM mart_features_daily")["n"].iloc[0])
    return {"rows": n, "sql_statements": len(stmts), "assets": int(
        db.read_sql("SELECT COUNT(*) AS n FROM mart_asset_latest")["n"].iloc[0])}


def step_build_mart():
    return _timed("build_mart", build_mart)


def step_retrieve():
    from .retrieve import retrieve

    return _timed("retrieve", retrieve)


def step_signals():
    from .signals import synthesize

    return _timed("signals", synthesize)


def step_evaluate():
    from .evaluate import evaluate

    return _timed("evaluate", evaluate)


STEPS = {"extract": step_extract, "load_raw": step_load_raw, "clean": step_clean,
         "quality_gate": step_quality_gate, "enrich": step_enrich, "build_mart": step_build_mart,
         "retrieve": step_retrieve, "signals": step_signals, "evaluate": step_evaluate}


def run_all():
    for name, fn in STEPS.items():
        fn()


def inject_bad_data():
    """Demo helper: copy 3 rows into the latest raw price batch with a negative price and a
    future date, as if the vendor sent a bad file. The next quality gate run fails."""
    import pandas as pd

    raw = db.read_sql("SELECT * FROM raw_prices WHERE _batch_id = (SELECT MAX(_batch_id) FROM raw_prices) LIMIT 3")
    raw.loc[0, "close"] = "-12.5"
    raw.loc[1, "date"] = "2099-01-02"
    raw.loc[2, "close"] = "-1"
    raw.loc[2, "date"] = "2099-01-03"
    db.write_df(raw, "raw_prices", if_exists="append")
    return {"injected_rows": len(raw), "symbols": raw["symbol"].tolist()}


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    class _NoGxTmp(logging.Filter):
        def filter(self, record):
            return "ephemeral docs site" not in record.getMessage()

    for h in logging.getLogger().handlers:
        h.addFilter(_NoGxTmp())
    for noisy in ("httpx", "chromadb", "great_expectations", "yfinance", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="msp.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sample", action="store_true", help="use the bundled synthetic sample only")
    s = sub.add_parser("step")
    s.add_argument("name", choices=list(STEPS))
    sub.add_parser("inject-bad-data")
    a = p.parse_args(argv)
    if getattr(a, "sample", False):
        config.DATA_MODE = "sample"
    try:
        if a.cmd == "run":
            run_all()
        elif a.cmd == "step":
            STEPS[a.name]()
        else:
            print(json.dumps(inject_bad_data()))
            print("Bad rows injected. Now run: python -m msp.pipeline step clean && "
                  "python -m msp.pipeline step quality_gate")
    except Exception:  # noqa: BLE001
        sys.exit(1)


if __name__ == "__main__":
    main()
