"""Airflow DAGs for the Market Signal ELT pipeline (Airflow 2.10).

Ingestion DAGs run on their own schedules and each publishes an Airflow Dataset when its
clean table is refreshed and has passed the quality gate. signals_refresh is scheduled on
those Datasets (data-aware scheduling), so any upstream update re-runs mart, retrieval,
signals and evaluation automatically.

    prices_15min      every 15 minutes on weekdays      -> clean_prices
    news_llm_30min    every 30 minutes, Llama + Chroma  -> clean_news_events
    earnings_event    daily sensor on earnings calendar -> clean_earnings, clean_filings
    macro_daily       daily                            -> clean_macro
    signals_refresh   when ANY of the Datasets updates  -> mart_* tables
"""
import os
from datetime import date, datetime, timedelta

from airflow import DAG
from airflow.datasets import Dataset
from airflow.operators.python import ExternalPythonOperator, PythonOperator
from airflow.sensors.python import PythonSensor

# The pipeline's own dependencies (Chroma, Great Expectations, pandas 2) live in a separate
# virtualenv inside the Airflow image, so they never clash with Airflow's pinned packages.
# Each task runs there through ExternalPythonOperator. Without MSP_PYTHON it runs in-process.
MSP_PYTHON = os.getenv("MSP_PYTHON")

PRICES = Dataset("warehouse://clean_prices")
NEWS = Dataset("warehouse://clean_news_events")
EARNINGS = Dataset("warehouse://clean_earnings")
MACRO = Dataset("warehouse://clean_macro")

DEFAULTS = {"owner": "data-eng", "retries": 1, "retry_delay": timedelta(minutes=3)}
# max_active_runs=1 on the scheduled DAGs: a run must finish before the next starts
START = datetime(2026, 1, 1)


def _call(step: str, sources=None, tables=None):
    """Runs inside the pipeline virtualenv. Keep imports inside: the function is shipped as source.
    No **kwargs, so Airflow does not inject its context into the call."""
    from msp import pipeline

    kw = {k: v for k, v in (("sources", sources), ("tables", tables)) if v is not None}
    pipeline.STEPS[step](**kw)


def PipelineTask(task_id: str, step: str, outlets=None, **kw):  # noqa: N802 - reads like an operator
    args = dict(task_id=task_id, python_callable=_call, op_kwargs={"step": step, **kw}, outlets=outlets or [])
    if MSP_PYTHON:
        return ExternalPythonOperator(python=MSP_PYTHON, expect_airflow=False, **args)
    return PythonOperator(**args)


def ingest_tasks(dag, sources, tables, outlet):
    extract = PipelineTask("extract", "extract", sources=sources)
    load = PipelineTask("load_raw", "load_raw", sources=sources)
    clean = PipelineTask("clean", "clean", sources=sources)
    gate = PipelineTask("quality_gate", "quality_gate", outlets=[] if outlet is None else [outlet], tables=tables)
    extract >> load >> clean >> gate
    return gate


with DAG("prices_15min", schedule="*/15 * * * 1-5", start_date=START, catchup=False, max_active_runs=1,
         default_args=DEFAULTS, tags=["ingest"],
         doc_md="Prices for 10 stocks and 4 metals every 15 minutes on weekdays.") as d1:
    ingest_tasks(d1, ["prices"], ["clean_prices"], PRICES)

with DAG("news_llm_30min", schedule="*/30 * * * *", start_date=START, catchup=False, max_active_runs=1,
         default_args=DEFAULTS, tags=["ingest", "ai"],
         doc_md="News every 30 minutes; Llama structures each new story, Chroma dedups and indexes.") as d2:
    gate = ingest_tasks(d2, ["news"], ["clean_news"], None)
    enrich = PipelineTask("enrich_news_llm_chroma", "enrich", outlets=[NEWS])
    gate >> enrich


def _earnings_due(**_):
    """True when any company in the universe reported in the last 2 days or reports today.
    Reads the warehouse with plain SQLAlchemy so it runs in Airflow's own environment."""
    from sqlalchemy import create_engine, inspect, text

    eng = create_engine(os.getenv("DATABASE_URL", "sqlite:////opt/airflow/project/data/warehouse.db"))
    if not inspect(eng).has_table("clean_earnings"):
        return True
    today = date.today()
    with eng.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM clean_earnings WHERE report_date BETWEEN :a AND :b"),
                         {"a": (today - timedelta(days=2)).isoformat(), "b": today.isoformat()}).scalar()
    return n > 0


with DAG("earnings_event", schedule="0 12 * * *", start_date=START, catchup=False, default_args=DEFAULTS,
         tags=["ingest", "event"], doc_md="Sensor waits for an earnings date, then pulls earnings and SEC filings.") as d3:
    wait = PythonSensor(task_id="wait_for_earnings_release", python_callable=_earnings_due, mode="reschedule",
                        poke_interval=3600, timeout=60 * 60 * 10, soft_fail=True)
    gate = ingest_tasks(d3, ["earnings", "filings"], ["clean_earnings", "clean_filings"], EARNINGS)
    enrich = PipelineTask("enrich_filings", "enrich", outlets=[NEWS])
    wait >> d3.get_task("extract")
    gate >> enrich

with DAG("macro_daily", schedule="0 13 * * 1-5", start_date=START, catchup=False, default_args=DEFAULTS,
         tags=["ingest"], doc_md="FRED rates, dollar and CPI used as metal drivers.") as d4:
    ingest_tasks(d4, ["macro"], ["clean_macro"], MACRO)

with DAG("signals_refresh", schedule=(PRICES | NEWS | EARNINGS | MACRO), start_date=START, catchup=False,
         max_active_runs=1, default_args=DEFAULTS, tags=["mart", "ai"],
         doc_md="Data-aware: runs whenever any upstream Dataset is updated.") as d5:
    mart = PipelineTask("build_mart_sql", "build_mart")
    retrieve = PipelineTask("retrieve_evidence_chroma", "retrieve")
    signals = PipelineTask("signal_synthesis_gemini", "signals")
    evaluate = PipelineTask("evaluate_judge_backtest", "evaluate")
    mart >> retrieve >> signals >> evaluate

with DAG("full_refresh_manual", schedule=None, start_date=START, catchup=False, default_args=DEFAULTS,
         tags=["manual"], doc_md="Trigger by hand: every source end to end. Use for the first load.") as d6:
    prev = None
    for step in ["extract", "load_raw", "clean", "quality_gate", "enrich", "build_mart", "retrieve", "signals",
                 "evaluate"]:
        t = PipelineTask(step, step)
        if prev:
            prev >> t
        prev = t
