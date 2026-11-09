# Market Signal ELT

**Purpose:** turn raw market, earnings and news data into explainable, auto-refreshing Buy / Hold / Sell research signals for 10 large US stocks and 4 metals.

An ELT pipeline with AI built into the steps rather than bolted on as a chatbot. Airflow orchestrates it, Postgres stores it, SQL builds the features, Python cleans and transforms, Great Expectations gates it, Llama structures the news, Chroma deduplicates and retrieves, Gemini decides and judges, and Streamlit shows the result.

> Research signals only, not investment advice.

## Roadmap

| # | Step (what it means) | Tool | Type | SQL / Python | Data lives in |
|---|---|---|---|---|---|
| Start | Free market data APIs | Yahoo Finance, Finnhub, SEC EDGAR, FRED | Source | – | External |
| 1 | Orchestrate and trigger (schedule every pipeline run) | Airflow 2.10 in Docker | Orchestration | Python DAGs | Airflow metadata DB |
| 2 | Extract (pull data from source APIs) | `yfinance`, `requests` | ETL: Extract | Python | `data/landing/<source>/<batch>.json` |
| 3 | Load raw (land an untouched copy in the warehouse) | Postgres (COPY) / SQLite (INSERT) | ETL: Load | SQL | `raw_*` + `_loaded_at`, `_source`, `_batch_id` |
| 4 | Clean (fix types, duplicates and gaps) | Pandas | Transform | Python | `clean_*` + `adj_close`, `ret_1d`, `is_trading_day` |
| 5 | Quality gate (block bad data before it moves on) | Great Expectations + SQL checks | Validation | Python + SQL | `dq_results`, `data/reports/quality_report.html` |
| 6 | News and filings enrichment (turn text into structured events) | Llama via Ollama + Chroma | AI: LLM extraction | Python | `clean_news_events`, Chroma `news_filings` |
| 7 | Build mart (join sources and compute features) | Postgres | Transform + aggregate | SQL (CTEs, window functions) | `mart_features_daily`, `mart_asset_latest` |
| 8 | Retrieve evidence (supporting news and past analogs) | Chroma + SQL | AI: RAG retrieval | Python + SQL | `mart_evidence`, `mart_analogs` |
| 9 | Signal synthesis (decide Buy, Hold or Sell) | Gemini + factor rules | AI: LLM reasoning | Python | `mart_signals`, `mart_decision_trace` |
| 10 | Evaluate (prove signals are backed and accurate) | Gemini as judge + SQL backtest | AI eval + backtest | Python + SQL | `mart_eval_results`, `mart_backtest_*` |
| 11 | Auto-refresh (re-run when new data lands) | Airflow Datasets | Orchestration | Python | all `mart_*` |
| End | Market signal dashboard | Streamlit | Product | – | reads `mart_*`, `dq_*`, `ops_*` |

**Uses:** daily pre-market research brief, earnings-season monitoring, explainable signals for risk and finance teams, and a template for any governed AI-on-data pipeline.

## Run it

### Option A: quick demo in GitHub Codespaces (no Docker, about 5 minutes)

1. On the repo page click **Code → Codespaces → Create codespace on main**. Setup installs packages and builds the database from the bundled sample.
2. Optional but recommended: open `.env` and paste your keys (`GEMINI_API_KEY`, `FINNHUB_API_KEY`). Never commit `.env`.
3. Load live data (falls back to the sample per source if an API fails):
   ```bash
   python -m msp.pipeline run
   ```
4. Start the dashboard:
   ```bash
   python -m streamlit run app/ui.py
   ```
   Click **Open in Browser**, or open the Ports tab and the globe icon on port 8501.

### Option B: full stack with Airflow, Postgres and Ollama

```bash
cp .env.example .env            # add keys; set AIRFLOW_UID=$(id -u)
docker compose up -d --build    # first build and model download take 10-15 minutes
```

| Service | URL | Notes |
|---|---|---|
| Dashboard | http://localhost:8501 | Streamlit |
| Airflow | http://localhost:8080 | login admin / admin. Trigger `full_refresh_manual` once for the first load |
| Postgres | localhost:5432 | user `msp`, password `msp`, databases `warehouse` and `airflow` |
| Ollama | localhost:11434 | `llama3.2:3b` and `nomic-embed-text` pulled by `ollama-pull` |

Needs about 8 GB of RAM for Ollama. In Codespaces choose a 4-core machine or larger.

### Commands

```bash
python -m msp.pipeline run              # all steps (live with sample fallback)
python -m msp.pipeline run --sample     # bundled sample only, no network
python -m msp.pipeline step build_mart  # one step: extract, load_raw, clean, quality_gate, enrich,
                                        #   build_mart, retrieve, signals, evaluate
python -m msp.pipeline inject-bad-data  # demo: plant bad rows so the quality gate fails
pytest -q                               # 12 tests, about 70 seconds
```

## Orchestration (dags/market_signal_dags.py)

| DAG | Schedule | Publishes Dataset |
|---|---|---|
| `prices_daily` | 21:15 UTC weekdays (after US close) | `clean_prices` |
| `news_hourly` | hourly, then Llama + Chroma enrichment | `clean_news_events` |
| `earnings_event` | daily `PythonSensor` waits for an earnings date | `clean_earnings` |
| `macro_daily` | 13:00 UTC weekdays | `clean_macro` |
| `signals_refresh` | **data-aware: when any Dataset above updates** | – |
| `full_refresh_manual` | manual, all steps end to end | – |

A dataset is published only after its quality gate passes, so bad data never triggers a refresh. Pipeline code runs in its own virtualenv inside the Airflow image through `ExternalPythonOperator`, so Chroma, Great Expectations and pandas 2+ never clash with Airflow's pinned packages.

## Where AI runs, and what happens when it can't

| Step | Model | Job | If the model is unavailable |
|---|---|---|---|
| 5 | Llama | Explains failed checks in plain English | Template explanation |
| 6 | Llama (bulk role) | Headline to JSON: event type, sentiment −1..1, materiality 1-5. Batched 10 per call, cached per story, last 30 days only | Keyword rules; history is always rules-backfilled to save calls |
| 6, 8 | Embeddings + Chroma | Collapse syndicated stories; top-k evidence; historical analogs (30+ days older, no look-ahead) | Embeddings fall back to ONNX MiniLM, then a hashing vectorizer |
| 9 | Gemini (reason role) | Decision, timing and a trace where every step cites evidence ids. May override the factor baseline only with a stated reason. Output is validated and invented evidence ids are dropped | Rules-based trace from the same factor scores |
| 10 | Gemini | Judge flags unsupported claims; deterministic checks always run | Deterministic checks only |

Routing (`LLM_PROVIDER=auto`): bulk work goes to local Llama first, reasoning goes to Gemini first, and each falls back to the other. Every call is logged to `ops_llm_calls` and shown on the AI usage tab.

## Data model

| Layer | Tables |
|---|---|
| raw | `raw_prices`, `raw_news`, `raw_earnings`, `raw_filings`, `raw_macro` (all text, last 3 batches kept) |
| clean | `clean_prices`, `clean_news`, `clean_earnings`, `clean_filings`, `clean_macro`, `clean_news_events` |
| mart | `mart_features_daily`, `mart_asset_latest`, `mart_evidence`, `mart_analog_candidates`, `mart_analogs`, `mart_signals`, `mart_signal_factors`, `mart_decision_trace`, `mart_eval_results`, `mart_backtest_daily`, `mart_backtest_summary`, `mart_backtest_factor_summary` |
| quality / ops | `dq_results`, `ops_pipeline_runs`, `ops_llm_calls` |

The same SQL runs on Postgres and SQLite; layers are table prefixes rather than schemas so the queries stay identical.

## The bundled sample

`msp/sample_data.py` generates a **synthetic** dataset (simulated prices and news for the same 14 assets) so the demo runs with no network or keys. It plants the mess a real feed has: duplicate rows, numbers stored as text, missing volumes, a 4:1 split, dividends, and the same story syndicated by several outlets. News moves later prices a little, so analogs and the backtest have something to find. The dashboard shows a warning whenever sample data is on screen.

## 5-minute demo script

| Time | Show | Say |
|---|---|---|
| 0:00 | README roadmap | "Raw APIs to explainable signals: ELT, a quality gate, AI inside the pipeline, orchestration." |
| 0:45 | Airflow UI: DAG list, `signals_refresh` scheduled on Datasets | "Ingestion DAGs publish Datasets only after the gate passes; the signal DAG is data-aware." |
| 1:30 | Dashboard → Data quality; sidebar **Inject bad data** | "Great Expectations plus SQL reconciliation. Bad vendor rows block the run, and the LLM explains why." |
| 2:15 | Pipeline tab: row counts raw → clean → mart, mart SQL | "Python cleans, SQL joins and computes RSI and moving averages with window functions." |
| 3:00 | AI usage tab | "Llama turns headlines into JSON; Chroma collapsed duplicate stories; model routing keeps costs down." |
| 3:30 | Signals → Asset detail (pick one stock, one metal) | "Evidence retrieved from Chroma, historical analogs with what happened next, then a cited decision trace." |
| 4:30 | Evaluation tab | "A judge checks every claim; the backtest and factor attribution show which inputs earn their place." |

## Project layout

```
msp/            pipeline package (config, extract, load_raw, clean, quality, llm, vectors,
                enrich, retrieve, signals, evaluate, pipeline CLI, sample_data)
sql/            mart_features.sql (portable Postgres / SQLite)
dags/           Airflow DAGs
app/ui.py       Streamlit dashboard
docker/         Dockerfiles and Postgres init
tests/          pytest suite
```
