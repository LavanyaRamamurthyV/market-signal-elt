"""Step 5 - Quality gate: block bad data before it moves on.

Great Expectations runs an expectation suite on every clean table (nulls, ranges,
uniqueness, allowed values). SQL reconciliation queries add checks across tables
(raw-to-clean row counts, freshness, referential integrity, future dates).
Any failed critical check raises QualityGateError, which fails the Airflow task and
stops everything downstream. Llama writes a plain-English explanation of each failure.
"""
import html
import json
import logging
from datetime import date, timedelta

import pandas as pd

from . import config, db, llm

logging.getLogger("great_expectations").setLevel(logging.ERROR)


class QualityGateError(RuntimeError):
    pass


def _suites():
    import great_expectations as gx

    E = gx.expectations
    syms = list(config.ASSET_BY_SYMBOL)
    stocks = [a["symbol"] for a in config.STOCKS]
    return {
        "clean_prices": [
            E.ExpectColumnValuesToNotBeNull(column="adj_close"),
            E.ExpectColumnValuesToBeBetween(column="adj_close", min_value=0.0001, strict_min=False),
            E.ExpectCompoundColumnsToBeUnique(column_list=["symbol", "date"]),
            E.ExpectColumnValuesToBeInSet(column="symbol", value_set=syms),
            E.ExpectColumnValuesToNotBeNull(column="volume", mostly=0.98),
            E.ExpectColumnValuesToBeBetween(column="ret_1d", min_value=-0.5, max_value=0.5, mostly=0.999),
        ],
        "clean_news": [
            E.ExpectColumnValuesToNotBeNull(column="headline"),
            E.ExpectCompoundColumnsToBeUnique(column_list=["symbol", "news_id"]),
            E.ExpectColumnValuesToBeInSet(column="symbol", value_set=syms),
        ],
        "clean_earnings": [
            E.ExpectColumnValuesToBeInSet(column="symbol", value_set=stocks),
            E.ExpectColumnValuesToBeBetween(column="surprise_pct", min_value=-200, max_value=200, mostly=0.95),
        ],
        "clean_filings": [E.ExpectColumnValuesToBeUnique(column="accession")],
        "clean_macro": [E.ExpectColumnValuesToNotBeNull(column="usd_index", mostly=0.95),
                        E.ExpectColumnValuesToNotBeNull(column="us_10y_yield", mostly=0.95)],
    }


def _run_gx(table: str, expectations: list) -> list[dict]:
    import great_expectations as gx

    df = db.read_sql(f'SELECT * FROM "{table}"')
    ctx = gx.get_context(mode="ephemeral")
    batch = (ctx.data_sources.add_pandas(f"ds_{table}").add_dataframe_asset(table)
             .add_batch_definition_whole_dataframe("all").get_batch(batch_parameters={"dataframe": df}))
    suite = gx.ExpectationSuite(name=f"{table}_suite")
    for e in expectations:
        suite.add_expectation(e)
    res = batch.validate(suite)
    rows = []
    for r in res.results:
        cfg = r.expectation_config
        kw = {k: v for k, v in cfg.kwargs.items() if k not in ("batch_id",)}
        rows.append({"table_name": table, "check_name": cfg.type, "tool": "Great Expectations",
                     "target": json.dumps(kw, default=str)[:300], "success": bool(r.success),
                     "observed": json.dumps({k: r.result.get(k) for k in
                                             ("element_count", "unexpected_count", "unexpected_percent",
                                              "partial_unexpected_list") if k in r.result}, default=str)[:500],
                     "critical": True})
    return rows


def _sql_checks() -> list[dict]:
    rows = []

    def add(name, table, sql, ok_fn, critical=True):
        val = db.read_sql(sql)
        v = val.iloc[0].to_dict() if len(val) else {}
        rows.append({"table_name": table, "check_name": name, "tool": "SQL", "target": sql.strip()[:300],
                     "success": bool(ok_fn(v)), "observed": json.dumps(v, default=str), "critical": critical})

    add("raw_to_clean_row_reconciliation", "clean_prices",
        """SELECT (SELECT COUNT(*) FROM (SELECT DISTINCT symbol, date FROM raw_prices
                   WHERE _batch_id = (SELECT MAX(_batch_id) FROM raw_prices)) r) AS raw_distinct_rows,
                  (SELECT COUNT(*) FROM clean_prices WHERE is_trading_day = 1) AS clean_trading_rows""",
        lambda v: v["raw_distinct_rows"] == v["clean_trading_rows"])
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    add("price_freshness_within_7_days", "clean_prices",
        "SELECT MAX(date) AS latest_date FROM clean_prices", lambda v: (v["latest_date"] or "") >= cutoff)
    add("no_future_price_dates", "clean_prices",
        f"SELECT COUNT(*) AS future_rows FROM clean_prices WHERE date > '{date.today().isoformat()}'",
        lambda v: v["future_rows"] == 0)
    add("all_assets_present", "clean_prices", "SELECT COUNT(DISTINCT symbol) AS assets FROM clean_prices",
        lambda v: v["assets"] == len(config.ASSETS))
    add("news_symbols_exist_in_prices", "clean_news",
        """SELECT COUNT(*) AS orphan_news FROM clean_news n
           WHERE NOT EXISTS (SELECT 1 FROM clean_prices p WHERE p.symbol = n.symbol)""",
        lambda v: v["orphan_news"] == 0)
    add("earnings_symbols_exist_in_prices", "clean_earnings",
        """SELECT COUNT(*) AS orphan_earnings FROM clean_earnings e
           WHERE NOT EXISTS (SELECT 1 FROM clean_prices p WHERE p.symbol = e.symbol)""",
        lambda v: v["orphan_earnings"] == 0)
    add("news_volume_last_30_days", "clean_news",
        f"SELECT COUNT(*) AS recent_news FROM clean_news WHERE date >= '{(date.today() - timedelta(days=30)).isoformat()}'",
        lambda v: v["recent_news"] > 0, critical=False)
    return rows


def _explain(failures: pd.DataFrame) -> tuple[str, str]:
    if failures.empty:
        return "All checks passed.", "n/a"
    items = failures[["table_name", "check_name", "observed"]].to_dict("records")
    prompt = ("You are a data engineer on call. These data quality checks failed in a market data pipeline. "
              "For each, explain in one plain-English sentence what is wrong and the most likely upstream cause, "
              "then one sentence on the fix. Return JSON: {\"explanations\": [{\"check\": str, \"what\": str, "
              f"\"likely_cause\": str, \"fix\": str}}]}}.\nFailures: {json.dumps(items)[:3500]}")
    out, provider = llm.complete_json(prompt, role="bulk", task="dq_explain")
    if out and isinstance(out, dict) and out.get("explanations"):
        text = "\n".join(f"- {e.get('check')}: {e.get('what')} Likely cause: {e.get('likely_cause')} "
                         f"Fix: {e.get('fix')}" for e in out["explanations"])
        return text, provider
    rules = []
    for f in items:
        rules.append(f"- {f['table_name']}.{f['check_name']} failed (observed {f['observed'][:160]}). "
                     "Check the latest raw batch for this source and re-run extract.")
    return "\n".join(rules), provider


def write_report(results: pd.DataFrame, explanation: str, provider: str) -> str:
    rows = "".join(
        f"<tr class='{'ok' if r.success else 'bad'}'><td>{html.escape(r.table_name)}</td><td>{html.escape(r.check_name)}</td>"
        f"<td>{r.tool}</td><td>{'PASS' if r.success else 'FAIL'}</td><td><code>{html.escape(str(r.observed))[:300]}</code></td></tr>"
        for r in results.itertuples())
    page = f"""<!doctype html><meta charset='utf-8'><title>Quality gate report</title>
<style>body{{font-family:Arial;margin:24px}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px;vertical-align:top}}
table{{border-collapse:collapse}}.ok td:nth-child(4){{color:#0F6E56}}.bad td:nth-child(4){{color:#A32D2D;font-weight:bold}}</style>
<h2>Quality gate report</h2><p>Run {results['run_id'].iloc[0]} - {int(results.success.sum())}/{len(results)} checks passed.</p>
<h3>Explanation ({html.escape(provider)})</h3><pre>{html.escape(explanation)}</pre>
<table><tr><th>Table</th><th>Check</th><th>Tool</th><th>Result</th><th>Observed</th></tr>{rows}</table>"""
    path = config.REPORT_DIR / "quality_report.html"
    path.write_text(page, encoding="utf-8")
    return str(path)


def quality_gate(tables: list[str] | None = None, raise_on_fail: bool = True) -> dict:
    run_id = db.now_utc()
    suites = _suites()
    results = []
    for table, exps in suites.items():
        if tables and table not in tables:
            continue
        if db.table_exists(table):
            results += _run_gx(table, exps)
    results += [r for r in _sql_checks() if not tables or r["table_name"] in tables]
    df = pd.DataFrame(results)
    df["run_id"] = run_id
    failures = df[(~df["success"]) & (df["critical"])]
    explanation, provider = _explain(df[~df["success"]])
    df["explanation"] = explanation
    df["explained_by"] = provider
    db.write_df(df, "dq_results", if_exists="append")
    report = write_report(df, explanation, provider)
    summary = {"run_id": run_id, "checks": len(df), "passed": int(df["success"].sum()),
               "critical_failures": len(failures), "report": report, "explained_by": provider}
    if len(failures) and raise_on_fail:
        raise QualityGateError(f"{len(failures)} critical data quality checks failed: "
                               + ", ".join(failures["table_name"] + "." + failures["check_name"])
                               + f"\n{explanation}")
    return summary
