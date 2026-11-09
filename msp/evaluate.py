"""Step 10 - Evaluate: prove signals are backed and accurate.

Judge: Gemini checks each signal's decision trace against the evidence and factor scores
it was given, and flags any claim that is not supported. Deterministic checks always run
too (cited ids exist, action consistent with score or override explained).
Backtest: the rules model is replayed over the last ~250 trading days in Python, then SQL
aggregates what actually happened 10 days after each Buy, Hold and Sell, both raw and
relative to each asset's own average return.
"""
import json

import pandas as pd

from . import db, llm
from .signals import baseline_action, score_frame

BACKTEST_SQL = """
DROP TABLE IF EXISTS mart_backtest_summary
;
CREATE TABLE mart_backtest_summary AS
SELECT asset_class, action,
       COUNT(*) AS signals,
       AVG(fwd_ret_10d) AS avg_fwd_ret_10d,
       AVG(excess_fwd_10d) AS avg_excess_ret_10d,
       AVG(CASE WHEN fwd_ret_10d > 0 THEN 1.0 ELSE 0.0 END) AS share_positive_10d,
       AVG(CASE WHEN (action = 'Buy' AND fwd_ret_10d > 0) OR (action = 'Sell' AND fwd_ret_10d < 0)
                THEN 1.0 ELSE 0.0 END) AS hit_rate
FROM mart_backtest_daily
WHERE fwd_ret_10d IS NOT NULL
GROUP BY asset_class, action
UNION ALL
SELECT 'all' AS asset_class, action, COUNT(*), AVG(fwd_ret_10d), AVG(excess_fwd_10d),
       AVG(CASE WHEN fwd_ret_10d > 0 THEN 1.0 ELSE 0.0 END),
       AVG(CASE WHEN (action = 'Buy' AND fwd_ret_10d > 0) OR (action = 'Sell' AND fwd_ret_10d < 0)
                THEN 1.0 ELSE 0.0 END)
FROM mart_backtest_daily
WHERE fwd_ret_10d IS NOT NULL
GROUP BY action
;
DROP TABLE IF EXISTS mart_backtest_factor_summary
;
CREATE TABLE mart_backtest_factor_summary AS
SELECT factor,
       CASE WHEN factor_score > 0 THEN 'positive' WHEN factor_score < 0 THEN 'negative' ELSE 'neutral' END AS reading,
       COUNT(*) AS days,
       AVG(excess_fwd_10d) AS avg_excess_ret_10d
FROM mart_backtest_factor_daily
WHERE excess_fwd_10d IS NOT NULL
GROUP BY factor, CASE WHEN factor_score > 0 THEN 'positive' WHEN factor_score < 0 THEN 'negative' ELSE 'neutral' END
"""


def _deterministic_checks(sig, trace: pd.DataFrame, ev_ids: set) -> list[str]:
    issues = []
    cited = {i for ids in trace["evidence_ids"].fillna("") for i in ids.split(",") if i}
    bad = cited - ev_ids
    if bad:
        issues.append(f"cites evidence that does not exist: {sorted(bad)}")
    if ev_ids and not cited:
        issues.append("no trace step cites any evidence")
    if sig["action"] != sig["baseline_action"] and not sig["override_reason"]:
        issues.append(f"action {sig['action']} differs from factor baseline {sig['baseline_action']} with no reason")
    if sig["action"] == "Buy" and sig["score"] < 0:
        issues.append("Buy with a negative factor score")
    if sig["action"] == "Sell" and sig["score"] > 0:
        issues.append("Sell with a positive factor score")
    return issues


def judge() -> pd.DataFrame:
    sigs = db.read_sql("SELECT * FROM mart_signals")
    traces = db.read_sql("SELECT * FROM mart_decision_trace")
    evidence = db.read_sql("SELECT * FROM mart_evidence")
    factors = db.read_sql("SELECT * FROM mart_signal_factors")
    prev = {}
    if db.table_exists("mart_eval_results"):
        pe = db.read_sql("SELECT * FROM mart_eval_results")
        if "input_key" in pe:
            prev = {r["symbol"]: r for r in pe.to_dict("records")}
    rows = []
    for _, s in sigs.iterrows():
        sym = s["symbol"]
        old = prev.get(sym)
        if old and old["input_key"] == s.get("input_key") and old["judged_by"] != "deterministic checks":
            rows.append(old)  # same decision as last run: keep its verdict, no new LLM call
            continue
        tr = traces[traces.symbol == sym].sort_values("step_no")
        ev = evidence[evidence.symbol == sym]
        issues = _deterministic_checks(s, tr, set(ev["evidence_id"]))
        verdict, provider, unsupported = None, "deterministic checks", []
        if not s["decided_by"].startswith("offline"):
            prompt = (
                "You are a model risk reviewer. Check whether each step of this decision trace is supported by the "
                "evidence and factor scores below. A claim is unsupported if it states a fact, number or event "
                "not present in the data. Return JSON: {\"supported\": bool, \"unsupported_claims\": [str], "
                "\"grounding_score\": float 0-1, \"comment\": str (max 25 words)}\n"
                f"Decision: {s['action']} ({s['timing']})\n"
                f"Trace: {tr[['step', 'finding', 'evidence_ids']].to_dict('records')}\n"
                f"Evidence: {ev[['evidence_id', 'headline', 'sentiment', 'materiality']].to_dict('records')}\n"
                f"Factor scores: {factors[factors.symbol == sym][['factor', 'score']].to_dict('records')}")
            out, provider = llm.complete_json(prompt, role="reason", task="judge")
            if isinstance(out, dict) and "supported" in out:
                verdict = out
                unsupported = [str(c) for c in out.get("unsupported_claims", [])][:5]
        grounding = float(verdict.get("grounding_score", 1.0)) if verdict else (1.0 if not issues else 0.5)
        supported = (bool(verdict["supported"]) if verdict else True) and not issues
        rows.append({"symbol": sym, "action": s["action"], "decided_by": s["decided_by"],
                     "supported": supported, "grounding_score": round(grounding, 2),
                     "issues": json.dumps(issues + unsupported),
                     "comment": (verdict or {}).get("comment", "Rule checks only: trace generated by rules."
                                                    if s["decided_by"].startswith("offline") else ""),
                     "judged_by": provider if verdict else "deterministic checks",
                     "input_key": s.get("input_key", "")})
    df = pd.DataFrame(rows)
    db.write_df(df, "mart_eval_results", if_exists="replace")
    return df


def backtest() -> pd.DataFrame:
    hist = db.read_sql("SELECT * FROM mart_features_daily WHERE is_trading_day = 1 AND rsi14 IS NOT NULL")
    hist = hist.sort_values(["symbol", "date"])
    hist = hist.groupby("symbol").tail(250)
    sc = score_frame(hist)
    hist["score"] = sc.sum(axis=1)
    hist["action"] = hist["score"].apply(baseline_action)
    # excess return vs the asset's own average removes the bias of high-drift assets
    hist["excess_fwd_10d"] = hist["fwd_ret_10d"] - hist.groupby("symbol")["fwd_ret_10d"].transform("mean")
    db.write_df(hist[["symbol", "asset_class", "date", "score", "action", "fwd_ret_10d", "excess_fwd_10d"]],
                "mart_backtest_daily",
                if_exists="replace")
    # factor attribution: one row per asset, day and factor, so SQL can show which factors earn their place
    long = sc.join(hist[["symbol", "date", "excess_fwd_10d"]]).melt(
        id_vars=["symbol", "date", "excess_fwd_10d"], var_name="factor", value_name="factor_score")
    db.write_df(long, "mart_backtest_factor_daily", if_exists="replace")
    with db.engine().begin() as conn:
        from sqlalchemy import text

        for stmt in [x for x in BACKTEST_SQL.split("\n;\n") if x.strip()]:
            conn.execute(text(stmt))
    return db.read_sql("SELECT * FROM mart_backtest_summary")


def evaluate() -> dict:
    j = judge()
    b = backtest()
    llm.flush_call_log()
    allb = b[b.asset_class == "all"].set_index("action")
    return {"signals_judged": len(j), "supported": int(j["supported"].sum()),
            "avg_grounding": round(float(j["grounding_score"].mean()), 2),
            "backtest": {a: {"signals": int(r.signals), "avg_fwd_ret_10d": round(float(r.avg_fwd_ret_10d), 4),
                             "avg_excess_ret_10d": round(float(r.avg_excess_ret_10d), 4),
                             "hit_rate": round(float(r.hit_rate), 3)} for a, r in allb.iterrows()}}
