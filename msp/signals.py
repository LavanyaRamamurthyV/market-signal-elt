"""Step 9 - Signal synthesis: decide Buy, Hold or Avoid, with a step-by-step decision trace.

1. Deterministic factor scores from the mart (momentum, trend, RSI, news sentiment,
   earnings surprise for stocks, rates and dollar for metals, historical analogs).
   These give a baseline action that is fully reproducible.
2. Gemini reads the factor scores, the retrieved evidence (E1..E6) and the analog outcomes
   and writes the decision, timing and a trace in which every step cites evidence.
   It may override the baseline only with a stated reason. Output is validated; if the model
   is unavailable or returns something invalid, the rules-based trace is used instead.
Research signals only, not investment advice.
"""
import json

import numpy as np
import pandas as pd

from . import config, db, llm

ACTIONS = ["Buy", "Hold", "Avoid"]
BUY_AT, AVOID_AT = 3, -2


def _bucket(x, hi2, hi1, lo1, lo2):
    return np.select([x > hi2, x > hi1, x < lo2, x < lo1], [2, 1, -2, -1], 0)


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised factor scores. Used for today's signals and for the backtest."""
    out = pd.DataFrame(index=df.index)
    out["momentum"] = _bucket(df["ret_20d"].fillna(0), 0.05, 0.01, -0.01, -0.05)
    above = df["adj_close"] > df["ma20"]
    out["trend"] = np.where((df["uptrend"] == 1) & above, 1, np.where((df["uptrend"] == 0) & ~above, -1, 0))
    out["rsi"] = np.select([df["rsi14"] > 70, df["rsi14"] < 30], [-1, 1], 0)
    out["news"] = _bucket(df["sentiment_7d"].fillna(0), 0.5, 0.15, -0.15, -0.5)
    is_stock = df["asset_class"] == "stock"
    out["earnings"] = np.where(is_stock, _bucket(df["last_eps_surprise_pct"].fillna(0), 10, 5, -5, -10), 0)
    macro = (np.select([df["yield_chg_20d"] < -0.15, df["yield_chg_20d"] > 0.15], [1, -1], 0)
             + np.select([df["usd_chg_20d"] < -0.01, df["usd_chg_20d"] > 0.01], [1, -1], 0))
    out["macro"] = np.where(is_stock, 0, macro)
    return out


def baseline_action(score: float) -> str:
    return "Buy" if score >= BUY_AT else "Avoid" if score <= AVOID_AT else "Hold"


def _timing(row, action: str) -> str:
    nxt = row.get("next_report_date")
    if row["asset_class"] == "stock" and isinstance(nxt, str) and nxt:
        days = (pd.Timestamp(nxt) - pd.Timestamp(row["date"])).days
        if 0 <= days <= 7:
            return f"Wait until after earnings on {nxt}; volatility risk is high before the report."
    if action == "Buy":
        if row["rsi14"] > 70 or row["dist_ma20"] > 0.05:
            return f"Wait for a pullback toward the 20-day average (about {row['ma20']:.2f}) before buying."
        return "Conditions support entering now; scale in rather than buying all at once."
    if action == "Avoid":
        return f"Avoid until price recovers above the 20-day average (about {row['ma20']:.2f}) and news turns neutral."
    return "Hold existing positions; reassess on the next earnings report or a material news event."


def _rules_trace(row, factors: dict, ev: pd.DataFrame, an: pd.DataFrame, action: str, score: int) -> list[dict]:
    top = ev.head(3)
    steps = [
        {"step": "Evidence", "finding": "; ".join(f"{r.evidence_id} {r.headline[:70]} (sentiment {r.sentiment:+.1f}, "
                                                  f"materiality {r.materiality})" for r in top.itertuples())
                                      or "No material news in the last 30 days.",
         "evidence_ids": top["evidence_id"].tolist()},
        {"step": "Factor scores", "finding": ", ".join(f"{k} {v:+d}" for k, v in factors.items()) + f" = total {score:+d}",
         "evidence_ids": []},
    ]
    if len(an):
        steps.append({"step": "Historical analogs",
                      "finding": f"{len(an)} similar past events; median 10-day return after them "
                                 f"{an['fwd_ret_10d'].median():+.1%}, positive in {(an['fwd_ret_10d'] > 0).mean():.0%} of cases.",
                      "evidence_ids": []})
    steps.append({"step": "Decision", "finding": f"Total score {score:+d} maps to {action} "
                                                f"(Buy at {BUY_AT:+d} or more, Avoid at {AVOID_AT:+d} or less).",
                  "evidence_ids": []})
    return steps


def _llm_decide(row, factors, score, base, ev: pd.DataFrame, an: pd.DataFrame, timing: str):
    payload = {
        "asset": f"{row['name']} ({row['symbol']}), {row['asset_class']}, {row['sector']}",
        "as_of": row["date"], "price": round(float(row["adj_close"]), 4),
        "indicators": {k: (None if pd.isna(row[k]) else round(float(row[k]), 4)) for k in
                       ("ret_20d", "ret_60d", "rsi14", "ma20", "ma50", "dist_ma20", "sentiment_7d",
                        "last_eps_surprise_pct", "yield_chg_20d", "usd_chg_20d")},
        "next_earnings_date": row.get("next_report_date"),
        "factor_scores": factors, "total_score": score, "baseline_action": base, "baseline_timing": timing,
        "evidence": ev[["evidence_id", "date", "headline", "event_type", "sentiment", "materiality"]].to_dict("records"),
        "analogs": (an[["analog_symbol", "analog_date", "analog_headline", "similarity", "fwd_ret_10d"]]
                    .head(8).round(4).to_dict("records")),
    }
    prompt = (
        "You are a sell-side research analyst writing an explainable signal for an internal risk committee. "
        "Use ONLY the data provided. Professional tone. Decide Buy, Hold or Avoid and when.\n"
        "Rules: start from baseline_action; you may change it only if the evidence clearly contradicts it, and say why. "
        "Every trace step must cite evidence ids (E1..) or name the factor it relies on. Do not invent numbers.\n"
        'Return JSON: {"action": "Buy|Hold|Avoid", "confidence": "low|medium|high", "timing": str, '
        '"summary": str (max 40 words), "trace": [{"step": str, "finding": str, "evidence_ids": [str]}] (4-6 steps: '
        'evidence, fundamentals or macro, technicals, analogs, decision), "risks": [str] (max 3), '
        '"override_reason": str or null}\n'
        f"Data: {json.dumps(payload, default=str)}")
    out, provider = llm.complete_json(prompt, role="reason", task="signal_synthesis")
    valid_ids = set(ev["evidence_id"])
    if not isinstance(out, dict) or out.get("action") not in ACTIONS or not isinstance(out.get("trace"), list):
        return None, provider
    trace = []
    for s in out["trace"][:8]:
        if not isinstance(s, dict):
            continue
        ids = [i for i in (s.get("evidence_ids") or []) if i in valid_ids]
        trace.append({"step": str(s.get("step", ""))[:60], "finding": str(s.get("finding", ""))[:600],
                      "evidence_ids": ids})
    if len(trace) < 2:
        return None, provider
    out["trace"] = trace
    return out, provider


def synthesize() -> dict:
    latest = db.read_sql("SELECT * FROM mart_asset_latest")
    evidence = db.read_sql("SELECT * FROM mart_evidence")
    analogs = db.read_sql("SELECT * FROM mart_analogs")
    fs = score_frame(latest)
    sig_rows, trace_rows, factor_rows = [], [], []
    for idx, row in latest.iterrows():
        sym = row["symbol"]
        factors = {k: int(v) for k, v in fs.loc[idx].items()}
        an = analogs[analogs.symbol == sym]
        if len(an) >= 3:
            med = an["fwd_ret_10d"].median()
            factors["analogs"] = 1 if med > 0.01 else -1 if med < -0.01 else 0
        else:
            factors["analogs"] = 0
        score = int(sum(factors.values()))
        base = baseline_action(score)
        timing = _timing(row, base)
        ev = evidence[evidence.symbol == sym].sort_values("rank")
        decided, provider = _llm_decide(row, factors, score, base, ev, an, timing)
        if decided:
            action, conf = decided["action"], decided.get("confidence", "medium")
            timing_out, summary = decided.get("timing") or timing, decided.get("summary", "")
            trace, risks, override = decided["trace"], decided.get("risks", [])[:3], decided.get("override_reason")
        else:
            action, conf = base, "high" if abs(score) >= 5 else "medium" if abs(score) >= 3 else "low"
            timing_out = timing
            trace = _rules_trace(row, factors, ev, an, action, score)
            summary = (f"{row['name']}: total factor score {score:+d} gives {action}. "
                       f"Strongest drivers: " + ", ".join(k for k, v in sorted(factors.items(), key=lambda kv: -abs(kv[1]))[:2]) + ".")
            risks, override = [], None
            provider = "offline rules"
        sig_rows.append({"symbol": sym, "name": row["name"], "asset_class": row["asset_class"], "sector": row["sector"],
                         "as_of": row["date"], "price": float(row["adj_close"]), "ret_1d": row["ret_1d"],
                         "ret_20d": row["ret_20d"], "rsi14": row["rsi14"], "action": action, "baseline_action": base,
                         "score": score, "confidence": conf, "timing": timing_out, "summary": summary,
                         "risks": json.dumps(risks), "override_reason": override or "",
                         "decided_by": provider, "data_origin": row["data_origin"],
                         "next_report_date": row.get("next_report_date")})
        for n, s in enumerate(trace, 1):
            trace_rows.append({"symbol": sym, "step_no": n, "step": s["step"], "finding": s["finding"],
                               "evidence_ids": ",".join(s.get("evidence_ids", []))})
        for k, v in factors.items():
            factor_rows.append({"symbol": sym, "factor": k, "score": v})
    db.write_df(pd.DataFrame(sig_rows), "mart_signals", if_exists="replace")
    db.write_df(pd.DataFrame(trace_rows), "mart_decision_trace", if_exists="replace")
    db.write_df(pd.DataFrame(factor_rows), "mart_signal_factors", if_exists="replace")
    llm.flush_call_log()
    s = pd.DataFrame(sig_rows)
    return {"assets": len(s), "actions": s["action"].value_counts().to_dict(),
            "decided_by": s["decided_by"].value_counts().to_dict(),
            "overrides": int((s["action"] != s["baseline_action"]).sum())}
