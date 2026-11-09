"""End-to-end and unit tests. Run: pytest -q"""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run(args, data_dir):
    env = {**os.environ, "DATA_DIR": str(data_dir), "DATA_MODE": "sample", "LLM_PROVIDER": "offline"}
    env.pop("DATABASE_URL", None)
    return subprocess.run([sys.executable, "-m", "msp.pipeline", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    r = _run(["run", "--sample"], d)
    assert r.returncode == 0, r.stderr[-3000:]
    return d


def q(d, sql):
    with sqlite3.connect(d / "warehouse.db") as c:
        return pd.read_sql(sql, c)


def test_all_14_assets_get_a_signal(built):
    s = q(built, "SELECT * FROM mart_signals")
    assert len(s) == 14
    assert set(s["action"]) <= {"Buy", "Hold", "Sell"}
    assert s["asset_class"].value_counts().to_dict() == {"stock": 10, "metal": 4}


def test_every_signal_has_a_trace_and_evidence(built):
    t = q(built, "SELECT symbol, COUNT(*) AS n FROM mart_decision_trace GROUP BY symbol")
    assert len(t) == 14 and (t["n"] >= 3).all()
    e = q(built, "SELECT COUNT(DISTINCT symbol) AS n FROM mart_evidence")
    assert e["n"].iloc[0] == 14


def test_clean_removes_duplicates_and_adjusts_split(built):
    dup = q(built, "SELECT symbol, date, COUNT(*) c FROM clean_prices GROUP BY 1, 2 HAVING c > 1")
    assert dup.empty
    big = q(built, "SELECT * FROM clean_prices WHERE symbol = 'NVDA' AND ABS(ret_1d) > 0.5")
    assert big.empty, "4:1 split was not back-adjusted"


def test_quality_gate_passed_and_raw_reconciles(built):
    dq = q(built, "SELECT * FROM dq_results")
    assert dq["success"].astype(bool).all()
    assert "Great Expectations" in set(dq["tool"]) and "SQL" in set(dq["tool"])


def test_chroma_dedup_collapsed_syndicated_stories(built):
    ev = q(built, "SELECT * FROM clean_news_events WHERE kind = 'news'")
    news = q(built, "SELECT COUNT(*) AS n FROM clean_news")["n"].iloc[0]
    assert len(ev) < news * 0.75
    # one story never contains the same outlet twice (that would mean two events were merged)
    assert not ev["outlets"].str.split(", ").apply(lambda x: len(x) != len(set(x))).any()


def test_analogs_have_no_lookahead(built):
    a = q(built, "SELECT a.analog_date, e.date AS query_date FROM mart_analogs a "
                 "JOIN clean_news_events e ON e.story_id = a.query_story_id")
    assert len(a) > 0
    assert (pd.to_datetime(a["analog_date"]) < pd.to_datetime(a["query_date"]) - pd.Timedelta(days=29)).all()


def test_judge_and_backtest_written(built):
    assert len(q(built, "SELECT * FROM mart_eval_results")) == 14
    bt = q(built, "SELECT * FROM mart_backtest_summary WHERE asset_class = 'all'")
    assert set(bt["action"]) >= {"Buy", "Hold"}


def test_quality_gate_blocks_bad_data(tmp_path):
    assert _run(["run", "--sample"], tmp_path).returncode == 0
    assert _run(["inject-bad-data"], tmp_path).returncode == 0
    assert _run(["step", "clean"], tmp_path).returncode == 0
    r = _run(["step", "quality_gate"], tmp_path)
    assert r.returncode == 1
    dq = q(tmp_path, "SELECT * FROM dq_results WHERE run_id = (SELECT MAX(run_id) FROM dq_results)")
    failed = set(dq.loc[~dq["success"].astype(bool), "check_name"])
    assert "no_future_price_dates" in failed
    assert "expect_column_values_to_be_between" in failed


# ------------------------------------------------------------------ LLM paths with a stub model
def test_news_extraction_parses_llm_json(monkeypatch):
    from msp import enrich, llm

    rows = [{"symbol": "AAPL", "headline": "Apple beats estimates"}, {"symbol": "GC=F", "headline": "Gold slides"}]
    fake = {"events": [{"i": 0, "event_type": "earnings", "sentiment": 0.7, "materiality": 4, "rationale": "beat"},
                       {"i": 1, "event_type": "macro", "sentiment": -0.4, "materiality": 3, "rationale": "yields"}]}
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: (fake, "Llama (stub)"))
    out, provider = enrich._llm_batch(rows)
    assert provider == "Llama (stub)" and out[0]["event_type"] == "earnings" and out[1]["sentiment"] == -0.4


def test_news_extraction_rejects_incomplete_llm_json(monkeypatch):
    from msp import enrich, llm

    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: ({"events": [{"i": 0}]}, "Llama (stub)"))
    out, _ = enrich._llm_batch([{"symbol": "AAPL", "headline": "a"}, {"symbol": "AAPL", "headline": "b"}])
    assert out is None  # caller falls back to rules


def test_signal_synthesis_validates_llm_output(monkeypatch):
    from msp import llm, signals

    row = pd.Series({"symbol": "AAPL", "name": "Apple", "asset_class": "stock", "sector": "Tech", "date": "2026-11-06",
                     "adj_close": 200.0, "ret_20d": 0.04, "ret_60d": 0.1, "rsi14": 60, "ma20": 195, "ma50": 190,
                     "dist_ma20": 0.025, "sentiment_7d": 0.4, "last_eps_surprise_pct": 6, "yield_chg_20d": 0.1,
                     "usd_chg_20d": 0.0, "next_report_date": None})
    ev = pd.DataFrame([{"evidence_id": "E1", "date": "2026-11-05", "headline": "Apple raises guidance",
                        "event_type": "guidance", "sentiment": 0.6, "materiality": 4}])
    an = pd.DataFrame(columns=["analog_symbol", "analog_date", "analog_headline", "similarity", "fwd_ret_10d"])
    fake = {"action": "Buy", "confidence": "medium", "timing": "Scale in", "summary": "s",
            "trace": [{"step": "Evidence", "finding": "guidance raised", "evidence_ids": ["E1", "E9"]},
                      {"step": "Decision", "finding": "Buy", "evidence_ids": []}], "risks": ["r"]}
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: (fake, "Gemini (stub)"))
    out, provider = signals._llm_decide(row, {"news": 1}, 3, "Buy", ev, an, "t")
    assert out["action"] == "Buy" and out["trace"][0]["evidence_ids"] == ["E1"]  # invented E9 dropped
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: ({"action": "Strong buy"}, "Gemini (stub)"))
    assert signals._llm_decide(row, {}, 3, "Buy", ev, an, "t")[0] is None


def test_rules_classifier_and_json_parser():
    from msp import enrich, llm

    r = enrich.rules_classify("Regulators open probe into Apple business practices")
    assert r["event_type"] == "legal" and r["sentiment"] < 0
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('noise {"b": [1,2]} trailing') == {"b": [1, 2]}
