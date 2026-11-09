"""Market Signal ELT - Streamlit dashboard (the end product of the pipeline).

Run: streamlit run app/ui.py
"""
import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from msp import config, db  # noqa: E402

st.set_page_config(page_title="Market Signal ELT", layout="wide")
COLORS = {"Buy": "#0F6E56", "Hold": "#854F0B", "Avoid": "#A32D2D"}
BG = {"Buy": "#E1F5EE", "Hold": "#FAEEDA", "Avoid": "#FCEBEB"}
st.markdown("""<style>
.card{border:1px solid #ddd;border-radius:10px;padding:10px 12px;margin-bottom:10px;background:#fff}
.card .sym{font-size:12px;color:#666}.card .nm{font-weight:600;font-size:15px}
.card .px{font-size:20px;font-weight:600;margin-top:2px}.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-weight:600;font-size:13px}
.small{font-size:12px;color:#666}.step{border-left:3px solid #7F77DD;padding:4px 10px;margin:6px 0;background:#F7F6FE}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def q(sql: str) -> pd.DataFrame:
    try:
        return db.read_sql(sql)
    except Exception:  # noqa: BLE001 - table not built yet
        return pd.DataFrame()


def pct(x, d=1):
    return "-" if x is None or pd.isna(x) else f"{x:+.{d}%}"


def pill(action):
    return f"<span class='pill' style='background:{BG.get(action, '#eee')};color:{COLORS.get(action, '#333')}'>{action}</span>"


def run_pipeline(args):
    with st.spinner("Running pipeline... (about 30 seconds on sample data, longer with live data and LLMs)"):
        r = subprocess.run([sys.executable, "-m", "msp.pipeline", *args], cwd=ROOT, capture_output=True, text=True)
    st.cache_data.clear()
    log = (r.stdout + r.stderr)[-4000:]
    (st.success if r.returncode == 0 else st.error)("Pipeline finished" if r.returncode == 0 else "Pipeline stopped")
    st.code("\n".join(l for l in log.splitlines() if "Calculating Metrics" not in l), language="text")


signals = q("SELECT * FROM mart_signals")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Market Signal ELT")
    st.caption("Raw market, earnings and news data into explainable, auto-refreshing Buy / Hold / Avoid "
               "research signals for 10 stocks and 4 metals.")
    if len(signals):
        origins = q("SELECT DISTINCT data_origin FROM clean_prices")["data_origin"].tolist()
        st.metric("As of", signals["as_of"].max())
        st.write("Price data:", ", ".join(origins))
        if "sample" in origins:
            st.warning("Showing the bundled SYNTHETIC sample (simulated prices and news). "
                       "Run with live data for real markets.")
        st.write("Decisions by:", ", ".join(signals["decided_by"].unique()))
    st.divider()
    st.subheader("Run")
    if st.button("Run full pipeline (live, sample fallback)", width="stretch"):
        run_pipeline(["run"])
    if st.button("Run full pipeline (sample only)", width="stretch"):
        run_pipeline(["run", "--sample"])
    with st.expander("Demo: break the data"):
        st.caption("Plants a negative price and a future date in the raw layer, then re-runs clean and the "
                   "quality gate. The gate fails and blocks downstream steps.")
        if st.button("Inject bad data and run quality gate"):
            subprocess.run([sys.executable, "-m", "msp.pipeline", "inject-bad-data"], cwd=ROOT)
            subprocess.run([sys.executable, "-m", "msp.pipeline", "step", "clean"], cwd=ROOT)
            run_pipeline(["step", "quality_gate"])
    st.divider()
    st.caption("Research signals only, not investment advice.")

if signals.empty:
    st.title("Market Signal ELT")
    st.info("No signals yet. Click **Run full pipeline** in the sidebar, or run `python -m msp.pipeline run`.")
    st.stop()

tabs = st.tabs(["Signals", "Asset detail", "Pipeline", "Data quality", "Evaluation", "AI usage"])

# ------------------------------------------------------------------ Signals
with tabs[0]:
    st.subheader(f"Signals for 14 assets, as of {signals['as_of'].max()}")
    c = signals["action"].value_counts()
    m = st.columns(4)
    m[0].metric("Buy", int(c.get("Buy", 0)))
    m[1].metric("Hold", int(c.get("Hold", 0)))
    m[2].metric("Avoid", int(c.get("Avoid", 0)))
    ev = q("SELECT AVG(grounding_score) AS g, SUM(CASE WHEN supported THEN 1 ELSE 0 END) AS s, COUNT(*) AS n "
           "FROM mart_eval_results")
    if len(ev):
        m[3].metric("Judge: traces supported", f"{int(ev['s'].iloc[0])}/{int(ev['n'].iloc[0])}")
    for cls, title in (("stock", "Stocks"), ("metal", "Metals")):
        st.markdown(f"#### {title}")
        sub = signals[signals.asset_class == cls].sort_values("score", ascending=False)
        cols = st.columns(5 if cls == "stock" else 4)
        for i, r in enumerate(sub.itertuples()):
            with cols[i % len(cols)]:
                st.markdown(
                    f"<div class='card'><div class='sym'>{r.symbol} &middot; {r.sector}</div>"
                    f"<div class='nm'>{r.name}</div><div class='px'>{r.price:,.2f}</div>"
                    f"<div class='small'>1d {pct(r.ret_1d)} &middot; 20d {pct(r.ret_20d)} &middot; RSI {r.rsi14:.0f}</div>"
                    f"<div style='margin-top:6px'>{pill(r.action)} <span class='small'>score {r.score:+d} "
                    f"&middot; {r.confidence}</span></div></div>", unsafe_allow_html=True)
    st.markdown("#### All signals")
    tbl = signals[["symbol", "name", "asset_class", "price", "action", "score", "confidence", "timing", "decided_by"]]
    st.dataframe(tbl, hide_index=True, width="stretch")

# ------------------------------------------------------------------ Asset detail
with tabs[1]:
    names = {f"{r.name} ({r.symbol})": r.symbol for r in signals.itertuples()}
    sym = names[st.selectbox("Asset", list(names))]
    s = signals[signals.symbol == sym].iloc[0]
    st.markdown(f"### {s['name']} ({sym}) {pill(s['action'])}", unsafe_allow_html=True)
    st.write(f"**When:** {s['timing']}")
    if s["summary"]:
        st.write(f"**Summary:** {s['summary']}")
    if s["override_reason"]:
        st.info(f"Model overrode the factor baseline ({s['baseline_action']}): {s['override_reason']}")
    left, right = st.columns([3, 2])
    with left:
        px = q(f"SELECT date, adj_close, ma20, ma50 FROM mart_features_daily WHERE symbol = '{sym}' "
               "AND is_trading_day = 1 ORDER BY date")
        px = px.tail(250).melt("date", var_name="series", value_name="value")
        px["date"] = pd.to_datetime(px["date"])
        st.altair_chart(alt.Chart(px).mark_line().encode(
            x=alt.X("date:T", title=None), y=alt.Y("value:Q", scale=alt.Scale(zero=False), title="Price"),
            color=alt.Color("series:N", legend=alt.Legend(orient="top"))).properties(height=300),
            width="stretch")
    with right:
        fac = q(f"SELECT factor, score FROM mart_signal_factors WHERE symbol = '{sym}'")
        st.altair_chart(alt.Chart(fac).mark_bar().encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[-2, 2])), y=alt.Y("factor:N", sort=None, title=None),
            color=alt.condition("datum.score > 0", alt.value("#1D9E75"), alt.value("#D85A30"))
        ).properties(height=260, title=f"Factor scores (total {s['score']:+d}, baseline {s['baseline_action']})"),
            width="stretch")
        st.caption(f"Decided by {s['decided_by']}. Next earnings: {s['next_report_date'] or 'n/a'}")

    st.markdown("#### Decision trace")
    tr = q(f"SELECT * FROM mart_decision_trace WHERE symbol = '{sym}' ORDER BY step_no")
    for r in tr.itertuples():
        cite = f" <span class='small'>[{r.evidence_ids}]</span>" if r.evidence_ids else ""
        st.markdown(f"<div class='step'><b>{r.step_no}. {r.step}</b><br>{r.finding}{cite}</div>",
                    unsafe_allow_html=True)
    risks = json.loads(s["risks"] or "[]")
    if risks:
        st.markdown("**Risks:** " + "; ".join(risks))
    jr = q(f"SELECT * FROM mart_eval_results WHERE symbol = '{sym}'")
    if len(jr):
        j = jr.iloc[0]
        (st.success if j["supported"] else st.warning)(
            f"Judge ({j['judged_by']}): {'supported' if j['supported'] else 'issues found'}, grounding "
            f"{j['grounding_score']:.2f}. {j['comment'] or ''} {'' if j['issues'] == '[]' else j['issues']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Evidence retrieved from Chroma")
        st.dataframe(q(f"SELECT evidence_id, date, kind, headline, outlets, event_type, sentiment, materiality, "
                       f"relevance FROM mart_evidence WHERE symbol = '{sym}' ORDER BY rank"),
                     hide_index=True, width="stretch")
    with c2:
        st.markdown("#### Historical analogs (similar past events, what happened next)")
        an = q(f"SELECT query_headline AS recent_event, analog_symbol, analog_date, analog_headline, similarity, "
               f"fwd_ret_5d, fwd_ret_10d, fwd_ret_20d FROM mart_analogs WHERE symbol = '{sym}' ORDER BY similarity DESC")
        if len(an):
            st.caption(f"Median 10-day return after analogs: {an['fwd_ret_10d'].median():+.2%} "
                       f"({(an['fwd_ret_10d'] > 0).mean():.0%} positive, n={len(an)})")
        st.dataframe(an, hide_index=True, width="stretch")

# ------------------------------------------------------------------ Pipeline
with tabs[2]:
    st.subheader("Pipeline runs (latest per step)")
    runs = q("SELECT * FROM ops_pipeline_runs")
    roadmap = [("1", "Orchestrate and trigger", "Airflow in Docker", "airflow"),
               ("2", "Extract", "yfinance, Finnhub, SEC EDGAR, FRED", "extract"),
               ("3", "Load raw", "Postgres / SQLite (COPY, INSERT)", "load_raw"),
               ("4", "Clean", "Pandas", "clean"),
               ("5", "Quality gate", "Great Expectations + SQL", "quality_gate"),
               ("6", "News and filings enrichment", "Llama (Ollama) + Chroma", "enrich"),
               ("7", "Build mart", "SQL: joins, CTEs, window functions", "build_mart"),
               ("8", "Retrieve evidence", "Chroma + SQL analog join", "retrieve"),
               ("9", "Signal synthesis", "Gemini + factor rules", "signals"),
               ("10", "Evaluate", "Gemini judge + SQL backtest", "evaluate"),
               ("11", "Auto-refresh", "Airflow Datasets", "airflow")]
    rows = []
    for n, name, tool, step in roadmap:
        r = runs[runs.step == step].tail(1) if len(runs) else pd.DataFrame()
        rows.append({"#": n, "step": name, "tool": tool,
                     "status": r["status"].iloc[0] if len(r) else ("see Airflow UI" if step == "airflow" else "-"),
                     "rows": int(r["rows_out"].iloc[0]) if len(r) else None,
                     "finished_at (UTC)": r["finished_at"].iloc[0] if len(r) else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("#### Row counts by layer")
    counts = []
    for t in ["raw_prices", "raw_news", "raw_earnings", "raw_filings", "raw_macro", "clean_prices", "clean_news",
              "clean_earnings", "clean_filings", "clean_macro", "clean_news_events", "mart_features_daily",
              "mart_evidence", "mart_analogs", "mart_signals", "mart_decision_trace"]:
        n = q(f'SELECT COUNT(*) AS n FROM "{t}"')
        counts.append({"layer": t.split("_")[0], "table": t, "rows": int(n["n"].iloc[0]) if len(n) else None})
    st.dataframe(pd.DataFrame(counts), hide_index=True, width="stretch")
    st.markdown("#### Step log")
    st.dataframe(runs.sort_values("finished_at", ascending=False).head(40) if len(runs) else runs,
                 hide_index=True, width="stretch")
    with st.expander("Mart SQL (sql/mart_features.sql)"):
        st.code((ROOT / "sql" / "mart_features.sql").read_text(), language="sql")

# ------------------------------------------------------------------ Data quality
with tabs[3]:
    dq = q("SELECT * FROM dq_results WHERE run_id = (SELECT MAX(run_id) FROM dq_results)")
    if dq.empty:
        st.info("No quality gate run yet.")
    else:
        ok = int(dq["success"].astype(bool).sum())
        crit = dq[(~dq["success"].astype(bool)) & (dq["critical"].astype(bool))]
        st.subheader(f"Quality gate: {ok}/{len(dq)} checks passed")
        if len(crit):
            st.error(f"BLOCKED: {len(crit)} critical checks failed, so downstream steps did not run.")
            st.markdown(f"**Explanation ({dq['explained_by'].iloc[0]}):**")
            st.text(dq["explanation"].iloc[0])
            st.caption("Recover: re-run the full pipeline. A fresh extract replaces the bad batch.")
        else:
            st.success("Gate passed. Data released to the mart.")
        st.dataframe(dq[["table_name", "check_name", "tool", "success", "observed", "target"]],
                     hide_index=True, width="stretch")
        rep = config.REPORT_DIR / "quality_report.html"
        if rep.exists():
            st.download_button("Download HTML quality report", rep.read_bytes(), "quality_report.html", "text/html")

# ------------------------------------------------------------------ Evaluation
with tabs[4]:
    st.subheader("LLM-as-judge: is every decision trace backed by its evidence?")
    st.dataframe(q("SELECT symbol, action, decided_by, supported, grounding_score, issues, comment, judged_by "
                   "FROM mart_eval_results"), hide_index=True, width="stretch")
    st.subheader("Backtest: rules model replayed over the last ~250 trading days")
    st.caption("Return 10 trading days after each signal. Excess return = vs the asset's own average, "
               "which removes the bias of assets that simply trended up.")
    bt = q("SELECT * FROM mart_backtest_summary ORDER BY asset_class, action")
    st.dataframe(bt, hide_index=True, width="stretch")
    st.subheader("Factor attribution: which inputs earn their place?")
    fa = q("SELECT * FROM mart_backtest_factor_summary WHERE reading <> 'neutral'")
    if len(fa):
        st.altair_chart(alt.Chart(fa).mark_bar().encode(
            x=alt.X("avg_excess_ret_10d:Q", axis=alt.Axis(format="%"), title="Avg excess 10-day return"),
            y=alt.Y("factor:N", title=None), color=alt.Color("reading:N", scale=alt.Scale(
                domain=["positive", "negative"], range=["#1D9E75", "#D85A30"])),
            yOffset="reading:N").properties(height=320), width="stretch")
        st.caption("A useful factor shows green bars right of zero and orange bars left of zero.")

# ------------------------------------------------------------------ AI usage
with tabs[5]:
    st.subheader("Where AI runs in this pipeline")
    st.dataframe(pd.DataFrame([
        {"step": "5 Quality gate", "model": "Llama (Ollama)", "job": "Explain failed checks in plain English"},
        {"step": "6 Enrichment", "model": "Llama (Ollama)", "job": "Headline to JSON: event type, sentiment, materiality"},
        {"step": "6 Enrichment", "model": "Embeddings + Chroma", "job": "Collapse the same story from several outlets"},
        {"step": "8 Retrieval", "model": "Chroma", "job": "Top-k evidence per asset; similar past events (analogs)"},
        {"step": "9 Synthesis", "model": "Gemini", "job": "Decision, timing and cited decision trace"},
        {"step": "10 Evaluate", "model": "Gemini", "job": "Judge: flag claims not backed by evidence"},
    ]), hide_index=True, width="stretch")
    calls = q("SELECT * FROM ops_llm_calls")
    if len(calls):
        g = calls.groupby(["provider", "task"]).agg(calls=("ok", "size"), ok=("ok", "sum"),
                                                     avg_latency_s=("latency_s", "mean")).reset_index()
        st.dataframe(g, hide_index=True, width="stretch")
    else:
        st.info("No LLM calls logged: no model was reachable, so every AI step used its offline rules fallback. "
                "Start Ollama or add GEMINI_API_KEY to .env and re-run.")
    ev = q("SELECT extracted_by, COUNT(*) AS stories FROM clean_news_events GROUP BY extracted_by")
    st.markdown("#### How each story was structured")
    st.dataframe(ev, hide_index=True)
    try:
        from msp import vectors

        meta = vectors.collection().metadata
        st.write(f"Chroma collection `{vectors.COLLECTION}`: {vectors.collection().count()} items, "
                 f"embeddings: {meta.get('embedding_backend')}")
    except Exception:  # noqa: BLE001
        pass
