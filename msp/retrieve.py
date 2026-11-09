"""Step 8 - Retrieve evidence: find supporting news and past analogs.

Evidence: for each asset, Chroma returns the stories and filings most relevant to the
asset's outlook from the last 30 days (retrieval-augmented generation without a chat box).
Analogs: for the asset's most important recent events, Chroma finds the most similar past
events across all assets (at least 30 days older, so no look-ahead). A SQL join then pulls
what the price did 5, 10 and 20 trading days after each analog.
"""
from datetime import date, timedelta

import pandas as pd

from . import config, db, vectors

EVIDENCE_K = 6
ANALOG_K = 5
ANALOG_SQL = """
DROP TABLE IF EXISTS mart_analogs
;
CREATE TABLE mart_analogs AS
SELECT c.*, f.date AS analog_trade_date, f.fwd_ret_5d, f.fwd_ret_10d, f.fwd_ret_20d
FROM mart_analog_candidates c
JOIN mart_features_daily f
  ON f.symbol = c.analog_symbol
 AND f.date = (SELECT MIN(p.date) FROM mart_features_daily p
                WHERE p.symbol = c.analog_symbol AND p.date >= c.analog_date)
WHERE f.fwd_ret_10d IS NOT NULL
"""


def retrieve() -> dict:
    col = vectors.collection()
    events = db.read_sql("SELECT * FROM clean_news_events")
    latest = db.read_sql("SELECT symbol, date FROM mart_asset_latest")
    asof = latest["date"].max()
    since = vectors.date_num((pd.Timestamp(asof) - timedelta(days=30)).strftime("%Y-%m-%d"))
    ev_by_id = events.set_index("story_id")
    evidence_rows, cand_rows = [], []

    for a in config.ASSETS:
        sym = a["symbol"]
        # ---- evidence: top-k relevant recent items for this asset
        qtext = (f"{a['name']} {a['sector']} outlook: earnings, guidance, demand, regulation, "
                 f"analyst ratings, rates and macro drivers")
        q = col.query(query_embeddings=vectors.embed([qtext]), n_results=EVIDENCE_K * 3,
                      where={"$and": [{"symbol": sym}, {"date_num": {"$gte": since}}]})
        hits = list(zip(q["ids"][0], q["distances"][0]))
        # re-rank: relevance x materiality, so a material event outranks a near-duplicate keyword match
        ranked = sorted(hits, key=lambda h: -(1 - h[1]) * (0.5 + ev_by_id.loc[h[0], "materiality"] / 5)
                        if h[0] in ev_by_id.index else 0)[:EVIDENCE_K]
        for rank, (sid, dist) in enumerate(ranked, 1):
            if sid not in ev_by_id.index:
                continue
            e = ev_by_id.loc[sid]
            evidence_rows.append({"symbol": sym, "rank": rank, "evidence_id": f"E{rank}", "story_id": sid,
                                  "date": e["date"], "kind": e["kind"], "headline": e["headline"],
                                  "outlets": e["outlets"], "event_type": e["event_type"],
                                  "sentiment": float(e["sentiment"]), "materiality": int(e["materiality"]),
                                  "relevance": round(1 - dist, 3), "url": e["url"]})

        # ---- analogs: similar past events for the 3 most material recent news events
        recent = events[(events.symbol == sym) & (events.kind == "news") &
                         (events.date >= (pd.Timestamp(asof) - timedelta(days=21)).strftime("%Y-%m-%d"))]
        recent = recent.assign(w=recent["impact_score"].abs()).sort_values(["w", "date"], ascending=False).head(3)
        for _, ev in recent.iterrows():
            limit = vectors.date_num((pd.Timestamp(ev["date"]) - timedelta(days=30)).strftime("%Y-%m-%d"))
            q2 = col.query(query_embeddings=vectors.embed([ev["headline"]]), n_results=ANALOG_K,
                           where={"$and": [{"kind": "news"}, {"date_num": {"$lt": limit}}]})
            for aid, dist, meta, doc in zip(q2["ids"][0], q2["distances"][0], q2["metadatas"][0], q2["documents"][0]):
                cand_rows.append({"symbol": sym, "query_story_id": ev["story_id"], "query_headline": ev["headline"],
                                  "query_sentiment": float(ev["sentiment"]), "analog_story_id": aid,
                                  "analog_symbol": meta["symbol"], "analog_date": meta["date"],
                                  "analog_headline": doc, "similarity": round(1 - dist, 3)})

    db.write_df(pd.DataFrame(evidence_rows), "mart_evidence", if_exists="replace")
    cand = pd.DataFrame(cand_rows, columns=["symbol", "query_story_id", "query_headline", "query_sentiment",
                                             "analog_story_id", "analog_symbol", "analog_date", "analog_headline",
                                             "similarity"])
    db.write_df(cand, "mart_analog_candidates", if_exists="replace")
    with db.engine().begin() as conn:
        from sqlalchemy import text

        for stmt in [s for s in ANALOG_SQL.split("\n;\n") if s.strip()]:
            conn.execute(text(stmt))
    n_analogs = int(db.read_sql("SELECT COUNT(*) AS n FROM mart_analogs")["n"].iloc[0])
    return {"as_of": asof, "evidence_rows": len(evidence_rows), "analog_candidates": len(cand),
            "analogs_with_outcomes": n_analogs, "embedding_backend": vectors.backend()}
