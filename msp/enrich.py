"""Step 6 - News and filings enrichment: turn text into structured events.

1. Embed every headline and filing, and store them in Chroma (collection news_filings).
2. Dedup: a headline whose nearest neighbour (same symbol, within 1 day) is above the
   similarity threshold is the same story from another outlet, so it joins that story.
3. Extract: Llama reads each recent story and returns JSON (event type, sentiment,
   materiality 1-5). Older history is backfilled with keyword rules to save LLM calls.
   Results are cached per story, so re-runs only send new stories to the model.
Output table: clean_news_events (one row per unique story or filing).
"""
import json
import re
from datetime import date, timedelta

import pandas as pd

from . import config, db, llm, vectors

EVENT_TYPES = ["earnings", "guidance", "product", "legal", "mna", "analyst", "operations",
               "macro", "supply", "demand", "flows", "filing", "other"]
LLM_ENRICH_DAYS = int(__import__("os").getenv("LLM_ENRICH_DAYS", "30"))
BATCH = 10

RULES = [
    ("earnings", r"earnings|eps|quarterly results|beats|misses"),
    ("guidance", r"guidance|outlook|forecast"),
    ("mna", r"acquire|acquisition|merger|deal"),
    ("legal", r"probe|lawsuit|regulator|antitrust|fine|investigation"),
    ("analyst", r"upgrade|downgrade|price target|analyst"),
    ("operations", r"layoff|restructur|recall|strike|outage"),
    ("product", r"unveil|launch|new product|release"),
    ("macro", r"fed|rate|yield|inflation|jobs|treasury|dollar"),
    ("supply", r"supply|mine|output|production"),
    ("demand", r"demand|purchases|buying|china"),
    ("flows", r"etf|inflow|outflow|pull money"),
]
POS = r"beat|raise|rais|upgrade|rall|surge|jump|record|strong|inflow|step up|tighten|unveil|acquire|cut.*rate|sooner"
NEG = r"miss|cut|downgrade|slide|fall|drop|probe|layoff|weak|pull money|dent|slow|lawsuit|concern"


def rules_classify(text: str) -> dict:
    t = text.lower()
    etype = next((e for e, pat in RULES if re.search(pat, t)), "other")
    pos, neg = len(re.findall(POS, t)), len(re.findall(NEG, t))
    if "rate cut" in t or "rate cuts" in t:
        pos += 1
    sentiment = 0.0 if pos == neg else (0.6 if pos > neg else -0.6)
    materiality = 4 if etype in ("earnings", "guidance", "mna", "legal") else 3 if etype != "other" else 1
    return {"event_type": etype, "sentiment": sentiment, "materiality": materiality,
            "rationale": "keyword rules"}


def _llm_batch(rows: list[dict]) -> tuple[list[dict] | None, str]:
    items = [{"i": i, "asset": config.ASSET_BY_SYMBOL[r["symbol"]]["name"], "headline": r["headline"]}
             for i, r in enumerate(rows)]
    prompt = (
        "You are a financial news analyst extracting structured events for a data pipeline.\n"
        f"For each item return event_type (one of {EVENT_TYPES}), sentiment for the named asset's price "
        "from -1.0 (very negative) to 1.0 (very positive), materiality from 1 (noise) to 5 (moves the price), "
        "and a rationale of at most 15 words.\n"
        'Return JSON only: {"events": [{"i": int, "event_type": str, "sentiment": float, '
        '"materiality": int, "rationale": str}]}\n'
        f"Items: {json.dumps(items)}")
    out, provider = llm.complete_json(prompt, role="bulk", task="news_extraction")
    if not out:
        return None, provider
    events = out.get("events", out) if isinstance(out, dict) else out
    by_i = {e.get("i"): e for e in events if isinstance(e, dict)}
    result = []
    for i in range(len(rows)):
        e = by_i.get(i)
        if not e:
            return None, provider
        et = e.get("event_type") if e.get("event_type") in EVENT_TYPES else "other"
        try:
            s = max(-1.0, min(1.0, float(e.get("sentiment", 0))))
            m = max(1, min(5, int(e.get("materiality", 1))))
        except (TypeError, ValueError):
            return None, provider
        result.append({"event_type": et, "sentiment": s, "materiality": m,
                       "rationale": str(e.get("rationale", ""))[:200]})
    return result, provider


def _filing_sentiment(desc: str) -> float:
    m = re.search(r"EPS ([-\d.]+) vs consensus ([-\d.]+)", desc or "")
    if m:
        a, e = float(m.group(1)), float(m.group(2))
        return 0.5 if a > e * 1.02 else -0.5 if a < e * 0.98 else 0.0
    return 0.0


def enrich() -> dict:
    news = db.read_sql("SELECT * FROM clean_news ORDER BY symbol, published_at")
    filings = db.read_sql("SELECT * FROM clean_filings ORDER BY symbol, filing_date")
    col = vectors.fresh_collection()
    thr = vectors.dedup_threshold()

    # ---- 1-2. embed and dedup headlines into stories, using Chroma nearest-neighbour search
    stories: list[dict] = []
    if len(news):
        vecs = vectors.embed((news["headline"]).tolist())
        for (_, r), v in zip(news.iterrows(), vecs):
            dn = vectors.date_num(r["date"])
            match = None
            if col.count():
                q = col.query(query_embeddings=[v], n_results=1,
                              where={"$and": [{"symbol": r["symbol"]}, {"kind": "news"},
                                              {"date_num": {"$gte": dn - 1}}, {"date_num": {"$lte": dn + 1}}]})
                if q["ids"][0] and 1 - q["distances"][0][0] >= thr:
                    match = q["ids"][0][0]
            if match:
                s = next(s for s in stories if s["story_id"] == match)
                s["outlets"].append(r["outlet"])
                s["news_ids"].append(r["news_id"])
                continue
            sid = f"n_{r['symbol']}_{r['news_id']}"
            col.add(ids=[sid], embeddings=[v], documents=[r["headline"]],
                    metadatas=[{"symbol": r["symbol"], "kind": "news", "date": r["date"], "date_num": dn}])
            stories.append({"story_id": sid, "symbol": r["symbol"], "date": r["date"], "kind": "news",
                            "headline": r["headline"], "url": r["url"], "outlets": [r["outlet"]],
                            "news_ids": [r["news_id"]], "data_origin": r["data_origin"]})

    # ---- 3. extract structured events: LLM for recent stories (cached), rules for history
    cache = {}
    if db.table_exists("clean_news_events"):
        prev = db.read_sql("SELECT story_id, headline, event_type, sentiment, materiality, rationale, extracted_by "
                           "FROM clean_news_events WHERE extracted_by <> 'keyword rules (history backfill)'")
        cache = {(r.story_id, r.headline): r._asdict() for r in prev.itertuples()}
    cutoff = (date.today() - timedelta(days=LLM_ENRICH_DAYS)).isoformat()
    recent = [s for s in stories if s["date"] >= cutoff]
    todo = [s for s in recent if (s["story_id"], s["headline"]) not in cache]
    llm_used, offline = 0, 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        res, provider = _llm_batch(chunk)
        for s, e in zip(chunk, res or [rules_classify(c["headline"]) for c in chunk]):
            s.update(e)
            s["extracted_by"] = provider if res else "offline rules"
        llm_used += bool(res)
        offline += not res
    for s in stories:
        key = (s["story_id"], s["headline"])
        if "event_type" in s:
            continue
        if key in cache:
            c = cache[key]
            s.update({k: c[k] for k in ("event_type", "sentiment", "materiality", "rationale", "extracted_by")})
        else:
            s.update(rules_classify(s["headline"]))
            s["extracted_by"] = "keyword rules (history backfill)" if s["date"] < cutoff else "offline rules"

    # ---- filings become events too, and go into the knowledge base
    fil_events = []
    if len(filings):
        texts = filings["description"].fillna("").tolist()
        fvecs = vectors.embed(texts)
        ids, metas = [], []
        for (_, f), v in zip(filings.iterrows(), fvecs):
            fid = f"f_{f['accession']}"
            ids.append(fid)
            metas.append({"symbol": f["symbol"], "kind": "filing", "date": f["filing_date"],
                          "date_num": vectors.date_num(f["filing_date"])})
            fil_events.append({"story_id": fid, "symbol": f["symbol"], "date": f["filing_date"], "kind": "filing",
                               "headline": f["description"], "url": "", "outlets": ["SEC EDGAR"],
                               "news_ids": [], "event_type": "filing", "sentiment": _filing_sentiment(f["description"]),
                               "materiality": 3 if f["form"] == "8-K" else 2, "rationale": f"SEC form {f['form']}",
                               "extracted_by": "SEC form type", "data_origin": f["data_origin"]})
        for i in range(0, len(ids), 500):
            col.add(ids=ids[i:i + 500], embeddings=fvecs[i:i + 500], documents=texts[i:i + 500],
                    metadatas=metas[i:i + 500])

    rows = stories + fil_events
    df = pd.DataFrame(rows)
    df["n_outlets"] = df["outlets"].apply(len)
    df["outlets"] = df["outlets"].apply(lambda x: ", ".join(dict.fromkeys(x)))
    df["news_ids"] = df["news_ids"].apply(lambda x: ",".join(map(str, x)))
    # numeric score used by the mart: direction x weight
    df["impact_score"] = df["sentiment"].astype(float) * df["materiality"].astype(float) / 5
    cols = ["story_id", "symbol", "date", "kind", "headline", "url", "outlets", "n_outlets", "news_ids",
            "event_type", "sentiment", "materiality", "impact_score", "rationale", "extracted_by", "data_origin"]
    db.write_df(df[cols], "clean_news_events", if_exists="replace")
    llm.flush_call_log()
    return {"headlines": len(news), "unique_stories": len(stories),
            "duplicates_collapsed": len(news) - len(stories), "filings": len(fil_events),
            "llm_extracted_recent": int(sum(1 for s in recent if "rules" not in s["extracted_by"])),
            "llm_batches": llm_used, "offline_batches": offline,
            "embedding_backend": vectors.backend(), "dedup_threshold": thr, "chroma_items": col.count()}
