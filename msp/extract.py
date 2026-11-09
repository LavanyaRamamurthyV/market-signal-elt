"""Step 2 - Extract: pull each source and land the response as raw JSON files.

Live sources: Yahoo Finance (yfinance) for prices and news, Finnhub for company news and
earnings, SEC EDGAR for filings and reported EPS, FRED for macro series.
In DATA_MODE=auto a source that fails falls back to the bundled synthetic sample, and the
landing file records which one was used.
"""
import io
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from . import config, sample_data

log = logging.getLogger(__name__)
_SAMPLE_CACHE: dict | None = None
HTTP_TIMEOUT = 20


def _sample(source: str) -> list[dict]:
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is None:
        _SAMPLE_CACHE = sample_data.generate()
    return _SAMPLE_CACHE[source]


def _get(url: str, params: dict | None = None, headers: dict | None = None, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"rate limited: {url}")


# --------------------------------------------------------------------------- live fetchers
def live_prices() -> list[dict]:
    import yfinance as yf

    symbols = [a["symbol"] for a in config.ASSETS]
    period = f"{max(1, config.HISTORY_DAYS // 365)}y"
    df = yf.download(symbols, period=period, interval="1d", actions=True, auto_adjust=False,
                     group_by="ticker", progress=False, threads=True)
    recs = []
    for sym in symbols:
        if sym not in df.columns.get_level_values(0):
            continue
        sub = df[sym].dropna(how="all")
        for d, row in sub.iterrows():
            recs.append({
                "symbol": sym, "date": d.strftime("%Y-%m-%d"),
                "open": _num(row.get("Open")), "high": _num(row.get("High")),
                "low": _num(row.get("Low")), "close": _num(row.get("Close")),
                "volume": _num(row.get("Volume")),
                "dividends": _num(row.get("Dividends")) or 0.0,
                "stock_splits": _num(row.get("Stock Splits")) or 0.0,
                "split_adjusted": True,  # Yahoo Close is already split adjusted
            })
    if len({r["symbol"] for r in recs}) < len(symbols) // 2:
        raise RuntimeError("yfinance returned too few symbols")
    return recs


def _num(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def live_news() -> list[dict]:
    end = date.today()
    start = end - timedelta(days=config.NEWS_DAYS)
    recs = []
    if config.FINNHUB_API_KEY:
        for a in config.STOCKS:
            r = _get("https://finnhub.io/api/v1/company-news",
                     {"symbol": a["symbol"].replace("-", "."), "from": start.isoformat(),
                      "to": end.isoformat(), "token": config.FINNHUB_API_KEY})
            for n in r.json()[:40]:
                recs.append({"id": str(n.get("id")), "symbol": a["symbol"],
                             "datetime": datetime.fromtimestamp(n["datetime"], timezone.utc).isoformat(),
                             "headline": n.get("headline", ""), "summary": n.get("summary", ""),
                             "source": n.get("source", "finnhub"), "url": n.get("url", "")})
            time.sleep(1.1)  # free tier: 60 calls per minute
    # Yahoo news covers metals, and stocks too when there is no Finnhub key
    import yfinance as yf

    targets = config.METALS if config.FINNHUB_API_KEY else config.ASSETS
    for a in targets:
        for n in (yf.Ticker(a["symbol"]).news or [])[:30]:
            c = n.get("content", n)
            ts = c.get("pubDate") or c.get("providerPublishTime")
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts, timezone.utc).isoformat()
            recs.append({"id": str(n.get("id") or c.get("id") or c.get("uuid")), "symbol": a["symbol"],
                         "datetime": ts, "headline": c.get("title", ""),
                         "summary": c.get("summary", "") or c.get("description", ""),
                         "source": (c.get("provider") or {}).get("displayName", c.get("publisher", "yahoo")),
                         "url": ((c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict)
                                 else c.get("link", ""))})
    if not recs:
        raise RuntimeError("no live news returned")
    return recs


def live_earnings() -> list[dict]:
    recs = []
    if config.FINNHUB_API_KEY:
        for a in config.STOCKS:
            r = _get("https://finnhub.io/api/v1/stock/earnings",
                     {"symbol": a["symbol"].replace("-", "."), "token": config.FINNHUB_API_KEY})
            for e in r.json():
                recs.append({"symbol": a["symbol"], "period": e.get("period"), "report_date": e.get("period"),
                             "eps_actual": e.get("actual"), "eps_estimate": e.get("estimate")})
            time.sleep(1.1)
        cal = _get("https://finnhub.io/api/v1/calendar/earnings",
                   {"from": date.today().isoformat(), "to": (date.today() + timedelta(days=60)).isoformat(),
                    "token": config.FINNHUB_API_KEY}).json().get("earningsCalendar", [])
        wanted = {a["symbol"].replace("-", "."): a["symbol"] for a in config.STOCKS}
        for e in cal:
            if e.get("symbol") in wanted:
                recs.append({"symbol": wanted[e["symbol"]], "period": None, "report_date": e.get("date"),
                             "eps_actual": None, "eps_estimate": e.get("epsEstimate")})
        return recs
    # No Finnhub key: reported EPS straight from SEC EDGAR XBRL (no estimates). Some filers do not
    # tag diluted EPS (Berkshire reports per Class A share), so try basic EPS too and skip a
    # company rather than failing the whole source.
    for a in config.STOCKS:
        for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
            try:
                r = _get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{a['cik']:010d}/us-gaap/{concept}.json",
                         headers={"User-Agent": config.SEC_USER_AGENT}, retries=1)
            except requests.RequestException:
                time.sleep(0.15)
                continue
            for f in r.json().get("units", {}).get("USD/shares", []):
                if f.get("form") == "10-Q" and f.get("fp", "").startswith("Q"):
                    recs.append({"symbol": a["symbol"], "period": f["end"], "report_date": f["filed"],
                                 "eps_actual": f["val"], "eps_estimate": None})
            time.sleep(0.15)  # SEC fair-access limit is 10 requests per second
            break
    if not recs:
        raise RuntimeError("SEC returned no EPS for any company")
    return recs


def live_filings() -> list[dict]:
    recs = []
    cutoff = (date.today() - timedelta(days=config.HISTORY_DAYS)).isoformat()
    for a in config.STOCKS:
        r = _get(f"https://data.sec.gov/submissions/CIK{a['cik']:010d}.json",
                 headers={"User-Agent": config.SEC_USER_AGENT})
        rec = r.json()["filings"]["recent"]
        for form, fdate, desc, acc in zip(rec["form"], rec["filingDate"],
                                          rec.get("primaryDocDescription", [""] * len(rec["form"])),
                                          rec["accessionNumber"]):
            if form in ("8-K", "10-Q", "10-K") and fdate >= cutoff:
                recs.append({"symbol": a["symbol"], "form": form, "filing_date": fdate,
                             "description": f"{a['name']} {form} {desc or ''}".strip(), "accession": acc})
        time.sleep(0.15)
    return recs


def live_macro() -> list[dict]:
    recs = []
    start = (date.today() - timedelta(days=config.HISTORY_DAYS + 60)).isoformat()
    for sid in config.FRED_SERIES:
        if config.FRED_API_KEY:
            r = _get("https://api.stlouisfed.org/fred/series/observations",
                     {"series_id": sid, "api_key": config.FRED_API_KEY, "file_type": "json",
                      "observation_start": start})
            rows = [(o["date"], o["value"]) for o in r.json()["observations"]]
        else:  # keyless CSV download
            r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": sid, "cosd": start})
            df = pd.read_csv(io.StringIO(r.text))
            rows = list(df.itertuples(index=False, name=None))
        for d, v in rows:
            recs.append({"series_id": sid, "date": str(d), "value": v})
    return recs


LIVE = {"prices": live_prices, "news": live_news, "earnings": live_earnings,
        "filings": live_filings, "macro": live_macro}


# --------------------------------------------------------------------------- step entry point
def extract(sources: list[str] | None = None, batch_id: str | None = None) -> dict:
    """Extract each source and write landing/<source>/<batch_id>.json. Returns a summary."""
    batch_id = batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    summary = {}
    for src in sources or config.SOURCES:
        origin, error = "sample", None
        if config.DATA_MODE in ("auto", "live"):
            try:
                recs = LIVE[src]()
                origin = "live"
                for r in recs:
                    r.setdefault("data_origin", "live")
            except Exception as e:  # noqa: BLE001 - any source failure falls back
                error = f"{type(e).__name__}: {str(e)[:160]}"
                if config.DATA_MODE == "live":
                    raise
                log.warning("live %s failed (%s); using bundled sample", src, error)
        if origin == "sample":
            recs = [dict(r) for r in _sample(src)]
        folder = config.LANDING_DIR / src
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{batch_id}.json"
        path.write_text(json.dumps({"source": src, "origin": origin, "batch_id": batch_id,
                                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                                    "live_error": error, "records": recs}, default=str))
        summary[src] = {"origin": origin, "records": len(recs), "file": str(path), "live_error": error}
    return {"batch_id": batch_id, "sources": summary}
