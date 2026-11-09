"""Bundled SYNTHETIC sample so the demo runs with no network or API keys.

Every record carries source="sample". The numbers are simulated, not real market data.
The generator plants realistic mess for the pipeline to handle: duplicate rows,
numbers stored as text, missing volumes, a stock split, dividends, and the same news
story syndicated by several outlets with different wording.
News events move later prices a little, so historical analogs and the backtest have
something real to find.
"""
import hashlib
import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import config

SEED = 42
OUTLETS = ["Market Wire", "Daily Ledger", "Finance Desk", "Street Brief"]

STOCK_EVENTS = [
    ("earnings", +1, "{name} beats quarterly earnings estimates as revenue climbs {pct}%"),
    ("earnings", -1, "{name} misses earnings expectations; revenue growth slows to {pct}%"),
    ("guidance", +1, "{name} raises full-year guidance on strong demand"),
    ("guidance", -1, "{name} cuts outlook, citing weaker orders and higher costs"),
    ("product", +1, "{name} unveils new flagship product line ahead of holiday season"),
    ("legal", -1, "Regulators open probe into {name} business practices"),
    ("mna", +1, "{name} agrees to acquire smaller rival in ${bn} billion deal"),
    ("analyst", +1, "Analysts upgrade {name} to buy, lifting price targets"),
    ("analyst", -1, "{name} downgraded to underperform on valuation concerns"),
    ("operations", -1, "{name} announces layoffs of {pct}% of workforce amid restructuring"),
]
METAL_EVENTS = [
    ("macro", +1, "{name} rallies as Fed signals rate cuts may come sooner"),
    ("macro", -1, "{name} slides as strong jobs data lifts Treasury yields"),
    ("supply", +1, "Mine disruptions tighten global {lname} supply"),
    ("demand", +1, "Central banks step up {lname} purchases, data shows"),
    ("demand", -1, "Weak factory data in China dents {lname} demand outlook"),
    ("flows", -1, "Investors pull money from {lname} ETFs for third straight week"),
    ("flows", +1, "{name} ETF inflows hit highest level in months"),
]
REWORDS = [
    lambda h: h,
    lambda h: "Report: " + h[0].lower() + h[1:],
    lambda h: h + ", according to people familiar with the matter",
    lambda h: "UPDATE 1-" + h,
]
BASE_PRICE = {"NVDA": 120, "AAPL": 190, "MSFT": 410, "GOOGL": 160, "AMZN": 180, "META": 500,
              "AVGO": 160, "TSLA": 240, "BRK-B": 430, "JPM": 210,
              "GC=F": 2400, "SI=F": 29, "HG=F": 4.3, "PL=F": 980}
VOL = {"TSLA": 0.035, "NVDA": 0.03, "AVGO": 0.026, "META": 0.024, "SI=F": 0.022,
       "PL=F": 0.018, "HG=F": 0.016, "GC=F": 0.009, "BRK-B": 0.01, "JPM": 0.014}


def _end_date() -> date:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def business_days() -> pd.DatetimeIndex:
    end = _end_date()
    return pd.bdate_range(end=end, periods=int(config.HISTORY_DAYS * 252 / 365))


def _id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def generate() -> dict[str, list[dict]]:
    rng = np.random.default_rng(SEED)
    rnd = random.Random(SEED)
    days = business_days()
    n = len(days)
    out = {s: [] for s in config.SOURCES}

    for asset in config.ASSETS:
        sym, name = asset["symbol"], asset["name"]
        is_stock = asset["asset_class"] == "stock"
        templates = STOCK_EVENTS if is_stock else METAL_EVENTS

        # ---- news events: about one story every 12 trading days, more in the last month
        event_idx = sorted(set(rng.choice(np.arange(5, n - 1), size=n // 12, replace=False).tolist()
                               + rng.choice(np.arange(n - 22, n), size=4, replace=False).tolist()))
        shock = np.zeros(n)
        for i in event_idx:
            etype, direction, tmpl = templates[rnd.randrange(len(templates))]
            headline = tmpl.format(name=name, lname=name.lower(), pct=rnd.randint(3, 25), bn=rnd.randint(2, 40))
            strength = rnd.uniform(0.004, 0.012)
            shock[i + 1: i + 8] += direction * strength / 2  # drift over the following week
            ts = datetime.combine(days[i].date(), datetime.min.time()) + timedelta(hours=rnd.randint(7, 20))
            outlets = rnd.sample(OUTLETS, k=rnd.choice([1, 1, 2, 3]))
            for k, outlet in enumerate(outlets):
                h = REWORDS[k](headline)
                out["news"].append({
                    "id": _id(sym, i, outlet), "symbol": sym,
                    "datetime": (ts + timedelta(minutes=17 * k)).isoformat(),
                    "headline": h, "summary": h + ".", "source": outlet,
                    "url": f"https://example.com/sample/{_id(sym, i, outlet)}",
                })

        # ---- prices: random walk plus the news drift above
        vol = VOL.get(sym, 0.017)
        rets = rng.normal(0.0004, vol, n) + shock
        close = BASE_PRICE[sym] * np.exp(np.cumsum(rets))
        split_at = n // 2 if sym == "NVDA" else None
        for i, d in enumerate(days):
            c = close[i] * (4 if split_at and i < split_at else 1)  # 4:1 split mid-history
            o = c * (1 + rng.normal(0, vol / 3))
            rec = {
                "symbol": sym, "date": d.strftime("%Y-%m-%d"),
                "open": round(o, 4), "high": round(max(o, c) * (1 + abs(rng.normal(0, vol / 2))), 4),
                "low": round(min(o, c) * (1 - abs(rng.normal(0, vol / 2))), 4), "close": round(c, 4),
                "volume": int(rng.integers(2e6, 9e7)) if is_stock else int(rng.integers(5e4, 4e5)),
                "dividends": 0.0, "stock_splits": 4.0 if split_at and i == split_at else 0.0,
            }
            if is_stock and sym in ("AAPL", "MSFT", "JPM") and i % 63 == 30:
                rec["dividends"] = round(c * 0.0025, 4)
            if rng.random() < 0.01:
                rec["volume"] = None  # missing volume
            if rng.random() < 0.02:
                rec["close"] = str(rec["close"])  # number delivered as text
            out["prices"].append(rec)
            if rng.random() < 0.005:
                out["prices"].append(dict(rec))  # duplicate row

        # ---- earnings and filings (stocks only)
        if is_stock:
            for q, i in enumerate(range(40, n, 63)):
                est = round(1.0 + 0.05 * q + rng.normal(0, 0.1), 2)
                act = round(est * (1 + rng.normal(0.03, 0.08)), 2)
                rdate = days[i].strftime("%Y-%m-%d")
                out["earnings"].append({"symbol": sym, "period": (days[i] - pd.Timedelta(days=25)).strftime("%Y-%m-%d"),
                                        "report_date": rdate, "eps_actual": act, "eps_estimate": est})
                out["filings"].append({"symbol": sym, "form": "8-K", "filing_date": rdate,
                                       "description": f"{name} results of operations and financial condition: "
                                                      f"diluted EPS {act} vs consensus {est}",
                                       "accession": _id(sym, "8k", q)})
                out["filings"].append({"symbol": sym, "form": "10-Q", "filing_date": (days[min(i + 3, n - 1)]).strftime("%Y-%m-%d"),
                                       "description": f"{name} quarterly report",
                                       "accession": _id(sym, "10q", q)})
            # next report date, so the earnings sensor has something upcoming
            nxt = (days[-1] + pd.Timedelta(days=rnd.randint(3, 40))).strftime("%Y-%m-%d")
            out["earnings"].append({"symbol": sym, "period": None, "report_date": nxt,
                                    "eps_actual": None, "eps_estimate": round(1.5 + rng.normal(0, 0.1), 2)})

    # ---- macro drivers
    y, usd = 4.2, 121.0
    for i, d in enumerate(days):
        y += rng.normal(0, 0.04)
        usd += rng.normal(0, 0.25)
        ds = d.strftime("%Y-%m-%d")
        out["macro"].append({"series_id": "DGS10", "date": ds, "value": round(y, 3)})
        out["macro"].append({"series_id": "DTWEXBGS", "date": ds, "value": round(usd, 3)})
        if d.day <= 3 and (i == 0 or days[i - 1].month != d.month):
            out["macro"].append({"series_id": "CPIAUCSL", "date": ds, "value": round(310 + i * 0.03, 2)})

    for recs in out.values():
        for r in recs:
            r["source"] = r.get("source", "sample") if "headline" in r else "sample"
            r["data_origin"] = "sample"
    return out
