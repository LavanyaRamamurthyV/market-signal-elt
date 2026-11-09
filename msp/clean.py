"""Step 4 - Clean: fix types, duplicates and gaps with pandas. Writes clean_* tables.

Deliberately NOT fixed here: impossible values such as negative prices. Those are left
for the quality gate to catch, so bad upstream data blocks the run instead of being hidden.
"""
from datetime import date

import numpy as np
import pandas as pd

from . import config, db


def _latest_batch(table: str) -> pd.DataFrame:
    return db.read_sql(f'SELECT * FROM "{table}" WHERE _batch_id = (SELECT MAX(_batch_id) FROM "{table}")')


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({".": None, "": None, "None": None, "nan": None}), errors="coerce")


def clean_prices() -> pd.DataFrame:
    df = _latest_batch("raw_prices")
    for c in ("open", "high", "low", "close", "volume", "dividends", "stock_splits"):
        df[c] = _num(df[c])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol"])
    df = df.sort_values(["symbol", "date", "_loaded_at"]).drop_duplicates(["symbol", "date"], keep="last")
    pre_adjusted = df.get("split_adjusted", pd.Series("False", index=df.index)).astype(str).eq("True")

    frames = []
    calendar = pd.bdate_range(df["date"].min(), df["date"].max())
    for sym, g in df.groupby("symbol"):
        already_split_adjusted = bool(pre_adjusted.loc[g.index].all())
        g = g.set_index("date").sort_index()
        # back-adjust splits (only when the source has not already done it)
        split_f = pd.Series(1.0, index=g.index)
        if not already_split_adjusted:
            for d, ratio in g.loc[g["stock_splits"].fillna(0) > 0, "stock_splits"].items():
                split_f[split_f.index < d] /= ratio
        # back-adjust dividends: earlier prices scaled by (1 - dividend / previous close)
        div_f = pd.Series(1.0, index=g.index)
        prev_close = g["close"].shift(1)
        for d, dv in g.loc[g["dividends"].fillna(0) > 0, "dividends"].items():
            pc = prev_close.get(d)
            if pc and pc > 0:
                div_f[div_f.index < d] *= 1 - dv / pc
        g["adj_close"] = g["close"] * split_f * div_f
        for c in ("open", "high", "low"):
            g[c] = g[c] * split_f
        # align to one business-day calendar so stocks and metals join cleanly
        g = g.reindex(g.index.union(calendar[(calendar >= g.index.min()) & (calendar <= g.index.max())]))
        g["is_trading_day"] = g["close"].notna().astype(int)
        g[["open", "high", "low", "close", "adj_close"]] = g[["open", "high", "low", "close", "adj_close"]].ffill()
        g["symbol"] = sym
        g["volume"] = g["volume"].where(g["is_trading_day"] == 1, 0)
        g["ret_1d"] = g["adj_close"].pct_change()
        g["data_origin"] = g["_source"].ffill()
        frames.append(g.reset_index(names="date"))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    meta = pd.DataFrame(config.ASSETS)[["symbol", "name", "asset_class", "sector"]]
    out = out.merge(meta, on="symbol", how="left")
    cols = ["symbol", "name", "asset_class", "sector", "date", "open", "high", "low", "close", "adj_close",
            "volume", "dividends", "stock_splits", "ret_1d", "is_trading_day", "data_origin"]
    return out[cols]


def clean_news() -> pd.DataFrame:
    df = db.read_sql('SELECT * FROM "raw_news"')  # news accumulates across batches
    df["headline"] = df["headline"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
    df = df[df["headline"].str.len() > 10]
    df["published_at"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"])
    df = df[df["symbol"].isin(config.ASSET_BY_SYMBOL)]
    df = df.sort_values("_loaded_at").drop_duplicates(["symbol", "id"], keep="last")
    df["date"] = df["published_at"].dt.strftime("%Y-%m-%d")
    df["published_at"] = df["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.rename(columns={"id": "news_id", "source": "outlet", "_source": "data_origin"})
    return df[["news_id", "symbol", "date", "published_at", "headline", "summary", "outlet", "url",
               "data_origin"]].reset_index(drop=True)


def clean_earnings() -> pd.DataFrame:
    df = _latest_batch("raw_earnings")
    for c in ("eps_actual", "eps_estimate"):
        df[c] = _num(df[c])
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["report_date"]).drop_duplicates(["symbol", "report_date"], keep="last")
    df["surprise_pct"] = np.where(df["eps_estimate"].abs() > 0,
                                  (df["eps_actual"] - df["eps_estimate"]) / df["eps_estimate"].abs() * 100, np.nan)
    df["is_upcoming"] = (df["eps_actual"].isna() & (df["report_date"] >= date.today().isoformat())).astype(int)
    df = df.rename(columns={"_source": "data_origin"})
    return df[["symbol", "period", "report_date", "eps_actual", "eps_estimate", "surprise_pct", "is_upcoming",
               "data_origin"]].sort_values(["symbol", "report_date"]).reset_index(drop=True)


def clean_filings() -> pd.DataFrame:
    df = _latest_batch("raw_filings").drop_duplicates("accession")
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"_source": "data_origin"})
    return df[["symbol", "form", "filing_date", "description", "accession", "data_origin"]].reset_index(drop=True)


def clean_macro() -> pd.DataFrame:
    df = _latest_batch("raw_macro")
    df["value"] = _num(df["value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    wide = (df.dropna(subset=["date"]).pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
              .rename(columns=config.FRED_SERIES))
    wide = wide.reindex(pd.bdate_range(wide.index.min(), wide.index.max())).ffill()  # monthly CPI to daily
    wide.index.name = "date"
    out = wide.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    for c in config.FRED_SERIES.values():
        if c not in out:
            out[c] = np.nan
    out["data_origin"] = df["_source"].iloc[0] if len(df) else "unknown"
    return out[["date", *config.FRED_SERIES.values(), "data_origin"]]


CLEANERS = {"prices": clean_prices, "news": clean_news, "earnings": clean_earnings,
            "filings": clean_filings, "macro": clean_macro}


def clean(sources: list[str] | None = None) -> dict:
    summary = {}
    for src in sources or config.SOURCES:
        raw_rows = int(db.read_sql(f'SELECT COUNT(*) AS n FROM "raw_{src}"')["n"].iloc[0])
        df = CLEANERS[src]()
        db.write_df(df, f"clean_{src}", if_exists="replace")
        summary[src] = {"table": f"clean_{src}", "rows_in_raw": raw_rows, "rows_out": len(df)}
    return summary
