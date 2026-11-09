-- Step 7 - Build mart: join prices x news events x earnings x macro and compute features.
-- Runs unchanged on Postgres and SQLite. Statements are separated by a line with only ';'.

DROP TABLE IF EXISTS mart_features_daily
;
CREATE TABLE mart_features_daily AS
WITH px AS (
    -- daily price change, used for RSI
    SELECT symbol, name, asset_class, sector, date, adj_close, volume, ret_1d, is_trading_day, data_origin,
           adj_close - LAG(adj_close) OVER w AS chg
    FROM clean_prices
    WINDOW w AS (PARTITION BY symbol ORDER BY date)
),
ind AS (
    -- technical indicators and forward returns (forward returns feed analogs and the backtest)
    SELECT px.*,
           AVG(adj_close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
           AVG(adj_close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma50,
           AVG(CASE WHEN chg > 0 THEN chg ELSE 0 END)
               OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain14,
           AVG(CASE WHEN chg < 0 THEN -chg ELSE 0 END)
               OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss14,
           adj_close / NULLIF(LAG(adj_close, 20) OVER (PARTITION BY symbol ORDER BY date), 0) - 1 AS ret_20d,
           adj_close / NULLIF(LAG(adj_close, 60) OVER (PARTITION BY symbol ORDER BY date), 0) - 1 AS ret_60d,
           LEAD(adj_close, 5)  OVER (PARTITION BY symbol ORDER BY date) / NULLIF(adj_close, 0) - 1 AS fwd_ret_5d,
           LEAD(adj_close, 10) OVER (PARTITION BY symbol ORDER BY date) / NULLIF(adj_close, 0) - 1 AS fwd_ret_10d,
           LEAD(adj_close, 20) OVER (PARTITION BY symbol ORDER BY date) / NULLIF(adj_close, 0) - 1 AS fwd_ret_20d
    FROM px
),
ev AS (
    -- map each event to the first price date on or after it (weekend news lands on Monday)
    SELECT e.symbol, e.impact_score,
           (SELECT MIN(p.date) FROM clean_prices p WHERE p.symbol = e.symbol AND p.date >= e.date) AS date
    FROM clean_news_events e
),
evd AS (
    SELECT symbol, date, SUM(impact_score) AS impact_day, COUNT(*) AS events_day
    FROM ev WHERE date IS NOT NULL
    GROUP BY symbol, date
),
base AS (
    SELECT i.*, COALESCE(evd.impact_day, 0) AS impact_day, COALESCE(evd.events_day, 0) AS events_day
    FROM ind i
    LEFT JOIN evd ON evd.symbol = i.symbol AND evd.date = i.date
),
feat AS (
    -- 5 trading days = 7 calendar days of news sentiment
    SELECT base.*,
           SUM(impact_day) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS sentiment_7d,
           SUM(events_day) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS events_7d
    FROM base
),
mac AS (
    SELECT date, us_10y_yield, usd_index, cpi,
           us_10y_yield - LAG(us_10y_yield, 20) OVER (ORDER BY date) AS yield_chg_20d,
           usd_index / NULLIF(LAG(usd_index, 20) OVER (ORDER BY date), 0) - 1 AS usd_chg_20d
    FROM clean_macro
)
SELECT f.symbol, f.name, f.asset_class, f.sector, f.date, f.adj_close, f.volume, f.ret_1d, f.is_trading_day,
       f.data_origin,
       f.ma20, f.ma50,
       CASE WHEN f.avg_loss14 = 0 THEN 100
            ELSE 100 - 100 / (1 + f.avg_gain14 / NULLIF(f.avg_loss14, 0)) END AS rsi14,
       f.adj_close / NULLIF(f.ma20, 0) - 1 AS dist_ma20,
       CASE WHEN f.ma20 > f.ma50 THEN 1 ELSE 0 END AS uptrend,
       f.ret_20d, f.ret_60d,
       f.impact_day, f.events_day, f.sentiment_7d, f.events_7d,
       (SELECT e.surprise_pct FROM clean_earnings e
         WHERE e.symbol = f.symbol AND e.eps_actual IS NOT NULL AND e.report_date <= f.date
         ORDER BY e.report_date DESC LIMIT 1) AS last_eps_surprise_pct,
       (SELECT e.report_date FROM clean_earnings e
         WHERE e.symbol = f.symbol AND e.eps_actual IS NOT NULL AND e.report_date <= f.date
         ORDER BY e.report_date DESC LIMIT 1) AS last_report_date,
       (SELECT MIN(e.report_date) FROM clean_earnings e
         WHERE e.symbol = f.symbol AND e.report_date > f.date) AS next_report_date,
       m.us_10y_yield, m.usd_index, m.yield_chg_20d, m.usd_chg_20d,
       f.fwd_ret_5d, f.fwd_ret_10d, f.fwd_ret_20d
FROM feat f
LEFT JOIN mac m ON m.date = f.date
;
CREATE INDEX ix_mart_features_symbol_date ON mart_features_daily (symbol, date)
;
DROP TABLE IF EXISTS mart_asset_latest
;
CREATE TABLE mart_asset_latest AS
SELECT f.*
FROM mart_features_daily f
JOIN (SELECT symbol, MAX(date) AS date FROM mart_features_daily WHERE is_trading_day = 1 GROUP BY symbol) l
  ON l.symbol = f.symbol AND l.date = f.date
