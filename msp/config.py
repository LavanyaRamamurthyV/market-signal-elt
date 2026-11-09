"""Central configuration: asset universe, paths, environment settings."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
LANDING_DIR = DATA_DIR / "landing"
CHROMA_DIR = DATA_DIR / "chroma"
REPORT_DIR = DATA_DIR / "reports"
SQL_DIR = ROOT / "sql"

# Warehouse: Postgres in Docker, SQLite file for the quick demo
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'warehouse.db'}")

# Data mode: "auto" tries live APIs and falls back to the bundled sample per source,
# "sample" uses the bundled synthetic sample only, "live" fails if a source is down.
DATA_MODE = os.getenv("DATA_MODE", "auto").lower()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# "auto" = Llama for bulk work, Gemini for reasoning, offline rules if neither is reachable
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "market-signal-elt research demo contact@example.com")

HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "730"))
NEWS_DAYS = int(os.getenv("NEWS_DAYS", "30"))

# 10 large US-listed companies (all SEC filers) + 4 metals. Edit to change the universe.
STOCKS = [
    {"symbol": "NVDA", "name": "NVIDIA", "cik": 1045810, "sector": "Semiconductors"},
    {"symbol": "AAPL", "name": "Apple", "cik": 320193, "sector": "Technology hardware"},
    {"symbol": "MSFT", "name": "Microsoft", "cik": 789019, "sector": "Software"},
    {"symbol": "GOOGL", "name": "Alphabet", "cik": 1652044, "sector": "Internet"},
    {"symbol": "AMZN", "name": "Amazon", "cik": 1018724, "sector": "Retail and cloud"},
    {"symbol": "META", "name": "Meta Platforms", "cik": 1326801, "sector": "Internet"},
    {"symbol": "AVGO", "name": "Broadcom", "cik": 1730168, "sector": "Semiconductors"},
    {"symbol": "TSLA", "name": "Tesla", "cik": 1318605, "sector": "Automotive"},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway", "cik": 1067983, "sector": "Financials"},
    {"symbol": "JPM", "name": "JPMorgan Chase", "cik": 19617, "sector": "Banking"},
]
METALS = [
    {"symbol": "GC=F", "name": "Gold", "sector": "Precious metal"},
    {"symbol": "SI=F", "name": "Silver", "sector": "Precious metal"},
    {"symbol": "HG=F", "name": "Copper", "sector": "Industrial metal"},
    {"symbol": "PL=F", "name": "Platinum", "sector": "Precious metal"},
]
for _s in STOCKS:
    _s["asset_class"] = "stock"
for _m in METALS:
    _m["asset_class"] = "metal"
ASSETS = STOCKS + METALS
ASSET_BY_SYMBOL = {a["symbol"]: a for a in ASSETS}

# FRED series used as metal drivers (10y Treasury yield, broad USD index, CPI)
FRED_SERIES = {"DGS10": "us_10y_yield", "DTWEXBGS": "usd_index", "CPIAUCSL": "cpi"}

SOURCES = ["prices", "news", "earnings", "filings", "macro"]

for _d in (DATA_DIR, LANDING_DIR, CHROMA_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
