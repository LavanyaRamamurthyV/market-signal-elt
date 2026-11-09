"""LLM client with model routing.

role="bulk"   high-volume extraction  -> Llama via Ollama first (local, free), then Gemini
role="reason" decisions and judging   -> Gemini first, then Llama
If no model is reachable, callers fall back to deterministic rules and label the output
"offline rules", so the pipeline never stops because an LLM is down.
Every call is logged to ops_llm_calls (provider, role, latency, success).
"""
import json
import re
import time

import pandas as pd
import requests

from . import config, db

_status: dict | None = None
_gemini_client = None
_gemini_model: str | None = None
_calls: list[dict] = []


def status(refresh: bool = False) -> dict:
    """Which providers are reachable right now."""
    global _status
    if _status is not None and not refresh:
        return _status
    s = {"ollama": False, "ollama_embed": False, "gemini": False}
    if config.LLM_PROVIDER in ("auto", "ollama"):
        try:
            tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2).json()
            names = {m["name"] for m in tags.get("models", [])}
            s["ollama"] = any(n.split(":")[0] == config.OLLAMA_MODEL.split(":")[0] for n in names)
            s["ollama_embed"] = any(n.split(":")[0] == config.OLLAMA_EMBED_MODEL.split(":")[0] for n in names)
        except requests.RequestException:
            pass
    if config.LLM_PROVIDER in ("auto", "gemini") and config.GEMINI_API_KEY:
        s["gemini"] = True
    _status = s
    return s


def _ollama(prompt: str) -> str:
    r = requests.post(f"{config.OLLAMA_URL}/api/generate",
                      json={"model": config.OLLAMA_MODEL, "prompt": prompt, "format": "json",
                            "stream": False, "options": {"temperature": 0.1}}, timeout=180)
    r.raise_for_status()
    return r.json()["response"]


def _gemini(prompt: str) -> str:
    global _gemini_client, _gemini_model
    from google import genai
    from google.genai import types

    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        _gemini_model = config.GEMINI_MODEL
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    for attempt in range(4):
        try:
            resp = _gemini_client.models.generate_content(model=_gemini_model, contents=prompt, config=cfg)
            return resp.text
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "404" in msg or "not found" in msg.lower():  # model renamed: pick a current flash model
                names = [m.name.split("/")[-1] for m in _gemini_client.models.list()]
                flash = [n for n in names if "flash" in n and "image" not in n and "tts" not in n]
                if not flash or _gemini_model == flash[0]:
                    raise
                _gemini_model = flash[0]
            elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                time.sleep(8 * (attempt + 1))
            else:
                raise
    raise RuntimeError("Gemini rate limit not cleared after retries")


def _order(role: str) -> list[str]:
    s = status()
    if config.LLM_PROVIDER == "offline":
        return []
    pref = ["ollama", "gemini"] if role == "bulk" else ["gemini", "ollama"]
    return [p for p in pref if s.get(p)]


def complete_json(prompt: str, role: str = "bulk", task: str = "") -> tuple[dict | list | None, str]:
    """Return (parsed JSON, provider). provider is 'offline rules' when no model answered."""
    for provider in _order(role):
        t0 = time.time()
        try:
            text = _ollama(prompt) if provider == "ollama" else _gemini(prompt)
            parsed = parse_json(text)
            _calls.append({"provider": provider if provider == "ollama" else f"gemini:{_gemini_model}",
                           "role": role, "task": task, "latency_s": round(time.time() - t0, 2),
                           "ok": parsed is not None, "at": db.now_utc()})
            if parsed is not None:
                label = f"Llama ({config.OLLAMA_MODEL})" if provider == "ollama" else f"Gemini ({_gemini_model})"
                return parsed, label
        except Exception as e:  # noqa: BLE001
            _calls.append({"provider": provider, "role": role, "task": task,
                           "latency_s": round(time.time() - t0, 2), "ok": False, "at": db.now_utc(),
                           "error": str(e)[:200]})
    return None, "offline rules"


def parse_json(text: str):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
    return None


def flush_call_log() -> int:
    """Persist call log for the Usage panel."""
    global _calls
    if not _calls:
        return 0
    df = pd.DataFrame(_calls)
    for c in ("error",):
        if c not in df:
            df[c] = ""
    db.write_df(df[["at", "provider", "role", "task", "latency_s", "ok", "error"]], "ops_llm_calls", if_exists="append")
    n, _calls = len(_calls), []
    return n
