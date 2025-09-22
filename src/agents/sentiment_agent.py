from typing import Dict, Any, List
import json
import time
import re
from openai import OpenAI

from src.langgraph_state import LangGraphMarketState
from src.config import OPENAI_API_KEY, OPENAI_MODEL


DEFAULT_MODEL = "gpt-4o-mini"
JSON_PATTERN = re.compile(r"\{[\s\S]*\}")


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = JSON_PATTERN.search(text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def _analyze_ticker_with_web_search(client: OpenAI, ticker: str) -> Dict[str, Any]:
    prompt = (
        "Analyze the CURRENT market sentiment for stock ticker "
        f"{ticker}. Use web_search to consult recent reputable sources (news, filings, blogs).\n"
        "Return ONLY JSON of shape: {\n"
        '  "ticker": "<TICKER>",\n'
        '  "label": "positive|neutral|negative",\n'
        '  "summary",\n'
        "}. No extra text."
    )

    resp = client.responses.create(
        model=DEFAULT_MODEL,
        tools=[{"type": "web_search"}],
        input=prompt,
    )
    time.sleep(2)

    text = getattr(resp, "output_text", None) or ""
    data = _parse_json(text)
    if data:
        data.setdefault("ticker", ticker)
        return data
    # Fallback to raw text if JSON not returned
    return {
        "ticker": ticker,
        "label": None,
        "score": None,
        "summary": text.strip()[:500],
        "sources": [],
    }


def analyze_sentiment(state: LangGraphMarketState) -> LangGraphMarketState:
    tickers: List[str] = state.get("tickers") or state.get("symbols") or []
    if not tickers:
        return {"sentiment": {"error": "No tickers to analyze"}}
    if not OPENAI_API_KEY:
        return {"sentiment": {"error": "OPENAI_API_KEY not set"}}

    client = OpenAI(api_key=OPENAI_API_KEY)
    results: Dict[str, Any] = {}
    for i, t in enumerate(tickers):
        try:
            results[t] = _analyze_ticker_with_web_search(client, t)
        except Exception as e:
            results[t] = {"ticker": t, "error": str(e)}
        if i < len(tickers) - 1:
            time.sleep(0.2)

    return {"sentiment": results}
