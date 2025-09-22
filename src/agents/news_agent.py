from typing import Dict, List
import time
from openai import OpenAI

from src.langgraph_state import LangGraphMarketState
from src.config import OPENAI_API_KEY, OPENAI_MODEL


def _web_search_summary_for_ticker(client: OpenAI, ticker: str) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set; skipping web_search."

    prompt = (
        "Use web_search to find the most recent market-moving news for the ticker.\n"
        "Return a concise 4-6 sentence summary followed by a short bullet list of 2-4 source URLs.\n"
        f"Ticker: {ticker}"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        tools=[{"type": "web_search"}],
        input=prompt,
    )
    time.sleep(2)
    # Some SDK versions expose .output_text; fallback to first text content if needed
    summary = getattr(resp, "output_text", None)
    if summary is None:
        try:
            # Minimal fallback extraction
            parts = resp.output[0].content if hasattr(resp, "output") else []
            texts = [
                p.get("text")
                for p in parts
                if isinstance(p, dict) and p.get("type") == "output_text"
            ]
            summary = "\n".join([t for t in texts if t])
        except Exception:
            summary = ""
    return summary or ""


def fetch_news(state: LangGraphMarketState) -> LangGraphMarketState:
    tickers = state.get("tickers") or []
    if not tickers:
        return {"news": []}

    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    results: List[Dict[str, str]] = []
    for tk in tickers:
        if client is None:
            results.append(
                {
                    "ticker": tk,
                    "summary": "OPENAI_API_KEY not set; skipping web_search.",
                }
            )
            continue
        summary = _web_search_summary_for_ticker(client, tk)
        results.append({"ticker": tk, "summary": summary})

    return {"news": results}
