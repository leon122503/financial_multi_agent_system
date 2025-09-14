from typing import Any, Dict, List
import json
import time
from openai import OpenAI
import openai

from src.state import MarketState
from src.config import OPENAI_API_KEY, OPENAI_MODEL


SYSTEM_PROMPT = (
    "You are a professional financial advisor.\n"
    "You will be given per-ticker fundamentals, recent news, and sentiment data.\n"
    "Your task:\n"
    "- For each ticker, classify it into one of: Strong Buy, Buy, Hold, Sell, Strong Sell.\n"
    "- Explain briefly how you reached that conclusion using fundamentals, news, and sentiment.\n"
    "- Mention key risks and potential catalysts.\n"
)


def _build_input_text(state: MarketState) -> str:
    tickers: List[str] = state.get("tickers") or []
    fundamentals = state.get("fundamentals", {}) or {}
    news = state.get("news", []) or []
    sentiment = state.get("sentiment", {}) or {}

    # Reindex news by ticker
    news_by_ticker: Dict[str, Any] = {}
    if isinstance(news, list):
        for entry in news:
            if isinstance(entry, dict) and entry.get("ticker"):
                news_by_ticker[entry["ticker"]] = entry

    # Build descriptive input text
    sections: List[str] = []
    for tk in tickers:
        f = fundamentals.get(tk, {})
        n = news_by_ticker.get(tk, {})
        s = sentiment.get(tk, {})

        section = [
            f"TICKER: {tk}",
            f"Fundamentals: {json.dumps(f)}",
            f"News: {json.dumps(n)}",
            f"Sentiment: {json.dumps(s)}",
        ]
        sections.append("\n".join(section))

    return "\n\n".join(sections) if sections else "No ticker data provided."


def advisor(state: MarketState) -> MarketState:
    if not OPENAI_API_KEY:
        return {"recommendation": "OPENAI_API_KEY not set; skipping LLM advisor."}

    client = OpenAI(api_key=OPENAI_API_KEY)
    input_text = _build_input_text(state)

    # Call with basic retry/backoff to handle TPM 429s
    last_err = None
    resp = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
            )
            break
        except openai.RateLimitError as e:
            last_err = e
            wait = 3 + attempt * 2
            time.sleep(wait)
        except Exception as e:
            last_err = e
            break
    else:
        return {"recommendation": f"LLM call failed: {last_err}"}

    time.sleep(2)  # Be polite with rate limits
    output_text = resp.choices[0].message.content.strip()

    return {"recommendation": output_text}
