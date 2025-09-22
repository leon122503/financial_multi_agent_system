from typing import Any, Dict, List
import json
import re
import time
from openai import OpenAI

from src.langgraph_state import LangGraphMarketState
from src.config import OPENAI_API_KEY, OPENAI_MODEL

DEFAULT_MODEL = OPENAI_MODEL or "gpt-4o-mini"
JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_obj(text: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def select_tickers(state: LangGraphMarketState) -> LangGraphMarketState:
    """
    Single-call selection logic:
    - If state already has tickers, return them without calling the LLM.
    - Otherwise, make one Responses API call that will EITHER:
        a) Extract tickers directly if they are clearly present in the prompt, OR
        b) Use web_search to find up to 3 relevant US-listed tickers.
    Writes:
        - state.tickers: selected tickers (up to 5 if extracted, else up to 3)
        - state.ticker_selection: { mode: "extracted"|"searched", reason, sources[], raw }
    """
    if not OPENAI_API_KEY:
        return {"ticker_selection_error": "OPENAI_API_KEY not set"}

    prompt: str = state.get("prompt") or ""
    context: Dict[str, Any] = state.get("context") or {}

    if not prompt.strip():
        return {"ticker_selection_error": "No prompt provided"}

    client = OpenAI(api_key=OPENAI_API_KEY)

    # If the user already provided tickers in state, honor them
    preexisting = state.get("tickers", []) or []
    if preexisting:
        return {"tickers": preexisting}

    # Single Responses call that handles extract-or-search
    system = (
        "You are a research assistant that selects US stock tickers from a user's request in a SINGLE call.\n"
        "Decision rule (one step): If the prompt clearly names tickers or companies, extract their primary US-listed tickers directly and DO NOT call web_search.\n"
        "Otherwise, call web_search to identify up to 3 highly relevant US-listed tickers that match the request (theme, sector, macro view, geography). Prefer liquid large/mid caps; avoid ETFs unless explicitly requested.\n"
        "Conventions: Use uppercase symbols; return common stocks (e.g., for Alphabet prefer GOOG); dedupe; exclude delisted or foreign-only listings unless the prompt requires otherwise.\n"
        "Return ONLY JSON (no extra text) with this shape:\n"
        "{\n"
        '  "mode": "extracted" | "searched",\n'
        '  "tickers": ["T1","T2","T3"],\n'
        '  "reason": "brief rationale for the selection or extraction",\n'
        '  "sources": [{"title":"...","url":"..."}]\n'
        "}\n"
        "When mode=extracted, sources may be an empty list. Ensure tickers are uppercase and valid US listings."
    )

    user = (
        "User prompt:\n"
        f"{prompt}\n\n"
        "Context (may be empty):\n"
        f"{json.dumps(context)}"
    )

    resp = client.responses.create(
        model=DEFAULT_MODEL,
        tools=[{"type": "web_search"}],
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    time.sleep(2)

    text = getattr(resp, "output_text", "") or ""
    obj = _parse_json_obj(text) or {}
    raw_tickers = obj.get("tickers") or []
    mode = (obj.get("mode") or "").strip().lower()
    if mode not in {"extracted", "searched"}:
        # Heuristic fallback: infer from presence of sources
        mode = "searched" if obj.get("sources") else "extracted"

    selected: List[str] = []
    limit = 5 if mode == "extracted" else 3
    for t in raw_tickers:
        if not isinstance(t, str):
            continue
        sym = t.strip().upper()
        if sym and re.fullmatch(r"[A-Z0-9.-]{1,8}", sym) and sym not in selected:
            selected.append(sym)
        if len(selected) >= limit:
            break

    # Merge (though typically none existed here)
    merged = sorted(set(selected))

    selection = {
        "mode": mode,
        "reason": obj.get("reason") or "",
        "sources": obj.get("sources") or [],
        "raw": text[:2000],
    }

    result: LangGraphMarketState = {"tickers": merged, "ticker_selection": selection}
    if not selected:
        result["ticker_selection_error"] = "No tickers extracted"
    return result
