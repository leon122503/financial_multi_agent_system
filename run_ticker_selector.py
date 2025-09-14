"""
Run the ticker selector agent for quick testing.

Usage:

"""

import json
import argparse
from typing import Any, Dict

from src.agents.ticker_selector_agent import select_tickers
from src.state import MarketState


def _run_once(prompt: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    initial: MarketState = {"prompt": prompt, "context": context or {}}
    result = select_tickers(initial)
    print("\n=== Ticker Selector Result ===")
    print("prompt:", prompt)
    print("tickers:", result.get("tickers"))
    sel = result.get("ticker_selection") or {}
    if sel:
        print("mode:", sel.get("mode"))
        print("reason:", sel.get("reason"))
        print("sources:")
        for src in sel.get("sources", []):
            print(" -", src)
    if "ticker_selection_error" in result:
        print("error:", result["ticker_selection_error"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ticker selector")
    parser.add_argument("--demo", action="store_true", help="Run predefined tests")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to run")
    parser.add_argument("--context", type=str, default=None, help="Context JSON string")
    args = parser.parse_args()

    # Demo mode runs one extraction-style and one search-style example (US only)
    if args.demo:
        _run_once("is Lululemon a buy")  # expect extraction -> LULU (US-listed)
        _run_once(
            "what are some defence stocks to buy", {"market": "US"}
        )  # expect search -> up to 3 US defense names
        return 0

    # CLI single-run mode
    if args.prompt:
        ctx: Dict[str, Any] = {}
        if args.context:
            try:
                ctx = json.loads(args.context)
            except Exception:
                ctx = {"raw": args.context}
        _run_once(args.prompt, ctx)
        return 0

    # Interactive fallback
    user_prompt = input("Enter your prompt: ").strip()
    if not user_prompt:
        print("❌ No prompt entered. Exiting.")
        return 1

    raw_context = input("Enter optional context as JSON (or leave blank): ").strip()
    try:
        ctx2: Dict[str, Any] = json.loads(raw_context) if raw_context else {}
    except Exception:
        ctx2 = {"raw": raw_context}

    _run_once(user_prompt, ctx2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
