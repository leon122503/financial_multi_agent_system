"""
Quick runner to test the code interpreter graphs agent.

Usage:
  python run_code_interpreter.py --prompt "plot revenue vs time for AAPL" --fundamentals path/to/fundamentals.json
  python run_code_interpreter.py --prompt "compare margins across tickers" --fundamentals '{"AAPL": {"dates": ["2024-01-01"], "revenue": [100]}}'
"""

import argparse
import json
import os
from typing import Any, Dict
import ast

from src.agents.code_interpreter_agent import code_interpreter_graphs
from src.state import MarketState


def _load_fundamentals(arg: str) -> Dict[str, Any]:
    # If arg is a path to a file, read it; else try to parse as JSON
    if os.path.exists(arg):
        with open(arg, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(arg)
    except Exception:
        # Fallback to Python literal parsing (supports single quotes)
        return ast.literal_eval(arg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run code interpreter graphs")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt")
    parser.add_argument(
        "--fundamentals",
        type=str,
        required=True,
        help="Path to JSON file or inline JSON for fundamentals",
    )
    args = parser.parse_args()

    fundamentals = _load_fundamentals(args.fundamentals)
    state: MarketState = {"prompt": args.prompt, "fundamentals": fundamentals}
    result = code_interpreter_graphs(state)

    print("\n=== Code Interpreter Result ===")
    if "code_interpreter_error" in result:
        print("error:", result["code_interpreter_error"])
    print("summary:\n", result.get("graph_summary"))
    imgs = result.get("graph_images") or []
    if imgs:
        print("images:")
        for p in imgs:
            print(" -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
