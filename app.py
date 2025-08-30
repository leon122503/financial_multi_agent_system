"""Application entry point. Uses structured multi-agent graph defined in src/.

Run with:
    python app.py
"""

from src.graph_builder import build_graph


def main():
    graph = build_graph()
    initial_state = {"tickers": ["AAPL", "MSFT"]}
    result = graph.invoke(initial_state)
    print("=== FINAL RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
