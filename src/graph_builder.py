from langgraph.graph import StateGraph, START, END
from src.state import MarketState
from src.agents.ticker_selector_agent import select_tickers
from src.agents.news_agent import fetch_news
from src.agents.fundamentals_agent import fetch_fundamentals
from src.agents.sentiment_agent import analyze_sentiment
from src.agents.advisor_agent import advisor


def build_graph():
    builder = StateGraph(MarketState)
    builder.add_node(select_tickers)
    builder.add_node(fetch_news)
    builder.add_node(fetch_fundamentals)
    builder.add_node(analyze_sentiment)
    builder.add_node(advisor)

    # Terminal node: if no tickers were selected, return a friendly message and end
    def no_tickers_message(state: MarketState) -> MarketState:
        msg = state.get("ticker_selection_error") or (
            "No US stock tickers could be selected for your request."
        )
        return {"message": msg}

    builder.add_node(no_tickers_message)

    builder.add_edge(START, "select_tickers")

    # After selecting tickers: continue if present; else end with a message
    def route_after_select_tickers(state: MarketState) -> str:
        tickers = state.get("tickers") or []
        return "fetch_news" if tickers else "no_tickers_message"

    builder.add_conditional_edges(
        "select_tickers",
        route_after_select_tickers,
        {"fetch_news": "fetch_news", "no_tickers_message": "no_tickers_message"},
    )
    builder.add_edge("fetch_news", "fetch_fundamentals")
    builder.add_edge("fetch_fundamentals", "analyze_sentiment")
    builder.add_edge("analyze_sentiment", "advisor")

    # No tickers route ends the graph
    builder.add_edge("no_tickers_message", END)

    return builder.compile()
