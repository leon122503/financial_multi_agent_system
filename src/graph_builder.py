from langgraph.graph import StateGraph, START
from src.state import MarketState
from src.agents.coordinator_agent import coordinator
from src.agents.news_agent import fetch_news
from src.agents.fundamentals_agent import fetch_fundamentals
from src.agents.sentiment_agent import analyze_sentiment
from src.agents.advisor_agent import advisor


def build_graph():
    builder = StateGraph(MarketState)
    builder.add_node(coordinator)
    builder.add_node(fetch_news)
    builder.add_node(fetch_fundamentals)
    builder.add_node(analyze_sentiment)
    builder.add_node(advisor)

    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "fetch_news")
    builder.add_edge("fetch_news", "fetch_fundamentals")
    builder.add_edge("fetch_fundamentals", "analyze_sentiment")
    builder.add_edge("analyze_sentiment", "advisor")

    return builder.compile()
