from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START

# 1) Define your state schema
class State(TypedDict):
    tickers: list[str]
    news: list
    fundamentals: dict
    sentiment: dict
    recommendation: str







def fetch_news(state: State):
    return {"news": [{"title": "Stub headline: AAPL beats earnings"}]}

def fetch_fundamentals(state: State):
    return {"fundamentals": {"pe": 28.5, "eps": 4.12}}

def analyze_sentiment(state: State):
    return {"sentiment": {"score": 0.3, "summary": "Positive sentiment"}}

def advisor(state: State):
    pe = state["fundamentals"]["pe"]
    if pe < 20:
        rec = "Buy"
    elif pe < 30:
        rec = "Hold"
    else:
        rec = "Sell"
    return {"recommendation": f"Stub recommendation: {rec}"}


# 3) Build the graph
builder = StateGraph(State)
builder.add_node(fetch_news)
builder.add_node(fetch_fundamentals)
builder.add_node(analyze_sentiment)
builder.add_node(advisor)

builder.add_edge(START, "fetch_news")
builder.add_edge("fetch_news", "fetch_fundamentals")
builder.add_edge("fetch_fundamentals", "analyze_sentiment")
builder.add_edge("analyze_sentiment", "advisor")

graph = builder.compile()


# 4) Run it
if __name__ == "__main__":
    initial_state = {"ticker": "AAPL"}
    result = graph.invoke(initial_state)
    print("=== FINAL RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")