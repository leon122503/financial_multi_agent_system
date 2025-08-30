from src.state import MarketState


def advisor(state: MarketState) -> MarketState:
    fundamentals = state.get("fundamentals", {})
    pe = fundamentals.get("pe")
    if pe is None:
        rec = "Insufficient data"
    elif pe < 20:
        rec = "Buy"
    elif pe < 30:
        rec = "Hold"
    else:
        rec = "Sell"
    return {"recommendation": f"Stub recommendation: {rec}"}
