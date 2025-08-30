from src.state import MarketState


def fetch_fundamentals(state: MarketState) -> MarketState:
    # TODO: Replace stub with real fundamentals source (e.g., yfinance)
    return {"fundamentals": {"pe": 28.5, "eps": 4.12}}
