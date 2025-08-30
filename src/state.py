from typing import TypedDict, List, Dict, Any


class MarketState(TypedDict, total=False):
    tickers: List[str]
    news: List[Dict[str, Any]]
    fundamentals: Dict[str, Any]
    sentiment: Dict[str, Any]
    recommendation: str
