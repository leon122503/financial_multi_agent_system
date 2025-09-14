from typing import TypedDict, List, Dict, Any, Optional


class MarketState(TypedDict, total=False):
    # Input
    prompt: Optional[str]
    context: Optional[Dict[str, Any]]

    # Working data
    tickers: List[str]
    news: List[Dict[str, Any]]
    fundamentals: Dict[str, Any]
    sentiment: Dict[str, Any]

    # Output
    reworded_prompt: Optional[str]
    recommendation: str
    message: Optional[str]
