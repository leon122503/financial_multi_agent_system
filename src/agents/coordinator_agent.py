from src.state import MarketState


def coordinator(state: MarketState) -> MarketState:
    """Coordinator decides orchestration or branching.
    Currently just passes through unchanged state.
    Extend here to add dynamic routing / conditional logic.
    """
    return state
