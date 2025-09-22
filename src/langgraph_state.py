from typing import TypedDict, List, Dict, Any, Optional, Literal
from typing_extensions import Annotated
from langgraph.graph.message import add_messages


class LangGraphMarketState(TypedDict, total=False):
    # Input
    prompt: Optional[str]
    context: Optional[Dict[str, Any]]

    # Workflow metadata for LangGraph
    workflow_step: Optional[str]
    execution_path: List[str]
    parallel_tasks: List[str]
    errors: List[Dict[str, Any]]
    next_actions: List[str]

    # Agent-specific states
    ticker_selection: Optional[Dict[str, Any]]
    fundamentals_complete: bool
    news_complete: bool
    sentiment_complete: bool
    recommendation_complete: bool
    visualization_complete: bool

    # Working data (from original MarketState)
    tickers: List[str]
    news: List[Dict[str, Any]]
    fundamentals: Dict[str, Any]
    sentiment: Dict[str, Any]

    # Enhanced output tracking
    reworded_prompt: Optional[str]
    recommendation: str
    message: Optional[str]
    graph_summary: Optional[str]
    graph_images: List[str]

    # LangGraph-specific fields for better orchestration
    requires_visualization: bool
    requires_news: bool
    requires_sentiment: bool
    requires_fundamentals: bool
    complexity_score: Optional[
        int
    ]  # 1-5 scale for determining parallel vs sequential execution
    user_intent: Optional[
        Literal[
            "investment_advice",
            "visualization",
            "news_analysis",
            "fundamental_analysis",
            "sentiment_analysis",
            "general_query",
        ]
    ]

    # Execution metadata
    ai_orchestrator_executed: List[str]
    ai_orchestrator_iterations: int


def create_initial_state(
    prompt: str, context: Optional[Dict[str, Any]] = None
) -> LangGraphMarketState:
    """Create a fresh LangGraph state for a new query."""
    return LangGraphMarketState(
        prompt=prompt,
        context=context or {},
        workflow_step="initialization",
        execution_path=[],
        parallel_tasks=[],
        errors=[],
        next_actions=[],
        # Agent completion tracking
        ticker_selection=None,
        fundamentals_complete=False,
        news_complete=False,
        sentiment_complete=False,
        recommendation_complete=False,
        visualization_complete=False,
        # Working data
        tickers=[],
        news=[],
        fundamentals={},
        sentiment={},
        # Output
        reworded_prompt=None,
        recommendation="",
        message=None,
        graph_summary=None,
        graph_images=[],
        # Analysis flags (will be determined by router)
        requires_visualization=False,
        requires_news=False,
        requires_sentiment=False,
        requires_fundamentals=False,
        complexity_score=None,
        user_intent=None,
        # Execution tracking
        ai_orchestrator_executed=[],
        ai_orchestrator_iterations=0,
    )
