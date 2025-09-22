"""
LangGraph-based orchestrator for the financial market analysis system.
Provides sophisticated workflow management with conditional routing, parallel execution,
and advanced state management capabilities.
"""

from typing import Dict, Any, List, Literal
import asyncio
from langgraph.graph import StateGraph, END, START

from src.langgraph_state import LangGraphMarketState, create_initial_state
from src.langgraph_nodes import (
    analysis_router_node,
    ticker_selector_node,
    fundamentals_node,
    news_node,
    sentiment_node,
    advisor_node,
    code_interpreter_node,
    final_synthesis_node,
)


class LangGraphOrchestrator:
    """
    Advanced orchestrator using LangGraph for sophisticated workflow management.

    Key features:
    - Conditional routing based on user intent analysis
    - Parallel execution of independent tasks
    - Comprehensive state management with error handling
    - Built-in workflow visualization and debugging
    - Automatic retry and fallback mechanisms
    """

    def __init__(self):
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with all nodes and conditional edges."""

        # Create the state graph
        workflow = StateGraph(LangGraphMarketState)

        # Add all nodes
        workflow.add_node("analysis_router", analysis_router_node)
        workflow.add_node("ticker_selector", ticker_selector_node)
        workflow.add_node("fundamentals", fundamentals_node)
        workflow.add_node("news", news_node)
        workflow.add_node("sentiment", sentiment_node)
        workflow.add_node("advisor", advisor_node)
        workflow.add_node("code_interpreter", code_interpreter_node)
        workflow.add_node("final_synthesis", final_synthesis_node)

        # Set entry point
        workflow.set_entry_point("analysis_router")

        # Add conditional routing from analysis_router
        workflow.add_conditional_edges(
            "analysis_router",
            self._route_from_analysis,
            {
                "ticker_selector": "ticker_selector",
                "parallel_analysis": "fundamentals",  # Start parallel execution
                "advisor": "advisor",
                "code_interpreter": "code_interpreter",
                "final_synthesis": "final_synthesis",
            },
        )

        # After ticker selection, route to parallel analysis or specific nodes
        workflow.add_conditional_edges(
            "ticker_selector",
            self._route_after_ticker_selection,
            {
                "parallel_analysis": "fundamentals",  # Start with fundamentals in parallel group
                "advisor": "advisor",
                "code_interpreter": "code_interpreter",
                "final_synthesis": "final_synthesis",
            },
        )

        # Parallel analysis routing - fundamentals can trigger news/sentiment in parallel
        workflow.add_conditional_edges(
            "fundamentals",
            self._route_parallel_analysis,
            {
                "news": "news",
                "sentiment": "sentiment",
                "advisor": "advisor",
                "code_interpreter": "code_interpreter",
                "final_synthesis": "final_synthesis",
            },
        )

        # News node routing
        workflow.add_conditional_edges(
            "news",
            self._route_after_news,
            {
                "sentiment": "sentiment",
                "advisor": "advisor",
                "code_interpreter": "code_interpreter",
                "final_synthesis": "final_synthesis",
            },
        )

        # Sentiment node routing
        workflow.add_conditional_edges(
            "sentiment",
            self._route_after_sentiment,
            {
                "advisor": "advisor",
                "code_interpreter": "code_interpreter",
                "final_synthesis": "final_synthesis",
            },
        )

        # Advisor routing
        workflow.add_conditional_edges(
            "advisor",
            self._route_after_advisor,
            {
                "code_interpreter": "code_interpreter",
                "final_synthesis": "final_synthesis",
            },
        )

        # Code interpreter routes to final synthesis
        workflow.add_edge("code_interpreter", "final_synthesis")

        # Final synthesis ends the workflow
        workflow.add_edge("final_synthesis", END)

        return workflow

    def _route_from_analysis(self, state: LangGraphMarketState) -> str:
        """Route from analysis router based on requirements and current state."""

        # If no tickers yet, need ticker selection first
        if not state.get("tickers"):
            return "ticker_selector"

        # If we have tickers, check what analysis is needed
        return self._determine_next_step(state)

    def _route_after_ticker_selection(self, state: LangGraphMarketState) -> str:
        """Route after ticker selection is complete."""
        return self._determine_next_step(state)

    def _route_parallel_analysis(self, state: LangGraphMarketState) -> str:
        """Route from fundamentals - can trigger parallel news/sentiment or move to synthesis."""

        # Check if news is needed and not done
        if state.get("requires_news") and not state.get("news_complete"):
            return "news"

        # Check if sentiment is needed and not done (and we have news if required)
        if (
            state.get("requires_sentiment")
            and not state.get("sentiment_complete")
            and (not state.get("requires_news") or state.get("news_complete"))
        ):
            return "sentiment"

        return self._determine_next_step(state)

    def _route_after_news(self, state: LangGraphMarketState) -> str:
        """Route after news analysis."""

        # If sentiment is needed and not done, do it next
        if state.get("requires_sentiment") and not state.get("sentiment_complete"):
            return "sentiment"

        return self._determine_next_step(state)

    def _route_after_sentiment(self, state: LangGraphMarketState) -> str:
        """Route after sentiment analysis."""
        return self._determine_next_step(state)

    def _route_after_advisor(self, state: LangGraphMarketState) -> str:
        """Route after advisor recommendation."""

        # If visualization is needed, do it next
        if state.get("requires_visualization") and not state.get(
            "visualization_complete"
        ):
            return "code_interpreter"

        return "final_synthesis"

    def _determine_next_step(self, state: LangGraphMarketState) -> str:
        """
        AI-driven determination of the next step based on current state and requirements.
        Uses OpenAI to make intelligent routing decisions.
        """

        from openai import OpenAI
        from src.config import OPENAI_API_KEY, OPENAI_MODEL

        user_intent = state.get("user_intent")

        # First, do explicit completion checks to prevent duplicates
        if state.get("requires_visualization", False) and state.get(
            "visualization_complete", False
        ):
            print("   ✅ Visualization already complete - skipping code_interpreter")

        if state.get("requires_fundamentals", False) and state.get(
            "fundamentals_complete", False
        ):
            print("   ✅ Fundamentals already complete")

        if state.get("requires_news", False) and state.get("news_complete", False):
            print("   ✅ News analysis already complete")

        if state.get("requires_sentiment", False) and state.get(
            "sentiment_complete", False
        ):
            print("   ✅ Sentiment analysis already complete")

        # Check if all required tasks are complete
        all_required_complete = True
        required_tasks = []

        if state.get("requires_fundamentals", False) and not state.get(
            "fundamentals_complete", False
        ):
            all_required_complete = False
            required_tasks.append("fundamentals")

        if state.get("requires_news", False) and not state.get("news_complete", False):
            all_required_complete = False
            required_tasks.append("news")

        if state.get("requires_sentiment", False) and not state.get(
            "sentiment_complete", False
        ):
            all_required_complete = False
            required_tasks.append("sentiment")

        if state.get("requires_visualization", False) and not state.get(
            "visualization_complete", False
        ):
            all_required_complete = False
            required_tasks.append("code_interpreter")

        # If advisor is needed and all data collection is done
        if (
            user_intent == "investment_advice"
            and not state.get("recommendation_complete", False)
            and state.get("fundamentals_complete", False)
        ):
            required_tasks.append("advisor")
            all_required_complete = False

        # If all required tasks are complete, go to synthesis
        if all_required_complete:
            print("   ✅ All required tasks complete - moving to final synthesis")
            return "final_synthesis"

        # If no AI available, fall back to simple rule-based routing
        if not OPENAI_API_KEY:
            print("   ⚠️  No AI available - using simple routing")
            return self._fallback_routing(state)

        # Prepare context for AI routing decision
        current_state_info = {
            "user_intent": user_intent,
            "tickers": state.get("tickers", []),
            "fundamentals_complete": state.get("fundamentals_complete", False),
            "news_complete": state.get("news_complete", False),
            "sentiment_complete": state.get("sentiment_complete", False),
            "recommendation_complete": state.get("recommendation_complete", False),
            "visualization_complete": state.get("visualization_complete", False),
            "requires_fundamentals": state.get("requires_fundamentals", False),
            "requires_news": state.get("requires_news", False),
            "requires_sentiment": state.get("requires_sentiment", False),
            "requires_visualization": state.get("requires_visualization", False),
            "execution_path": state.get("execution_path", []),
            "remaining_tasks": required_tasks,
        }

        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            system_prompt = """
            You are an intelligent workflow router for financial analysis. Based on the current state and requirements, determine the next agent to run.

            Available next steps:
            - "parallel_analysis": Run fundamentals analysis (can trigger news/sentiment in parallel)
            - "news": Get recent news articles
            - "sentiment": Analyze market sentiment  
            - "advisor": Generate investment recommendations
            - "code_interpreter": Create charts and visualizations
            - "final_synthesis": Complete the analysis and provide final results

            CRITICAL RULES:
            1. NEVER run an agent that is already marked as complete (e.g., if visualization_complete=True, never return "code_interpreter")
            2. Only run tasks that are in the remaining_tasks list
            3. Fundamentals usually needed before advisor recommendations
            4. Charts/visualization typically come after data collection
            5. Move to final_synthesis when remaining_tasks is empty
            6. Prioritize data collection (fundamentals, news, sentiment) before recommendations or visualization

            Respond with ONLY the next step name (one of the options above).
            """

            user_prompt = f"""
            Current state: {current_state_info}
            
            What should be the next step?
            """

            response = client.chat.completions.create(
                model=OPENAI_MODEL or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Removed temperature parameter as it's not supported by this model
            )

            next_step = response.choices[0].message.content.strip().lower()

            # Validate AI response
            valid_steps = [
                "parallel_analysis",
                "news",
                "sentiment",
                "advisor",
                "code_interpreter",
                "final_synthesis",
            ]
            if next_step not in valid_steps:
                print(f"   ⚠️  AI returned invalid step '{next_step}' - using fallback")
                return self._fallback_routing(state)

            print(f"   🤖 AI routing decision: {next_step}")
            return next_step

        except Exception as e:
            print(f"   ⚠️  AI routing failed ({e}) - using fallback")
            return self._fallback_routing(state)

    def _fallback_routing(self, state: LangGraphMarketState) -> str:
        """Fallback rule-based routing when AI is not available."""
        user_intent = state.get("user_intent")

        # COMMENTED OUT: Original rule-based routing logic (now using AI)
        # For investment advice, we need fundamentals -> advisor -> optional visualization
        if user_intent == "investment_advice":
            if not state.get("fundamentals_complete"):
                return "parallel_analysis"  # Start with fundamentals
            if not state.get("recommendation_complete"):
                return "advisor"
            if state.get("requires_visualization") and not state.get(
                "visualization_complete"
            ):
                return "code_interpreter"
            return "final_synthesis"

        # For visualization requests, get data first then visualize
        if user_intent == "visualization":
            if not state.get("fundamentals_complete") and state.get(
                "requires_fundamentals"
            ):
                return "parallel_analysis"
            if state.get("requires_visualization") and not state.get(
                "visualization_complete"
            ):
                return "code_interpreter"
            return "final_synthesis"

        # For news/sentiment analysis
        if user_intent in ["news_analysis", "sentiment_analysis"]:
            if state.get("requires_news") and not state.get("news_complete"):
                return "news"
            if state.get("requires_sentiment") and not state.get("sentiment_complete"):
                return "sentiment"
            return "final_synthesis"

        # Default route to final synthesis
        return "final_synthesis"

    def execute(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> LangGraphMarketState:
        """
        Execute the LangGraph workflow for a given prompt.

        Args:
            prompt: User query/request
            context: Optional additional context

        Returns:
            Final state with all analysis results
        """
        print(f"\n🚀 LangGraph Orchestrator starting analysis...")
        print(f"📝 Prompt: {prompt}")

        # Create initial state
        initial_state = create_initial_state(prompt, context)

        # Execute the compiled graph
        final_state = self.compiled_graph.invoke(initial_state)

        print(f"\n✅ LangGraph execution complete!")
        print(f"📊 Execution path: {' → '.join(final_state.get('execution_path', []))}")
        print(
            f"🔄 Total iterations: {final_state.get('ai_orchestrator_iterations', 0)}"
        )

        return final_state

    async def execute_async(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> LangGraphMarketState:
        """
        Asynchronous execution of the LangGraph workflow.
        Useful for handling multiple concurrent requests.
        """
        return await asyncio.to_thread(self.execute, prompt, context)

    def get_workflow_visualization(self) -> str:
        """
        Get a text representation of the workflow structure.
        LangGraph provides built-in visualization capabilities.
        """
        try:
            # This would generate a visual graph in a real implementation
            # For now, return a text description
            return """
LangGraph Workflow Structure:
════════════════════════════

START → analysis_router → [Decision Point]
                         ├── ticker_selector → [Parallel Analysis]
                         ├── fundamentals ⟷ news ⟷ sentiment
                         ├── advisor → [Optional Visualization]
                         ├── code_interpreter → final_synthesis
                         └── final_synthesis → END

Parallel Execution Groups:
- Data Collection: fundamentals ∥ news ∥ sentiment
- Analysis: advisor (after data) → code_interpreter (if needed)

Conditional Routing:
- User intent determines required analysis components
- Complexity score influences parallel vs sequential execution
- State tracking prevents redundant operations
            """
        except Exception as e:
            return f"Visualization error: {e}"

    def get_execution_stats(self, state: LangGraphMarketState) -> Dict[str, Any]:
        """Get detailed execution statistics from a completed workflow."""

        execution_path = state.get("execution_path", [])
        executed_agents = state.get("ai_orchestrator_executed", [])
        errors = state.get("errors", [])

        return {
            "total_nodes_executed": len(execution_path),
            "unique_agents_used": len(set(executed_agents)),
            "execution_path": execution_path,
            "agents_executed": executed_agents,
            "user_intent": state.get("user_intent"),
            "complexity_score": state.get("complexity_score"),
            "iterations": state.get("ai_orchestrator_iterations", 0),
            "errors_encountered": len(errors),
            "error_details": errors,
            "parallel_tasks_identified": state.get("parallel_tasks", []),
            "workflow_completion": state.get("workflow_step") == "complete",
            "data_generated": {
                "tickers_found": len(state.get("tickers", [])),
                "fundamentals_analyzed": len(state.get("fundamentals", {})),
                "news_articles": len(state.get("news", [])),
                "sentiment_results": len(state.get("sentiment", {})),
                "charts_generated": len(state.get("graph_images", [])),
                "recommendation_provided": bool(state.get("recommendation")),
            },
        }


# Convenience function for easy usage
def create_langgraph_orchestrator() -> LangGraphOrchestrator:
    """Create and return a configured LangGraph orchestrator."""
    return LangGraphOrchestrator()


# Example usage
if __name__ == "__main__":
    orchestrator = create_langgraph_orchestrator()

    # Print workflow structure
    print(orchestrator.get_workflow_visualization())

    # Example execution
    result = orchestrator.execute(
        "Should I invest in Apple? Show me a chart with their financials."
    )

    # Print execution stats
    stats = orchestrator.get_execution_stats(result)
    print(f"\nExecution Statistics: {stats}")
