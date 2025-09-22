"""
LangGraph workflow visualization and debugging tools.
Provides visual representations of workflow execution and performance analysis.
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from src.langgraph_state import LangGraphMarketState
from src.langgraph_orchestrator import LangGraphOrchestrator


class WorkflowVisualizer:
    """
    Advanced visualization and debugging tools for LangGraph workflows.
    """

    def __init__(self, orchestrator: LangGraphOrchestrator):
        self.orchestrator = orchestrator

    def generate_execution_diagram(self, state: LangGraphMarketState) -> str:
        """Generate a text-based execution flow diagram."""

        execution_path = state.get("execution_path", [])
        user_intent = state.get("user_intent", "unknown")
        complexity = state.get("complexity_score", 0)

        diagram = f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                          LANGGRAPH EXECUTION FLOW                                ║
║  Intent: {user_intent:<20} │ Complexity: {complexity}/5 │ Nodes: {len(execution_path):<3}           ║
╠═══════════════════════════════════════════════════════════════════════════════════╣"""

        # Build execution flow
        for i, node in enumerate(execution_path):
            is_last = i == len(execution_path) - 1
            arrow = "└──" if is_last else "├──"

            # Add node with status
            node_status = self._get_node_status(node, state)
            diagram += f"\n║ {arrow} {node:<20} │ {node_status:<40} ║"

        diagram += "\n╚═══════════════════════════════════════════════════════════════════════════════════╝"

        return diagram

    def _get_node_status(self, node: str, state: LangGraphMarketState) -> str:
        """Get the status description for a specific node."""

        status_map = {
            "analysis_router": f"Intent: {state.get('user_intent', 'unknown')}",
            "ticker_selector": f"Found: {len(state.get('tickers', []))} tickers",
            "fundamentals": f"Analyzed: {len(state.get('fundamentals', {}))} companies",
            "news": f"Articles: {len(state.get('news', []))}",
            "sentiment": f"Sentiment: {len(state.get('sentiment', {}))} analyzed",
            "advisor": (
                "Recommendation generated"
                if state.get("recommendation")
                else "No recommendation"
            ),
            "code_interpreter": f"Charts: {len(state.get('graph_images', []))}",
            "final_synthesis": "Response compiled",
        }

        return status_map.get(node, "Executed")

    def generate_performance_report(self, state: LangGraphMarketState) -> str:
        """Generate a detailed performance analysis report."""

        stats = self.orchestrator.get_execution_stats(state)

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              PERFORMANCE REPORT                                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║ Execution Overview                                                               ║
║   • Total Nodes Executed: {stats['total_nodes_executed']:<8}                                          ║
║   • Unique Agents Used:   {stats['unique_agents_used']:<8}                                          ║
║   • Iterations:           {stats['iterations']:<8}                                          ║
║   • Workflow Complete:    {str(stats['workflow_completion']):<8}                                          ║
║   • Errors Encountered:   {stats['errors_encountered']:<8}                                          ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║ Analysis Breakdown                                                               ║
║   • User Intent:          {stats['user_intent']:<20}                                    ║
║   • Complexity Score:     {stats['complexity_score']}/5                                                ║
║   • Parallel Tasks:       {len(stats['parallel_tasks_identified']):<8}                                          ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║ Data Generation Results                                                          ║
║   • Tickers Found:        {stats['data_generated']['tickers_found']:<8}                                          ║
║   • Fundamentals:         {stats['data_generated']['fundamentals_analyzed']:<8}                                          ║
║   • News Articles:        {stats['data_generated']['news_articles']:<8}                                          ║
║   • Sentiment Results:    {stats['data_generated']['sentiment_results']:<8}                                          ║
║   • Charts Generated:     {stats['data_generated']['charts_generated']:<8}                                          ║
║   • Recommendation:       {str(stats['data_generated']['recommendation_provided']):<8}                                          ║
╚══════════════════════════════════════════════════════════════════════════════════╝"""

        return report

    def generate_comparison_report(
        self,
        langgraph_state: LangGraphMarketState,
        ai_orchestrator_result: Dict[str, Any],
    ) -> str:
        """Compare LangGraph execution with previous AI orchestrator results."""

        lg_stats = self.orchestrator.get_execution_stats(langgraph_state)
        ai_agents = ai_orchestrator_result.get("ai_orchestrator_executed", [])
        ai_iterations = ai_orchestrator_result.get("ai_orchestrator_iterations", 0)

        comparison = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           ORCHESTRATOR COMPARISON                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║ Metric                    │ LangGraph      │ AI Orchestrator │ Difference      ║
╠═══════════════════════════╪════════════════╪═════════════════╪═════════════════╣
║ Agents Executed           │ {lg_stats['unique_agents_used']:<14} │ {len(ai_agents):<15} │ {lg_stats['unique_agents_used'] - len(ai_agents):+3}             ║
║ Total Iterations          │ {lg_stats['iterations']:<14} │ {ai_iterations:<15} │ {lg_stats['iterations'] - ai_iterations:+3}             ║
║ Workflow Structure        │ Graph-based    │ Sequential      │ More Advanced   ║
║ Parallel Execution        │ ✓ Supported    │ ✗ No            │ LG Advantage    ║
║ Error Handling            │ ✓ Built-in     │ ✓ Basic         │ LG Superior     ║
║ State Management          │ ✓ Rich         │ ✓ Basic         │ LG Superior     ║
║ Visualization             │ ✓ Built-in     │ ✗ No            │ LG Advantage    ║
║ Debuggability             │ ✓ Excellent    │ ✓ Good          │ LG Advantage    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Key LangGraph Advantages:
• Conditional routing based on comprehensive state analysis
• Parallel execution of independent analysis tasks  
• Built-in error handling and retry mechanisms
• Rich state management with detailed execution tracking
• Visual workflow representation and debugging tools
• Modular node structure for easy maintenance and extension
• Better separation of concerns between routing logic and execution
        """

        return comparison

    def export_execution_log(
        self, state: LangGraphMarketState, filename: Optional[str] = None
    ) -> str:
        """Export detailed execution log to JSON file."""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/langgraph_execution_{timestamp}.json"

        # Prepare comprehensive log data
        log_data = {
            "execution_metadata": {
                "timestamp": datetime.now().isoformat(),
                "workflow_type": "langgraph",
                "execution_path": state.get("execution_path", []),
                "user_intent": state.get("user_intent"),
                "complexity_score": state.get("complexity_score"),
                "parallel_tasks": state.get("parallel_tasks", []),
                "iterations": state.get("ai_orchestrator_iterations", 0),
            },
            "performance_stats": self.orchestrator.get_execution_stats(state),
            "full_state": dict(state),  # Complete state for debugging
            "workflow_visualization": self.orchestrator.get_workflow_visualization(),
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, default=str)
            return f"Execution log exported to: {filename}"
        except Exception as e:
            return f"Failed to export log: {e}"

    def generate_debug_summary(self, state: LangGraphMarketState) -> str:
        """Generate a concise debug summary for troubleshooting."""

        errors = state.get("errors", [])
        execution_path = state.get("execution_path", [])

        summary = f"""
🔧 DEBUG SUMMARY
================
• Status: {'✅ Success' if not errors else '❌ Errors detected'}
• Execution Path: {' → '.join(execution_path)}
• User Intent: {state.get('user_intent', 'Unknown')}
• Tickers: {state.get('tickers', [])}
• Data Generated: F:{len(state.get('fundamentals', {}))}, N:{len(state.get('news', []))}, S:{len(state.get('sentiment', {}))}, G:{len(state.get('graph_images', []))}"""

        if errors:
            summary += f"\n\n❌ ERRORS:\n"
            for error in errors:
                summary += f"   • {error.get('node', 'Unknown')}: {error.get('error', 'Unknown error')}\n"

        return summary


def create_visualizer(orchestrator: LangGraphOrchestrator) -> WorkflowVisualizer:
    """Create and return a workflow visualizer."""
    return WorkflowVisualizer(orchestrator)
