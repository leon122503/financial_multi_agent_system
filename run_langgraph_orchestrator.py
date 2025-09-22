#!/usr/bin/env python3
"""
LangGraph Orchestrator Runner

This script demonstrates the LangGraph-based orchestrator system for financial market analysis.
Provides comprehensive testing, comparison with the AI orchestrator, and detailed reporting.

Usage:
    python run_langgraph_orchestrator.py "Your query here" [max_iterations]
    python run_langgraph_orchestrator.py "Should I invest in Apple?" 5
    python run_langgraph_orchestrator.py "Compare Tesla vs Ford PE ratios in a chart"
"""

import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

from src.langgraph_orchestrator import create_langgraph_orchestrator
from src.langgraph_visualizer import create_visualizer


def run_langgraph_analysis(
    prompt: str, context: Optional[Dict[str, Any]] = None, max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Run a complete LangGraph-based financial analysis.

    Args:
        prompt: User query to analyze
        context: Optional additional context
        max_iterations: Maximum workflow iterations (unused in LangGraph but kept for compatibility)

    Returns:
        Dictionary containing all analysis results and execution metadata
    """

    print("=" * 80)
    print("🚀 LANGGRAPH FINANCIAL MARKET ORCHESTRATOR")
    print("=" * 80)

    start_time = time.time()

    # Create orchestrator and visualizer
    orchestrator = create_langgraph_orchestrator()
    visualizer = create_visualizer(orchestrator)

    try:
        # Execute the workflow
        print(f"\n📝 Query: {prompt}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Show workflow structure
        print("\n📊 Workflow Structure:")
        print(orchestrator.get_workflow_visualization())

        # Execute the analysis
        final_state = orchestrator.execute(prompt, context)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Generate visualizations and reports
        print("\n" + "=" * 60)
        print("📈 EXECUTION ANALYSIS")
        print("=" * 60)

        # Show execution flow diagram
        print(visualizer.generate_execution_diagram(final_state))

        # Show performance report
        print(visualizer.generate_performance_report(final_state))

        # Show debug summary
        print(visualizer.generate_debug_summary(final_state))

        # Prepare comprehensive results
        results = {
            # Original state data
            **dict(final_state),
            # Execution metadata
            "execution_metadata": {
                "orchestrator_type": "langgraph",
                "execution_time_seconds": round(execution_time, 2),
                "timestamp": datetime.now().isoformat(),
                "total_nodes_executed": len(final_state.get("execution_path", [])),
                "workflow_completed": final_state.get("workflow_step") == "complete",
            },
            # Performance statistics
            "performance_stats": orchestrator.get_execution_stats(final_state),
        }

        # Export detailed log
        log_file = visualizer.export_execution_log(final_state)
        print(f"\n💾 {log_file}")

        print(f"\n✅ Analysis completed in {execution_time:.2f} seconds")

        return results

    except Exception as e:
        print(f"\n❌ Error during LangGraph execution: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "execution_metadata": {
                "orchestrator_type": "langgraph",
                "execution_time_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "workflow_completed": False,
            },
        }


def display_final_results(results: Dict[str, Any]):
    """Display the final analysis results in a user-friendly format."""

    print("\n" + "=" * 80)
    print("📋 FINAL ANALYSIS RESULTS")
    print("=" * 80)

    # Show execution summary
    metadata = results.get("execution_metadata", {})
    print(f"⏰ Execution Time: {metadata.get('execution_time_seconds', 0)} seconds")
    print(f"🎯 Nodes Executed: {metadata.get('total_nodes_executed', 0)}")
    print(f"✅ Completed: {metadata.get('workflow_completed', False)}")

    # Show tickers found
    tickers = results.get("tickers", [])
    if tickers:
        print(f"\n🎯 Tickers Analyzed: {', '.join(tickers)}")

    # Show recommendation if available
    recommendation = results.get("recommendation", "")
    if recommendation:
        print(f"\n💡 Investment Recommendation:")
        print("─" * 40)
        # Show first few lines of recommendation
        rec_lines = recommendation.split("\n")[:10]
        for line in rec_lines:
            print(f"   {line}")
        if len(recommendation.split("\n")) > 10:
            print("   ... (see full results in JSON output)")

    # Show charts generated
    images = results.get("graph_images", [])
    if images:
        print(f"\n📈 Charts Generated: {len(images)}")
        for img in images:
            print(f"   📊 {img}")

    # Show final message
    message = results.get("message", "")
    if message and message != recommendation:  # Avoid duplication
        print(f"\n📝 Summary Message:")
        print("─" * 40)
        # Show full message
        for line in message.split("\n"):
            print(f"   {line}")


def save_results_to_file(results: Dict[str, Any], prompt: str):
    """Save results to a timestamped JSON file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/langgraph_results_{timestamp}.json"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return None


def main():
    """Main execution function."""

    # Parse command line arguments
    if len(sys.argv) < 2:
        print(
            'Usage: python run_langgraph_orchestrator.py "Your query here" [max_iterations]'
        )
        print("\nExamples:")
        print('  python run_langgraph_orchestrator.py "Should I invest in Apple?"')
        print(
            '  python run_langgraph_orchestrator.py "Compare Tesla vs Ford PE ratios in a chart"'
        )
        print(
            '  python run_langgraph_orchestrator.py "What\'s the latest news on NVIDIA?" 3'
        )
        sys.exit(1)

    prompt = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Run the analysis
    results = run_langgraph_analysis(prompt, max_iterations=max_iterations)

    # Display results
    display_final_results(results)

    # Save results
    save_results_to_file(results, prompt)

    print("\n" + "=" * 80)
    print("🎉 LangGraph Financial Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
