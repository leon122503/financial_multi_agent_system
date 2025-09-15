#!/usr/bin/env python3
"""
AI Orchestrator Runner - Use AI to intelligently decide which agents to run
"""

import sys
import json
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.ai_orchestrator import ai_analyze_request


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_ai_orchestrator.py '<your question>' [max_iterations]")
        print("\nThis orchestrator uses AI to intelligently decide which agents to run!")
        print("\nExample queries:")
        print("  'Should I buy Apple stock right now?'")
        print("  'Compare Tesla vs Ford financial performance and show charts'") 
        print("  'Plot the revenue growth of top 3 tech companies'")
        print("  'What's the current sentiment around NVIDIA and should I invest?'")
        print("  'Analyze Microsoft's latest financials and news'")
        print("  'Show me a PE ratio comparison between Amazon and Google'")
        sys.exit(1)
    
    user_prompt = " ".join(sys.argv[1:-1]) if len(sys.argv) > 2 and sys.argv[-1].isdigit() else " ".join(sys.argv[1:])
    max_iterations = int(sys.argv[-1]) if len(sys.argv) > 2 and sys.argv[-1].isdigit() else 10
    
    print(f"🤖 AI-Driven Analysis: {user_prompt}")
    print(f"⚙️  Max iterations: {max_iterations}")
    print("=" * 70)
    print("🧠 The AI will decide which agents to run based on your request...")
    print("=" * 70)
    
    # Run the AI-driven orchestrated analysis
    result = ai_analyze_request(user_prompt, max_iterations=max_iterations)
    
    print("\n" + "=" * 70)
    print("🎯 AI ORCHESTRATION RESULTS")
    print("=" * 70)
    
    # Display key results
    if "message" in result:
        print(f"\n💬 FINAL RESPONSE:")
        print(result["message"])
    
    if "ai_orchestrator_executed" in result:
        executed = result["ai_orchestrator_executed"]
        print(f"\n🤖 AI DECIDED TO RUN: {' → '.join(executed)}")
    
    if "ai_orchestrator_iterations" in result:
        print(f"\n🔄 COMPLETED IN: {result['ai_orchestrator_iterations']} iterations")
    
    if "tickers" in result and result["tickers"]:
        print(f"\n📈 ANALYZED STOCKS: {', '.join(result['tickers'])}")
    
    # Show detailed results by category
    if "fundamentals" in result and result["fundamentals"]:
        print(f"\n📊 FUNDAMENTAL ANALYSIS:")
        fundamentals = result["fundamentals"]
        for ticker, data in fundamentals.items():
            if isinstance(data, dict):
                print(f"  📈 {ticker}:")
                if data.get("market_cap"):
                    print(f"    Market Cap: ${data['market_cap']:,.0f}")
                if data.get("valuation", {}).get("pe"):
                    print(f"    P/E Ratio: {data['valuation']['pe']}")
                if data.get("valuation", {}).get("price_to_book"):
                    print(f"    P/B Ratio: {data['valuation']['price_to_book']}")
                if data.get("revenue"):
                    print(f"    Revenue: ${data['revenue']:,.0f}")
    
    if "recommendation" in result and result["recommendation"]:
        print(f"\n💡 INVESTMENT RECOMMENDATION:")
        print(result["recommendation"])
    
    if "graph_images" in result and result["graph_images"]:
        print(f"\n📊 GENERATED VISUALIZATIONS:")
        for i, img in enumerate(result["graph_images"], 1):
            print(f"  {i}. {img}")
    
    if "graph_summary" in result and result["graph_summary"]:
        print(f"\n📈 CHART ANALYSIS:")
        print(result["graph_summary"])
    
    if "news" in result and result["news"]:
        print(f"\n📰 NEWS ANALYSIS: {len(result['news'])} articles analyzed")
    
    if "sentiment" in result and result["sentiment"]:
        print(f"\n😊 SENTIMENT ANALYSIS: {len(result['sentiment'])} sentiment assessments")
    
    # Show AI decision-making process
    if "ticker_selection" in result:
        selection = result["ticker_selection"]
        if isinstance(selection, dict) and selection.get("mode"):
            print(f"\n🎯 TICKER SELECTION: {selection['mode']} - {selection.get('reason', 'N/A')}")
    
    # Show any errors or warnings
    error_keys = [k for k in result.keys() if k.endswith("_error")]
    if error_keys:
        print(f"\n⚠️  WARNINGS/ERRORS:")
        for error_key in error_keys:
            print(f"  - {error_key}: {result[error_key]}")
    
    # Show execution stats
    data_summary = []
    if result.get("fundamentals"):
        data_summary.append(f"✅ Fundamentals for {len(result['fundamentals'])} stocks")
    if result.get("news"):
        data_summary.append(f"✅ {len(result['news'])} news items")
    if result.get("sentiment"):
        data_summary.append(f"✅ Sentiment for {len(result['sentiment'])} stocks")
    if result.get("graph_images"):
        data_summary.append(f"✅ {len(result['graph_images'])} visualizations")
    
    if data_summary:
        print(f"\n📋 DATA COLLECTED:")
        for summary in data_summary:
            print(f"  {summary}")
    
    # Save results
    output_file = Path("outputs") / "ai_orchestrator_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    # Clean up the result for JSON serialization
    json_result = {k: v for k, v in result.items() if k not in ["code_interpreter_raw"]}
    
    with open(output_file, "w") as f:
        json.dump(json_result, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to: {output_file}")
    
    # Show the power of AI orchestration
    print(f"\n🧠 AI ORCHESTRATION SUMMARY:")
    print(f"   The AI analyzed your request and intelligently chose which agents to run.")
    print(f"   No predefined workflows - pure AI decision making!")
    if "ai_orchestrator_executed" in result:
        agents = result["ai_orchestrator_executed"]
        print(f"   Agents selected: {', '.join(agents)}")
        print(f"   Total agents available: 6 (ticker_selector, fundamentals, news, sentiment, advisor, code_interpreter)")
        print(f"   AI efficiency: {len(agents)}/6 agents used ({len(agents)/6*100:.1f}%)")


if __name__ == "__main__":
    main()
