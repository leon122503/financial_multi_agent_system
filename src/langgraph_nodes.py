"""
LangGraph node implementations that wrap existing agents for use in the LangGraph workflow.
Each node function follows the LangGraph pattern: takes state dict, returns state dict updates.
"""

from typing import Dict, Any
import copy

from src.langgraph_state import LangGraphMarketState
from src.agents.ticker_selector_agent import select_tickers as _select_tickers
from src.agents.fundamentals_agent import fetch_fundamentals as _fetch_fundamentals
from src.agents.news_agent import fetch_news as _fetch_news
from src.agents.sentiment_agent import analyze_sentiment as _analyze_sentiment
from src.agents.advisor_agent import advisor as _advisor
from src.agents.code_interpreter_agent import (
    code_interpreter_graphs as _code_interpreter_graphs,
)


def ticker_selector_node(state: LangGraphMarketState) -> Dict[str, Any]:
    """LangGraph node for ticker selection."""
    print("🎯 Executing ticker selection...")

    # Track execution
    execution_path = state.get("execution_path", [])
    execution_path.append("ticker_selector")

    # Call the existing agent
    result = _select_tickers(state)

    # Update LangGraph-specific state
    updates = {
        "execution_path": execution_path,
        "workflow_step": "ticker_selection_complete",
        "ticker_selection": result.get("ticker_selection"),
        "tickers": result.get("tickers", []),
        "ai_orchestrator_executed": state.get("ai_orchestrator_executed", [])
        + ["ticker_selector"],
    }

    # Handle errors
    if "ticker_selection_error" in result:
        errors = state.get("errors", [])
        errors.append(
            {"node": "ticker_selector", "error": result["ticker_selection_error"]}
        )
        updates["errors"] = errors

    return updates


def fundamentals_node(state: LangGraphMarketState) -> Dict[str, Any]:
    """LangGraph node for fundamental analysis."""
    print("📊 Executing fundamental analysis...")

    execution_path = state.get("execution_path", [])
    execution_path.append("fundamentals")

    result = _fetch_fundamentals(state)

    updates = {
        "execution_path": execution_path,
        "workflow_step": "fundamentals_complete",
        "fundamentals_complete": True,
        "fundamentals": result.get("fundamentals", {}),
        "ai_orchestrator_executed": state.get("ai_orchestrator_executed", [])
        + ["fundamentals"],
    }

    return updates


def news_node(state: LangGraphMarketState) -> Dict[str, Any]:
    """LangGraph node for news analysis."""
    print("📰 Executing news analysis...")

    execution_path = state.get("execution_path", [])
    execution_path.append("news")

    result = _fetch_news(state)

    updates = {
        "execution_path": execution_path,
        "workflow_step": "news_complete",
        "news_complete": True,
        "news": result.get("news", []),
        "ai_orchestrator_executed": state.get("ai_orchestrator_executed", [])
        + ["news"],
    }

    return updates


def sentiment_node(state: LangGraphMarketState) -> Dict[str, Any]:
    """LangGraph node for sentiment analysis."""
    print("🎭 Executing sentiment analysis...")

    execution_path = state.get("execution_path", [])
    execution_path.append("sentiment")

    result = _analyze_sentiment(state)

    updates = {
        "execution_path": execution_path,
        "workflow_step": "sentiment_complete",
        "sentiment_complete": True,
        "sentiment": result.get("sentiment", {}),
        "ai_orchestrator_executed": state.get("ai_orchestrator_executed", [])
        + ["sentiment"],
    }

    return updates


def advisor_node(state: LangGraphMarketState) -> Dict[str, Any]:
    """LangGraph node for investment recommendation."""
    print("💡 Generating investment recommendation...")

    execution_path = state.get("execution_path", [])
    execution_path.append("advisor")

    result = _advisor(state)

    updates = {
        "execution_path": execution_path,
        "workflow_step": "recommendation_complete",
        "recommendation_complete": True,
        "recommendation": result.get("recommendation", ""),
        "ai_orchestrator_executed": state.get("ai_orchestrator_executed", [])
        + ["advisor"],
    }

    return updates


def code_interpreter_node(state: LangGraphMarketState) -> Dict[str, Any]:
    """LangGraph node for chart generation."""

    # Check if visualization is already complete to prevent duplicates
    if state.get("visualization_complete", False):
        print("📈 Visualization already complete - skipping duplicate generation")
        return {"workflow_step": "visualization_already_complete"}

    print("📈 Generating charts and visualizations...")

    execution_path = state.get("execution_path", [])
    execution_path.append("code_interpreter")

    result = _code_interpreter_graphs(state)

    updates = {
        "execution_path": execution_path,
        "workflow_step": "visualization_complete",
        "visualization_complete": True,
        "graph_summary": result.get("graph_summary", ""),
        "graph_images": result.get("graph_images", []),
        "ai_orchestrator_executed": state.get("ai_orchestrator_executed", [])
        + ["code_interpreter"],
    }

    return updates


def analysis_router_node(state: LangGraphMarketState) -> Dict[str, Any]:
    """
    LangGraph node that uses AI to determine what types of analysis are needed.
    Uses OpenAI to intelligently decide which agents to run based on user intent.
    """
    print("🧠 Using AI to analyze user intent and routing workflow...")

    prompt = state.get("prompt", "")

    # AI-driven agent selection
    from openai import OpenAI
    from src.config import OPENAI_API_KEY, OPENAI_MODEL

    if not OPENAI_API_KEY:
        print("   ⚠️  No OpenAI API key - falling back to basic routing")
        # Fallback to simple logic if no AI available
        requires_visualization = "chart" in prompt.lower() or "graph" in prompt.lower()
        requires_fundamentals = (
            "invest" in prompt.lower() or "financial" in prompt.lower()
        )
        user_intent = (
            "investment_advice" if "invest" in prompt.lower() else "general_query"
        )
        complexity_score = 2
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_prompt = """
        You are an intelligent financial analysis router. Analyze the user's request and determine:
        1. What type of analysis they want (user_intent)
        2. Which specific data components are needed
        3. The complexity of the request (1-5 scale)
        
        Available agents:
        - ticker_selector: Find stock tickers from company names
        - fundamentals: Get financial data (earnings, ratios, balance sheet)
        - news: Get recent news articles about companies
        - sentiment: Analyze market sentiment
        - advisor: Generate investment recommendations (BUY/SELL/HOLD recommendations)
        - code_interpreter: Create charts and visualizations
        
        IMPORTANT: If the user asks about investing, buying, selling, or wants investment advice, 
        ALWAYS set requires_advisor=true and user_intent="investment_advice".
        
        Investment question keywords: "should I invest", "buy", "sell", "invest in", "purchase", 
        "investment advice", "recommend", "worth buying", "good investment"
        
        Respond with ONLY a JSON object in this format:
        {
            "user_intent": "investment_advice|visualization|news_analysis|fundamental_analysis|sentiment_analysis|general_query",
            "requires_fundamentals": true/false,
            "requires_news": true/false,
            "requires_sentiment": true/false,
            "requires_visualization": true/false,
            "requires_advisor": true/false,
            "complexity_score": 1-5,
            "reasoning": "brief explanation of decisions"
        }
        """

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User request: {prompt}"},
                ],
                # Removed temperature parameter as it's not supported by this model
            )

            import json

            ai_decision = json.loads(response.choices[0].message.content.strip())

            user_intent = ai_decision.get("user_intent", "general_query")
            requires_fundamentals = ai_decision.get("requires_fundamentals", False)
            requires_news = ai_decision.get("requires_news", False)
            requires_sentiment = ai_decision.get("requires_sentiment", False)
            requires_visualization = ai_decision.get("requires_visualization", False)
            requires_advisor = ai_decision.get("requires_advisor", False)
            complexity_score = ai_decision.get("complexity_score", 2)
            reasoning = ai_decision.get("reasoning", "AI analysis")

            # Auto-enable advisor for investment advice
            if user_intent == "investment_advice":
                requires_advisor = True
                requires_fundamentals = (
                    True  # Always need fundamentals for investment advice
                )

            print(f"   🤖 AI Reasoning: {reasoning}")
            print(f"   🎯 Requires Advisor: {requires_advisor}")

        except Exception as e:
            print(f"   ⚠️  AI routing failed ({e}) - using fallback")
            # Fallback logic with better investment detection
            requires_visualization = (
                "chart" in prompt.lower() or "graph" in prompt.lower()
            )
            requires_fundamentals = (
                "invest" in prompt.lower() or "financial" in prompt.lower()
            )
            requires_news = "news" in prompt.lower()
            requires_sentiment = "sentiment" in prompt.lower()

            # Better investment advice detection
            investment_keywords = [
                "should i invest",
                "buy",
                "sell",
                "invest in",
                "purchase",
                "investment advice",
                "recommend",
                "worth buying",
                "good investment",
            ]
            requires_advisor = any(
                keyword in prompt.lower() for keyword in investment_keywords
            )

            if requires_advisor:
                user_intent = "investment_advice"
                requires_fundamentals = (
                    True  # Always need fundamentals for investment advice
                )
            else:
                user_intent = (
                    "fundamental_analysis" if requires_fundamentals else "general_query"
                )

            complexity_score = 2

    # COMMENTED OUT: Original rule-based routing logic
    """
    # Original keyword-based routing (now replaced by AI)
    requires_visualization = any(
        word in prompt.lower()
        for word in [
            "chart", "graph", "plot", "visualiz", "show", "display", "draw", "image",
        ]
    )

    requires_news = any(
        word in prompt.lower()
        for word in [
            "news", "latest", "recent", "headlines", "announcement", "development",
        ]
    )

    requires_sentiment = (
        any(
            word in prompt.lower()
            for word in [
                "sentiment", "opinion", "feeling", "market mood", "bullish", "bearish",
            ]
        )
        or requires_news
    )  # News often implies sentiment analysis

    requires_fundamentals = (
        any(
            word in prompt.lower()
            for word in [
                "fundamental", "financial", "earnings", "revenue", "pe ratio",
                "valuation", "balance sheet", "cash flow", "invest", "buy", "sell",
            ]
        )
        or "should i" in prompt.lower()
    )

    # Determine user intent
    user_intent = "general_query"
    if any(
        word in prompt.lower() for word in ["should i", "invest", "buy", "sell", "recommend"]
    ):
        user_intent = "investment_advice"
    elif requires_visualization:
        user_intent = "visualization"
    elif requires_news and not requires_fundamentals:
        user_intent = "news_analysis"
    elif requires_fundamentals and not requires_news:
        user_intent = "fundamental_analysis"
    elif requires_sentiment and not (requires_fundamentals or requires_news):
        user_intent = "sentiment_analysis"

    # Calculate complexity score (1-5) for determining execution strategy
    complexity_score = 1
    if requires_fundamentals:
        complexity_score += 1
    if requires_news:
        complexity_score += 1
    if requires_sentiment:
        complexity_score += 1
    if requires_visualization:
        complexity_score += 1
    """

    # Determine next actions based on AI requirements
    next_actions = []
    if not state.get("tickers"):
        next_actions.append("ticker_selector")

    # Parallel tasks (can run simultaneously after ticker selection)
    parallel_tasks = []
    if requires_fundamentals and not state.get("fundamentals_complete"):
        parallel_tasks.append("fundamentals")
    if requires_news and not state.get("news_complete"):
        parallel_tasks.append("news")
    if requires_sentiment and not state.get("sentiment_complete"):
        parallel_tasks.append("sentiment")

    # Sequential tasks (must run after parallel tasks)
    if requires_advisor and not state.get("recommendation_complete"):
        next_actions.append("advisor")
    if requires_visualization and not state.get("visualization_complete"):
        next_actions.append("code_interpreter")

    execution_path = state.get("execution_path", [])
    execution_path.append("analysis_router")

    updates = {
        "execution_path": execution_path,
        "workflow_step": "routing_complete",
        "user_intent": user_intent,
        "requires_visualization": requires_visualization,
        "requires_news": requires_news,
        "requires_sentiment": requires_sentiment,
        "requires_fundamentals": requires_fundamentals,
        "requires_advisor": requires_advisor,
        "complexity_score": complexity_score,
        "next_actions": next_actions,
        "parallel_tasks": parallel_tasks,
        "ai_orchestrator_iterations": state.get("ai_orchestrator_iterations", 0) + 1,
    }

    print(f"   📋 User intent: {user_intent}")
    print(f"   🎯 Complexity: {complexity_score}/5")
    print(f"   ⚡ Parallel tasks: {parallel_tasks}")
    print(f"   📝 Next actions: {next_actions}")

    return updates


def final_synthesis_node(state: LangGraphMarketState) -> Dict[str, Any]:
    """
    Final node that synthesizes all analysis results into a comprehensive response.
    """
    print("🎁 Synthesizing final response...")

    prompt = state.get("prompt", "")
    user_intent = state.get("user_intent", "general_query")

    # Build comprehensive message based on available data
    message_parts = []

    # Add fundamentals if available and relevant
    fundamentals = state.get("fundamentals", {})
    if fundamentals and user_intent in [
        "fundamental_analysis",
        "investment_advice",
        "general_query",
    ]:
        message_parts.append("📊 FUNDAMENTAL ANALYSIS RESULTS:\n")
        for ticker, data in fundamentals.items():
            # Helper function to safely format numbers
            def safe_format(value, default=0):
                """Safely format a number, handling None values"""
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default

            message_parts.append(
                f"\n🏢 {ticker} ({data.get('sector', 'N/A')} - {data.get('industry', 'N/A')})"
            )
            message_parts.append(
                f"Market Cap: ${safe_format(data.get('market_cap')):,.0f}"
            )

            # Valuation metrics
            valuation = data.get("valuation", {})
            if valuation:
                message_parts.append(f"\n📈 Valuation:")
                message_parts.append(f"  • P/E Ratio: {valuation.get('pe', 'N/A')}")
                message_parts.append(
                    f"  • Forward P/E: {valuation.get('forward_pe', 'N/A')}"
                )
                message_parts.append(f"  • PEG Ratio: {valuation.get('peg', 'N/A')}")
                message_parts.append(
                    f"  • Price-to-Book: {valuation.get('price_to_book', 'N/A')}"
                )
                message_parts.append(
                    f"  • Price-to-Sales: {valuation.get('price_to_sales', 'N/A')}"
                )

            # Profitability metrics
            profitability = data.get("profitability", {})
            if profitability:
                message_parts.append(f"\n💰 Profitability:")
                message_parts.append(
                    f"  • Net Margin: {safe_format(profitability.get('margins'))*100:.1f}%"
                )
                message_parts.append(
                    f"  • Operating Margin: {safe_format(profitability.get('operating_margin'))*100:.1f}%"
                )
                message_parts.append(
                    f"  • ROE: {safe_format(profitability.get('roe'))*100:.1f}%"
                )
                message_parts.append(
                    f"  • ROA: {safe_format(profitability.get('roa'))*100:.1f}%"
                )

            # Growth metrics
            growth = data.get("growth", {})
            if growth:
                message_parts.append(f"\n📈 Growth:")
                message_parts.append(
                    f"  • Revenue YoY: {safe_format(growth.get('revenue_yoy'))*100:.1f}%"
                )
                message_parts.append(
                    f"  • Earnings YoY: {safe_format(growth.get('earnings_yoy'))*100:.1f}%"
                )

            # Helper function to safely format numbers
            def safe_format(value, default=0):
                """Safely format a number, handling None values"""
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default

            # Financial health
            message_parts.append(f"\n💵 Financial Health:")
            message_parts.append(
                f"  • Revenue: ${safe_format(data.get('revenue')):,.0f}"
            )
            message_parts.append(
                f"  • Net Income: ${safe_format(data.get('net_income')):,.0f}"
            )
            message_parts.append(
                f"  • Total Debt: ${safe_format(data.get('debt')):,.0f}"
            )
            message_parts.append(
                f"  • Free Cash Flow: ${safe_format(data.get('free_cashflow')):,.0f}"
            )

            # Recent performance
            quarterly_eps = data.get("quarterly_eps", [])
            if quarterly_eps:
                message_parts.append(f"\n📊 Recent Quarterly EPS:")
                for q in quarterly_eps[:4]:  # Show last 4 quarters
                    surprise = q.get("surprise", 0)
                    surprise_text = (
                        f" ({surprise:+.1f}% surprise)" if surprise != 0 else ""
                    )
                    message_parts.append(
                        f"  • {q.get('date', 'N/A')}: ${q.get('eps', 'N/A')}{surprise_text}"
                    )

    # Add recommendation if available
    recommendation = state.get("recommendation", "")
    if recommendation:
        message_parts.append(f"\n\n💡 INVESTMENT RECOMMENDATION:\n{recommendation}")

    # Add news if available
    news = state.get("news", [])
    if news and user_intent in ["news_analysis", "investment_advice"]:
        message_parts.append(f"\n\n📰 RECENT NEWS:")
        for article in news[:3]:  # Show top 3 news items
            title = article.get("title", "No title")
            summary = article.get("summary", "No summary")
            message_parts.append(f"\n  • {title}")
            if summary and len(summary) < 200:
                message_parts.append(f"    {summary}")

    # Add sentiment if available
    sentiment = state.get("sentiment", {})
    if sentiment and user_intent in ["sentiment_analysis", "investment_advice"]:
        message_parts.append(f"\n\n🎭 SENTIMENT ANALYSIS:")
        for ticker, sent_data in sentiment.items():
            score = sent_data.get("score", "N/A")
            label = sent_data.get("label", "N/A")
            message_parts.append(f"  • {ticker}: {label} (Score: {score})")

    # Add chart summary if available
    graph_summary = state.get("graph_summary", "")
    if graph_summary:
        message_parts.append(f"\n\n📊 VISUALIZATION SUMMARY:\n{graph_summary}")

    # Add images if available
    graph_images = state.get("graph_images", [])
    if graph_images:
        message_parts.append(f"\n\n📈 Generated {len(graph_images)} chart(s):")
        for img in graph_images:
            message_parts.append(f"  • {img}")

    # Combine all parts
    raw_message = "\n".join(message_parts) if message_parts else "Analysis complete."

    # Add AI-powered summarization for investment advice questions
    final_message = raw_message
    if user_intent == "investment_advice" and message_parts:
        try:
            from openai import OpenAI
            from src.config import OPENAI_API_KEY, OPENAI_MODEL

            if OPENAI_API_KEY:
                client = OpenAI(api_key=OPENAI_API_KEY)

                summarization_prompt = f"""
                You are a financial advisor providing a clear, actionable summary for a client's investment question.
                
                Original Question: {prompt}
                
                Analysis Data:
                {raw_message}
                
                Provide a concise, professional summary that:
                1. Directly answers their investment question
                2. Highlights the key financial metrics that matter most
                3. Gives a clear recommendation if available
                4. Mentions important risks or considerations
                5. Keeps it under 300 words
                
                Format as a natural, conversational response to their question.
                """

                response = client.chat.completions.create(
                    model=OPENAI_MODEL or "gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional financial advisor providing clear, actionable investment guidance.",
                        },
                        {"role": "user", "content": summarization_prompt},
                    ],
                )

                ai_summary = response.choices[0].message.content.strip()
                final_message = f"🤖 AI INVESTMENT SUMMARY:\n\n{ai_summary}\n\n---\n\nDETAILED ANALYSIS:\n{raw_message}"
                print(f"   🤖 Added AI summary for investment advice")

        except Exception as e:
            print(f"   ⚠️  AI summarization failed ({e}) - using raw message")
            final_message = raw_message

    execution_path = state.get("execution_path", [])
    execution_path.append("final_synthesis")

    updates = {
        "execution_path": execution_path,
        "workflow_step": "complete",
        "message": final_message,
    }

    print(f"   ✅ Final message length: {len(final_message)} characters")
    print(f"   📊 Execution path: {' → '.join(execution_path)}")

    return updates
