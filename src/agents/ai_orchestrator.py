from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Set
from openai import OpenAI

from src.state import MarketState
from src.config import OPENAI_API_KEY, OPENAI_MODEL

# Import all available agents
from .ticker_selector_agent import select_tickers
from .fundamentals_agent import fetch_fundamentals
from .news_agent import fetch_news
from .sentiment_agent import analyze_sentiment
from .advisor_agent import advisor
from .code_interpreter_agent import code_interpreter_graphs

DEFAULT_MODEL = OPENAI_MODEL or "gpt-4o-mini"


class AIOrchestrator:
    """
    AI-powered orchestrator that uses LLM to decide which agents to run
    based on the user's prompt and current state
    """
    
    def __init__(self):
        # Available agents with their descriptions
        self.available_agents = {
            "ticker_selector": {
                "function": select_tickers,
                "description": "Extracts or searches for stock tickers from user prompts. Required for most workflows.",
                "inputs": ["prompt"],
                "outputs": ["tickers", "ticker_selection"],
                "dependencies": [],
                "typical_use": "When user mentions company names or asks about stocks"
            },
            "fundamentals": {
                "function": fetch_fundamentals,
                "description": "Fetches financial fundamentals (revenue, EPS, ratios, etc.) for selected tickers.",
                "inputs": ["tickers"],
                "outputs": ["fundamentals"],
                "dependencies": ["ticker_selector"],
                "typical_use": "For financial analysis, comparisons, valuations, investment decisions"
            },
            "news": {
                "function": fetch_news,
                "description": "Gathers recent news and market-moving events for selected tickers.",
                "inputs": ["tickers"],
                "outputs": ["news"],
                "dependencies": ["ticker_selector"],
                "typical_use": "For latest developments, market sentiment, recent events"
            },
            "sentiment": {
                "function": analyze_sentiment,
                "description": "Analyzes market sentiment and social media buzz for selected tickers.",
                "inputs": ["tickers"],
                "outputs": ["sentiment"],
                "dependencies": ["ticker_selector"],
                "typical_use": "For understanding market mood, investor sentiment, social trends"
            },
            "advisor": {
                "function": advisor,
                "description": "Provides buy/sell/hold recommendations based on fundamentals, news, and sentiment.",
                "inputs": ["fundamentals", "news", "sentiment"],
                "outputs": ["recommendation"],
                "dependencies": ["fundamentals"],
                "typical_use": "For investment advice, buy/sell decisions, portfolio recommendations"
            },
            "code_interpreter": {
                "function": code_interpreter_graphs,
                "description": "Creates charts, graphs, and data visualizations using Python/matplotlib.",
                "inputs": ["fundamentals", "prompt"],
                "outputs": ["graph_images", "graph_summary"],
                "dependencies": ["fundamentals"],
                "typical_use": "For plotting data, creating charts, visual comparisons, trend analysis"
            }
        }
    
    def decide_next_agents(self, prompt: str, current_state: MarketState, executed_agents: Set[str]) -> List[str]:
        """
        Use AI to decide which agents to run next based on:
        - User's original prompt
        - Current state of analysis
        - Which agents have already been executed
        """
        if not OPENAI_API_KEY:
            return []
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build context for AI decision
        available_agents_info = []
        for agent_name, agent_info in self.available_agents.items():
            if agent_name not in executed_agents:
                # Check if dependencies are met
                deps_met = all(dep in executed_agents for dep in agent_info["dependencies"])
                status = "✅ Ready" if deps_met else f"⏳ Waiting for: {', '.join(agent_info['dependencies'])}"
                
                available_agents_info.append(f"""
{agent_name}:
  Description: {agent_info['description']}
  Inputs needed: {', '.join(agent_info['inputs'])}
  Outputs: {', '.join(agent_info['outputs'])}
  Dependencies: {', '.join(agent_info['dependencies']) if agent_info['dependencies'] else 'None'}
  Status: {status}
  Use case: {agent_info['typical_use']}""")
        
        # Current state summary
        state_summary = []
        if current_state.get("tickers"):
            state_summary.append(f"Tickers identified: {', '.join(current_state['tickers'])}")
        if current_state.get("fundamentals"):
            state_summary.append("Fundamentals data available")
        if current_state.get("news"):
            state_summary.append("News data available")
        if current_state.get("sentiment"):
            state_summary.append("Sentiment data available")
        if current_state.get("recommendation"):
            state_summary.append("Investment recommendation available")
        if current_state.get("graph_images"):
            state_summary.append("Visualizations created")
        
        system_prompt = f"""
You are an AI orchestrator that decides which analysis agents to run next for financial analysis.

USER'S ORIGINAL REQUEST: "{prompt}"

AGENTS ALREADY EXECUTED: {', '.join(executed_agents) if executed_agents else 'None'}

CURRENT STATE:
{chr(10).join(state_summary) if state_summary else 'No data collected yet'}

AVAILABLE AGENTS:
{chr(10).join(available_agents_info)}

DECISION RULES:
1. Always consider what the user is actually asking for
2. Only suggest agents whose dependencies are satisfied
3. Prioritize agents that directly address the user's request
4. Consider the logical flow: tickers → data → analysis/visualization
5. Don't run redundant agents unless specifically needed
6. If user wants investment advice, include advisor
7. If user wants charts/plots/visualizations, include code_interpreter
8. If user asks about recent events, include news and/or sentiment

RESPOND with a JSON array of agent names to run next, in order of priority.
If no more agents are needed, return an empty array [].

Examples:
- For "Should I buy Apple?": ["ticker_selector", "fundamentals", "news", "sentiment", "advisor"]
- For "Plot Tesla vs Ford revenue": ["ticker_selector", "fundamentals", "code_interpreter"]
- For "What's happening with NVIDIA?": ["ticker_selector", "news", "sentiment"]
- For visualizations/charts: always include "code_interpreter"
- For investment decisions: always include "advisor"

Return ONLY the JSON array, no other text.
"""

        try:
            response = client.responses.create(
                model=DEFAULT_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "What agents should run next?"}
                ]
            )
            
            response_text = getattr(response, "output_text", "[]").strip()
            
            # Try to parse JSON response
            try:
                agents_to_run = json.loads(response_text)
                if isinstance(agents_to_run, list):
                    # Filter out invalid agents and already executed ones
                    valid_agents = []
                    for agent in agents_to_run:
                        if (agent in self.available_agents and 
                            agent not in executed_agents):
                            # Check dependencies
                            deps = self.available_agents[agent]["dependencies"]
                            if all(dep in executed_agents for dep in deps):
                                valid_agents.append(agent)
                    
                    return valid_agents
                else:
                    print(f"⚠️ AI returned non-list: {response_text}")
                    return []
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse AI response as JSON: {response_text}")
                return []
                
        except Exception as e:
            print(f"❌ Error in AI decision making: {e}")
            return []
    
    def should_continue(self, prompt: str, current_state: MarketState, executed_agents: Set[str]) -> bool:
        """
        Ask AI if the analysis is complete or if more agents should be run
        """
        if not OPENAI_API_KEY:
            return False
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build summary of what we have
        available_data = []
        if current_state.get("tickers"):
            available_data.append(f"Stock tickers: {', '.join(current_state['tickers'])}")
        if current_state.get("fundamentals"):
            available_data.append("Financial fundamentals data")
        if current_state.get("news"):
            available_data.append("Recent news data")
        if current_state.get("sentiment"):
            available_data.append("Market sentiment analysis")
        if current_state.get("recommendation"):
            available_data.append("Investment recommendation")
        if current_state.get("graph_images"):
            available_data.append(f"{len(current_state['graph_images'])} visualization(s)")
        
        system_prompt = f"""
You are evaluating whether a financial analysis is complete.

USER'S ORIGINAL REQUEST: "{prompt}"

AGENTS EXECUTED: {', '.join(executed_agents)}

DATA AVAILABLE:
{chr(10).join(available_data) if available_data else 'No data available yet'}

QUESTION: Is this analysis sufficient to answer the user's request?

Consider:
- Does the user want investment advice? (need recommendation)
- Does the user want charts/visualizations? (need graphs)
- Does the user want news/sentiment? (need those analyses)
- Does the user want comparisons? (need data for multiple stocks)
- Is there enough data to provide a meaningful answer?

Respond with ONLY "YES" if the analysis is complete, or "NO" if more work is needed.
"""

        try:
            response = client.responses.create(
                model=DEFAULT_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Is the analysis complete?"}
                ]
            )
            
            response_text = getattr(response, "output_text", "NO").strip().upper()
            return "NO" in response_text
            
        except Exception as e:
            print(f"❌ Error checking completion: {e}")
            return False
    
    def generate_final_response(self, prompt: str, current_state: MarketState) -> str:
        """
        Generate a comprehensive final response based on all collected data
        """
        if not OPENAI_API_KEY:
            return "Analysis completed. Please check the detailed results."
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Build comprehensive context
        context_parts = []
        
        if current_state.get("tickers"):
            context_parts.append(f"Analyzed stocks: {', '.join(current_state['tickers'])}")
        
        if current_state.get("fundamentals"):
            fundamentals = current_state["fundamentals"]
            if isinstance(fundamentals, dict):
                context_parts.append("Financial fundamentals:")
                for ticker, data in fundamentals.items():
                    if isinstance(data, dict):
                        metrics = []
                        if data.get("market_cap"):
                            metrics.append(f"Market Cap: ${data['market_cap']:,.0f}")
                        if data.get("valuation", {}).get("pe"):
                            metrics.append(f"P/E: {data['valuation']['pe']}")
                        if data.get("revenue"):
                            metrics.append(f"Revenue: ${data['revenue']:,.0f}")
                        if metrics:
                            context_parts.append(f"  {ticker}: {', '.join(metrics)}")
        
        if current_state.get("news"):
            news_items = current_state["news"]
            if isinstance(news_items, list) and news_items:
                context_parts.append(f"Latest news: {len(news_items)} articles analyzed")
        
        if current_state.get("sentiment"):
            sentiment_data = current_state["sentiment"]
            if isinstance(sentiment_data, dict):
                context_parts.append("Market sentiment analysis completed")
        
        if current_state.get("recommendation"):
            context_parts.append(f"Investment recommendation: {current_state['recommendation']}")
        
        if current_state.get("graph_images"):
            context_parts.append(f"Created {len(current_state['graph_images'])} visualization(s)")
        
        if current_state.get("graph_summary"):
            context_parts.append(f"Chart analysis: {current_state['graph_summary'][:200]}...")
        
        system_prompt = f"""
You are a professional financial advisor providing a final comprehensive response.

USER'S ORIGINAL REQUEST: "{prompt}"

ANALYSIS RESULTS:
{chr(10).join(context_parts)}

COMPLETE DATA AVAILABLE:
{json.dumps({k: v for k, v in current_state.items() if k not in ['code_interpreter_raw']}, indent=2, default=str)}

Your task:
1. Directly answer the user's original question
2. Provide actionable insights based on the analysis
3. If investment advice was requested, clearly state recommendations
4. If comparisons were requested, highlight key differences
5. If visualizations were created, reference them
6. Use a professional but accessible tone
7. Be specific and data-driven
8. Include any important caveats or disclaimers

Structure your response to be clear and valuable to the user.
"""

        try:
            response = client.responses.create(
                model=DEFAULT_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please provide the final comprehensive response."}
                ]
            )
            
            return getattr(response, "output_text", "Analysis completed. Please check the detailed results.")
            
        except Exception as e:
            return f"Analysis completed with comprehensive data. Error generating summary: {e}"
    
    def orchestrate(self, state: MarketState, max_iterations: int = 10) -> MarketState:
        """
        Main AI-driven orchestration method
        """
        prompt = state.get("prompt", "")
        
        if not prompt.strip():
            return {
                **state,
                "ai_orchestrator_error": "No prompt provided",
                "message": "Please provide a question or request to analyze."
            }
        
        print(f"🤖 AI Orchestrator starting analysis...")
        print(f"📝 User request: {prompt}")
        
        current_state = state.copy()
        executed_agents: Set[str] = set()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")
            
            # Ask AI what to do next
            next_agents = self.decide_next_agents(prompt, current_state, executed_agents)
            
            if not next_agents:
                print("🤖 AI says: No more agents needed")
                break
            
            print(f"🤖 AI decided to run: {', '.join(next_agents)}")
            
            # Execute the suggested agents
            for agent_name in next_agents:
                if agent_name in self.available_agents:
                    try:
                        print(f"  🔄 Running {agent_name}...")
                        agent_func = self.available_agents[agent_name]["function"]
                        result = agent_func(current_state)
                        
                        if isinstance(result, dict):
                            current_state.update(result)
                            executed_agents.add(agent_name)
                            
                            # Check for errors
                            error_key = f"{agent_name}_error"
                            if error_key in result:
                                print(f"    ⚠️  Warning: {result[error_key]}")
                        else:
                            print(f"    ⚠️  Unexpected result type: {type(result)}")
                            
                    except Exception as e:
                        error_msg = f"Error in {agent_name}: {str(e)}"
                        print(f"    ❌ {error_msg}")
                        current_state[f"{agent_name}_error"] = error_msg
                else:
                    print(f"    ⚠️  Unknown agent: {agent_name}")
            
            # Ask AI if we should continue
            if not self.should_continue(prompt, current_state, executed_agents):
                print("🤖 AI says: Analysis is complete")
                break
        
        if iteration >= max_iterations:
            print(f"⏰ Reached maximum iterations ({max_iterations})")
        
        # Generate final response
        print("📝 Generating final response...")
        try:
            final_response = self.generate_final_response(prompt, current_state)
            current_state["message"] = final_response
            current_state["ai_orchestrator_executed"] = list(executed_agents)
            current_state["ai_orchestrator_iterations"] = iteration
        except Exception as e:
            current_state["ai_orchestrator_error"] = f"Error generating final response: {e}"
            current_state["message"] = "Analysis completed. Please check individual agent results."
        
        print("✅ AI orchestration complete!")
        return current_state


# Convenience function
def ai_analyze_request(prompt: str, context: Optional[Dict[str, Any]] = None, max_iterations: int = 10) -> MarketState:
    """
    Analyze a user request using AI-driven orchestration
    """
    orchestrator = AIOrchestrator()
    
    initial_state: MarketState = {
        "prompt": prompt,
        "context": context or {}
    }
    
    return orchestrator.orchestrate(initial_state, max_iterations)
