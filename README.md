# Financial Market Multi-Agent LangGraph

A starter multi-agent architecture using LangGraph for financial market analysis.

## Agents
- Coordinator: Orchestrates flow, future place for conditional routing.
- NewsAgent (`fetch_news`): Fetch or summarize recent news (stub).
- FundamentalsAgent (`fetch_fundamentals`): Pulls fundamental metrics (stub).
- SentimentAgent (`analyze_sentiment`): Generates sentiment metrics (stub).
- AdvisorAgent (`advisor`): Produces basic recommendation from fundamentals.

## State Schema
Defined in `src/state.py` as `MarketState` containing:
- tickers
- news
- fundamentals
- sentiment
- recommendation

## Run
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Next Steps
- Replace stubs with real data (yfinance, news API, sentiment LLM).
- Add branching logic in `coordinator`.
- Implement caching & error handling.
- Add tests.
