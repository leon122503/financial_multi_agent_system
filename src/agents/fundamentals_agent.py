from typing import Any, Dict, List
import traceback
import requests

from src.langgraph_state import LangGraphMarketState
from src.config import ALPHAVANTAGE_API_KEY


BASE_URL = "https://www.alphavantage.co/query"


def _safe_float(x: Any):
    try:
        return float(x) if x not in (None, "None", "") else None
    except Exception:
        return None


def _av_request(params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to query Alpha Vantage safely"""
    if not ALPHAVANTAGE_API_KEY:
        return {}
    try:
        params["apikey"] = ALPHAVANTAGE_API_KEY
        r = requests.get(BASE_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json() or {}
    except Exception:
        return {}


def _av_overview(ticker: str) -> Dict[str, Any]:
    """Slim overview: sector, valuation, profitability"""
    data = _av_request({"function": "OVERVIEW", "symbol": ticker})
    if not data:
        return {}

    return {
        "symbol": data.get("Symbol"),
        "sector": data.get("Sector"),
        "industry": data.get("Industry"),
        "market_cap": _safe_float(data.get("MarketCapitalization")),
        "valuation": {
            "pe": _safe_float(data.get("PERatio")),
            "forward_pe": _safe_float(data.get("ForwardPE")),
            "peg": _safe_float(data.get("PEGRatio")),
            "price_to_book": _safe_float(data.get("PriceToBookRatio")),
            "price_to_sales": _safe_float(data.get("PriceToSalesRatioTTM")),
        },
        "profitability": {
            "margins": _safe_float(data.get("ProfitMargin")),
            "operating_margin": _safe_float(data.get("OperatingMarginTTM")),
            "roe": _safe_float(data.get("ReturnOnEquityTTM")),
            "roa": _safe_float(data.get("ReturnOnAssetsTTM")),
        },
        "dividends": {
            "yield": _safe_float(data.get("DividendYield")),
            "per_share": _safe_float(data.get("DividendPerShare")),
        },
        "growth": {
            "revenue_yoy": _safe_float(data.get("QuarterlyRevenueGrowthYOY")),
            "earnings_yoy": _safe_float(data.get("QuarterlyEarningsGrowthYOY")),
        },
        "52w": {
            "high": _safe_float(data.get("52WeekHigh")),
            "low": _safe_float(data.get("52WeekLow")),
        },
        "shares_outstanding": _safe_float(data.get("SharesOutstanding")),
    }


def _av_earnings(ticker: str) -> Dict[str, Any]:
    """Slimmed EPS history (last 5 annual + last 4 quarters)"""
    data = _av_request({"function": "EARNINGS", "symbol": ticker})
    out: Dict[str, Any] = {}
    if not data:
        return out

    ann = data.get("annualEarnings", [])
    qtr = data.get("quarterlyEarnings", [])

    if ann and isinstance(ann, list):
        out["annual_eps"] = [
            {
                "date": e.get("fiscalDateEnding"),
                "eps": _safe_float(e.get("reportedEPS")),
            }
            for e in ann[:5]  # keep last 5 years
        ]

    if qtr and isinstance(qtr, list):
        out["quarterly_eps"] = [
            {
                "date": e.get("fiscalDateEnding"),
                "eps": _safe_float(e.get("reportedEPS")),
                "surprise": _safe_float(e.get("surprisePercentage")),
            }
            for e in qtr[:4]  # keep last 4 quarters
        ]

    return out


def _av_financials(ticker: str) -> Dict[str, Any]:
    """Slim key financials from last annual reports"""
    out: Dict[str, Any] = {}

    # Income statement
    income = _av_request({"function": "INCOME_STATEMENT", "symbol": ticker})
    if income.get("annualReports"):
        last = income["annualReports"][0]
        out.update(
            {
                "revenue": _safe_float(last.get("totalRevenue")),
                "net_income": _safe_float(last.get("netIncome")),
                "eps": _safe_float(last.get("eps")),
            }
        )

    # Balance sheet
    balance = _av_request({"function": "BALANCE_SHEET", "symbol": ticker})
    if balance.get("annualReports"):
        last = balance["annualReports"][0]
        out.update(
            {
                "assets": _safe_float(last.get("totalAssets")),
                "liabilities": _safe_float(last.get("totalLiabilities")),
                "debt": _safe_float(last.get("longTermDebt")),
                "equity": _safe_float(last.get("totalShareholderEquity")),
            }
        )

    # Cash flow
    cashflow = _av_request({"function": "CASH_FLOW", "symbol": ticker})
    if cashflow.get("annualReports"):
        last = cashflow["annualReports"][0]
        ocf = _safe_float(last.get("operatingCashflow"))
        capex = _safe_float(last.get("capitalExpenditures"))
        out.update(
            {
                "operating_cashflow": ocf,
                "capex": capex,
                "free_cashflow": (
                    (ocf - capex) if ocf is not None and capex is not None else None
                ),
            }
        )

    return out


def _fundamentals_for_ticker(ticker: str) -> Dict[str, Any]:
    try:
        data: Dict[str, Any] = {}
        data.update(_av_overview(ticker) or {})
        data.update(_av_earnings(ticker) or {})
        data.update(_av_financials(ticker) or {})
        return data or {"error": f"No fundamentals available for {ticker}"}
    except Exception:
        return {
            "error": f"Failed to fetch fundamentals for {ticker}",
            "trace": traceback.format_exc(),
        }


def fetch_fundamentals(state: LangGraphMarketState) -> LangGraphMarketState:
    tickers: List[str] = state.get("tickers") or []
    results: Dict[str, Dict[str, Any]] = {}
    for tk in tickers:
        results[tk] = _fundamentals_for_ticker(tk)
    return {"fundamentals": results}
