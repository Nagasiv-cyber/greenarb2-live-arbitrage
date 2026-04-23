"""
live_feed.py — GreenArb 2.0 | Nifty 50 Scale Engine
Fetches baseline prices ONCE, then simulates live ticks via Brownian motion.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple

# ─── Constants ────────────────────────────────────────────────────────────────
FXRATE_TICKER         = "INR=X"
FEE_RATE              = 0.0005
ENERGY_COST_INR       = 0.50
TRADE_QTY             = 1_000
EAA_THRESHOLD         = 50.0
CO2_PER_UNDERCLOCK_KG = 0.15
BROWNIAN_VOL          = 0.002   # ±0.2% per tick

# ─── Nifty 50 Universe ────────────────────────────────────────────────────────
# nyse: real NYSE/OTC ticker or None (simulated ADR)
# ratio: NSE shares per 1 ADR unit
# premium: baseline ADR premium fraction (for simulated ADRs)
NIFTY_50_UNIVERSE = {
    "ADANIENT":   {"nse":"ADANIENT.NS",   "nyse":None,    "ratio":1,"name":"Adani Enterprises",   "sector":"Conglomerate",   "premium": 0.003},
    "ADANIPORTS": {"nse":"ADANIPORTS.NS", "nyse":None,    "ratio":1,"name":"Adani Ports",          "sector":"Infrastructure", "premium":-0.002},
    "APOLLOHOSP": {"nse":"APOLLOHOSP.NS", "nyse":None,    "ratio":1,"name":"Apollo Hospitals",     "sector":"Healthcare",     "premium": 0.005},
    "ASIANPAINT": {"nse":"ASIANPAINT.NS", "nyse":None,    "ratio":1,"name":"Asian Paints",         "sector":"Consumer",       "premium":-0.001},
    "AXISBANK":   {"nse":"AXISBANK.NS",   "nyse":None,    "ratio":1,"name":"Axis Bank",            "sector":"Banking",        "premium": 0.004},
    "BAJAJ-AUTO": {"nse":"BAJAJ-AUTO.NS", "nyse":None,    "ratio":1,"name":"Bajaj Auto",           "sector":"Auto",           "premium":-0.003},
    "BAJAJFINSV": {"nse":"BAJAJFINSV.NS", "nyse":None,    "ratio":1,"name":"Bajaj Finserv",        "sector":"Finance",        "premium": 0.002},
    "BAJFINANCE": {"nse":"BAJFINANCE.NS", "nyse":None,    "ratio":1,"name":"Bajaj Finance",        "sector":"Finance",        "premium": 0.001},
    "BHARTIARTL": {"nse":"BHARTIARTL.NS", "nyse":None,    "ratio":1,"name":"Bharti Airtel",        "sector":"Telecom",        "premium": 0.003},
    "BPCL":       {"nse":"BPCL.NS",       "nyse":None,    "ratio":1,"name":"BPCL",                 "sector":"Energy",         "premium":-0.002},
    "BRITANNIA":  {"nse":"BRITANNIA.NS",  "nyse":None,    "ratio":1,"name":"Britannia",            "sector":"FMCG",           "premium": 0.001},
    "CIPLA":      {"nse":"CIPLA.NS",      "nyse":None,    "ratio":1,"name":"Cipla",                "sector":"Pharma",         "premium": 0.004},
    "COALINDIA":  {"nse":"COALINDIA.NS",  "nyse":None,    "ratio":1,"name":"Coal India",           "sector":"Mining",         "premium":-0.001},
    "DIVISLAB":   {"nse":"DIVISLAB.NS",   "nyse":None,    "ratio":1,"name":"Divi's Laboratories",  "sector":"Pharma",         "premium": 0.003},
    "DRREDDY":    {"nse":"DRREDDY.NS",    "nyse":"RDY",   "ratio":1,"name":"Dr. Reddy's",          "sector":"Pharma",         "premium": 0.002},
    "EICHERMOT":  {"nse":"EICHERMOT.NS",  "nyse":None,    "ratio":1,"name":"Eicher Motors",        "sector":"Auto",           "premium":-0.002},
    "GRASIM":     {"nse":"GRASIM.NS",     "nyse":None,    "ratio":1,"name":"Grasim Industries",    "sector":"Materials",      "premium": 0.001},
    "HCLTECH":    {"nse":"HCLTECH.NS",    "nyse":None,    "ratio":1,"name":"HCL Technologies",     "sector":"IT",             "premium": 0.002},
    "HDFCBANK":   {"nse":"HDFCBANK.NS",   "nyse":"HDB",   "ratio":3,"name":"HDFC Bank",            "sector":"Banking",        "premium": 0.003},
    "HDFCLIFE":   {"nse":"HDFCLIFE.NS",   "nyse":None,    "ratio":1,"name":"HDFC Life Insurance",  "sector":"Insurance",      "premium":-0.001},
    "HEROMOTOCO": {"nse":"HEROMOTOCO.NS", "nyse":None,    "ratio":1,"name":"Hero MotoCorp",        "sector":"Auto",           "premium": 0.001},
    "HINDALCO":   {"nse":"HINDALCO.NS",   "nyse":None,    "ratio":1,"name":"Hindalco Industries",  "sector":"Metals",         "premium":-0.002},
    "HINDUNILVR": {"nse":"HINDUNILVR.NS", "nyse":None,    "ratio":1,"name":"Hindustan Unilever",   "sector":"FMCG",           "premium": 0.002},
    "ICICIBANK":  {"nse":"ICICIBANK.NS",  "nyse":"IBN",   "ratio":2,"name":"ICICI Bank",           "sector":"Banking",        "premium": 0.004},
    "INDUSINDBK": {"nse":"INDUSINDBK.NS", "nyse":None,    "ratio":1,"name":"IndusInd Bank",        "sector":"Banking",        "premium":-0.003},
    "INFY":       {"nse":"INFY.NS",       "nyse":"INFY",  "ratio":1,"name":"Infosys",              "sector":"IT",             "premium": 0.002},
    "ITC":        {"nse":"ITC.NS",        "nyse":None,    "ratio":1,"name":"ITC Limited",          "sector":"FMCG",           "premium": 0.001},
    "JSWSTEEL":   {"nse":"JSWSTEEL.NS",   "nyse":None,    "ratio":1,"name":"JSW Steel",            "sector":"Metals",         "premium":-0.001},
    "KOTAKBANK":  {"nse":"KOTAKBANK.NS",  "nyse":None,    "ratio":1,"name":"Kotak Mahindra Bank",  "sector":"Banking",        "premium": 0.002},
    "LT":         {"nse":"LT.NS",         "nyse":None,    "ratio":1,"name":"Larsen & Toubro",      "sector":"Infrastructure", "premium": 0.003},
    "MM":         {"nse":"M&M.NS",        "nyse":None,    "ratio":1,"name":"Mahindra & Mahindra",  "sector":"Auto",           "premium":-0.002},
    "MARUTI":     {"nse":"MARUTI.NS",     "nyse":None,    "ratio":1,"name":"Maruti Suzuki",        "sector":"Auto",           "premium": 0.001},
    "NESTLEIND":  {"nse":"NESTLEIND.NS",  "nyse":None,    "ratio":1,"name":"Nestle India",         "sector":"FMCG",           "premium": 0.003},
    "NTPC":       {"nse":"NTPC.NS",       "nyse":None,    "ratio":1,"name":"NTPC Limited",         "sector":"Energy",         "premium":-0.001},
    "ONGC":       {"nse":"ONGC.NS",       "nyse":None,    "ratio":1,"name":"ONGC",                 "sector":"Energy",         "premium": 0.002},
    "POWERGRID":  {"nse":"POWERGRID.NS",  "nyse":None,    "ratio":1,"name":"Power Grid Corp",      "sector":"Utilities",      "premium":-0.002},
    "RELIANCE":   {"nse":"RELIANCE.NS",   "nyse":"RLNIY", "ratio":1,"name":"Reliance Industries",  "sector":"Conglomerate",   "premium": 0.001},
    "SBILIFE":    {"nse":"SBILIFE.NS",    "nyse":None,    "ratio":1,"name":"SBI Life Insurance",   "sector":"Insurance",      "premium": 0.003},
    "SBIN":       {"nse":"SBIN.NS",       "nyse":None,    "ratio":1,"name":"State Bank of India",  "sector":"Banking",        "premium":-0.001},
    "SHRIRAMFIN": {"nse":"SHRIRAMFIN.NS", "nyse":None,    "ratio":1,"name":"Shriram Finance",      "sector":"Finance",        "premium": 0.003},
    "SUNPHARMA":  {"nse":"SUNPHARMA.NS",  "nyse":None,    "ratio":1,"name":"Sun Pharmaceutical",   "sector":"Pharma",         "premium": 0.004},
    "TATACONSUM": {"nse":"TATACONSUM.NS", "nyse":None,    "ratio":1,"name":"Tata Consumer",        "sector":"FMCG",           "premium": 0.001},
    "TATAMOTORS": {"nse":"TATAMOTORS.NS", "nyse":"TTM",   "ratio":1,"name":"Tata Motors",          "sector":"Auto",           "premium": 0.002},
    "TATASTEEL":  {"nse":"TATASTEEL.NS",  "nyse":None,    "ratio":1,"name":"Tata Steel",           "sector":"Metals",         "premium":-0.003},
    "TCS":        {"nse":"TCS.NS",        "nyse":None,    "ratio":1,"name":"Tata Consultancy",     "sector":"IT",             "premium": 0.003},
    "TECHM":      {"nse":"TECHM.NS",      "nyse":None,    "ratio":1,"name":"Tech Mahindra",        "sector":"IT",             "premium": 0.001},
    "TITAN":      {"nse":"TITAN.NS",      "nyse":None,    "ratio":1,"name":"Titan Company",        "sector":"Consumer",       "premium": 0.002},
    "TRENT":      {"nse":"TRENT.NS",      "nyse":None,    "ratio":1,"name":"Trent Limited",        "sector":"Retail",         "premium": 0.004},
    "ULTRACEMCO": {"nse":"ULTRACEMCO.NS", "nyse":None,    "ratio":1,"name":"UltraTech Cement",     "sector":"Materials",      "premium":-0.001},
    "UPL":        {"nse":"UPL.NS",        "nyse":None,    "ratio":1,"name":"UPL Limited",          "sector":"Chemicals",      "premium":-0.002},
    "WIPRO":      {"nse":"WIPRO.NS",      "nyse":"WIT",   "ratio":1,"name":"Wipro Limited",        "sector":"IT",             "premium": 0.001},
}

# ─── Agent ID assignments ─────────────────────────────────────────────────────
AGENT_IDS = {t: f"AGT-{i+1:02d}" for i, t in enumerate(NIFTY_50_UNIVERSE.keys())}


def _get_fx_rate() -> float:
    try:
        t = yf.Ticker(FXRATE_TICKER)
        r = getattr(t.fast_info, "last_price", None) or getattr(t.fast_info, "previous_close", None)
        return float(r) if r else 83.50
    except Exception:
        return 83.50


def _safe_price(symbol: str) -> float | None:
    try:
        tkr   = yf.Ticker(symbol)
        info  = tkr.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
        if not price:
            hist  = tkr.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price else None
    except Exception:
        return None


# ─── Baseline Fetch (called ONCE on startup) ──────────────────────────────────

def fetch_baseline_prices() -> Tuple[pd.DataFrame, float]:
    """
    Batch-fetches the last closing price for all 50 NSE tickers + real ADR tickers.
    Returns (baseline_df, usd_inr_rate).
    Called ONCE; result stored in st.session_state.
    """
    usd_inr = _get_fx_rate()

    # Batch-download NSE prices
    nse_symbols = [cfg["nse"] for cfg in NIFTY_50_UNIVERSE.values()]
    # Also collect real ADR tickers
    adr_symbols = [cfg["nyse"] for cfg in NIFTY_50_UNIVERSE.values() if cfg["nyse"]]

    all_symbols = nse_symbols + adr_symbols + [FXRATE_TICKER]

    try:
        raw = yf.download(all_symbols, period="2d", auto_adjust=True, progress=False)
        closes = raw["Close"].iloc[-1] if "Close" in raw.columns else pd.Series(dtype=float)
    except Exception:
        closes = pd.Series(dtype=float)

    rows = []
    for ticker, cfg in NIFTY_50_UNIVERSE.items():
        # NSE price
        nse_sym    = cfg["nse"]
        nse_price  = float(closes.get(nse_sym, 0) or 0)
        if nse_price <= 0:
            nse_price = _safe_price(nse_sym) or 1000.0   # individual fallback

        # ADR price (INR equiv)
        if cfg["nyse"]:
            adr_usd = float(closes.get(cfg["nyse"], 0) or 0)
            if adr_usd <= 0:
                adr_usd = _safe_price(cfg["nyse"]) or (nse_price / usd_inr)
            adr_inr = (adr_usd * usd_inr) / cfg["ratio"]
        else:
            # Simulated ADR: use NSE price + fixed premium
            adr_inr = nse_price * (1.0 + cfg["premium"])

        rows.append({
            "Ticker":       ticker,
            "Name":         cfg["name"],
            "Sector":       cfg["sector"],
            "NSE_base":     round(nse_price, 2),
            "ADR_base_INR": round(adr_inr,   2),
            "ratio":        cfg["ratio"],
            "has_real_adr": cfg["nyse"] is not None,
        })

    return pd.DataFrame(rows), usd_inr


# ─── Brownian Motion Tick ─────────────────────────────────────────────────────

def simulate_brownian_tick(baseline_df: pd.DataFrame, vol: float = BROWNIAN_VOL) -> pd.DataFrame:
    """
    Apply independent Gaussian noise to NSE and ADR base prices.
    Returns a new DataFrame with current-tick prices.
    """
    df = baseline_df.copy()
    n  = len(df)
    df["NSE ₹"]        = df["NSE_base"]     * (1 + np.random.normal(0, vol, n))
    df["ADR ₹ (conv)"] = df["ADR_base_INR"] * (1 + np.random.normal(0, vol, n))
    df["NSE ₹"]        = df["NSE ₹"].round(2)
    df["ADR ₹ (conv)"] = df["ADR ₹ (conv)"].round(2)
    return df


# ─── EAA Computation ─────────────────────────────────────────────────────────

def compute_radar(tick_df: pd.DataFrame) -> pd.DataFrame:
    """Compute spread + EAA for every stock in the tick DataFrame."""
    df = tick_df.copy()
    df["Spread ₹"]    = (df["ADR ₹ (conv)"] - df["NSE ₹"]).round(2)
    df["Gross Profit"] = (df["Spread ₹"] * TRADE_QTY).round(2)
    df["Total Fees"]   = (df["NSE ₹"] * TRADE_QTY * FEE_RATE).round(2)
    df["Net EAA ₹"]    = (df["Gross Profit"] - df["Total Fees"] - ENERGY_COST_INR).round(2)
    df["Decision"]     = df["Net EAA ₹"].apply(
        lambda x: "EXECUTE ✅" if x > EAA_THRESHOLD else "UNDERCLOCKING ⚡"
    )
    df["Power Mode"]   = df["Net EAA ₹"].apply(
        lambda x: "1000W (Peak)" if x > EAA_THRESHOLD else "150W (Eco)"
    )
    df["CCU (kg)"]     = df["Net EAA ₹"].apply(
        lambda x: 0.0 if x > EAA_THRESHOLD else CO2_PER_UNDERCLOCK_KG
    )
    df["Status"]       = df["has_real_adr"].apply(lambda x: "LIVE" if x else "SIMULATED")
    df["Agent_ID"]     = df["Ticker"].map(AGENT_IDS)
    return df.sort_values("Net EAA ₹", ascending=False).reset_index(drop=True)


# ─── Legacy single-stock (INFY metric cards) ──────────────────────────────────

def get_adr_arbitrage_data(tick_df: pd.DataFrame | None = None) -> Tuple[float, float, float, float, str]:
    """Returns INFY row from a pre-computed tick_df, or fetches live if None."""
    if tick_df is not None:
        row = tick_df[tick_df["Ticker"] == "INFY"]
        if not row.empty:
            r = row.iloc[0]
            fx = round(r["ADR ₹ (conv)"] / (r["ADR ₹ (conv)"] / 83.5), 4)  # approx
            return r["NSE ₹"], r["ADR ₹ (conv)"], 83.5, r["Spread ₹"], "SIMULATED"

    # Fallback: live fetch
    nse = _safe_price("INFY.NS") or 1425.50
    adr = _safe_price("INFY")    or 17.35
    fx  = _get_fx_rate()
    return round(nse, 2), round(adr * fx, 2), round(fx, 4), round(adr * fx - nse, 2), "LIVE"
