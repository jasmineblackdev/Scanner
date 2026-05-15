"""
================================================================
  ALPACA-POWERED OPTIONS MOMENTUM × VOLATILITY SCANNER
================================================================

Scans a watchlist of liquid options tickers and ranks them by setup
quality using the same scoring framework as the React scanner:
  • Trend (price vs 20/50 EMA)
  • RSI position
  • Volume confirmation (vs 20-day avg)
  • Realized volatility (as IV proxy — see note below)
  • ATR % of price (movement potential)

The signal column tells you CALL, PUT, or WAIT.

----------------------------------------------------------------
ONE-TIME SETUP
----------------------------------------------------------------
1. Regenerate your Alpaca secret key in the dashboard (because it
   was exposed in a screenshot — never reuse a leaked secret).

2. Install the dependencies:
     pip install alpaca-py pandas python-dotenv

3. Create a file named `.env` in the SAME folder as this script.
   Paste these two lines into it (with your NEW keys):

     ALPACA_API_KEY=your_new_key_here
     ALPACA_SECRET_KEY=your_new_secret_here

   Do NOT commit .env to GitHub. Add it to .gitignore.

4. Run it:
     python options_scanner.py

----------------------------------------------------------------
NOTE ON IV DATA
----------------------------------------------------------------
Alpaca's free tier doesn't expose options Greeks / IV rank cleanly.
This script uses *realized volatility* (historical price-based vol)
as a proxy. For true IV rank you'd add a second data source like
Tradier (free, has options chains) or a paid Polygon plan. The
scoring logic accepts whatever vol metric you feed it.
================================================================
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame


# ============================================================
# CONFIG
# ============================================================
WATCHLIST = [
    "SPY", "QQQ", "NVDA", "TSLA", "AMD", "AAPL", "META", "MSFT",
    "AMZN", "GOOGL", "PLTR", "COIN", "BA", "NFLX", "SOFI", "AVGO",
    "MU", "CRWD", "MARA", "RIVN",
]

LOOKBACK_DAYS = 60   # enough history for EMA50 + indicators
ATR_PERIOD    = 14
RSI_PERIOD    = 14
EMA_FAST      = 20
EMA_SLOW      = 50
VOL_AVG_DAYS  = 20
HV_PERIOD     = 20   # realized volatility window


# ============================================================
# ANSI colors for terminal output
# ============================================================
class C:
    RESET   = "\033[0m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    GREEN   = "\033[38;5;114m"
    RED     = "\033[38;5;167m"
    AMBER   = "\033[38;5;215m"
    CREAM   = "\033[38;5;230m"
    GRAY    = "\033[38;5;245m"
    DARKGRY = "\033[38;5;238m"


# ============================================================
# Indicator math
# ============================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift()
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def realized_vol(close: pd.Series, period: int = 20) -> float:
    """Annualized realized volatility in % — used as IV proxy."""
    returns = close.pct_change().dropna().tail(period)
    if len(returns) < 2:
        return float("nan")
    return float(returns.std() * np.sqrt(252) * 100)


# ============================================================
# Scoring — mirrors the React scanner exactly
# ============================================================
def analyze(metrics: dict) -> dict:
    signal, direction, score = "WAIT", None, 0
    reasons, cautions = [], []

    rsi_v       = metrics["rsi"]
    above_20    = metrics["above_ema20"]
    above_50    = metrics["above_ema50"]
    trend       = metrics["trend"]
    vol_mult    = metrics["vol_mult"]
    iv_proxy    = metrics["hv"]       # realized vol % (IV proxy)
    atr_pct     = metrics["atr_pct"]

    # ---- Bullish setup ----
    if trend == "up" and above_20 and above_50:
        if 50 <= rsi_v <= 70:
            signal, direction = "CALL", "bullish"
            reasons.append("Above 20 & 50 EMA (uptrend intact)")
            reasons.append(f"RSI {rsi_v:.0f} in healthy trend zone")
            score += 35
        elif rsi_v > 70:
            signal, direction = "CALL", "bullish"
            reasons.append("Strong uptrend, above key EMAs")
            cautions.append(f"RSI {rsi_v:.0f} overbought — wait for pullback to 20 EMA")
            score += 15
        elif 40 <= rsi_v < 50:
            signal, direction = "CALL", "bullish"
            reasons.append("Pullback inside uptrend (buy-the-dip zone)")
            score += 25

    # ---- Bearish setup ----
    if trend == "down" and not above_20 and not above_50:
        if 30 <= rsi_v <= 50:
            signal, direction = "PUT", "bearish"
            reasons.append("Below 20 & 50 EMA (downtrend intact)")
            reasons.append(f"RSI {rsi_v:.0f} in trend zone")
            score += 35
        elif rsi_v < 30:
            signal, direction = "PUT", "bearish"
            reasons.append("Strong downtrend")
            cautions.append(f"RSI {rsi_v:.0f} oversold — bounce risk")
            score += 15

    # ---- Volume confirmation ----
    if vol_mult >= 1.3:
        reasons.append(f"Volume {vol_mult:.1f}× the 20-day average")
        score += 20
    elif vol_mult < 1.0:
        cautions.append("Below-average volume — weak conviction")
        score -= 5

    # ---- IV proxy (realized vol) ----
    # Convert to rough "rank" feel: <25% annualized = calm, 25-40 = normal,
    # 40-60 = elevated, >60 = pricey.
    if iv_proxy < 25:
        reasons.append(f"HV {iv_proxy:.0f}% — calm regime, cheap to be wrong")
        score += 25
    elif iv_proxy < 40:
        reasons.append(f"HV {iv_proxy:.0f}% — normal regime")
        score += 10
    elif iv_proxy < 60:
        cautions.append(f"HV {iv_proxy:.0f}% — elevated, consider debit spread")
    else:
        cautions.append(f"HV {iv_proxy:.0f}% — extreme. Avoid long singles")
        score -= 10

    # ---- Movement potential ----
    if atr_pct >= 2.5:
        reasons.append(f"Wide daily range (ATR {atr_pct:.1f}%) — room to move")
        score += 15
    elif atr_pct < 1.2:
        cautions.append("Tight daily range — limited movement")

    if not direction:
        signal = "WAIT"
        score = min(score, 35)

    score = max(0, min(100, score))
    return {
        "signal":    signal,
        "direction": direction,
        "reasons":   reasons,
        "cautions":  cautions,
        "score":     score,
    }


# ============================================================
# Data fetch
# ============================================================
def fetch_metrics(client: StockHistoricalDataClient, ticker: str) -> dict | None:
    end   = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS * 2)  # buffer for weekends/holidays

    try:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(req).df
    except Exception as e:
        print(f"{C.RED}  ✗ {ticker}: data fetch failed → {e}{C.RESET}")
        return None

    if bars.empty:
        return None

    # alpaca-py returns multi-index (symbol, timestamp) — flatten
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level=0)
    bars = bars.tail(LOOKBACK_DAYS).copy()

    if len(bars) < 50:
        print(f"{C.GRAY}  · {ticker}: not enough history ({len(bars)} bars){C.RESET}")
        return None

    close = bars["close"]
    high  = bars["high"]
    low   = bars["low"]
    vol   = bars["volume"]

    ema20 = ema(close, EMA_FAST).iloc[-1]
    ema50 = ema(close, EMA_SLOW).iloc[-1]
    rsi_v = rsi(close, RSI_PERIOD).iloc[-1]
    atr_v = atr(high, low, close, ATR_PERIOD).iloc[-1]
    hv    = realized_vol(close, HV_PERIOD)

    price = float(close.iloc[-1])
    prev  = float(close.iloc[-2])
    change_pct = (price - prev) / prev * 100

    vol_today = float(vol.iloc[-1])
    vol_avg   = float(vol.tail(VOL_AVG_DAYS).mean())
    vol_mult  = vol_today / vol_avg if vol_avg > 0 else 1.0

    above_20 = price > ema20
    above_50 = price > ema50

    # Trend classification: both EMAs agree → trending, else neutral
    if above_20 and above_50 and ema20 > ema50:
        trend = "up"
    elif (not above_20) and (not above_50) and ema20 < ema50:
        trend = "down"
    else:
        trend = "neutral"

    return {
        "ticker":      ticker,
        "price":       price,
        "change_pct":  change_pct,
        "rsi":         float(rsi_v),
        "ema20":       float(ema20),
        "ema50":       float(ema50),
        "above_ema20": bool(above_20),
        "above_ema50": bool(above_50),
        "trend":       trend,
        "atr_pct":     float(atr_v) / price * 100,
        "hv":          hv,
        "vol_mult":    vol_mult,
    }


# ============================================================
# Output formatting
# ============================================================
def signal_color(sig: str) -> str:
    return {"CALL": C.GREEN, "PUT": C.RED, "WAIT": C.GRAY}[sig]


def score_bar(score: int, width: int = 20) -> str:
    filled = int(round(score / 100 * width))
    return "█" * filled + C.DARKGRY + "░" * (width - filled) + C.RESET


def print_row(rank: int, m: dict, a: dict, show_detail: bool):
    col = signal_color(a["signal"])
    chg_col = C.GREEN if m["change_pct"] >= 0 else C.RED
    chg_sym = "+" if m["change_pct"] >= 0 else ""

    print(
        f"{C.DARKGRY}{rank:02d}{C.RESET}  "
        f"{C.BOLD}{m['ticker']:<6}{C.RESET}"
        f"{C.GRAY}${m['price']:>8.2f}{C.RESET}  "
        f"{chg_col}{chg_sym}{m['change_pct']:>5.2f}%{C.RESET}   "
        f"{C.GRAY}RSI{C.RESET} {m['rsi']:>5.1f}  "
        f"{C.GRAY}HV{C.RESET} {m['hv']:>4.0f}%  "
        f"{C.GRAY}ATR{C.RESET} {m['atr_pct']:>4.1f}%  "
        f"{C.GRAY}Vol{C.RESET} {m['vol_mult']:>4.1f}x   "
        f"{col}{a['signal']:>4}{C.RESET}  "
        f"{score_bar(a['score'])} {col}{a['score']:>3}{C.RESET}"
    )

    if show_detail and (a["reasons"] or a["cautions"]):
        for r in a["reasons"]:
            print(f"      {C.GREEN}▸{C.RESET} {C.CREAM}{r}{C.RESET}")
        for c in a["cautions"]:
            print(f"      {C.AMBER}▸{C.RESET} {C.CREAM}{c}{C.RESET}")
        print()


# ============================================================
# Main
# ============================================================
def main():
    load_dotenv()
    api_key    = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not api_secret:
        print(f"{C.RED}Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env file.{C.RESET}")
        print("Create a .env file in this folder with:")
        print("  ALPACA_API_KEY=your_key")
        print("  ALPACA_SECRET_KEY=your_secret")
        sys.exit(1)

    print(f"\n{C.BOLD}{C.CREAM}  MOMENTUM × VOLATILITY SCANNER{C.RESET}")
    print(f"{C.DIM}  Alpaca daily bars · {len(WATCHLIST)} tickers · {datetime.now():%Y-%m-%d %H:%M}{C.RESET}\n")

    client = StockHistoricalDataClient(api_key, api_secret)

    print(f"{C.DIM}  Fetching data...{C.RESET}")
    results = []
    for t in WATCHLIST:
        m = fetch_metrics(client, t)
        if m is None:
            continue
        a = analyze(m)
        results.append((m, a))

    if not results:
        print(f"{C.RED}\nNo data returned. Check your keys and internet connection.{C.RESET}")
        return

    # Sort by score, then by absolute change
    results.sort(key=lambda x: (x[1]["score"], abs(x[0]["change_pct"])), reverse=True)

    # Header row
    print()
    print(
        f"{C.DIM}{'#':<3} {'TICKER':<6}{'PRICE':>9} {'CHG':>8}   "
        f"{'RSI':<8} {'HV':<7}  {'ATR':<8}  {'VOL':<8} "
        f"{'SIG':>4}  {'SCORE':<24}{C.RESET}"
    )
    print(f"{C.DARKGRY}{'─' * 110}{C.RESET}\n")

    for i, (m, a) in enumerate(results, 1):
        # Show full reasons/cautions only for top scoring CALL/PUT signals
        show_detail = a["signal"] in ("CALL", "PUT") and a["score"] >= 50
        print_row(i, m, a, show_detail)

    print(f"\n{C.DARKGRY}{'─' * 110}{C.RESET}")
    print(f"{C.DIM}  Reminder: this is a SCREEN, not an entry signal.{C.RESET}")
    print(f"{C.DIM}  Still wait for trigger candle close + volume at a key level before entering.{C.RESET}\n")


if __name__ == "__main__":
    main()
