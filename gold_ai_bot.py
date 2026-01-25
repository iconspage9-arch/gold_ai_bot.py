import pandas as pd
import numpy as np
import yfinance as yf
import json, os
import threading
import time
import requests
from datetime import datetime
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from flask import Flask

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_TOKEN")
TIMEFRAMES = {"M30": "30m", "H1": "60m", "H4": "4h"}
TRADE_HISTORY_FILE = "trade_history.json"
JOURNAL_FILE = "trade_journal.json"

# Available trading pairs organized by type
PAIRS_BY_TYPE = {
    "COMMODITIES": {
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "OIL": "CL=F",
        "COPPER": "HG=F",
        "NATGAS": "NG=F"
    },
    "FOREX": {
        # Each base maps to a list of possible quote currencies (we will pick the first quote by default)
        "EUR": ["USD", "GBP", "CHF", "JPY"],
        "GBP": ["USD", "JPY"],
        "JPY": ["USD", "EUR", "GBP", "AUD"],
        "AUD": ["USD", "JPY"],
        "NZD": ["USD"],
        "CAD": ["USD", "CHF"],
        "CHF": ["USD", "EUR", "CAD"]
    },
    "CRYPTO": {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "LTC": "LTC-USD",
        "XRP": "XRP-USD",
        "ADA": "ADA-USD",
        "DOGE": "DOGE-USD",
        "BNB": "BNB-USD",
        "SOL": "SOL-USD",
        "POLKA": "DOT-USD",
        "LINK": "LINK-USD"
    },
    "INDICES": {
        "SP500": "^GSPC",
        "DOWJONES": "^DJI",
        "NASDAQ": "^IXIC",
        "DAX": "^GDAXI",
        "FTSE": "^FTSE",
        "NIKKEI": "^N225",
        "HSI": "^HSI"
    }
}

# Helper function to detect swing highs and lows
def find_swing_levels(df, lookback=20):
    """Find actual swing highs and lows (local extremes)"""
    if len(df) < 5:
        return [], []
    
    highs = []
    lows = []
    recent = df.iloc[-lookback:] if len(df) >= lookback else df
    
    for i in range(1, len(recent) - 1):
        if recent['High'].iloc[i] > recent['High'].iloc[i-1] and recent['High'].iloc[i] > recent['High'].iloc[i+1]:
            highs.append(recent['High'].iloc[i])
        if recent['Low'].iloc[i] < recent['Low'].iloc[i-1] and recent['Low'].iloc[i] < recent['Low'].iloc[i+1]:
            lows.append(recent['Low'].iloc[i])
    
    return highs, lows

# Helper function to convert pips to decimal
def pips_to_decimal(pair_name, pips):
    """Convert pips to decimal price movement"""
    if "JPY" in pair_name:
        return pips * 0.01  # JPY pairs = 2 decimals
    else:
        return pips * 0.0001  # Standard pairs = 4 decimals

# Helper function to build forex pair symbol
def build_forex_symbol(base, quote):
    """Convert base/quote to yfinance symbol (e.g., EUR + USD -> EURUSD=X)"""
    return f"{base}{quote}=X"

# ================= DATA FUNCTIONS =================
def fetch(tf, symbol="GC=F"):
    try:
        df = yf.download(symbol, period="30d", interval=tf, progress=False)
        if df is None or len(df) == 0:
            print(f"fetch: no rows for {symbol} interval={tf}")
            return None
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Check if we have required columns
        if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
            print(f"fetch: missing OHLC columns for {symbol} interval={tf} cols={list(df.columns)}")
            return None
        
        # Remove NaN values first
        df = df.dropna()
        if len(df) < 50:  # lowered minimum bars to 50 (EMA200 optional)
            print(f"fetch: insufficient bars for {symbol} interval={tf} rows={len(df)} (need>=50)")
            return None
        
        # Convert to numeric, handle any string values
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        
        # Calculate indicators only if we have valid data
        try:
            df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
            # Compute ema200 only if we have enough bars; otherwise leave as NaN
            if len(df) >= 200:
                df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
            else:
                df["ema200"] = np.nan

            df["rsi"] = RSIIndicator(df["Close"], window=14).rsi()
            
            macd = MACD(df["Close"], window_fast=12, window_slow=26, window_sign=9)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            
            df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
            
            # Bollinger Bands
            bb = BollingerBands(df["Close"], window=20, window_dev=2)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_mid"] = bb.bollinger_mavg()
            
            # Stochastic
            stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
            
            # Remove NaN from indicators where possible (but keep rows if some indicators are NaN)
            df = df.dropna(subset=['Close', 'High', 'Low'])
            if len(df) == 0:
                return None
            
            return df
        except Exception as e:
            print(f"Indicator calculation error: {e}")
            return None
    except Exception as e:
        print(f"Fetch error for {symbol} interval={tf}: {e}")
        return None

# ================= FEATURE: DIVERGENCE DETECTION =================
def detect_divergence(df):
    """Detect RSI and MACD divergence (bullish/bearish)"""
    if len(df) < 20:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # RSI Divergence: Price lower but RSI higher (bullish) or price higher but RSI lower (bearish)
    rsi_div = None
    if len(df) >= 50:
        low_20 = df['Low'].iloc[-20:].min()
        low_50 = df['Low'].iloc[-50:].min()
        rsi_low_20 = df['rsi'].iloc[-20:].min()
        rsi_low_50 = df['rsi'].iloc[-50:].min()
        
        if low_20 < low_50 and rsi_low_20 > rsi_low_50:
            rsi_div = "BULLISH"  # Lower low but higher RSI = potential reversal up
        elif len(df) >= 20:
            high_20 = df['High'].iloc[-20:].max()
            high_50 = df['High'].iloc[-50:].max()
            rsi_high_20 = df['rsi'].iloc[-20:].max()
            rsi_high_50 = df['rsi'].iloc[-50:].max()
            
            if high_20 > high_50 and rsi_high_20 < rsi_high_50:
                rsi_div = "BEARISH"  # Higher high but lower RSI = potential reversal down
    
    # MACD Divergence
    macd_div = None
    if len(df) >= 30:
        macd_momentum = df['macd'].iloc[-1] - df['macd'].iloc[-30]
        if df['Close'].iloc[-1] < df['Close'].iloc[-30] and macd_momentum > 0:
            macd_div = "BULLISH"
        elif df['Close'].iloc[-1] > df['Close'].iloc[-30] and macd_momentum < 0:
            macd_div = "BEARISH"
    
    return {"rsi": rsi_div, "macd": macd_div}

# ================= FEATURE: FIBONACCI LEVELS =================
def calculate_fibonacci(high, low, direction="UP"):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    if direction == "UP":
        levels = {
            "0%": low,
            "23.6%": low + (diff * 0.236),
            "38.2%": low + (diff * 0.382),
            "50%": low + (diff * 0.5),
            "61.8%": low + (diff * 0.618),
            "78.6%": low + (diff * 0.786),
            "100%": high
        }
    else:
        levels = {
            "0%": high,
            "23.6%": high - (diff * 0.236),
            "38.2%": high - (diff * 0.382),
            "50%": high - (diff * 0.5),
            "61.8%": high - (diff * 0.618),
            "78.6%": high - (diff * 0.786),
            "100%": low
        }
    return levels

# ================= FEATURE: PATTERN RECOGNITION =================
def detect_patterns(df):
    """Detect chart patterns: double top/bottom, triangle, flag, head & shoulders, wedge, cup & handle"""
    if len(df) < 50:
        return []
    
    patterns = []
    recent = df.iloc[-50:]
    
    # Double Top: 2 peaks at similar levels
    peaks = []
    for i in range(2, len(recent) - 2):
        if recent['High'].iloc[i] > recent['High'].iloc[i-2] and recent['High'].iloc[i] > recent['High'].iloc[i+2]:
            peaks.append((i, recent['High'].iloc[i]))
    
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if abs(p1[1] - p2[1]) / p1[1] < 0.02:  # Within 2%
            patterns.append("DOUBLE_TOP")
    
    # Double Bottom
    troughs = []
    for i in range(2, len(recent) - 2):
        if recent['Low'].iloc[i] < recent['Low'].iloc[i-2] and recent['Low'].iloc[i] < recent['Low'].iloc[i+2]:
            troughs.append((i, recent['Low'].iloc[i]))
    
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if abs(t1[1] - t2[1]) / t1[1] < 0.02:  # Within 2%
            patterns.append("DOUBLE_BOTTOM")
    
    # Triangle: contracting high/low range
    if len(recent) >= 20:
        high_range = recent['High'].iloc[-20:].max() - recent['High'].iloc[-20:].min()
        low_range = recent['Low'].iloc[-20:].max() - recent['Low'].iloc[-20:].min()
        curr_high = recent['High'].iloc[-10:].max() - recent['High'].iloc[-10:].min()
        curr_low = recent['Low'].iloc[-10:].max() - recent['Low'].iloc[-10:].min()
        
        if curr_high < high_range * 0.5 and curr_low < low_range * 0.5:
            patterns.append("TRIANGLE")
    
    # Head & Shoulders: peak-valley-peak with middle peak highest
    if len(peaks) >= 3:
        left_shoulder = peaks[-3]
        head = peaks[-2]
        right_shoulder = peaks[-1]
        
        if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
            if abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.03:
                patterns.append("HEAD_SHOULDERS")
    
    # Wedge: converging trend lines (ascending or descending)
    if len(recent) >= 20:
        highs_range = recent['High'].iloc[-20:].max() - recent['High'].iloc[-10:].max()
        lows_range = recent['Low'].iloc[-20:].min() - recent['Low'].iloc[-10:].min()
        
        # Ascending wedge: higher lows, converging highs
        if lows_range > 0 and highs_range < lows_range * 0.3:
            patterns.append("WEDGE_ASCENDING")
        # Descending wedge: lower highs, converging lows
        elif highs_range < 0 and abs(lows_range) < abs(highs_range) * 0.3:
            patterns.append("WEDGE_DESCENDING")
    
    # Cup & Handle: U-shaped recovery with small pullback
    if len(recent) >= 30:
        low_idx = recent['Low'].iloc[-30:].argmin()  # Get position, not label
        left_high = recent['High'].iloc[-30:low_idx].max() if low_idx > 0 else 0
        right_high = recent['High'].iloc[low_idx:].max()
        
        if left_high and abs(left_high - right_high) / left_high < 0.02 and left_high > recent['Low'].iloc[-1]:
            # Check for handle (small pullback)
            handle_low = recent['Low'].iloc[-5:].min()
            if handle_low > recent['Low'].iloc[low_idx]:
                patterns.append("CUP_HANDLE")
    
    # Flag: small consolidation after strong directional move
    if len(recent) >= 25:
        early = recent.iloc[-25:-15]
        recent_part = recent.iloc[-15:]
        
        early_range = (early['High'].max() - early['Low'].min()) / early['Close'].mean()
        recent_range = (recent_part['High'].max() - recent_part['Low'].min()) / recent_part['Close'].mean()
        
        # Consolidation is much tighter than prior move
        if recent_range < early_range * 0.4:
            patterns.append("FLAG")
    
    return patterns

# ================= FEATURE: TRADE HISTORY & WIN RATE =================
def load_trades():
    """Load trade history from file"""
    if os.path.exists(TRADE_HISTORY_FILE):
        with open(TRADE_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_trade(pair, signal, tp, sl, entry):
    """Log a new trade signal"""
    trades = load_trades()
    trade = {
        "timestamp": datetime.now().isoformat(),
        "pair": pair,
        "signal": signal,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "status": "OPEN"
    }
    trades.append(trade)
    with open(TRADE_HISTORY_FILE, 'w') as f:
        json.dump(trades, f, indent=2)

def get_win_rate():
    """Calculate bot's win rate from closed trades"""
    trades = load_trades()
    closed = [t for t in trades if t["status"] != "OPEN"]
    if not closed:
        return None
    
    wins = sum(1 for t in closed if t["status"] == "WIN")
    return {
        "total_trades": len(closed),
        "wins": wins,
        "losses": len(closed) - wins,
        "win_rate": (wins / len(closed) * 100) if closed else 0
    }

# ================= FEATURE: BACKTESTING =================
def backtest_strategy(symbol, timeframe="H1", days=30):
    """Backtest strategy on historical data"""
    try:
        df = yf.download(symbol, period=f"{days}d", interval=timeframe, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        
        if len(df) < 20:
            return None
        
        wins = 0
        losses = 0
        total_profit = 0
        
        # Simulate signals on historical data
        for i in range(20, len(df)):
            window = df.iloc[max(0, i-50):i+1]
            last = window.iloc[-1]
            
            # Simple signal check - use shorter periods for small datasets
            rsi = RSIIndicator(window["Close"], 14).rsi().iloc[-1]
            ema9 = EMAIndicator(window["Close"], 9).ema_indicator().iloc[-1]
            ema21 = EMAIndicator(window["Close"], 21).ema_indicator().iloc[-1]
            
            # Relaxed signal: just check EMA9 > EMA21 + RSI not extreme
            signal = None
            if ema9 > ema21 and rsi < 80:
                signal = "BUY"
            elif ema9 < ema21 and rsi > 20:
                signal = "SELL"
            
            if signal and i + 1 < len(df):
                entry = last['Close']
                # Check next 3 candles for TP/SL hit (more likely with small lookback)
                for j in range(1, min(4, len(df) - i)):
                    future = df.iloc[i+j]
                    
                    if signal == "BUY":
                        if future['High'] >= entry * 1.015:  # TP hit (1.5%)
                            wins += 1
                            total_profit += entry * 0.015
                            break
                        elif future['Low'] <= entry * 0.985:  # SL hit (1.5%)
                            losses += 1
                            total_profit -= entry * 0.015
                            break
                    else:  # SELL
                        if future['Low'] <= entry * 0.985:  # TP hit (1.5%)
                            wins += 1
                            total_profit += entry * 0.015
                            break
                        elif future['High'] >= entry * 1.015:  # SL hit (1.5%)
                            losses += 1
                            total_profit -= entry * 0.015
                            break
        
        total_trades = wins + losses
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
            "total_profit": total_profit
        }
    except Exception as e:
        print(f"Backtest error: {e}")
        return None

# ================= FEATURE: TRADE JOURNAL =================
def log_journal(pair, signal, tp, sl, entry, note):
    """Log trade to personal journal with notes"""
    journal = []
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, 'r') as f:
            journal = json.load(f)
    
    entry_log = {
        "timestamp": datetime.now().isoformat(),
        "pair": pair,
        "signal": signal,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "user_note": note
    }
    journal.append(entry_log)
    with open(JOURNAL_FILE, 'w') as f:
        json.dump(journal, f, indent=2)

def get_journal_entries(limit=5):
    """Get recent journal entries"""
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, 'r') as f:
            journal = json.load(f)
        return journal[-limit:] if journal else []
    return []

def analyze(df, symbol=None):
    if df is None or len(df) == 0:
        return None
    
    last = df.iloc[-1]
    signals = []
    
    # Signal 1: EMA Trend (50/200 crossover)
    # Handle missing ema200: use fallback
    ema200_val = getattr(last, 'ema200', np.nan)
    if pd.notna(ema200_val):
        ema_signal = "BUY" if last.ema50 > last.ema200 else "SELL"
    else:
        # Fallback: compare ema50 to price if ema200 not available
        ema_signal = "BUY" if last.ema50 > last.Close else "SELL"
    signals.append(ema_signal)
    
    # Signal 2: RSI (oversold/overbought)
    rsi_signal = "BUY" if last.rsi < 30 else ("SELL" if last.rsi > 70 else "NEUTRAL")
    if rsi_signal != "NEUTRAL":
        signals.append(rsi_signal)
    
    # Signal 3: MACD Momentum
    macd_signal = "BUY" if last.macd > last.macd_signal else "SELL"
    signals.append(macd_signal)
    
    # Signal 4: Bollinger Bands
    bb_signal = "NEUTRAL"
    if pd.notna(last.bb_lower) and pd.notna(last.bb_upper):
        if last.Close <= last.bb_lower:
            bb_signal = "BUY"  # Price at/below lower band = oversold
        elif last.Close >= last.bb_upper:
            bb_signal = "SELL"  # Price at/above upper band = overbought
    if bb_signal != "NEUTRAL":
        signals.append(bb_signal)
    
    # Signal 5: Stochastic
    stoch_signal = "NEUTRAL"
    if pd.notna(last.stoch_k):
        if last.stoch_k < 20:
            stoch_signal = "BUY"  # Oversold
        elif last.stoch_k > 80:
            stoch_signal = "SELL"  # Overbought
    if stoch_signal != "NEUTRAL":
        signals.append(stoch_signal)
    
    # Count agreement
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    total = len(signals)
    
    # Determine final signal
    if buy_count > sell_count:
        final_signal = "BUY"
        agreement = (buy_count / total) * 100 if total > 0 else 0
    else:
        final_signal = "SELL"
        agreement = (sell_count / total) * 100 if total > 0 else 0
    
    # Detect divergence (bonus confirmation)
    divergence = detect_divergence(df)
    
    # Detect patterns
    patterns = detect_patterns(df)
    
    # Calculate Fibonacci levels
    high_50 = df['High'].iloc[-50:].max()
    low_50 = df['Low'].iloc[-50:].min()
    fib_levels = calculate_fibonacci(high_50, low_50, "UP" if final_signal == "BUY" else "DOWN")
    
    # Calculate TP and SL
    atr = last.atr if pd.notna(last.atr) else 0
    entry = last.Close

    # Forex-specific: use multi-factor approach
    is_forex = False
    if isinstance(symbol, str) and symbol.endswith("=X"):
        is_forex = True

    if is_forex:
        entry = last.Close
        
        # Factor 1: Swing highs/lows (actual turning points)
        swing_highs, swing_lows = find_swing_levels(df, lookback=20)
        
        # Factor 2: Recent highs/lows (tighter window)
        range_window = min(len(df), 15)
        recent_high = df['High'].iloc[-range_window:].max()
        recent_low = df['Low'].iloc[-range_window:].min()
        
        # Factor 3: Risk/Reward ratio (1:2)
        # SL will be X pips, TP will be 2X pips
        
        # Factor 4: Fixed pip distances (fallback)
        # 50 pips SL, 100 pips TP for standard pairs
        # 5 pips SL, 10 pips TP for JPY pairs
        pair_name_str = str(symbol)
        is_jpy = "JPY" in pair_name_str
        if is_jpy:
            pip_sl = 5
            pip_tp = 10
        else:
            pip_sl = 50
            pip_tp = 100
        
        # Factor 5: Multi-timeframe confirmation (use recent levels as anchors)
        consolidation = False
        if atr > 0 and (recent_high - recent_low) < (atr * 1.5):
            consolidation = True

        if final_signal == "BUY":
            # TP: prefer swing high, else recent high, else entry + 2x pip distance
            if swing_highs:
                candidates = [h for h in swing_highs if h > entry]
                if candidates:
                    tp_swing = min(candidates)  # Nearest swing high
                    tp = tp_swing
                else:
                    tp = entry + pips_to_decimal(pair_name_str, pip_tp)
            elif consolidation:
                tp = recent_high
            else:
                tp = entry + pips_to_decimal(pair_name_str, pip_tp)

            # SL: prefer swing low, else recent low, else entry - 1x pip distance
            if swing_lows:
                candidates = [l for l in swing_lows if l < entry]
                if candidates:
                    sl_swing = max(candidates)  # Nearest swing low
                    sl = max(sl_swing, entry - pips_to_decimal(pair_name_str, pip_sl))
                else:
                    sl = entry - pips_to_decimal(pair_name_str, pip_sl)
            else:
                sl = max(recent_low, entry - pips_to_decimal(pair_name_str, pip_sl))
            
            # Enforce risk/reward: TP must be at least 2x SL distance
            tp_distance = tp - entry
            sl_distance = entry - sl
            if tp_distance < sl_distance * 2 and atr > 0:
                tp = entry + (sl_distance * 2)
        else:
            # SELL
            if swing_lows:
                candidates = [l for l in swing_lows if l < entry]
                if candidates:
                    tp_swing = max(candidates)  # Nearest swing low
                    tp = tp_swing
                else:
                    tp = entry - pips_to_decimal(pair_name_str, pip_tp)
            elif consolidation:
                tp = recent_low
            else:
                tp = entry - pips_to_decimal(pair_name_str, pip_tp)

            if swing_highs:
                candidates = [h for h in swing_highs if h > entry]
                if candidates:
                    sl_swing = min(candidates)  # Nearest swing high
                    sl = min(sl_swing, entry + pips_to_decimal(pair_name_str, pip_sl))
                else:
                    sl = entry + pips_to_decimal(pair_name_str, pip_sl)
            else:
                sl = min(recent_high, entry + pips_to_decimal(pair_name_str, pip_sl))
            
            # Enforce risk/reward: TP must be at least 2x SL distance
            tp_distance = entry - tp
            sl_distance = sl - entry
            if tp_distance < sl_distance * 2 and atr > 0:
                tp = entry - (sl_distance * 2)
    else:
        # Non-forex: fallback to ATR distances
        if final_signal == "BUY":
            tp = entry + (atr * 2)
            sl = entry - atr
        else:
            tp = entry - (atr * 2)
            sl = entry + atr
    
    return {
        "signal": final_signal,
        "agreement": agreement,
        "ema": ema_signal,
        "rsi": round(last.rsi, 2),
        "macd": "UP" if last.macd > last.macd_signal else "DOWN",
        "bb": bb_signal,
        "stoch": round(last.stoch_k, 2) if pd.notna(last.stoch_k) else 0,
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "divergence": divergence,
        "patterns": patterns,
        "fib_levels": {k: round(v, 5) for k, v in fib_levels.items()}
    }

# ================= TELEGRAM =================
async def send_signal(update: Update, choice, symbol="GC=F", pair_name="GOLD"):
    
    if choice == "MULTI":
        df_h4 = fetch(TIMEFRAMES["H4"], symbol)
        df_h1 = fetch(TIMEFRAMES["H1"], symbol)
        df_m30 = fetch(TIMEFRAMES["M30"], symbol)
        if not df_h4 or not df_h1 or not df_m30:
            await update.message.reply_text("❌ Not enough data for this pair.")
            return
        sig_h4 = analyze(df_h4, symbol)
        sig_h1 = analyze(df_h1, symbol)
        sig_m30 = analyze(df_m30, symbol)
        note = f"H4: {sig_h4['signal']} ({sig_h4['agreement']:.0f}%) | H1: {sig_h1['signal']} ({sig_h1['agreement']:.0f}%)"
        sigs = sig_m30
        df = df_m30
    else:
        # choice is expected to be a timeframe key like "H1", "M30", etc.
        interval = TIMEFRAMES.get(choice, choice)
        df = fetch(interval, symbol)
        if not df or len(df) == 0:
            await update.message.reply_text("❌ Not enough data for this pair. Try a different one.")
            return
        sigs = analyze(df, symbol)
        if not sigs:
            await update.message.reply_text("❌ Error analyzing data. Try again.")
            return
        note = ""

    emoji = {"BUY":"🟢 BUY", "SELL":"🔴 SELL"}
    signal = sigs["signal"]

    # Choose decimal precision based on price magnitude
    price = df.iloc[-1]['Close']
    if price >= 100:
        dec = 2
    elif price >= 10:
        dec = 3
    elif price >= 1:
        dec = 4
    else:
        dec = 5

    entry_s = f"${price:.{dec}f}"
    tp_s = f"${sigs['tp']:.{dec}f}"
    sl_s = f"${sigs['sl']:.{dec}f}"
    
    # Build message with all features
    div_text = ""
    if sigs['divergence'] and (sigs['divergence'].get('rsi') or sigs['divergence'].get('macd')):
        div_text = f"🔷 DIVERGENCE: RSI {sigs['divergence']['rsi'] or 'None'} | MACD {sigs['divergence']['macd'] or 'None'}\n"
    
    pattern_text = ""
    if sigs['patterns']:
        pattern_text = f"🔶 PATTERNS: {', '.join(sigs['patterns'])}\n"
    
    fib_text = ""
    if sigs['fib_levels']:
        fib_important = ["38.2%", "50%", "61.8%"]
        fib_text = "📊 FIBONACCI LEVELS:\n"
        for level in fib_important:
            if level in sigs['fib_levels']:
                fib_text += f"  {level}: ${sigs['fib_levels'][level]:.{dec}f}\n"

    await update.message.reply_text(
        f"🟡 {pair_name} SIGNAL\n\n"
        f"{emoji[signal]} {signal} - {sigs['agreement']:.0f}% Agreement\n\n"
        f"📊 INDICATORS:\n"
        f"EMA (50/200): {sigs['ema']}\n"
        f"RSI (14): {sigs['rsi']}\n"
        f"MACD: {sigs['macd']}\n"
        f"Bollinger Bands: {sigs['bb']}\n"
        f"Stochastic: {sigs['stoch']}\n\n"
        f"{div_text}"
        f"{pattern_text}"
        f"{fib_text}"
        f"🎯 TARGETS:\n"
        f"Entry: {entry_s}\n"
        f"Take Profit: {tp_s}\n"
        f"Stop Loss: {sl_s}\n\n"
        f"{note}"
    )
    
    # Auto-log trade to history
    save_trade(pair_name, signal, sigs['tp'], sigs['sl'], price)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_data = context.user_data
    
    # Step 1: START - Show main menu
    if text.lower() == "/start" or text.lower() == "start":
        kb = [["Trade History"], ["Backtest"], ["View Notes"], ["Make Note"], ["Get Signal"]]
        await update.message.reply_text(
            "🤖 Trading Bot Menu:\n\nChoose an option:",
            reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
        )
        user_data.clear()
        user_data["stage"] = "main_menu"
        return
    
    # Step 2: MAIN MENU - User chose an option
    if user_data.get("stage") == "main_menu":
        if text == "Trade History":
            try:
                wr = get_win_rate()
                if wr and wr['total_trades'] > 0:
                    msg = f"📈 WIN RATE:\n✅ Wins: {wr['wins']}\n❌ Losses: {wr['losses']}\nTotal: {wr['total_trades']}\n📊 Rate: {wr['win_rate']:.1f}%"
                else:
                    msg = "❌ No closed trades yet."
            except Exception as e:
                msg = f"❌ Error: {str(e)}"
            await update.message.reply_text(msg)
            kb = [["Trade History"], ["Backtest"], ["View Notes"], ["Make Note"], ["Get Signal"]]
            await update.message.reply_text("Choose another option:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            return
        
        if text == "View Notes":
            try:
                entries = get_journal_entries(10)
                if entries:
                    msg = "📔 TRADE JOURNAL (Last 10):\n"
                    for e in entries:
                        msg += f"\n• {e.get('pair', 'Unknown')}: {e.get('user_note', 'No note')}"
                else:
                    msg = "❌ No journal entries yet."
            except Exception as e:
                msg = f"❌ Error: {str(e)}"
            await update.message.reply_text(msg)
            kb = [["Trade History"], ["Backtest"], ["View Notes"], ["Make Note"], ["Get Signal"]]
            await update.message.reply_text("Choose another option:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            return
        
        if text == "Backtest":
            kb = [["BTC-USD"], ["Gold"], ["EUR/USD"]]
            await update.message.reply_text("Choose pair to backtest:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            user_data["stage"] = "backtest_pair"
            return
        
        if text == "Make Note":
            await update.message.reply_text("📝 Type your note:")
            user_data["stage"] = "waiting_for_note"
            return
        
        if text == "Get Signal":
            kb = [["COMMODITIES", "FOREX"], ["CRYPTO", "INDICES"]]
            await update.message.reply_text("Choose pair type:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            user_data["stage"] = "choosing_type"
            return
    
    # Step 3: BACKTEST - Select pair
    if user_data.get("stage") == "backtest_pair":
        pair_map = {"BTC-USD": "BTC-USD", "Gold": "GC=F", "EUR/USD": "EURUSD=X"}
        if text in pair_map:
            user_data["backtest_pair"] = pair_map[text]
            kb = [["7"], ["14"], ["30"]]
            await update.message.reply_text("How many days back? (7/14/30):", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            user_data["stage"] = "backtest_days"
            return
    
    # Step 3b: BACKTEST - Select days
    if user_data.get("stage") == "backtest_days":
        if text in ["7", "14", "30"]:
            days = int(text)
            pair = user_data.get("backtest_pair", "BTC-USD")
            try:
                await update.message.reply_text(f"⏳ Running backtest on {days}d {pair}...")
                result = backtest_strategy(pair, "1h", days)
                if result and result['total_trades'] > 0:
                    msg = (f"✅ BACKTEST COMPLETE ({days}d {pair} H1):\n\n"
                           f"💰 Trades: {result['total_trades']}\n✅ Wins: {result['wins']}\n"
                           f"❌ Losses: {result['losses']}\n📊 Win Rate: {result['win_rate']:.2f}%\n"
                           f"💵 Total Profit: {result['total_profit']:.2f}")
                else:
                    msg = f"❌ Backtest: No trades found for {pair}"
            except Exception as e:
                msg = f"❌ Backtest error: {str(e)}"
            await update.message.reply_text(msg)
            kb = [["Trade History"], ["Backtest"], ["View Notes"], ["Make Note"], ["Get Signal"]]
            await update.message.reply_text("Choose another option:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            user_data["stage"] = "main_menu"
            return
    
    # Step 4: WAITING FOR NOTE
    if user_data.get("stage") == "waiting_for_note":
        note = text
        try:
            log_journal("Manual Note", "USER_NOTE", 0, 0, 0, note)
            await update.message.reply_text("✅ Note saved!")
        except Exception as e:
            await update.message.reply_text(f"❌ Error saving note: {str(e)}")
        kb = [["Trade History"], ["Backtest"], ["View Notes"], ["Make Note"], ["Get Signal"]]
        await update.message.reply_text("Choose another option:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
        user_data["stage"] = "main_menu"
        return
    
    # Step 5: CHOOSING TYPE - for Get Signal
    if user_data.get("stage") == "choosing_type":
        if text.upper() in PAIRS_BY_TYPE:
            user_data["type"] = text.upper()
            pairs_dict = PAIRS_BY_TYPE[text.upper()]
            pair_names = list(pairs_dict.keys())  # Get dictionary keys
            kb = [[p] for p in pair_names[:4]]  # Show first 4
            await update.message.reply_text(f"Choose {text} pair:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            user_data["stage"] = "choosing_pair"
            return
    
    # Step 6: CHOOSING PAIR
    if user_data.get("stage") == "choosing_pair":
        pair_type = user_data.get("type", "COMMODITIES")
        pairs_dict = PAIRS_BY_TYPE.get(pair_type, {})
        if text in pairs_dict:
            user_data["pair_name"] = text
            val = pairs_dict[text]
            # Handle FOREX lists (base -> [quotes]) by building a symbol using the first quote by default
            if isinstance(val, list):
                quote = val[0]
                symbol = build_forex_symbol(text, quote)
            else:
                symbol = val
            user_data["pair"] = symbol  # Get symbol from dict
            
            kb = [["M30", "H1"], ["H4", "D1"]]
            await update.message.reply_text("Choose timeframe:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            user_data["stage"] = "choosing_timeframe"
            return
    
    # Step 7: CHOOSING TIMEFRAME - Analyze and send signal
    if user_data.get("stage") == "choosing_timeframe":
        if text in ["M30", "H1", "H4", "D1"]:
            user_data["timeframe"] = text
            pair_type = user_data.get("type", "COMMODITIES")
            pair_name = user_data.get("pair_name", "Unknown")
            pair_symbol = user_data.get("pair", "GC=F")
            
            try:
                await update.message.reply_text("⏳ Analyzing... This may take 5-10 seconds...")
                # map key (H1 etc.) to actual interval string (30m, 60m, etc.)
                interval = TIMEFRAMES.get(text, text)
                df = fetch(interval, pair_symbol)
                if df is not None and len(df) > 0:
                    result = analyze(df, pair_symbol)
                    # send_signal expects the timeframe key (e.g., "H1") or "MULTI"
                    await send_signal(update, text, pair_symbol, pair_name)
                else:
                    await update.message.reply_text(f"❌ No data for {pair_name} (symbol={pair_symbol}, interval={interval})")
            except Exception as e:
                await update.message.reply_text(f"❌ Error: {str(e)}")
            
            kb = [["Trade History"], ["Backtest"], ["View Notes"], ["Make Note"], ["Get Signal"]]
            await update.message.reply_text("Choose another option:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
            user_data["stage"] = "main_menu"
            return
    


# ================= TELEGRAM BOT SETUP =================
app_bot = ApplicationBuilder().token(TOKEN).build()
app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ================= FLASK (Render Keep-Alive) =================
flask_app = Flask(__name__)
@flask_app.route("/")
def home():
    return "Bot running"

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)

# ================= KEEP-ALIVE PING =================
def keep_alive():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    while True:
        try:
            requests.get(url)
        except:
            pass
        time.sleep(300)  # Ping every 5 minutes

# ================= RUN BOTH =================
threading.Thread(target=run_flask, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()
app_bot.run_polling()
