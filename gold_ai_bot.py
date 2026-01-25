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
    """Convert base/quote to yfinance symbol (e.g., USD+GBP -> USDGBP=X)"""
    return f"{base}{quote}=X"

# ================= DATA FUNCTIONS =================
def fetch(tf, symbol="GC=F"):
    df = yf.download(symbol, period="30d", interval=tf, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    
    # Calculate indicators
    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
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
    
    return df

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
        low_point = recent['Low'].iloc[-30:].idxmin()
        left_high = recent['High'].iloc[-30:low_point].max()
        right_high = recent['High'].iloc[low_point:].max()
        
        if abs(left_high - right_high) / left_high < 0.02 and left_high > recent['Low'].iloc[-1]:
            # Check for handle (small pullback)
            handle_low = recent['Low'].iloc[-5:].min()
            if handle_low > recent['Low'].iloc[low_point]:
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
        
        if len(df) < 50:
            return None
        
        wins = 0
        losses = 0
        total_profit = 0
        
        # Simulate signals on historical data
        for i in range(50, len(df)):
            window = df.iloc[i-50:i]
            last = window.iloc[-1]
            
            # Quick signal check
            ema50 = EMAIndicator(window["Close"], 50).ema_indicator().iloc[-1]
            ema200 = EMAIndicator(window["Close"], 200).ema_indicator().iloc[-1]
            rsi = RSIIndicator(window["Close"], 14).rsi().iloc[-1]
            
            signal = "BUY" if ema50 > ema200 and rsi < 70 else "SELL" if ema50 < ema200 and rsi > 30 else None
            
            if signal and i + 1 < len(df):
                entry = last['Close']
                future = df.iloc[i+1]
                
                if signal == "BUY":
                    if future['High'] > entry * 1.02:  # TP hit (2%)
                        wins += 1
                        total_profit += entry * 0.02
                    elif future['Low'] < entry * 0.99:  # SL hit (1%)
                        losses += 1
                        total_profit -= entry * 0.01
                else:
                    if future['Low'] < entry * 0.98:  # TP hit (2%)
                        wins += 1
                        total_profit += entry * 0.02
                    elif future['High'] > entry * 1.01:  # SL hit (1%)
                        losses += 1
                        total_profit -= entry * 0.01
        
        return {
            "total_trades": wins + losses,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
            "total_profit": total_profit
        }
    except Exception as e:
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
    last = df.iloc[-1]
    signals = []
    
    # Signal 1: EMA Trend (50/200 crossover)
    ema_signal = "BUY" if last.ema50 > last.ema200 else "SELL"
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
        sig_h4 = analyze(df_h4, symbol)
        sig_h1 = analyze(df_h1, symbol)
        sig_m30 = analyze(df_m30, symbol)
        note = f"H4: {sig_h4['signal']} ({sig_h4['agreement']:.0f}%) | H1: {sig_h1['signal']} ({sig_h1['agreement']:.0f}%)"
        sigs = sig_m30
        df = df_m30
    else:
        df = fetch(TIMEFRAMES[choice], symbol)
        sigs = analyze(df, symbol)
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
    if sigs['divergence']['rsi'] or sigs['divergence']['macd']:
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
    
    # Commands
    if text.lower() == "/history":
        wr = get_win_rate()
        if wr:
            msg = f"📈 WIN RATE:\n✅ Wins: {wr['wins']}\n❌ Losses: {wr['losses']}\nTotal: {wr['total_trades']}\n📊 Rate: {wr['win_rate']:.1f}%"
        else:
            msg = "No closed trades yet."
        await update.message.reply_text(msg)
        return
    
    if text.lower() == "/backtest":
        result = backtest_strategy("BTC-USD", "1h", 30)
        if result:
            msg = f"🔬 BACKTEST (30d BTC):\n💰 Trades: {result['total_trades']}\n✅ Wins: {result['wins']}\n❌ Losses: {result['losses']}\n📊 Rate: {result['win_rate']:.1f}%\n💵 Profit: ${result['total_profit']:.2f}"
        else:
            msg = "Backtest failed."
        await update.message.reply_text(msg)
        return
    
    if text.lower().startswith("/journal"):
        entries = get_journal_entries(5)
        if entries:
            msg = "📔 TRADE JOURNAL (Last 5):\n"
            for e in entries[-5:]:
                msg += f"\n{e['pair']} {e['signal']} @ ${e['entry']:.5f}\nNote: {e['user_note']}\n"
        else:
            msg = "No journal entries yet."
        await update.message.reply_text(msg)
        return
    
    if text.lower().startswith("/note "):
        note = text[6:]
        if user_data.get("pair"):
            log_journal(user_data.get("pair_name", "Unknown"), "USER_NOTE", 0, 0, 0, note)
            await update.message.reply_text("✅ Note logged!")
        return
    
    # Step 1: Start - ask for type
    if text.lower() == "start":
        kb = [["COMMODITIES", "FOREX", "CRYPTO", "INDICES"]]
        await update.message.reply_text(
            "Choose pair type:\n\n/history - Win rate\n/backtest - Test strategy\n/journal - Trade notes\n/note [text] - Log note",
            reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
        )
        user_data.clear()
        return
    
    # Step 2: Type selected - ask for base/pair
    if text.upper() in PAIRS_BY_TYPE and "type" not in user_data:
        user_data["type"] = text.upper()
        pair_type = user_data["type"]
        
        if pair_type == "FOREX":
            bases = list(PAIRS_BY_TYPE["FOREX"].keys())
            kb = [bases[i:i+3] for i in range(0, len(bases), 3)]  # 3 per row
            await update.message.reply_text(
                "Choose base currency:",
                reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
            )
        else:
            # Commodities, Crypto, Indices - show all options
            pairs = list(PAIRS_BY_TYPE[pair_type].keys())
            kb = [pairs[i:i+3] for i in range(0, len(pairs), 3)]
            await update.message.reply_text(
                f"Choose {pair_type.lower()}:",
                reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
            )
        return
    
    # Step 3: Forex - base selected, ask for quote
    if user_data.get("type") == "FOREX" and "base" not in user_data:
        if text.upper() in PAIRS_BY_TYPE["FOREX"]:
            user_data["base"] = text.upper()
            quotes = PAIRS_BY_TYPE["FOREX"][user_data["base"]]
            kb = [quotes[i:i+3] for i in range(0, len(quotes), 3)]
            await update.message.reply_text(
                f"Choose quote currency for {user_data['base']}:",
                reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
            )
            return
    
    # Step 4: Forex - quote selected, build symbol
    if user_data.get("type") == "FOREX" and user_data.get("base") and "pair" not in user_data:
        if text.upper() in PAIRS_BY_TYPE["FOREX"][user_data["base"]]:
            user_data["quote"] = text.upper()
            symbol = build_forex_symbol(user_data["base"], user_data["quote"])
            user_data["pair"] = symbol
            user_data["pair_name"] = f"{user_data['base']}{user_data['quote']}"
            await ask_for_timeframe(update)
            return
    
    # Step 3 (non-Forex): Pair selected directly
    if user_data.get("type") in ["COMMODITIES", "CRYPTO", "INDICES"] and "pair" not in user_data:
        pair_type = user_data["type"]
        if text.upper() in PAIRS_BY_TYPE[pair_type]:
            user_data["pair_name"] = text.upper()
            user_data["pair"] = PAIRS_BY_TYPE[pair_type][text.upper()]
            await ask_for_timeframe(update)
            return
    
    # Step 5: Timeframe selected - analyze and send signal
    if user_data.get("pair") and text.upper() in TIMEFRAMES or text.upper() == "MULTI":
        symbol = user_data["pair"]
        pair_name = user_data["pair_name"]
        await send_signal(update, text.upper(), symbol, pair_name)
        return
    
    await update.message.reply_text("Invalid input. Type *start* to begin.", parse_mode="Markdown")

async def ask_for_timeframe(update: Update):
    kb = [["M30","H1","H4","MULTI"]]
    await update.message.reply_text(
        "Choose timeframe:",
        reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
    )

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
