import pandas as pd
import numpy as np
import yfinance as yf
import json, os
import threading
import time
import requests
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from flask import Flask

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_TOKEN")
TIMEFRAMES = {"M30": "30m", "H1": "60m", "H4": "4h"}

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
        "USD": ["EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"],
        "EUR": ["USD", "GBP", "CHF", "JPY"],
        "GBP": ["USD", "JPY"],
        "AUD": ["USD", "JPY"],
        "NZD": ["USD"],
        "CAD": ["USD", "CHF"],
        "CHF": ["USD", "EUR", "CAD"],
        "JPY": ["USD", "EUR", "GBP", "AUD"]
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
        "sl": round(sl, 2)
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

    await update.message.reply_text(
        f"🟡 {pair_name} SIGNAL\n\n"
        f"{emoji[signal]} {signal} - {sigs['agreement']:.0f}% Agreement\n\n"
        f"📊 INDICATORS:\n"
        f"EMA (50/200): {sigs['ema']}\n"
        f"RSI (14): {sigs['rsi']}\n"
        f"MACD: {sigs['macd']}\n"
        f"Bollinger Bands: {sigs['bb']}\n"
        f"Stochastic: {sigs['stoch']}\n\n"
        f"🎯 TARGETS:\n"
        f"Entry: {entry_s}\n"
        f"Take Profit: {tp_s}\n"
        f"Stop Loss: {sl_s}\n\n"
        f"{note}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_data = context.user_data
    
    # Step 1: Start - ask for type
    if text.lower() == "start":
        kb = [["COMMODITIES", "FOREX", "CRYPTO", "INDICES"]]
        await update.message.reply_text(
            "Choose pair type:",
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
