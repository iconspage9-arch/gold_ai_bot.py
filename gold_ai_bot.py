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
SYMBOL = "GC=F"
TIMEFRAMES = {"M30": "30m", "H1": "60m", "H4": "4h"}

# ================= DATA FUNCTIONS =================
def fetch(tf):
    df = yf.download(SYMBOL, period="7d", interval=tf, progress=False)
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
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_k=3, smooth_d=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    return df

def analyze(df):
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
    
    # Calculate TP and SL based on ATR
    atr = last.atr if pd.notna(last.atr) else 0
    if final_signal == "BUY":
        tp = last.Close + (atr * 2)  # TP = 2 ATR above entry
        sl = last.Close - atr  # SL = 1 ATR below entry
    else:
        tp = last.Close - (atr * 2)  # TP = 2 ATR below entry
        sl = last.Close + atr  # SL = 1 ATR above entry
    
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
async def send_signal(update: Update, choice):
    if choice == "MULTI":
        df_h4 = fetch(TIMEFRAMES["H4"])
        df_h1 = fetch(TIMEFRAMES["H1"])
        df_m30 = fetch(TIMEFRAMES["M30"])
        sig_h4 = analyze(df_h4)
        sig_h1 = analyze(df_h1)
        sig_m30 = analyze(df_m30)
        note = f"H4: {sig_h4['signal']} ({sig_h4['agreement']:.0f}%) | H1: {sig_h1['signal']} ({sig_h1['agreement']:.0f}%)"
        sigs = sig_m30
        df = df_m30
    else:
        df = fetch(TIMEFRAMES[choice])
        sigs = analyze(df)
        note = ""

    emoji = {"BUY":"🟢 BUY", "SELL":"🔴 SELL"}
    signal = sigs["signal"]

    await update.message.reply_text(
        f"🟡 GOLD SIGNAL (XAUUSD)\n\n"
        f"{emoji[signal]} {signal} - {sigs['agreement']:.0f}% Agreement\n\n"
        f"📊 INDICATORS:\n"
        f"EMA (50/200): {sigs['ema']}\n"
        f"RSI (14): {sigs['rsi']}\n"
        f"MACD: {sigs['macd']}\n"
        f"Bollinger Bands: {sigs['bb']}\n"
        f"Stochastic: {sigs['stoch']}\n\n"
        f"🎯 TARGETS:\n"
        f"Entry: ${df.iloc[-1]['Close']:.2f}\n"
        f"Take Profit: ${sigs['tp']:.2f}\n"
        f"Stop Loss: ${sigs['sl']:.2f}\n\n"
        f"{note}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    if text == "start":
        kb = [["M30","H1","H4","MULTI"]]
        await update.message.reply_text(
            "Choose timeframe:",
            reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
        )
        return
    if text.upper() in TIMEFRAMES or text.upper() == "MULTI":
        await send_signal(update, text.upper())
        return
    await update.message.reply_text("Type *start* to begin.", parse_mode="Markdown")

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
