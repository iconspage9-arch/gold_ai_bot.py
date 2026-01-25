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

# ✅ FIX 1: D1 timeframe fixed
TIMEFRAMES = {
    "M30": "30m",
    "H1": "60m",
    "H4": "4h",
    "D1": "1d"
}

TRADE_HISTORY_FILE = "trade_history.json"
JOURNAL_FILE = "trade_journal.json"

# ================= DATA FUNCTIONS =================
def fetch(tf, symbol="GC=F"):
    try:
        df = yf.download(symbol, period="30d", interval=tf, progress=False)
        if df is None or len(df) == 0:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()
        if len(df) < 50:
            return None

        df[['Close','High','Low']] = df[['Close','High','Low']].apply(pd.to_numeric, errors='coerce')

        df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
        df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator() if len(df) >= 200 else np.nan
        df["rsi"] = RSIIndicator(df["Close"], 14).rsi()

        macd = MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()

        bb = BollingerBands(df["Close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()

        stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["stoch_k"] = stoch.stoch()

        return df.dropna(subset=["Close","High","Low"])
    except:
        return None

# ================= DIVERGENCE =================
def detect_divergence(df):
    if len(df) < 30:
        return None

    rsi_div = None
    macd_div = None

    if df['Low'].iloc[-1] < df['Low'].iloc[-20] and df['rsi'].iloc[-1] > df['rsi'].iloc[-20]:
        rsi_div = "BULLISH"
    elif df['High'].iloc[-1] > df['High'].iloc[-20] and df['rsi'].iloc[-1] < df['rsi'].iloc[-20]:
        rsi_div = "BEARISH"

    # ✅ FIX 3: honest naming (momentum, not true divergence)
    macd_momentum = df['macd'].iloc[-1] - df['macd'].iloc[-30]
    if macd_momentum > 0:
        macd_div = "MOMENTUM_BULLISH"
    elif macd_momentum < 0:
        macd_div = "MOMENTUM_BEARISH"

    return {"rsi": rsi_div, "macd": macd_div}

# ================= ANALYSIS =================
def analyze(df, symbol=None):
    last = df.iloc[-1]
    signals = []

    # ✅ FIX 2: EMA fallback fixed
    if pd.notna(last.ema200):
        ema_signal = "BUY" if last.ema50 > last.ema200 else "SELL"
    else:
        ema_signal = "BUY" if last.ema50 > df["ema50"].iloc[-10:].mean() else "SELL"
    signals.append(ema_signal)

    rsi_signal = "BUY" if last.rsi < 30 else "SELL" if last.rsi > 70 else "NEUTRAL"
    if rsi_signal != "NEUTRAL":
        signals.append(rsi_signal)

    signals.append("BUY" if last.macd > last.macd_signal else "SELL")

    if last.Close <= last.bb_lower:
        signals.append("BUY")
    elif last.Close >= last.bb_upper:
        signals.append("SELL")

    if last.stoch_k < 20:
        signals.append("BUY")
    elif last.stoch_k > 80:
        signals.append("SELL")

    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")

    final_signal = "BUY" if buy_count > sell_count else "SELL"
    agreement = max(buy_count, sell_count) / len(signals) * 100

    atr = last.atr
    entry = last.Close

    if final_signal == "BUY":
        tp = entry + atr * 2
        sl = entry - atr
    else:
        tp = entry - atr * 2
        sl = entry + atr

    return {
        "signal": final_signal,
        "agreement": agreement,
        "ema": ema_signal,
        "rsi": round(last.rsi, 2),
        "macd": "UP" if last.macd > last.macd_signal else "DOWN",
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "divergence": detect_divergence(df)
    }

# ================= TELEGRAM =================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()

    if text == "START" or text == "/START":
        kb = [["M30","H1"],["H4","D1"]]
        await update.message.reply_text("Choose timeframe:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
        return

    if text in TIMEFRAMES:
        tf = TIMEFRAMES[text]
        df = fetch(tf, "GC=F")
        if not df:
            await update.message.reply_text("❌ No data.")
            return
        sig = analyze(df)
        await update.message.reply_text(
            f"🟡 GOLD SIGNAL\n\n"
            f"{sig['signal']} ({sig['agreement']:.0f}%)\n"
            f"Entry: {df.iloc[-1]['Close']:.2f}\n"
            f"TP: {sig['tp']}\n"
            f"SL: {sig['sl']}"
        )

# ================= RUN =================
app_bot = ApplicationBuilder().token(TOKEN).build()
app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

flask_app = Flask(__name__)
@flask_app.route("/")
def home():
    return "Bot running"

threading.Thread(target=lambda: flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000))), daemon=True).start()
app_bot.run_polling()
