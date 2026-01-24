import pandas as pd
import numpy as np
import yfinance as yf
import json, os
import threading
import time
import requests
from ta.trend import EMAIndicator
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
    df["ema50"] = EMAIndicator(df["Close"],50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"],200).ema_indicator()
    return df

def analyze(df):
    sigs = {}
    last, prev = df.iloc[-1], df.iloc[-2]
    sigs["Trend"] = "BUY" if last.ema50 > last.ema200 else "SELL"
    sigs["EMA"] = "BUY" if last.Close > last.ema50 else "SELL"
    high = df["High"].iloc[-20:].max()
    low  = df["Low"].iloc[-20:].min()
    sigs["Support/Resistance"] = "BUY" if abs(last.Close - low) < abs(last.Close - high) else "SELL"
    sigs["Breakout"] = "BUY" if last.Close > prev.High else "SELL"
    hour = last.name.hour
    sigs["Session"] = "BUY" if 7 <= hour <= 16 else "OFF"
    return sigs, last.Close, high, low

# ================= TRADE CALCULATION =================
def determine_direction(sigs):
    counts = {"BUY":0, "SELL":0}
    for v in sigs.values():
        if v in counts:
            counts[v] +=1
    if counts["BUY"] > counts["SELL"]:
        return "BUY"
    elif counts["SELL"] > counts["BUY"]:
        return "SELL"
    else:
        return None

def calculate_trade_levels(direction, entry, high, low):
    buffer = 0.5  # small buffer for SL
    if direction == "BUY":
        sl = low - buffer
        tp = entry + (entry - sl)*2  # RR=1:2
    elif direction == "SELL":
        sl = high + buffer
        tp = entry - (sl - entry)*2
    else:
        sl, tp = None, None
    return sl, tp

# ================= TELEGRAM =================
async def send_signal(update: Update, choice):
    if choice == "MULTI":
        df_h4 = fetch(TIMEFRAMES["H4"])
        df_h1 = fetch(TIMEFRAMES["H1"])
        df_m30 = fetch(TIMEFRAMES["M30"])
        sig_h4, _, _, _ = analyze(df_h4)
        sig_h1, _, _, _ = analyze(df_h1)
        sigs, entry, high, low = analyze(df_m30)
        note = f"H4 Trend: {sig_h4['Trend']} | H1 Trend: {sig_h1['Trend']}"
    else:
        df = fetch(TIMEFRAMES[choice])
        sigs, entry, high, low = analyze(df)
        note = ""

    direction = determine_direction(sigs)
    sl, tp = calculate_trade_levels(direction, entry, high, low)

    emoji = {"BUY":"🟢 BUY", "SELL":"🔴 SELL", "OFF":"⚪ OFF"}
    signal_text = "\n".join([f"{k}: {emoji[v]}" for k,v in sigs.items()])

    trade_text = f"Direction: {direction}\nEntry: {entry:.2f}\nStop-Loss: {sl:.2f}\nTake-Profit: {tp:.2f}" if direction else "No clear direction"

    await update.message.reply_text(
        f"🟡 GOLD SIGNAL BREAKDOWN (XAUUSD)\n\n"
        f"{note}\n\n"
        f"{signal_text}\n\n"
        f"💰 Trade Suggestion:\n{trade_text}\n\n"
        f"🧠 Signals only — use discretion"
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
        time.sleep(300)  # ping every 5 minutes

# ================= RUN BOTH =================
threading.Thread(target=run_flask, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()
app_bot.run_polling()
