import pandas as pd
import numpy as np
import yfinance as yf
import json, os
from ta.trend import EMAIndicator
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters, CommandHandler
from flask import Flask
import threading

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_TOKEN")
SYMBOL = "GC=F"
WEIGHT_FILE = "ai_weights.json"

TIMEFRAMES = {"M30": "30m", "H1": "60m", "H4": "4h"}

# ================= DATA =================
def fetch(tf):
    df = yf.download(SYMBOL, period="7d", interval=tf, progress=False)

    # Fix for GC=F MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
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

    hour = last.name.hour  # UTC on Render
    sigs["Session"] = "BUY" if 7 <= hour <= 16 else "OFF"

    return sigs

# ================= TELEGRAM =================
async def send_signal(update: Update, choice):
    if choice == "MULTI":
        df_h4 = fetch(TIMEFRAMES["H4"])
        df_h1 = fetch(TIMEFRAMES["H1"])
        df_m30 = fetch(TIMEFRAMES["M30"])

        sig_h4 = analyze(df_h4)
        sig_h1 = analyze(df_h1)
        sig_m30 = analyze(df_m30)

        note = f"H4 Trend: {sig_h4['Trend']} | H1 Trend: {sig_h1['Trend']}"
        sigs = sig_m30
    else:
        df = fetch(TIMEFRAMES[choice])
        sigs = analyze(df)
        note = ""

    emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "OFF": "⚪ OFF"}
    signal_text = "\n".join([f"{k}: {emoji[v]}" for k, v in sigs.items()])

    await update.message.reply_text(
        f"🟡 GOLD SIGNAL BREAKDOWN (XAUUSD)\n\n"
        f"{note}\n\n"
        f"{signal_text}\n\n"
        f"🧠 Each line shows what that signal suggests"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()

    if text == "start":
        kb = [["M30", "H1", "H4", "MULTI"]]
        await update.message.reply_text(
            "Choose timeframe:",
            reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
        )
        return

    if text.upper() in TIMEFRAMES or text.upper() == "MULTI":
        await send_signal(update, text.upper())
        return

    await update.message.reply_text("Type *start* to begin.", parse_mode="Markdown")

# ================= BOT =================
app_bot = ApplicationBuilder().token(TOKEN).build()
app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ================= FLASK (Render) =================
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "Bot running"

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)

# ================= RUN BOTH =================
threading.Thread(target=run_flask, daemon=True).start()
app_bot.run_polling()
