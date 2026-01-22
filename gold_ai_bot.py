import pandas as pd
import numpy as np
import yfinance as yf
import json, os
import pytz, datetime as dt
from ta.trend import EMAIndicator
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from flask import Flask

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_TOKEN")  # must be set in Render environment
SYMBOL = "GC=F"  # Gold futures proxy
WEIGHT_FILE = "ai_weights.json"

TIMEFRAMES = {
    "M30": "30m",
    "H1": "60m",
    "H4": "4h"
}

DEFAULT_WEIGHTS = {
    "trend": 1.0,
    "ema": 1.0,
    "sr": 1.0,
    "break": 1.0,
    "session": 1.0
}

# ================= AI MEMORY =================
def load_weights():
    if os.path.exists(WEIGHT_FILE):
        return json.load(open(WEIGHT_FILE))
    return DEFAULT_WEIGHTS.copy()

def save_weights(w):
    json.dump(w, open(WEIGHT_FILE, "w"), indent=2)

weights = load_weights()
last_signals = []
results = []

# ================= NEWS FILTER =================
HIGH_IMPACT = ["CPI", "NFP", "FOMC", "FED", "INFLATION", "RATE"]

def news_safe():
    # simplified USD news filter (can expand with API)
    hour = dt.datetime.now(pytz.UTC).hour
    return not (12 <= hour <= 14)

# ================= DATA =================
def fetch(tf):
    df = yf.download(SYMBOL, period="7d", interval=tf, progress=False)
    df.dropna(inplace=True)
    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
    return df

# ================= SIGNAL ENGINE =================
def analyze(df):
    sigs = {}
    last, prev = df.iloc[-1], df.iloc[-2]

    # Trend
    sigs["trend"] = "BUY" if last.ema50 > last.ema200 else "SELL"

    # EMA pullback
    sigs["ema"] = "BUY" if last.Close > last.ema50 else "SELL"

    # Support / Resistance
    high, low = df.High[-20:].max(), df.Low[-20:].min()
    sigs["sr"] = "BUY" if abs(last.Close - low) < abs(last.Close - high) else "SELL"

    # Break & retest
    sigs["break"] = "BUY" if last.Close > prev.High else "SELL"

    # Session
    hour = last.name.hour
    sigs["session"] = "BUY" if (7 <= hour <= 16) else None

    return sigs

def ai_decide(sigs):
    score = {"BUY": 0, "SELL": 0}
    for k, v in sigs.items():
        if v:
            score[v] += weights[k]

    if max(score.values()) >= 3:
        return max(score, key=score.get)
    return None

# ================= TELEGRAM =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [["M30", "H1", "H4", "MULTI"]]
    await update.message.reply_text(
        "Choose timeframe:",
        reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
    )

async def timeframe_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = update.message.text.upper()

    if not news_safe():
        await update.message.reply_text("⚠️ High-impact USD news window — NO TRADE")
        return

    if choice == "MULTI":
        df_h4 = fetch(TIMEFRAMES["H4"])
        df_h1 = fetch(TIMEFRAMES["H1"])
        df_m30 = fetch(TIMEFRAMES["M30"])

        sig_h4 = analyze(df_h4)
        sig_h1 = analyze(df_h1)
        sig_m30 = analyze(df_m30)

        # Require H4 and H1 trend alignment
        if sig_h4["trend"] != sig_h1["trend"]:
            await update.message.reply_text("❌ Timeframes not aligned — NO TRADE")
            return

        sigs = sig_m30
        multi_note = f"H4 Trend: {sig_h4['trend']}, H1 Confirm: {sig_h1['trend']}"
    else:
        df = fetch(TIMEFRAMES[choice])
        sigs = analyze(df)
        multi_note = ""

    direction = ai_decide(sigs)

    if not direction:
        await update.message.reply_text("❌ No strong confluence — NO TRADE")
        return

    last_signals.clear()
    last_signals.extend(sigs.keys())

    await update.message.reply_text(
        f"""🟡 GOLD SIGNAL (XAUUSD)

{multi_note}

📍 Direction: {direction}
🧠 Confidence: {round(sum(weights.values()),2)}
📊 Signals: {sigs}

Reply /win or /loss after trade
"""
    )

# ================= WIN / LOSS =================
async def win(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for s in last_signals:
        weights[s] += 0.05
    save_weights(weights)
    results.append(1)
    await update.message.reply_text("✅ Win recorded. AI updated.")

async def loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for s in last_signals:
        weights[s] = max(0.1, weights[s] - 0.05)
    save_weights(weights)
    results.append(0)
    await update.message.reply_text("❌ Loss recorded. AI updated.")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if results:
        winrate = round(sum(results)/len(results)*100, 2)
    else:
        winrate = 0
    await update.message.reply_text(
        f"📊 Win rate: {winrate}%\n🧠 Weights:\n{weights}"
    )

async def reset_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    weights.update(DEFAULT_WEIGHTS)
    save_weights(weights)
    await update.message.reply_text("♻️ AI reset to default.")

# ================= RUN TELEGRAM BOT =================
app_bot = ApplicationBuilder().token(TOKEN).build()
app_bot.add_handler(CommandHandler("start", start))
app_bot.add_handler(CommandHandler("win", win))
app_bot.add_handler(CommandHandler("loss", loss))
app_bot.add_handler(CommandHandler("stats", stats))
app_bot.add_handler(CommandHandler("reset_ai", reset_ai))
app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, timeframe_choice))

# ================= FAKE WEB SERVER (for free Render) =================
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "Bot running"

# ================= RUN BOTH =================
import threading

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)

# Start Flask in a thread
threading.Thread(target=run_flask).start()

# Start Telegram bot (blocking)
app_bot.run_polling()
