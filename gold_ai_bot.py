import pandas as pd
import numpy as np
import yfinance as yf
import json, os, threading
from ta.trend import EMAIndicator
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, MessageHandler,
    ContextTypes, filters, CommandHandler
)
from flask import Flask

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_TOKEN")
SYMBOL = "GC=F"
WEIGHT_FILE = "/tmp/ai_weights.json"  # Render-safe

TIMEFRAMES = {"M30": "30m", "H1": "60m", "H4": "4h"}
DEFAULT_WEIGHTS = {"trend":1.0,"ema":1.0,"sr":1.0,"break":1.0,"session":1.0}

# ================= AI MEMORY =================
def load_weights():
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_WEIGHTS.copy()

def save_weights(w):
    with open(WEIGHT_FILE, "w") as f:
        json.dump(w, f, indent=2)

weights = load_weights()
last_signals = []
results = []

# ================= DATA & SIGNALS =================
def fetch(tf):
    df = yf.download(SYMBOL, period="7d", interval=tf, progress=False)

    # FIX: yfinance MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
    return df

def analyze(df):
    sigs = {}
    last, prev = df.iloc[-1], df.iloc[-2]

    sigs["trend"] = "BUY" if last.ema50 > last.ema200 else "SELL"
    sigs["ema"] = "BUY" if last.Close > last.ema50 else "SELL"

    high = df["High"].iloc[-20:].max()
    low  = df["Low"].iloc[-20:].min()
    sigs["sr"] = "BUY" if abs(last.Close - low) < abs(last.Close - high) else "SELL"

    sigs["break"] = "BUY" if last.Close > prev.High else "SELL"

    hour = last.name.hour  # UTC on Render
    sigs["session"] = "BUY" if 7 <= hour <= 16 else None

    return sigs

def ai_decide(sigs):
    score = {"BUY":0,"SELL":0}
    for k, v in sigs.items():
        if v:
            score[v] += weights[k]
    return max(score, key=score.get) if max(score.values()) >= 3 else None

# ================= TELEGRAM =================
async def send_signal(update: Update, choice):
    if choice == "MULTI":
        df_h4 = fetch(TIMEFRAMES["H4"])
        df_h1 = fetch(TIMEFRAMES["H1"])
        df_m30 = fetch(TIMEFRAMES["M30"])

        sig_h4 = analyze(df_h4)
        sig_h1 = analyze(df_h1)
        sig_m30 = analyze(df_m30)

        if sig_h4["trend"] != sig_h1["trend"]:
            await update.message.reply_text("❌ Timeframes not aligned — NO TRADE")
            return

        sigs = sig_m30
        note = f"H4 Trend: {sig_h4['trend']} | H1 Confirm: {sig_h1['trend']}"
    else:
        df = fetch(TIMEFRAMES[choice])
        sigs = analyze(df)
        note = ""

    direction = ai_decide(sigs)
    if not direction:
        await update.message.reply_text("❌ No strong confluence — NO TRADE")
        return

    last_signals.clear()
    last_signals.extend([k for k,v in sigs.items() if v == direction])

    await update.message.reply_text(
        f"🟡 GOLD SIGNAL (XAUUSD)\n\n"
        f"{note}\n\n"
        f"📍 Direction: {direction}\n"
        f"📊 Signals: {sigs}\n\n"
        f"Reply /win or /loss after trade"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.upper()

    if text == "START":
        kb = [["M30", "H1", "H4", "MULTI"]]
        await update.message.reply_text(
            "Choose timeframe:",
            reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
        )
        return

    if text in TIMEFRAMES or text == "MULTI":
        await send_signal(update, text)
        return

    await update.message.reply_text("Type START to begin.")

# ================= COMMANDS =================
async def win(update: Update, context):
    for s in last_signals:
        weights[s] += 0.05
    save_weights(weights)
    results.append(1)
    await update.message.reply_text("✅ Win recorded. AI updated.")

async def loss(update: Update, context):
    for s in last_signals:
        weights[s] = max(0.1, weights[s] - 0.05)
    save_weights(weights)
    results.append(0)
    await update.message.reply_text("❌ Loss recorded. AI updated.")

async def stats(update: Update, context):
    winrate = round(sum(results)/len(results)*100,2) if results else 0
    await update.message.reply_text(f"📊 Win rate: {winrate}%\n🧠 Weights:\n{weights}")

async def reset_ai(update: Update, context):
    weights.update(DEFAULT_WEIGHTS)
    save_weights(weights)
    await update.message.reply_text("♻️ AI reset.")

# ================= BOT =================
app_bot = ApplicationBuilder().token(TOKEN).build()
app_bot.add_handler(CommandHandler("win", win))
app_bot.add_handler(CommandHandler("loss", loss))
app_bot.add_handler(CommandHandler("stats", stats))
app_bot.add_handler(CommandHandler("reset_ai", reset_ai))
app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ================= FLASK (Render) =================
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "Bot running"

def run_bot():
    app_bot.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_bot).start()
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)
