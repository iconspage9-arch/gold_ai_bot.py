import pandas as pd
import numpy as np
import yfinance as yf
import json, os
import pytz, datetime as dt
from ta.trend import EMAIndicator
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from flask import Flask
import threading

# ================= CONFIG =================
TOKEN = os.getenv("TELEGRAM_TOKEN")
SYMBOL = "GC=F"
WEIGHT_FILE = "ai_weights.json"

TIMEFRAMES = {"M30": "30m", "H1": "60m", "H4": "4h"}
DEFAULT_WEIGHTS = {"trend": 1.0, "ema": 1.0, "sr": 1.0, "break": 1.0, "session": 1.0}

weights = {}
last_signals = []
results = []

def load_weights():
    global weights
    if os.path.exists(WEIGHT_FILE):
        weights = json.load(open(WEIGHT_FILE))
    else:
        weights = DEFAULT_WEIGHTS.copy()
    return weights

def save_weights(w):
    json.dump(w, open(WEIGHT_FILE, "w"), indent=2)

load_weights()

# ================= DATA =================
def fetch(tf):
    df = yf.download(SYMBOL, period="7d", interval=tf, progress=False)
    df.dropna(inplace=True)
    df["ema50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["Close"], 200).ema_indicator()
    return df

def analyze(df):
    sigs = {}
    last, prev = df.iloc[-1], df.iloc[-2]
    sigs["trend"] = "BUY" if last.ema50 > last.ema200 else "SELL"
    sigs["ema"] = "BUY" if last.Close > last.ema50 else "SELL"
    high, low = df.High[-20:].max(), df.Low[-20:].min()
    sigs["sr"] = "BUY" if abs(last.Close - low) < abs(last.Close - high) else "SELL"
    sigs["break"] = "BUY" if last.Close > prev.High else "SELL"
    hour = last.name.hour
    sigs["session"] = "BUY" if (7 <= hour <= 16) else None
    return sigs

def ai_decide(sigs):
    score = {"BUY": 0, "SELL": 0}
    for k, v in sigs.items():
        if v:
            score[v] += weights[k]
    return max(score, key=score.get) if max(score.values()) >= 3 else None

# ================= TELEGRAM HANDLER =================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()

    # Typing "start"
    if text == "start":
        kb = [["M30", "H1", "H4", "MULTI"]]
        await update.message.reply_text(
            "Choose timeframe:",
            reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
        )
        return

    # Waiting for Yes/No confirmation
    if text in ["yes", "no"]:
        if "next_choice" in context.user_data:
            choice = context.user_data.pop("next_choice")
            if text == "yes":
                await send_signal(update, context, choice)
            else:
                await update.message.reply_text("❌ Trade cancelled due to news.")
        return

    # Timeframe choice
    if text.upper() in TIMEFRAMES or text.upper() == "MULTI":
        if not news_safe():
            kb = [["Yes", "No"]]
            context.user_data["next_choice"] = text.upper()
            await update.message.reply_text(
                "⚠️ High-impact USD news detected. Proceed?",
                reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True)
            )
            return
        await send_signal(update, context, text.upper())
        return

    await update.message.reply_text("Type 'start' to begin or choose a valid timeframe.")

# ================= SIGNAL SENDER =================
def news_safe():
    hour = dt.datetime.now(pytz.UTC).hour
    return not (12 <= hour <= 14)

async def send_signal(update, context, choice):
    if choice == "MULTI":
        df_h4, df_h1, df_m30 = fetch(TIMEFRAMES["H4"]), fetch(TIMEFRAMES["H1"]), fetch(TIMEFRAMES["M30"])
        sig_h4, sig_h1, sig_m30 = analyze(df_h4), analyze(df_h1), analyze(df_m30)
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
        f"🟡 GOLD SIGNAL (XAUUSD)\n\n{multi_note}\n\n📍 Direction: {direction}\n🧠 Confidence: {round(sum(weights.values()),2)}\n📊 Signals: {sigs}\n\nReply /win or /loss after trade"
    )

# ================= WIN/LOSS =================
async def win(update, context):
    for s in last_signals:
        weights[s] += 0.05
    save_weights(weights)
    results.append(1)
    await update.message.reply_text("✅ Win recorded. AI updated.")

async def loss(update, context):
    for s in last_signals:
        weights[s] = max(0.1, weights[s] - 0.05)
    save_weights(weights)
    results.append(0)
    await update.message.reply_text("❌ Loss recorded. AI updated.")

async def stats(update, context):
    winrate = round(sum(results)/len(results)*100, 2) if results else 0
    await update.message.reply_text(f"📊 Win rate: {winrate}%\n🧠 Weights:\n{weights}")

async def reset_ai(update, context):
    weights.update(DEFAULT_WEIGHTS)
    save_weights(weights)
    await update.message.reply_text("♻️ AI reset to default.")

# ================= FLASK =================
flask_app = Flask(__name__)
@flask_app.route("/")
def home():
    return "Bot running"

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)

# ================= RUN =================
app_bot = ApplicationBuilder().token(TOKEN).build()
app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app_bot.add_handler(MessageHandler(filters.TEXT & filters.Regex("^/win$"), win))
app_bot.add_handler(MessageHandler(filters.TEXT & filters.Regex("^/loss$"), loss))
app_bot.add_handler(MessageHandler(filters.TEXT & filters.Regex("^/stats$"), stats))
app_bot.add_handler(MessageHandler(filters.TEXT & filters.Regex("^/reset_ai$"), reset_ai))

threading.Thread(target=run_flask, daemon=True).start()
app_bot.run_polling(stop_signals=None)
