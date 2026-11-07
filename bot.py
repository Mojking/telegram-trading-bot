"""
5M BOS/CHoCH + Fair Value Gap alert bot for Telegram.

Strategy it watches for:
  1. A Break of Structure (BOS) or Change of Character (CHoCH): a 5M candle
     CLOSES beyond the last confirmed swing high/low (internal or swing pivots).
  2. The Fair Value Gap (3-candle imbalance) created by the impulse leg that
     caused that break.
  3. ALERT the moment price first comes back INTO that FVG.
  (Optional) second alert when a 5M candle closes back out of the FVG in the
  direction of the break, i.e. the FVG was respected.

It never places trades. You confirm and execute yourself.

Telegram commands: /status  /zones  /help
"""
import json
import os
import time
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

# ============================ CONFIG ============================
def env(name, default=None, cast=str):
    v = os.environ.get(name, default)
    if v is None:
        raise SystemExit(f"Missing required setting: {name} (put it in .env)")
    return cast(v)

TG_TOKEN = env("TELEGRAM_TOKEN")
TG_CHAT = env("TELEGRAM_CHAT_ID")
SOURCE = env("SOURCE", "twelvedata").lower()
TD_KEY = os.environ.get("TWELVEDATA_KEY", "")
SYMBOLS = [s.strip() for s in env("SYMBOLS", "XAU/USD").split(",") if s.strip()]
POLL_SEC = env("POLL_SEC", "120", int)
TZ_OFFSET = env("TZ_OFFSET_HOURS", "8", float)
FOREX_WEEKEND_PAUSE = env("FOREX_WEEKEND_PAUSE", "1") == "1"

INTERNAL_LEN = env("INTERNAL_LEN", "3", int)     # pivot strength for internal highs/lows
SWING_LEN = env("SWING_LEN", "10", int)          # pivot strength for swing highs/lows
MIN_FVG_ATR = env("MIN_FVG_ATR", "0.10", float)  # ignore tiny gaps (fraction of ATR)
MAX_AGE_BARS = env("MAX_AGE_BARS", "48", int)    # FVG expires if untouched this long (48 x 5m = 4h)
CONFIRM_ALERT = env("CONFIRM_ALERT", "0") == "1"

TF_SEC = 300  # 5 minutes
BARS = 300
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
LOCAL_TZ = timezone(timedelta(hours=TZ_OFFSET))

# ============================ STATE ============================
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"alerted": [], "confirmed": [], "tg_offset": 0}

def save_state(st):
    st["alerted"] = st["alerted"][-1000:]
    st["confirmed"] = st["confirmed"][-1000:]
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)

STATE = load_state()
STATUS = {"last_scan": None, "zones": {}, "errors": 0}

# ============================ TELEGRAM ============================
def tg_send(text):
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": text}, timeout=15)
        if r.status_code != 200:
            print("Telegram error:", r.text)
    except Exception as e:
        print("Telegram send failed:", e)

def tg_poll_commands():
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                         params={"offset": STATE["tg_offset"], "timeout": 0}, timeout=15).json()
    except Exception:
        return
    for u in r.get("result", []):
        STATE["tg_offset"] = u["update_id"] + 1
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != str(TG_CHAT):
            continue  # ignore strangers
        cmd = (msg.get("text") or "").strip().split()[0].lower() if msg.get("text") else ""
        if cmd.startswith("/status"):
            last = STATUS["last_scan"].astimezone(LOCAL_TZ).strftime("%H:%M:%S") if STATUS["last_scan"] else "never"
            n = sum(len(v) for v in STATUS["zones"].values())
            tg_send(f"✅ Running\nSymbols: {', '.join(SYMBOLS)} (5M)\nLast scan: {last}\n"
                    f"Fresh FVGs being watched: {n}\nPoll every {POLL_SEC}s")
        elif cmd.startswith("/zones"):
            lines = []
            for sym, zs in STATUS["zones"].items():
                for z in zs:
                    lines.append(f"{'🟢' if z['dir'] > 0 else '🔴'} {sym} {fmt(z['bot'])}–{fmt(z['top'])} "
                                 f"({z['label']} {z['kind']}, formed {ts(z['t'])})")
            tg_send("Fresh FVGs (not yet tapped):\n" + ("\n".join(lines) if lines else "None right now"))
        elif cmd.startswith("/help") or cmd.startswith("/start"):
            tg_send("Commands:\n/status - is the bot alive\n/zones - FVGs waiting for price\n/help - this list")
    save_state(STATE)

# ============================ DATA ============================
def fetch(symbol):
    """Returns (closed candles dict of arrays, live candle dict or None)."""
    rows = []
    if SOURCE == "binance":
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol.replace("/", ""), "interval": "5m", "limit": BARS},
                         timeout=15).json()
        if isinstance(r, dict):
            raise RuntimeError(f"Binance {symbol}: {r}")
        for k in r:
            rows.append((int(k[0]) // 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4])))
    else:
        r = requests.get("https://api.twelvedata.com/time_series",
                         params={"symbol": symbol, "interval": "5min", "outputsize": BARS,
                                 "timezone": "UTC", "apikey": TD_KEY}, timeout=15).json()
        if r.get("status") != "ok":
            raise RuntimeError(f"Twelve Data {symbol}: {r.get('message', r)}")
        for v in r["values"]:
            t = int(datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
            rows.append((t, float(v["open"]), float(v["high"]), float(v["low"]), float(v["close"])))
        rows.sort()

    now = time.time()
    live = None
    if rows and rows[-1][0] + TF_SEC > now:
        t, o, h, l, c = rows.pop()
        live = {"t": t, "o": o, "h": h, "l": l, "c": c}
    a = np.array(rows, dtype=float)
    return {"t": a[:, 0].astype(int), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4]}, live

# ============================ STRATEGY ============================
def atr(d, n=14):
    h, l, c = d["h"], d["l"], d["c"]
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    out = np.convolve(tr, np.ones(n) / n, mode="full")[: len(tr)]
    out[: n - 1] = tr[: n - 1].mean() if n > 1 else tr[: n - 1]
    return out

def structure_breaks(d, L, label):
    """Detect BOS/CHoCH with pivots of strength L. No look-ahead: a pivot at k
    is only known once k+L candles have closed."""
    h, l, c = d["h"], d["l"], d["c"]
    n = len(c)
    events, ph, pl = [], None, None
    ph_used = pl_used = True
    trend = 0
    for j in range(n):
        k = j - L
        if k >= L:
            if h[k] == h[k - L:k + L + 1].max():
                ph, ph_used = (k, h[k]), False
            if l[k] == l[k - L:k + L + 1].min():
                pl, pl_used = (k, l[k]), False
        if ph and not ph_used and c[j] > ph[1]:
            events.append({"dir": 1, "kind": "CHoCH" if trend == -1 else "BOS", "label": label,
                           "idx": j, "level": ph[1], "origin": ph[0] + int(np.argmin(l[ph[0]:j + 1]))})
            trend, ph_used = 1, True
        if pl and not pl_used and c[j] < pl[1]:
            events.append({"dir": -1, "kind": "CHoCH" if trend == 1 else "BOS", "label": label,
                           "idx": j, "level": pl[1], "origin": pl[0] + int(np.argmax(h[pl[0]:j + 1]))})
            trend, pl_used = -1, True
    return events

def find_zones(d):
    """For each structure break, take the most recent FVG in the impulse leg
    that caused it (from leg origin up to the candle after the break)."""
    h, l, c, t = d["h"], d["l"], d["c"], d["t"]
    n = len(c)
    a = atr(d)
    events = structure_breaks(d, INTERNAL_LEN, "Internal") + structure_breaks(d, SWING_LEN, "Swing")
    zones = {}
    for ev in events:
        best = None
        for i in range(max(ev["origin"] + 2, 2), min(ev["idx"] + 1, n - 1) + 1):
            if ev["dir"] > 0 and l[i] > h[i - 2]:
                best = (i, h[i - 2], l[i])            # bottom, top
            elif ev["dir"] < 0 and h[i] < l[i - 2]:
                best = (i, h[i], l[i - 2])
        if not best:
            continue
        i, bot, top = best
        if top - bot < MIN_FVG_ATR * a[i]:
            continue
        zid = f"{ev['dir']}|{t[i]}"
        # Swing label wins if both pivot sizes produce the same gap
        if zid in zones and zones[zid]["label"] == "Swing":
            continue
        zones[zid] = {"id": zid, "dir": ev["dir"], "i": i, "t": int(t[i]), "bot": bot, "top": top,
                      "kind": ev["kind"], "label": ev["label"], "level": ev["level"],
                      "break_t": int(t[ev["idx"]])}

    # Evaluate each zone against candles AFTER it formed
    for z in zones.values():
        z["touch"], z["dead"] = None, None
        for k in range(z["i"] + 1, n):
            if z["dir"] > 0 and l[k] <= z["top"]:
                z["touch"] = k
                break
            if z["dir"] < 0 and h[k] >= z["bot"]:
                z["touch"] = k
                break
        if z["touch"] is None and (n - 1 - z["i"]) > MAX_AGE_BARS:
            z["dead"] = "expired"
    return list(zones.values())

# ============================ ALERTS ============================
def fmt(x):
    if x >= 1000:
        return f"{x:.2f}"   # gold, indices, BTC
    if x >= 20:
        return f"{x:.3f}"   # JPY pairs, silver
    return f"{x:.5f}"       # majors

def ts(epoch):
    return datetime.fromtimestamp(epoch, LOCAL_TZ).strftime("%H:%M")

def entry_msg(sym, z, price, on_live):
    bull = z["dir"] > 0
    return (f"{'🟢 BULLISH' if bull else '🔴 BEARISH'} FVG TAPPED | {sym} 5M\n\n"
            f"Structure: {z['label']} {z['kind']} {'above' if bull else 'below'} {fmt(z['level'])} "
            f"(closed {ts(z['break_t'] + TF_SEC)})\n"
            f"FVG: {fmt(z['bot'])} – {fmt(z['top'])}\n"
            f"Price now: {fmt(price)}{'' if on_live else ' (tapped on last closed candle)'}\n\n"
            f"Watch for confirmation that the gap holds.\n"
            f"Invalid on a 5M close {'below ' + fmt(z['bot']) if bull else 'above ' + fmt(z['top'])}.")

def confirm_msg(sym, z, close):
    bull = z["dir"] > 0
    return (f"✅ FVG RESPECTED | {sym} 5M\n"
            f"5M candle closed {'back above' if bull else 'back below'} the FVG "
            f"({fmt(z['bot'])} – {fmt(z['top'])}) at {fmt(close)}.\n"
            f"Your call: check the chart before entering.")

def process(sym, d, live):
    n = len(d["c"])
    zones = find_zones(d)
    fresh = []
    for z in zones:
        zid = f"{sym}|{z['id']}"
        bull = z["dir"] > 0
        if z["dead"]:
            continue
        # 1) Price enters the FVG
        if zid not in STATE["alerted"]:
            hit_live = (z["touch"] is None and live is not None and
                        ((bull and live["l"] <= z["top"]) or (not bull and live["h"] >= z["bot"])))
            hit_last = (z["touch"] == n - 1 and
                        not ((bull and d["c"][n - 1] < z["bot"]) or (not bull and d["c"][n - 1] > z["top"])))
            if hit_live or hit_last:
                price = live["c"] if (hit_live and live) else d["c"][n - 1]
                tg_send(entry_msg(sym, z, price, hit_live))
                STATE["alerted"].append(zid)
            elif z["touch"] is None:
                fresh.append(z)
        # 2) Optional: FVG respected (5M close back out in break direction)
        if CONFIRM_ALERT and zid in STATE["alerted"] and zid not in STATE["confirmed"] and z["touch"] is not None:
            for k in range(z["touch"], n):
                ck = d["c"][k]
                if (bull and ck < z["bot"]) or (not bull and ck > z["top"]):
                    STATE["confirmed"].append(zid)      # failed, stop watching
                    break
                if (bull and ck > z["top"]) or (not bull and ck < z["bot"]):
                    if k == n - 1:
                        tg_send(confirm_msg(sym, z, ck))
                    STATE["confirmed"].append(zid)
                    break
    STATUS["zones"][sym] = fresh

def forex_closed():
    if SOURCE != "twelvedata" or not FOREX_WEEKEND_PAUSE:
        return False
    u = datetime.now(timezone.utc)
    return u.weekday() == 5 or (u.weekday() == 4 and u.hour >= 21) or (u.weekday() == 6 and u.hour < 21)

# ============================ MAIN LOOP ============================
def main():
    if SOURCE == "twelvedata" and not TD_KEY:
        raise SystemExit("SOURCE=twelvedata needs TWELVEDATA_KEY in .env")
    tg_send(f"🤖 FVG bot started\nWatching: {', '.join(SYMBOLS)} on 5M\nPoll: every {POLL_SEC}s\nSend /help for commands.")
    first = True
    while True:
        start = time.time()
        tg_poll_commands()
        if forex_closed():
            STATUS["last_scan"] = datetime.now(timezone.utc)
            time.sleep(300)
            continue
        for idx, sym in enumerate(SYMBOLS):
            try:
                d, live = fetch(sym)
                if first:
                    # Don't fire on taps that already happened before the bot started.
                    for z in find_zones(d):
                        if z["touch"] is not None or (live and ((z["dir"] > 0 and live["l"] <= z["top"]) or
                                                               (z["dir"] < 0 and live["h"] >= z["bot"]))):
                            zid = f"{sym}|{z['id']}"
                            if zid not in STATE["alerted"]:
                                STATE["alerted"].append(zid)
                                STATE["confirmed"].append(zid)
                process(sym, d, live)
            except Exception as e:
                STATUS["errors"] += 1
                print(f"[{sym}] error: {e}")
                traceback.print_exc()
            if SOURCE == "twelvedata" and idx < len(SYMBOLS) - 1:
                time.sleep(8)  # free plan: max 8 requests/minute
        first = False
        STATUS["last_scan"] = datetime.now(timezone.utc)
        save_state(STATE)
        time.sleep(max(5, POLL_SEC - (time.time() - start)))

if __name__ == "__main__":
    main()
