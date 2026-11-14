#!/usr/bin/env python3
"""
Fibonacci sinyal botu -> Telegram.

Her çalıştığında (GitHub Actions ile 5 dakikada bir) şunu yapar:
  1. Her hisse için bugünün TAMAMLANMIŞ 5 dakikalık barlarını alır.
  2. fibo_backtest.py'deki aynı nedensel swing/bacak mantığıyla Fibonacci seviyelerini çizer.
  3. Fiyat son birkaç bar içinde bir seviyeye değdiyse sinyal üretir:
       Giriş (limit) = dokunulan Fibonacci seviyesi
       Stop          = bacağın başlangıcı
       TP1           = bacağın ucu (önceki tepe/dip)
       TP2, TP3, TP4 = bacağın 1,272 / 1,618 / 2,0 uzantısı
  4. Aynı sinyali iki kez göndermemek için state.json tutar.

Gerekli ortam değişkenleri:  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
Deneme (mesaj atmaz):        python bot.py --dry-run
Geçmiş bir ana oynatma:      python bot.py --dry-run --asof "2026-11-12 11:30"

Not: Bu yatırım tavsiyesi değildir. Strateji backtest ile doğrulanmadan gerçek parayla kullanılmamalı.
"""
import argparse
import json
import os

import pandas as pd
from zoneinfo import ZoneInfo

from fibo_backtest import find_legs, load

ET = ZoneInfo("America/New_York")
TR = ZoneInfo("Europe/Istanbul")
EXT = [1.272, 1.618, 2.0]  # TP2, TP3, TP4 uzantıları
DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,AMZN,TSLA,META,GOOGL,AMD,SPY,QQQ"


def views_of(g):
    """Long için orijinal, short için ters çevrilmiş görünüm."""
    o, h, l, c = (g[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    return (o, h, l, c), (-o, -l, -h, -c)


def scan(sym, g, a):
    views = views_of(g)
    n = len(g)
    out = []
    for v, lo, hi, st in find_legs(views, a.k, a.min_pct):
        if a.long_only and v == 1:
            continue
        O, H, L, C = views[v]
        sign = 1 if v == 0 else -1
        for r in a.levels:
            level = hi - r * (hi - lo)
            j = -1
            for t in range(st, min(n, st + a.max_wait)):
                if L[t] <= level < O[t]:  # limit emir dolar
                    j = t
                    break
                if L[t] <= lo or H[t] > hi:  # bacak bozuldu ya da yeni uç yapıldı
                    break
            if j < 0 or j < n - a.recent:  # sadece taze temaslar
                continue
            risk = level - lo
            if risk <= 0 or L[j] <= lo:
                continue
            if any(L[t] <= lo or H[t] >= hi for t in range(j + 1, n)):  # stop/TP1 zaten gerçekleşmiş
                continue
            tps = [hi] + [lo + e * (hi - lo) for e in EXT]
            direction = "LONG" if v == 0 else "SHORT"
            out.append(
                dict(
                    sym=sym,
                    dir=direction,
                    r=r,
                    entry=sign * level,
                    stop=sign * lo,
                    tps=[sign * x for x in tps],
                    rr1=(hi - level) / risk,
                    bar=g.index[j],
                    key=f"{sym}|{direction}|{sign * lo:.2f}|{sign * hi:.2f}|{r}",
                )
            )
    return out


def message(s):
    emoji = "🟢" if s["dir"] == "LONG" else "🔴"
    t_et = s["bar"]
    t_tr = t_et.tz_convert(TR)
    ratio = f"{s['r'] * 100:.1f}".replace(".", ",")
    lines = [
        f"{emoji} {s['sym']} - {s['dir']} (Fib %{ratio})",
        f"Giriş (limit): {s['entry']:.2f}",
        f"Stop: {s['stop']:.2f}",
        f"TP1: {s['tps'][0]:.2f}",
        f"TP2: {s['tps'][1]:.2f}",
        f"TP3: {s['tps'][2]:.2f}",
        f"TP4: {s['tps'][3]:.2f}",
        f"TP1 R:R = {s['rr1']:.2f}",
        f"Temas barı: {t_et:%H:%M} ET / {t_tr:%H:%M} TR",
        "Fiyat girişten uzaklaştıysa işleme girme. Yatırım tavsiyesi değildir.",
    ]
    return "\n".join(lines)


def make_sender(a):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")

    def send(text):
        if a.dry_run:
            print("-----\n" + text)
            return True
        if not (token and chat):
            print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil, mesaj gönderilmedi.")
            return False
        import requests

        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text},
            timeout=20,
        )
        if not r.ok:
            print("Telegram hatası:", r.status_code, r.text[:200])
            return False
        return True

    return send


def run_once(a, sent, now, send):
    today = str(now.date())
    sent = {k for k in sent if k.split("|", 1)[0] == today}  # eski günlerin kayıtlarını at
    msgs = []
    for sym in a.symbols:
        try:
            df = load(sym, a.csv_dir, "5d")
        except Exception as e:
            print(f"{sym}: okunamadı ({e})")
            continue
        df = df[df.index + pd.Timedelta(minutes=5) <= now]  # sadece tamamlanmış barlar
        g = df[df.index.date == now.date()]
        if len(g) < 2 * a.k + 2:
            continue
        lag = now - (g.index[-1] + pd.Timedelta(minutes=5))
        if lag > pd.Timedelta(minutes=a.max_lag):
            print(f"{sym}: veri {lag} gecikmeli, atlandı")
            continue
        for s in scan(sym, g, a):
            key = f"{today}|{s['key']}"
            if key in sent:
                continue
            text = message(s)
            if send(text):
                sent.add(key)
                msgs.append(text)
    return sent, msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=os.environ.get("SYMBOLS", DEFAULT_SYMBOLS), help="virgülle ayrılmış liste")
    p.add_argument("--levels", default=os.environ.get("LEVELS", "0.382,0.5,0.618"), help="Fibonacci oranları")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--min-pct", type=float, default=0.3)
    p.add_argument("--max-wait", type=int, default=24)
    p.add_argument("--recent", type=int, default=3, help="son kaç bardaki temaslar sinyal sayılır")
    p.add_argument("--max-lag", type=int, default=20, help="veri en fazla kaç dakika gecikmeli olabilir")
    p.add_argument("--long-only", action="store_true", default=os.environ.get("LONG_ONLY") == "1")
    p.add_argument("--csv-dir", default=None)
    p.add_argument("--asof", default=None, help="ET saatiyle 'YYYY-MM-DD HH:MM' (oynatma/deneme)")
    p.add_argument("--state", default="state.json")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    a.levels = [float(x) for x in a.levels.split(",")]

    now = pd.Timestamp(a.asof).tz_localize(ET) if a.asof else pd.Timestamp.now(tz=ET)
    sent = set(json.load(open(a.state))) if os.path.exists(a.state) else set()
    sent, msgs = run_once(a, sent, now, make_sender(a))
    if not a.dry_run:
        with open(a.state, "w") as f:
            json.dump(sorted(sent), f)
    print(f"{len(msgs)} sinyal gönderildi.")


if __name__ == "__main__":
    main()
