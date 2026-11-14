#!/usr/bin/env python3
"""
Fibonacci geri çekilme backtesti - ABD hisseleri, 5 dakikalık bar.

Mantık (hepsi nedensel, yani geleceğe bakmadan):
  1. Her işlem günü ayrı işlenir (gün içi, geceye pozisyon taşınmaz).
  2. Swing tepe/dip: bir bar, solundaki ve sağındaki k barın en yükseği/en düşüğüyse pivot;
     pivot ancak k bar sonra "onaylanmış" sayılır, işlem ancak ondan sonra aranır.
  3. Bacak (leg): onaylı bir tepeye kadar olan yükseliş (long) ya da onaylı bir dibe kadar
     olan düşüş (short). Fibonacci seviyeleri bu bacağın geri çekilmesine çizilir.
  4. Giriş: fiyat seviyeye limit emirle değer. Stop: bacağın başlangıcı. Hedef: bacağın ucu.
     Gün sonu ya da max_hold bar dolunca kapanıştan çıkış. Sonuç R katı olarak ölçülür.
  5. Karşılaştırma: aynı bacaklarda Fibonacci yerine rastgele oranlar (0.236-0.786 arası)
     denenir. Fibonacci rastgeleden iyi değilse seviyelerde özel bir şey yok demektir.

Kurulum:   pip install yfinance pandas numpy
Çalıştır:  python fibo_backtest.py
           python fibo_backtest.py --symbols AAPL MSFT NVDA --reps 100
Uzun veri: yfinance 5 dakikalıkta yalnızca son ~60 günü verir. Daha uzun test için
           her hisse için SEMBOL.csv hazırla (ilk sütun zaman, sonra open,high,low,close)
           ve:  python fibo_backtest.py --csv-dir veri --symbols AAPL MSFT
           Saat dilimi bilgisi yoksa zaman damgaları New York saati kabul edilir.

Varsayımlar: stop fiyatından doldurulur (gap'te kayma yok), komisyon+spread
--cost-bps ile gidiş-dönüş toplam olarak düşülür. Bu yatırım tavsiyesi değildir.
"""
import argparse
import os

import numpy as np
import pandas as pd

FIB = [0.236, 0.382, 0.5, 0.618, 0.786]
EXT = [1.272, 1.618, 2.0]  # TP2, TP3, TP4 uzantıları (TP1 = bacağın ucu); bot.py ile aynı
TZ = "America/New_York"


# ----------------------------------------------------------------------------- veri
def load(sym, csv_dir, period):
    if csv_dir:
        df = pd.read_csv(os.path.join(csv_dir, f"{sym}.csv"), index_col=0)
        s = df.index.astype(str)
        if s.str.contains(r"(?:[+-]\d{2}:\d{2}|Z)$").any():
            idx = pd.to_datetime(s, utc=True).tz_convert(TZ)
        else:
            idx = pd.to_datetime(s).tz_localize(TZ)
    else:
        import yfinance as yf

        df = yf.download(sym, period=period, interval="5m", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize("UTC").tz_convert(TZ) if idx.tz is None else idx.tz_convert(TZ)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = idx
    df = df[["open", "high", "low", "close"]].dropna().sort_index()
    return df.between_time("09:30", "15:55")  # sadece normal seans


def resample(df, tf):
    """5 dakikalık barları tf dakikalığa birleştirir (seans 09:30'a hizalı)."""
    if tf <= 5:
        return df
    return (
        df.resample(f"{tf}min", offset="30min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def day_views(df, tf=5):
    """Her gün için iki görünüm: (long için orijinal, short için ters çevrilmiş)."""
    out = []
    for day, g in df.groupby(df.index.date):
        if len(g) < max(8, 150 // tf):  # eksik gün
            continue
        o, h, l, c = (g[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        out.append((day, ((o, h, l, c), (-o, -l, -h, -c))))
    return out


# ----------------------------------------------------------------------------- bacaklar
def find_legs(views, k, min_pct):
    """Nedensel bacak listesi: (view_id, lo, hi, start_bar). View 1'de short da 'long' gibi görünür."""
    O, H, L, C = views[0]
    n = len(H)
    legs, prev_h, prev_l = [], [], []
    for i in range(k, n - k):
        wh, wl = H[i - k : i + k + 1], L[i - k : i + k + 1]
        if H[i] == wh.max() and H[i] > wh.min():  # swing tepe -> long bacağı
            s = prev_h[-1] if prev_h else 0
            lo = L[s : i + 1].min()
            hi = H[i]
            if hi > lo and (hi - lo) / hi * 100 >= min_pct:
                legs.append((0, lo, hi, i + k + 1))
            prev_h.append(i)
        if L[i] == wl.min() and L[i] < wl.max():  # swing dip -> short bacağı
            s = prev_l[-1] if prev_l else 0
            top = H[s : i + 1].max()
            bot = L[i]
            if top > bot and (top - bot) / top * 100 >= min_pct:
                legs.append((1, -top, -bot, i + k + 1))  # ters görünümde: lo=-top, hi=-bot
            prev_l.append(i)
    return legs


# ----------------------------------------------------------------------------- işlem
def trade(V, start, level, lo, hi, max_wait, max_hold, cost_bps):
    O, H, L, C = V
    n = len(C)
    j = -1
    for t in range(start, min(n, start + max_wait)):
        if L[t] <= level < O[t]:  # limit emir dolar
            j = t
            break
        if L[t] <= lo or H[t] > hi:  # bacak bozuldu / yeni uç yapıldı, işlem yok
            return None
    if j < 0:
        return None
    risk = level - lo
    if risk <= 0:
        return None
    cost = cost_bps * 1e-4 * level / risk
    if L[j] <= lo:  # aynı barda stop
        return -1.0 - cost
    last = min(n - 1, j + max_hold)
    r = None
    for t in range(j + 1, last + 1):
        if L[t] <= lo:  # aynı barda hem stop hem hedef olursa stop sayılır (temkinli)
            r = -1.0
            break
        if H[t] >= hi:
            r = (hi - level) / risk
            break
    if r is None:
        r = (C[last] - level) / risk
    return r - cost


def trade_detail(V, start, level, lo, hi, max_wait, max_hold, cost_bps):
    """trade() ile aynı giriş/stop mantığı. Ek olarak stop'a değmeden kaç hedefe (TP1-4) ulaşıldığını verir.
    Dönüş: (ulaşılan TP sayısı 0-4, stop oldu mu, [TP1..TP4'ün her birinde tamamen çıkılsa R]) ya da None."""
    O, H, L, C = V
    n = len(C)
    j = -1
    for t in range(start, min(n, start + max_wait)):
        if L[t] <= level < O[t]:
            j = t
            break
        if L[t] <= lo or H[t] > hi:
            return None
    if j < 0:
        return None
    risk = level - lo
    if risk <= 0:
        return None
    cost = cost_bps * 1e-4 * level / risk
    span = hi - lo
    tps = [hi] + [lo + e * span for e in EXT]
    if L[j] <= lo:
        return 0, True, [-1.0 - cost] * 4
    last = min(n - 1, j + max_hold)
    reached, stopped = 0, False
    for t in range(j + 1, last + 1):
        if L[t] <= lo:  # aynı barda hem stop hem hedef olursa stop sayılır (temkinli)
            stopped = True
            break
        while reached < 4 and H[t] >= tps[reached]:
            reached += 1
    rs = []
    for k in range(4):
        if reached > k:
            rs.append((tps[k] - level) / risk - cost)
        elif stopped:
            rs.append(-1.0 - cost)
        else:
            rs.append((C[last] - level) / risk - cost)
    return reached, stopped, rs


def tp_report(legs, a):
    rows = []  # (gün, oran, ulaşılan tp, stop, [R x4])
    for views, v, lo, hi, st, day in legs:
        for r in FIB:
            d = trade_detail(views[v], st, hi - r * (hi - lo), lo, hi, a.max_wait, a.max_hold, a.cost_bps)
            if d is not None:
                rows.append((day, r) + d)
    if not rows:
        print("\nTP raporu: işlem yok.")
        return

    def block(name, sel):
        if not sel:
            return
        n = len(sel)
        pre = np.mean([x[3] and x[2] == 0 for x in sel]) * 100
        tout = np.mean([(not x[3]) and x[2] == 0 for x in sel]) * 100
        hit = [np.mean([x[2] >= k for x in sel]) * 100 for k in range(1, 5)]
        avg = [np.mean([x[4][k] for x in sel]) for k in range(4)]
        print(f"{name}  n={n}")
        print(f"  TP1 öncesi stop %{pre:.0f} | süre dolumu %{tout:.0f}")
        print("  hedefe ulaşma: " + " ".join(f"TP{k + 1} %{hit[k]:.0f}" for k in range(4)))
        print("  ort.R (hepsi TPn'de kapanırsa):")
        print("   " + " ".join(f"TP{k + 1} {avg[k]:+.2f}" for k in range(4)))

    print("\n=== Hedef bazında başarı (hedefe stop'tan ÖNCE ulaşan işlem oranı) ===")
    for r in FIB:
        block(f"Fib %{r * 100:.1f}", [x for x in rows if x[1] == r])
    block("TÜMÜ", rows)

    days = sorted(set(x[0] for x in rows))
    span = (days[-1] - days[0]).days
    keyf = (lambda d: str(d.year)) if span > 540 else (lambda d: f"{d.year}-{d.month:02d}")
    groups = {}
    for x in rows:
        groups.setdefault(keyf(x[0]), []).append(x[4][0])
    print("\n=== Dönem bazında (TP1'de çıkış, tüm Fib seviyeleri) ===")
    for k in sorted(groups):
        g = np.array(groups[k])
        print(f"  {k} n={len(g):5d} kazanma(R>0) %{np.mean(g > 0) * 100:3.0f} ort.R {g.mean():+.3f}")
    print("  Sonuç dönemden döneme tutarlı değilse tek bir rejime uymuş olabilir.")


def summarize(x):
    x = np.asarray(x)
    if len(x) == 0:
        return "n=0"
    ci = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return f"n={len(x):5d}  kazanma={np.mean(x > 0) * 100:5.1f}%  ort.R={x.mean():+.3f} (±{ci:.3f})"


# ----------------------------------------------------------------------------- ana akış
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "SPY", "QQQ"])
    p.add_argument("--csv-dir", default=None)
    p.add_argument("--period", default="60d")
    p.add_argument("--k", type=int, default=3, help="pivot için sağ/sol bar sayısı")
    p.add_argument("--min-pct", type=float, default=0.3, help="minimum bacak boyu, fiyatın %%'si")
    p.add_argument("--tf", type=int, default=5, help="bar süresi, dakika (5, 15, 30, 60)")
    p.add_argument("--max-wait", type=int, default=None, help="seviyeye temas için en fazla bar (varsayılan: 2 saat)")
    p.add_argument("--max-hold", type=int, default=None, help="pozisyonda en fazla bar (varsayılan: 2 saat)")
    p.add_argument("--cost-bps", type=float, default=2.0, help="gidiş-dönüş maliyet, baz puan")
    p.add_argument("--reps", type=int, default=50, help="rastgele seviye tekrar sayısı")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.tf < 5 or a.tf % 5 != 0:
        p.error("--tf 5'in katı olmalı")
    if a.max_wait is None:
        a.max_wait = max(4, 120 // a.tf)
    if a.max_hold is None:
        a.max_hold = max(4, 120 // a.tf)
    rng = np.random.default_rng(a.seed)

    print(f"Zaman dilimi: {a.tf} dk | bekleme/tutma: {a.max_wait}/{a.max_hold} bar | maliyet: {a.cost_bps} bp")
    legs = []  # (views, view_id, lo, hi, start)
    for sym in a.symbols:
        try:
            df = resample(load(sym, a.csv_dir, a.period), a.tf)
        except Exception as e:
            print(f"{sym}: okunamadı ({e})")
            continue
        days = day_views(df, a.tf)
        cnt = 0
        for day, views in days:
            for v, lo, hi, st in find_legs(views, a.k, a.min_pct):
                legs.append((views, v, lo, hi, st, day))
                cnt += 1
        print(f"{sym}: {len(days)} gün, {cnt} bacak")
    if not legs:
        print("Hiç bacak bulunamadı.")
        return

    def run(ratios_fn):
        res = []
        for views, v, lo, hi, st, _day in legs:
            for r in ratios_fn():
                level = hi - r * (hi - lo)
                R = trade(views[v], st, level, lo, hi, a.max_wait, a.max_hold, a.cost_bps)
                if R is not None:
                    res.append((r, R))
        return res

    fib = run(lambda: FIB)
    print("\n=== Fibonacci seviyeleri ===")
    for r in FIB:
        print(f"  %{r * 100:5.1f}  {summarize([x for rr, x in fib if rr == r])}")
    fib_R = [x for _, x in fib]
    print(f"  TÜMÜ    {summarize(fib_R)}")
    tp_report(legs, a)

    print(f"\n=== Rastgele seviyeler ({a.reps} tekrar) hesaplanıyor... ===")
    means = []
    for _ in range(a.reps):
        res = run(lambda: rng.uniform(FIB[0], FIB[-1], size=len(FIB)))
        if res:
            means.append(np.mean([x for _, x in res]))
    means = np.array(means)
    fm = float(np.mean(fib_R)) if fib_R else float("nan")
    print(f"  Rastgele ort.R: {means.mean():+.3f}  (5-95 yüzdelik: {np.percentile(means, 5):+.3f} .. {np.percentile(means, 95):+.3f})")
    print(f"  Fibonacci ort.R: {fm:+.3f}  -> rastgele dağılımda %{np.mean(means < fm) * 100:.0f}. yüzdelikte")
    print("\nYorum: Fibonacci, rastgele dağılımın üst %95'inin belirgin üstünde değilse")
    print("seviyelerin rastgele oranlardan farkı yoktur. Ort.R negatifse maliyet sonrası kaybettiriyor demektir.")
    print("Not: işlemler birbirinden bağımsız değil (örtüşen bacaklar), güven aralıkları iyimser kalır.")


if __name__ == "__main__":
    main()
