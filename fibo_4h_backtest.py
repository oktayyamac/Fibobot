#!/usr/bin/env python3
"""
4 saatlik grafikte Fibonacci geri çekilme backtesti (küçük hisseler için). İşlem gün sonunda kapanır.

Mantık (hepsi nedensel, geleceğe bakmaz):
  1. 5 dakikalık veriden 4 saatlik bar üretilir (her gün 09:30-13:30 ve 13:30-16:00), günler arası tek seri.
  2. Swing tepe: solundaki ve sağındaki k barın en yükseği; k bar sonra onaylanır.
     Bacak = önceki swing tepeden bu tepeye kadarki en düşük dipten, tepeye kadar olan yükseliş.
  3. Fib seviyesi = tepe - oran * (tepe - dip). Bacak onaylandıktan sonraki --wait-days işlem günü boyunca
     her gün bu seviyede bir limit alım emri bekler (gün sonunda biter, ertesi gün yenilenir).
     Fiyat seviyeye inerse dolar. Dolmadan önce tepe aşılırsa işlem açılmaz.
  4. Çıkış (AYNI GÜN): hedef = girişin %--target üstü; stop = bacağın dibi (ya da --stop-pct);
     ikisi de olmazsa seans kapanışı. Aynı barda hem stop hem hedef olursa stop sayılır (temkinli).
  5. Aynı bacaklarda rastgele oranlarla karşılaştırma: Fibonacci rastgeleden iyi değilse seviyelerde özel bir şey yok.

Sadece long (küçük hisselerde açığa satış çoğu zaman mümkün değil).
Getiri yüzde olarak, maliyet (gidiş-dönüş, baz puan) düşüldükten sonra verilir.
Kullanım: python fibo_4h_backtest.py --csv-dir veri --symbols SOUN PLUG RIOT
Not: Bu yatırım tavsiyesi değildir.
"""
import argparse

import numpy as np
import pandas as pd

from fibo_backtest import FIB, TZ, load


# ----------------------------------------------------------------------------- 4 saatlik bar ve bacaklar
def make_4h(df):
    """5 dk -> 4 saatlik (09:30-13:30 ve 13:30-16:00). Döndürür: yüksekler, düşükler, bar bitiş zamanları (UTC ns)."""
    t = df.index
    second = ((t.hour * 60 + t.minute) >= 13 * 60 + 30).astype(int)
    g = df.groupby([t.date, second])
    h, l = g["high"].max(), g["low"].min()
    ends = [
        pd.Timestamp(f"{d} {'13:30' if hf == 0 else '16:00'}").tz_localize(TZ).value
        for d, hf in h.index
    ]
    return h.to_numpy(float), l.to_numpy(float), np.array(ends, dtype=np.int64)


def find_legs(H, L, ends, k, min_pct):
    legs, prev = [], 0
    for i in range(k, len(H) - k):
        w = H[i - k : i + k + 1]
        if H[i] == w.max() and H[i] > w.min():  # swing tepe
            lo, hi = L[prev : i + 1].min(), H[i]
            if hi > lo and (hi - lo) / hi * 100 >= min_pct:
                legs.append((lo, hi, ends[i + k]))  # k bar kapandıktan sonra onaylı
            prev = i
    return legs


def to_ns(idx):
    """Zaman damgalarını her zaman nanosaniye (UTC) olarak ver; pandas sürümüne göre birim değişebiliyor."""
    try:
        return idx.as_unit("ns").asi8
    except AttributeError:
        return idx.asi8


def prepare_df(df, a):
    idx = df.index
    dates = np.array(idx.date)
    day = pd.factorize(dates)[0]
    day_end = np.r_[np.flatnonzero(np.diff(day)), len(day) - 1]
    mod = idx.hour * 60 + idx.minute
    H4, L4, ends = make_4h(df)
    return dict(
        ts=to_ns(idx),
        O=df["open"].to_numpy(float), H=df["high"].to_numpy(float),
        L=df["low"].to_numpy(float), C=df["close"].to_numpy(float),
        day=day, ndays=int(day[-1]) + 1, day_end=day_end, date=dates,
        early=np.asarray(mod < 15 * 60 + 30),  # gün 15:30'dan önce bitiyorsa (seyrek veri) işaretle
        legs=find_legs(H4, L4, ends, a.k, a.min_leg_pct),
    )


# ----------------------------------------------------------------------------- işlem
def simulate(S, level, lo, hi, avail, a):
    """Tek bir seviye için işlem. Dönüş: (getiri %, sonuç, stop mesafesi %, ay, erken_bitti) ya da None."""
    n = len(S["ts"])
    s = int(np.searchsorted(S["ts"], avail, side="left"))
    if s >= n:
        return None
    dl = min(S["day"][s] + a.wait_days - 1, S["ndays"] - 1)
    e = S["day_end"][dl] + 1
    touch = np.flatnonzero(S["L"][s:e] <= level)
    if touch.size == 0:
        return None
    ft = touch[0]
    hib = np.flatnonzero(S["H"][s:e] > hi)
    if hib.size and hib[0] < ft:  # seviyeye inmeden yeni tepe yapıldı
        return None
    j = s + ft
    E = min(level, S["O"][j])  # açılış seviyenin altındaysa daha iyi fiyattan dolar
    stop = E * (1 - a.stop_pct / 100) if a.stop_pct else lo
    if stop >= E:
        return None
    T = E * (1 + a.target / 100)
    de = S["day_end"][S["day"][j]]
    Lx, Hx = S["L"][j : de + 1], S["H"][j : de + 1]
    fs = np.flatnonzero(Lx <= stop)
    fh = np.flatnonzero(Hx[1:] >= T)  # giriş barında hedef sayılmaz (sıra belirsiz)
    fs0 = fs[0] if fs.size else None
    fh0 = fh[0] + 1 if fh.size else None
    if fs0 is not None and (fh0 is None or fs0 <= fh0):
        t = j + fs0
        ex = stop if fs0 == 0 else min(stop, S["O"][t])  # stop altında açılırsa açılıştan çıkılır
        out = "stop"
    elif fh0 is not None:
        ex, out = T, "hedef"
    else:
        ex, out = S["C"][de], "gün sonu"
    ret = (ex / E - 1) * 100 - a.cost_bps / 100
    return ret, out, (E - stop) / E * 100, str(S["date"][j])[:7], bool(S["early"][de])


def run(syms, ratios_fn, a):
    res = []
    for S in syms:
        for lo, hi, av in S["legs"]:
            for r in ratios_fn():
                x = simulate(S, hi - r * (hi - lo), lo, hi, av, a)
                if x is not None:
                    res.append((r,) + x)
    return res


# ----------------------------------------------------------------------------- rapor
def block(name, sel):
    if not sel:
        return
    ret = np.array([x[1] for x in sel])
    out = [x[2] for x in sel]
    pct = lambda o: np.mean([v == o for v in out]) * 100
    print(f"{name}  n={len(sel)}")
    print(f"  hedef %{pct('hedef'):.0f} | stop %{pct('stop'):.0f} | gün sonu %{pct('gün sonu'):.0f}")
    print(f"  ort.getiri %{ret.mean():+.2f} | medyan %{np.median(ret):+.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--csv-dir", default=None)
    p.add_argument("--period", default="60d", help="CSV yoksa Yahoo'dan alınacak süre")
    p.add_argument("--target", type=float, default=10.0, help="kâr hedefi, giriş fiyatının yüzdesi")
    p.add_argument("--stop-pct", type=float, default=0.0, help="sabit stop yüzdesi; 0 = bacağın dibi")
    p.add_argument("--cost-bps", type=float, default=20.0, help="gidiş-dönüş maliyet, baz puan")
    p.add_argument("--k", type=int, default=2, help="4 saatlik pivot için sağ/sol bar sayısı")
    p.add_argument("--min-leg-pct", type=float, default=5.0, help="minimum bacak boyu, tepenin %%'si")
    p.add_argument("--wait-days", type=int, default=3, help="onaydan sonra kaç işlem günü emir beklesin")
    p.add_argument("--reps", type=int, default=30, help="rastgele seviye tekrar sayısı")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    stop_txt = f"%{a.stop_pct:g}" if a.stop_pct else "bacak dibi"
    print(f"4 saatlik Fib | long | hedef %{a.target:g} | stop {stop_txt} | gün sonu çıkış | maliyet {a.cost_bps:g} bp")
    print(f"pivot k={a.k} | min bacak %{a.min_leg_pct:g} | emir {a.wait_days} gün bekler")
    syms = []
    for sym in a.symbols:
        try:
            df = load(sym, a.csv_dir, a.period)
            if len(df) < 200:
                print(f"{sym}: çok az veri ({len(df)} bar), atlandı")
                continue
            S = prepare_df(df, a)
        except Exception as e:
            print(f"{sym}: okunamadı ({e})")
            continue
        print(f"{sym}: {S['ndays']} gün, {len(S['legs'])} bacak")
        syms.append(S)

    fib = run(syms, lambda: FIB, a)
    if not fib:
        print("Hiç işlem oluşmadı.")
        return

    print("\n=== Seviye bazında (net getiri, işlem başına) ===")
    for r in FIB:
        block(f"Fib %{r * 100:.1f}", [x for x in fib if x[0] == r])
    block("TÜMÜ", fib)
    ret = np.array([x[1] for x in fib])
    ci = 1.96 * ret.std(ddof=1) / np.sqrt(len(ret)) if len(ret) > 1 else float("nan")
    sd = np.mean([x[3] for x in fib])
    early = np.mean([x[5] for x in fib]) * 100
    print(f"  TÜMÜ ort.getiri %{ret.mean():+.2f} (±{ci:.2f})")
    print(f"Ort. stop mesafesi %{sd:.1f}. Hedefe stop'tan önce")
    print(f"ulaşma oranı kabaca %{sd / (a.target + sd) * 100:.0f} üstündeyse kârlı (gün sonu")
    print("çıkışlar ve maliyet hariç).")
    print(f"Seyrek veri: işlemlerin %{early:.0f}'i 15:30'dan önce biten günlerde.")

    print("\n=== Dönem bazında (tüm seviyeler, net getiri) ===")
    groups = {}
    for x in fib:
        groups.setdefault(x[4], []).append((x[1], x[2]))
    for k in sorted(groups):
        g = groups[k]
        rr = np.array([v[0] for v in g])
        hit = np.mean([v[1] == "hedef" for v in g]) * 100
        print(f"  {k} n={len(g):4d} ort.getiri %{rr.mean():+.2f} hedef %{hit:.0f}")

    print(f"\n=== Rastgele seviyeler ({a.reps} tekrar) hesaplanıyor... ===")
    means = []
    for _ in range(a.reps):
        res = run(syms, lambda: rng.uniform(FIB[0], FIB[-1], size=len(FIB)), a)
        if res:
            means.append(np.mean([x[1] for x in res]))
    means = np.array(means)
    fm = float(ret.mean())
    print(f"  Rastgele ort.getiri: %{means.mean():+.2f} (5-95 yüzdelik: {np.percentile(means, 5):+.2f} .. {np.percentile(means, 95):+.2f})")
    print(f"  Fibonacci ort.getiri: %{fm:+.2f} -> rastgele dağılımda %{np.mean(means < fm) * 100:.0f}. yüzdelikte")
    print("\nYorum: Fibonacci, rastgele dağılımın üst %95'inin belirgin üstünde değilse")
    print("seviyelerin rastgele oranlardan farkı yoktur. Ort.getiri negatifse maliyet sonrası kaybettiriyor.")
    print("Not: işlemler birbirinden bağımsız değil (örtüşen bacaklar), güven aralıkları iyimser kalır.")


if __name__ == "__main__":
    main()
