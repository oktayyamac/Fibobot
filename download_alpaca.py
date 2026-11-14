#!/usr/bin/env python3
"""
Alpaca'dan 5 dakikalık geçmiş veriyi indirir ve fibo_backtest.py'nin okuduğu biçimde
SEMBOL.csv dosyaları olarak kaydeder.

Kullanım:
  export ALPACA_KEY=...  ALPACA_SECRET=...
  python download_alpaca.py --symbols AAPL MSFT NVDA --years 2 --out veri
  python fibo_backtest.py --csv-dir veri --symbols AAPL MSFT NVDA

Ücretsiz planda veri IEX borsasından gelir (feed=iex) ve son 15 dakika alınamaz;
bu yüzden bitiş tarihi dünden başlar. Fiyatlar bölünme/temettüye göre düzeltilmiştir.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

URL = "https://data.alpaca.markets/v2/stocks/bars"


def fetch(sym, start, end, feed, headers):
    rows, token = [], None
    while True:
        params = dict(
            symbols=sym, timeframe="5Min", start=start, end=end,
            limit=10000, adjustment="all", feed=feed, sort="asc",
        )
        if token:
            params["page_token"] = token
        for attempt in range(6):
            r = requests.get(URL, params=params, headers=headers, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(3 * (attempt + 1))
                continue
            break
        if r.status_code in (401, 403):
            sys.exit(f"{sym}: Alpaca erişimi reddetti ({r.status_code}): {r.text[:200]}")
        r.raise_for_status()
        j = r.json()
        rows += (j.get("bars") or {}).get(sym, [])
        token = j.get("next_page_token")
        if not token:
            return rows


def save(rows, path):
    df = pd.DataFrame(rows).rename(columns={"t": "datetime", "o": "open", "h": "high", "l": "low", "c": "close"})
    df[["datetime", "open", "high", "low", "close"]].to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument("--out", default="veri")
    p.add_argument("--feed", default="iex", choices=["iex", "sip"])
    a = p.parse_args()
    key, secret = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
    if not (key and secret):
        sys.exit("ALPACA_KEY ve ALPACA_SECRET tanımlı değil.")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=int(365 * a.years))
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    os.makedirs(a.out, exist_ok=True)
    for sym in a.symbols:
        sym = sym.upper()
        rows = fetch(sym, start.strftime(fmt), end.strftime(fmt), a.feed, headers)
        if not rows:
            print(f"{sym}: veri gelmedi")
            continue
        save(rows, os.path.join(a.out, f"{sym}.csv"))
        print(f"{sym}: {len(rows)} bar indirildi")


if __name__ == "__main__":
    main()
