#!/usr/bin/env python3
"""Bir metin dosyasını Telegram'a gönderir (uzunsa parçalara böler, sabit aralıklı yazıyla).
Kullanım: python send_report.py report.txt      (TELEGRAM_TOKEN ve TELEGRAM_CHAT_ID gerekir)"""
import html
import os
import sys

import requests


def chunks(text, limit=3500):
    out, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) + 1 > limit:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


def main():
    try:
        text = open(sys.argv[1], encoding="utf-8").read().strip()
    except OSError:
        text = ""
    text = text or "Rapor oluşmadı. GitHub'da Actions sekmesinden çalıştırma kaydına bak."
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        sys.exit("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil.")
    for part in chunks(text):
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": "<pre>" + html.escape(part) + "</pre>", "parse_mode": "HTML"},
            timeout=20,
        )
        if not r.ok:
            sys.exit(f"Telegram hatası: {r.status_code} {r.text[:200]}")


if __name__ == "__main__":
    main()
