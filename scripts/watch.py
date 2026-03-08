"""
scripts/watch.py
=================
AMACI:
    Paper trading sirasinda log dosyasini canli izler.
    Bot calisirken ikinci bir terminalde bu scripti ac:
        python scripts/watch.py

    Neler gosterir:
        - Son N tick ozeti (fiyat, sinyal, karar)
        - Sinyal dagilimi (AL/SAT/BEKLE kac kez)
        - Acilan/kapanan pozisyonlar
        - Equity degisimi

CALISTIRMAK ICIN:
    Terminal 1: python -m trading.main_loop --interval 3600
    Terminal 2: python scripts/watch.py
"""

from __future__ import annotations
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path
from datetime import datetime

# Proje koku
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR      = PROJECT_ROOT / "logs"

# Son N tick bilgisi bellekte tutulur
WINDOW = 20


def find_latest_log() -> Path | None:
    """En yeni log dosyasini bul."""
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_line(line: str) -> dict | None:
    """
    Log satirini parse eder, anlamli bilgi cikartir.
    Returns None eger ilgisiz bir satir ise.
    """
    result = {}

    # Timestamp: 2026-03-08 06:29:26
    ts_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if ts_match:
        result["ts"] = ts_match.group(1)

    # Fiyat + rejim
    if "Fiyat:" in line and "Rejim:" in line:
        price_m = re.search(r"Fiyat: \$([0-9,]+\.\d+)", line)
        regime_m = re.search(r"Rejim: (\S+)", line)
        if price_m and regime_m:
            result["type"]   = "tick"
            result["price"]  = price_m.group(1)
            result["regime"] = regime_m.group(1)

    # Sinyal (RSI + PA + ML)
    elif "Sinyal:" in line and "Guven:" in line and "RSI:" in line:
        sig_m = re.search(r"Sinyal: (\w+) \| Guven: ([\d.]+)", line)
        rsi_m = re.search(r"RSI: (\w+) \(([\d.]+)\)", line)
        pa_m  = re.search(r"PA: (\w+) \(([\d.]+)\)", line)
        ml_m  = re.search(r"ML: (\w+) \(([\d.]+)\)", line)
        if sig_m:
            result["type"]       = "signal"
            result["action"]     = sig_m.group(1)
            result["confidence"] = float(sig_m.group(2))
            result["rsi"]        = f"{rsi_m.group(1)}({rsi_m.group(2)})" if rsi_m else "-"
            result["pa"]         = f"{pa_m.group(1)}({pa_m.group(2)})"   if pa_m  else "-"
            result["ml"]         = f"{ml_m.group(1)}({ml_m.group(2)})"   if ml_m  else "yok"

    # Risk karari
    elif "Risk karari:" in line:
        dec_m = re.search(r"TradeDecision\((\w+)", line)
        if dec_m:
            result["type"]     = "decision"
            result["approved"] = dec_m.group(1) == "ONAYLANDI"

    # Equity
    elif "Equity:" in line and "Unrealized" in line:
        eq_m  = re.search(r"Equity: \$([\d,]+\.\d+)", line)
        pnl_m = re.search(r"Unrealized PnL: \$([-\d,]+\.\d+)", line)
        if eq_m:
            result["type"]    = "equity"
            result["equity"]  = eq_m.group(1)
            result["pnl"]     = pnl_m.group(1) if pnl_m else "0.00"

    # Pozisyon acildi / kapandi
    elif "Pozisyon acildi" in line or "open_position" in line.lower():
        result["type"] = "pos_open"
    elif "kapandi" in line.lower() or "close_position" in line.lower():
        result["type"] = "pos_close"

    # Kill switch uyari
    elif "SARI" in line or "TURUNCU" in line or "KIRMIZI" in line:
        if "Kill" in line or "alarm" in line.lower():
            result["type"]  = "alert"
            result["level"] = "SARI" if "SARI" in line else ("TURUNCU" if "TURUNCU" in line else "KIRMIZI")
            result["msg"]   = line.strip()[-80:]

    return result if result.get("type") else None


def render(ticks: deque, signal_counts: Counter, trade_log: list) -> None:
    """Temizlenip yeniden yazilir."""
    # Terminal temizle (Windows + Unix)
    print("\033[2J\033[H", end="")

    now = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*65}")
    print(f"  ALGO TRADE CODEX — Paper Trading Izleme       [{now}]")
    print(f"{'='*65}")

    # Son tick bilgisi
    if ticks:
        last = ticks[-1]
        print(f"\n  Son Fiyat  : ${last.get('price', '?')}  |  Rejim: {last.get('regime', '?')}")
        print(f"  Son Sinyal : {last.get('action', '?')}  (Guven: {last.get('confidence', 0):.2f})")
        print(f"  RSI: {last.get('rsi','?')}  PA: {last.get('pa','?')}  ML: {last.get('ml','?')}")
        print(f"  Equity     : ${last.get('equity', '?')}  |  Unrealized: ${last.get('pnl', '0.00')}")

    # Sinyal dagilimi
    total = sum(signal_counts.values())
    if total > 0:
        print(f"\n  {'─'*45}")
        print(f"  Sinyal Dagilimi ({total} tick):")
        for act in ["AL", "SAT", "BEKLE"]:
            n = signal_counts.get(act, 0)
            bar = "#" * n
            pct = n / total * 100
            print(f"    {act:<6} : {bar:<20} {n:>3} ({pct:.0f}%)")

    # Islem gecmisi
    print(f"\n  {'─'*45}")
    print(f"  Son Islemler:")
    if trade_log:
        for t in trade_log[-5:]:
            print(f"    {t}")
    else:
        print("    Henuz islem yok.")

    print(f"\n{'='*65}")
    print("  [Ctrl+C ile cik]")


def tail_log(path: Path, callback) -> None:
    """Dosya sonundan canli okuma (tail -f gibi)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # Once dosya sonuna git
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                callback(line)
            else:
                time.sleep(0.5)


def main() -> None:
    log_path = find_latest_log()
    if not log_path:
        print(f"[HATA] {LOG_DIR} icinde log dosyasi bulunamadi.")
        print("Once botu calistir: python -m trading.main_loop --interval 3600")
        sys.exit(1)

    print(f"Log izleniyor: {log_path.name}")
    time.sleep(1)

    ticks: deque = deque(maxlen=WINDOW)   # Son WINDOW tick bilgisi
    signal_counts: Counter = Counter()
    trade_log: list = []
    current_tick: dict = {}

    def on_line(line: str) -> None:
        nonlocal current_tick
        parsed = parse_line(line)
        if not parsed:
            return

        t = parsed["type"]

        if t == "tick":
            current_tick = {**parsed}

        elif t == "signal":
            current_tick.update(parsed)
            signal_counts[parsed["action"]] += 1

        elif t == "equity":
            current_tick.update(parsed)
            ticks.append(dict(current_tick))
            render(ticks, signal_counts, trade_log)

        elif t == "pos_open":
            ts = parsed.get("ts", "?")
            trade_log.append(f"[{ts}] POZISYON ACILDI")

        elif t == "pos_close":
            ts = parsed.get("ts", "?")
            trade_log.append(f"[{ts}] POZISYON KAPATILDI")

        elif t == "alert":
            ts = parsed.get("ts", "?")
            trade_log.append(f"[{ts}] !!! ALARM: {parsed.get('level')} — {parsed.get('msg','')}")
            render(ticks, signal_counts, trade_log)

    try:
        tail_log(log_path, on_line)
    except KeyboardInterrupt:
        print("\nIzleme durduruldu.")


if __name__ == "__main__":
    main()
