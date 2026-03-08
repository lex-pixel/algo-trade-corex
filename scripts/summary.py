"""
scripts/summary.py
===================
AMACI:
    Bot durdurulmus olsa bile son durumu gosterir.
    data/bot_state.json dosyasini okur.

CALISTIRMAK ICIN:
    python scripts/summary.py

CIKTI:
    - Sermaye ve getiri
    - Kapanmis islem listesi (son 20)
    - Win rate, toplam P&L
    - Her istemin detayi
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
STATE_FILE   = PROJECT_ROOT / "data" / "bot_state.json"


def fmt_pnl(v: float) -> str:
    """Renk yoksa +/- isaretli format."""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}"


def main() -> None:
    if not STATE_FILE.exists():
        print(f"[HATA] State dosyasi bulunamadi: {STATE_FILE}")
        print("Bot hic calistirilmamis veya farkli dizinde.")
        sys.exit(1)

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        s = json.load(f)

    saved_at      = s.get("saved_at", "?")[:19].replace("T", " ")
    capital       = s.get("capital", 0)
    init_capital  = s.get("initial_capital", capital)
    equity_peak   = s.get("equity_peak", capital)
    max_dd        = s.get("max_drawdown", 0) * 100
    iteration     = s.get("iteration", 0)
    paper_mode    = s.get("paper", True)
    trades        = s.get("trades", [])
    open_pos      = s.get("open_positions", [])

    total_pnl     = sum(t["realized_pnl"] for t in trades)
    return_pct    = (capital - init_capital) / init_capital * 100 if init_capital else 0
    winners       = [t for t in trades if t["realized_pnl"] > 0]
    losers        = [t for t in trades if t["realized_pnl"] <= 0]
    win_rate      = len(winners) / len(trades) * 100 if trades else 0

    mod_str = "PAPER" if paper_mode else "LIVE"

    print(f"\n{'='*65}")
    print(f"  ALGO TRADE CODEX - Bot Ozeti  [{mod_str}]")
    print(f"  Son kayit: {saved_at} UTC  |  Tick: {iteration}")
    print(f"{'='*65}")

    # Sermaye ozeti
    print(f"\n  SERMAYE")
    print(f"  {'-'*45}")
    print(f"  Baslangic    : ${init_capital:>12,.2f}")
    print(f"  Guncel       : ${capital:>12,.2f}")
    print(f"  Toplam Getiri: {fmt_pnl(total_pnl):>12}  ({fmt_pnl(return_pct)}%)")
    print(f"  Equity Peak  : ${equity_peak:>12,.2f}")
    print(f"  Max Drawdown : {max_dd:>11.2f}%")

    # Islem ozeti
    print(f"\n  ISLEM OZETI ({len(trades)} toplam)")
    print(f"  {'-'*45}")
    if trades:
        avg_win  = sum(t["realized_pnl"] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t["realized_pnl"] for t in losers)  / len(losers)  if losers  else 0
        print(f"  Kazanan      : {len(winners):>4}   Ort. +${avg_win:.2f}")
        print(f"  Kaybeden     : {len(losers):>4}   Ort.  ${avg_loss:.2f}")
        print(f"  Win Rate     : {win_rate:>8.1f}%")
    else:
        print("  Henuz tamamlanmis islem yok.")

    # Acik pozisyonlar
    if open_pos:
        print(f"\n  ACIK POZISYONLAR ({len(open_pos)})")
        print(f"  {'-'*45}")
        for p in open_pos:
            opened = p.get("opened_at", "?")[:16].replace("T", " ")
            print(
                f"  {p['direction']:<5} {p['quantity']:.5f} BTC | "
                f"Giris: ${p['entry_price']:,.2f} | "
                f"SL: ${p['stop_loss']:,.2f}" if p.get("stop_loss") else
                f"  {p['direction']:<5} {p['quantity']:.5f} BTC | "
                f"Giris: ${p['entry_price']:,.2f} | SL: yok"
            )
            print(f"         Strateji: {p['strategy']} | Acilis: {opened}")

    # Son islemler tablosu
    print(f"\n  SON {min(20, len(trades))} ISLEM")
    print(f"  {'-'*65}")
    if trades:
        print(f"  {'#':>3}  {'YON':<5}  {'MIKTAR':>9}  {'GIRIS':>10}  {'CIKIS':>10}  {'PnL':>9}  SEBEP")
        print(f"  {'-'*65}")
        for i, t in enumerate(trades[-20:], 1):
            pnl_str = fmt_pnl(t["realized_pnl"])
            print(
                f"  {i:>3}  {t['direction']:<5}  "
                f"{t['quantity']:>9.5f}  "
                f"${t['entry_price']:>9,.2f}  "
                f"${t['exit_price']:>9,.2f}  "
                f"{pnl_str:>9}  "
                f"{t['exit_reason']}"
            )
    else:
        print("  Henuz islem yok.")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
