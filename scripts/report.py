"""
scripts/report.py
==================
AMACI:
    bot_state.json'dan otomatik performans raporu uretir.
    - Sharpe, Sortino, Profit Factor, Win Rate, Max DD hesaplar
    - reports/report_TARIH.txt metin raporu yazar
    - reports/report_TARIH.json JSON raporu yazar
    - Terminalde ozet gosterir

CALISTIRMAK ICIN:
    python scripts/report.py
    python scripts/report.py --json          # Sadece JSON
    python scripts/report.py --txt           # Sadece TXT
    python scripts/report.py --output ozet   # Ozel dosya adi
"""

from __future__ import annotations
import json
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
STATE_FILE   = PROJECT_ROOT / "data" / "bot_state.json"
REPORTS_DIR  = PROJECT_ROOT / "reports"

# UTC+3 (Turkiye)
TZ_TR = timezone(timedelta(hours=3))


# ─────────────────────────────────────────────────────────────────────────────
# HESAPLAMALAR
# ─────────────────────────────────────────────────────────────────────────────

def calc_sharpe(equity_list: list[float], annual_factor: int = 365) -> float:
    """
    Equity listesinden Sharpe oranini hesaplar.
    Formul: (ort. gunluk getiri / std) * sqrt(annual_factor)
    """
    if len(equity_list) < 10:
        return 0.0
    returns = []
    for i in range(1, len(equity_list)):
        prev = equity_list[i - 1]
        curr = equity_list[i]
        if prev > 0:
            returns.append((curr - prev) / prev)

    if len(returns) < 5:
        return 0.0

    n    = len(returns)
    mean = sum(returns) / n
    std  = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))

    if std == 0:
        return 0.0
    return round((mean / std) * math.sqrt(annual_factor), 4)


def calc_sortino(equity_list: list[float], annual_factor: int = 365) -> float:
    """
    Sortino orani: sadece asagi volatilite cezalandirilir.
    """
    if len(equity_list) < 10:
        return 0.0
    returns = []
    for i in range(1, len(equity_list)):
        prev = equity_list[i - 1]
        if prev > 0:
            returns.append((equity_list[i] - prev) / prev)

    if not returns:
        return 0.0

    mean     = sum(returns) / len(returns)
    neg      = [r for r in returns if r < 0]
    if not neg:
        return float("inf") if mean > 0 else 0.0

    n_neg    = len(neg)
    down_std = math.sqrt(sum(r ** 2 for r in neg) / n_neg)
    if down_std == 0:
        return 0.0
    return round((mean / down_std) * math.sqrt(annual_factor), 4)


def calc_max_drawdown(equity_list: list[float]) -> float:
    """
    Equity listesinden maksimum drawdown hesaplar.
    Doner: yuzde (ornek: -12.4 = -%12.4)
    """
    if len(equity_list) < 2:
        return 0.0
    peak = equity_list[0]
    max_dd = 0.0
    for eq in equity_list:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (eq - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
    return round(max_dd, 4)


def calc_profit_factor(trades: list[dict]) -> float:
    """Toplam kazanc / Toplam kayip."""
    wins   = sum(t["realized_pnl"] for t in trades if t.get("realized_pnl", 0) > 0)
    losses = abs(sum(t["realized_pnl"] for t in trades if t.get("realized_pnl", 0) < 0))
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / losses, 4)


def calc_calmar(total_return_pct: float, max_dd_pct: float) -> float:
    """Calmar orani: Toplam getiri / |Max Drawdown|."""
    if max_dd_pct == 0:
        return 0.0
    return round(total_return_pct / abs(max_dd_pct), 4)


def calc_avg_duration(trades: list[dict]) -> float:
    """Ortalama islem suresi (saat)."""
    durations = []
    for t in trades:
        try:
            opened  = datetime.fromisoformat(t.get("opened_at") or t.get("entry_time", ""))
            closed  = datetime.fromisoformat(t.get("closed_at") or t.get("exit_time", ""))
            durations.append((closed - opened).total_seconds() / 3600)
        except Exception:
            pass
    if not durations:
        return 0.0
    return round(sum(durations) / len(durations), 2)


def calc_max_consec_losses(trades: list[dict]) -> int:
    """Ust uste en fazla kayipli islem sayisi."""
    max_seq = cur = 0
    for t in trades:
        if t.get("realized_pnl", 0) <= 0:
            cur += 1
            max_seq = max(max_seq, cur)
        else:
            cur = 0
    return max_seq


def calc_expectancy(trades: list[dict]) -> float:
    """
    Beklenti (Expectancy): her islemde ortalama beklenen kazan/kayip.
    Formul: (Win Rate * Ort. Kazanc) + (Loss Rate * Ort. Kayip)
    Pozitif olmali — ne kadar yuksekse o kadar iyi.
    """
    if not trades:
        return 0.0
    n      = len(trades)
    wins   = [t["realized_pnl"] for t in trades if t.get("realized_pnl", 0) > 0]
    losses = [t["realized_pnl"] for t in trades if t.get("realized_pnl", 0) <= 0]
    wr     = len(wins) / n
    lr     = 1 - wr
    avg_w  = sum(wins) / len(wins) if wins else 0.0
    avg_l  = sum(losses) / len(losses) if losses else 0.0
    return round(wr * avg_w + lr * avg_l, 4)


# ─────────────────────────────────────────────────────────────────────────────
# ANA RAPOR
# ─────────────────────────────────────────────────────────────────────────────

def build_report(state: dict) -> dict:
    """
    bot_state.json'dan tum metrikleri hesaplar ve dict dondurur.
    """
    initial_cap  = state.get("initial_capital", 10000.0)
    capital      = state.get("capital", initial_cap)
    equity_peak  = state.get("equity_peak", capital)
    iteration    = state.get("iteration", 0)
    paper_mode   = state.get("paper", True)
    trades       = state.get("trades", [])
    eq_history   = state.get("equity_history", [])

    # Equity listesi
    eq_threshold = initial_cap * 0.85
    eq_list = [e["equity"] for e in eq_history if e["equity"] >= eq_threshold]

    # Acik pozisyonlar — true equity
    open_positions  = state.get("open_positions", [])
    locked_notional = sum(p.get("entry_price", 0) * p.get("quantity", 0) for p in open_positions)
    last_eq         = eq_list[-1] if eq_list else capital
    true_equity     = capital + locked_notional + (last_eq - capital) if locked_notional > 0 else capital

    # Temel getiri
    total_return_usd = true_equity - initial_cap
    total_return_pct = (total_return_usd / initial_cap) * 100 if initial_cap > 0 else 0.0

    # Trade istatistikleri
    n_trades   = len(trades)
    wins       = [t for t in trades if t.get("realized_pnl", 0) > 0]
    losses     = [t for t in trades if t.get("realized_pnl", 0) <= 0]
    win_rate   = len(wins) / n_trades * 100 if n_trades > 0 else 0.0
    total_pnl  = sum(t.get("realized_pnl", 0) for t in trades)
    avg_win    = sum(t["realized_pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss   = sum(t["realized_pnl"] for t in losses) / len(losses) if losses else 0.0
    best_trade = max((t.get("realized_pnl", 0) for t in trades), default=0.0)
    worst_trd  = min((t.get("realized_pnl", 0) for t in trades), default=0.0)

    # Risk metrikleri
    sharpe      = calc_sharpe(eq_list)
    sortino     = calc_sortino(eq_list)
    max_dd      = calc_max_drawdown(eq_list)
    pf          = calc_profit_factor(trades)
    calmar      = calc_calmar(total_return_pct, max_dd)
    avg_dur     = calc_avg_duration(trades)
    max_cons_l  = calc_max_consec_losses(trades)
    expectancy  = calc_expectancy(trades)

    # Tarih araligi
    start_ts = eq_history[0]["ts"][:10] if eq_history else "?"
    end_ts   = eq_history[-1]["ts"][:10] if eq_history else "?"

    now_tr = datetime.now(TZ_TR).strftime("%Y-%m-%d %H:%M")

    return {
        "report_time"       : now_tr,
        "mode"              : "PAPER" if paper_mode else "LIVE",
        "period"            : f"{start_ts} -> {end_ts}",
        "iteration"         : iteration,

        # Sermaye
        "initial_capital"   : round(initial_cap, 2),
        "current_capital"   : round(capital, 2),
        "true_equity"       : round(true_equity, 2),
        "equity_peak"       : round(equity_peak, 2),

        # Getiri
        "total_return_usd"  : round(total_return_usd, 2),
        "total_return_pct"  : round(total_return_pct, 3),

        # Risk
        "sharpe_ratio"      : sharpe,
        "sortino_ratio"     : sortino,
        "max_drawdown_pct"  : max_dd,
        "calmar_ratio"      : calmar,

        # Trade istatistikleri
        "total_trades"      : n_trades,
        "winning_trades"    : len(wins),
        "losing_trades"     : len(losses),
        "win_rate_pct"      : round(win_rate, 2),
        "profit_factor"     : pf,
        "total_pnl_usd"     : round(total_pnl, 2),
        "avg_win_usd"       : round(avg_win, 2),
        "avg_loss_usd"      : round(avg_loss, 2),
        "best_trade_usd"    : round(best_trade, 2),
        "worst_trade_usd"   : round(worst_trd, 2),
        "avg_duration_hours": avg_dur,
        "max_consec_losses" : max_cons_l,
        "expectancy_usd"    : expectancy,

        # Acik pozisyonlar ozeti
        "open_positions"    : len(open_positions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CIKTI FORMATLARI
# ─────────────────────────────────────────────────────────────────────────────

def _metric_grade(sharpe: float, win_rate: float, max_dd: float, pf: float) -> str:
    """
    Toplam sistem puani:
    Sharpe >1.5 + WR >55 + DD >-10 + PF >1.5 = A
    """
    score = 0
    if sharpe > 1.5:    score += 3
    elif sharpe > 1.0:  score += 2
    elif sharpe > 0.5:  score += 1

    if win_rate > 55:   score += 3
    elif win_rate > 45: score += 2
    elif win_rate > 35: score += 1

    if max_dd > -5:     score += 3
    elif max_dd > -10:  score += 2
    elif max_dd > -20:  score += 1

    if pf > 2.0:        score += 3
    elif pf > 1.5:      score += 2
    elif pf > 1.0:      score += 1

    if score >= 10: return "A (Mukemmel)"
    if score >= 7:  return "B (Iyi)"
    if score >= 4:  return "C (Orta)"
    return "D (Gelistirme Gerekiyor)"


def format_txt(r: dict) -> str:
    """Metin raporu formatlar."""
    grade = _metric_grade(
        r["sharpe_ratio"], r["win_rate_pct"],
        r["max_drawdown_pct"], r["profit_factor"]
    )

    lines = [
        "=" * 65,
        "  ALGO TRADE CODEX — Otomatik Performans Raporu",
        f"  Olusturulma: {r['report_time']} (UTC+3)",
        f"  Mod: {r['mode']}  |  Donem: {r['period']}  |  Tick: {r['iteration']}",
        "=" * 65,
        "",
        "  SERMAYE",
        "  " + "-" * 45,
        f"  Baslangic Sermayesi : ${r['initial_capital']:>12,.2f}",
        f"  Guncel Sermaye      : ${r['current_capital']:>12,.2f}",
        f"  Gercek Equity       : ${r['true_equity']:>12,.2f}",
        f"  Equity Tepe         : ${r['equity_peak']:>12,.2f}",
        f"  Toplam Getiri       : ${r['total_return_usd']:>+12,.2f}  "
        f"(%{r['total_return_pct']:+.2f})",
        "",
        "  RISK METRIKLERI",
        "  " + "-" * 45,
        f"  Sharpe Orani        : {r['sharpe_ratio']:>12.4f}",
        f"  Sortino Orani       : {r['sortino_ratio']:>12.4f}",
        f"  Calmar Orani        : {r['calmar_ratio']:>12.4f}",
        f"  Max Drawdown        : {r['max_drawdown_pct']:>11.2f}%",
        "",
        "  ISLEM ISTATISTIKLERI",
        "  " + "-" * 45,
        f"  Toplam Islem        : {r['total_trades']:>12}",
        f"  Kazanan / Kaybeden  : {r['winning_trades']:>5} / {r['losing_trades']:<5}",
        f"  Win Rate            : {r['win_rate_pct']:>11.1f}%",
        f"  Profit Factor       : {r['profit_factor']:>12.4f}",
        f"  Toplam PnL          : ${r['total_pnl_usd']:>+12,.2f}",
        f"  Ort. Kazanc         : ${r['avg_win_usd']:>+12,.2f}",
        f"  Ort. Kayip          : ${r['avg_loss_usd']:>+12,.2f}",
        f"  En Iyi Islem        : ${r['best_trade_usd']:>+12,.2f}",
        f"  En Kotu Islem       : ${r['worst_trade_usd']:>+12,.2f}",
        f"  Ort. Islem Suresi   : {r['avg_duration_hours']:>10.1f} saat",
        f"  Maks. Ard. Kayip    : {r['max_consec_losses']:>12}",
        f"  Beklenti (Exp.)     : ${r['expectancy_usd']:>+12,.4f}",
        "",
        "  ACIK POZISYONLAR",
        "  " + "-" * 45,
        f"  Acik Pozisyon       : {r['open_positions']:>12}",
        "",
        "  SISTEM DEGERLENDIRMESI",
        "  " + "-" * 45,
        f"  Genel Not           :  {grade}",
        "",
        "  NOT: Sharpe >1 iyi | >2 mukemmel | WR >55% hedef | DD <%10 guvenli",
        "=" * 65,
    ]
    return "\n".join(lines)


def format_txt_interpretation(r: dict) -> str:
    """Metriklerin Turkce aciklamasi."""
    lines = ["\n  METRIK YORUMLARI", "  " + "-" * 45]

    sh = r["sharpe_ratio"]
    if sh >= 2:
        lines.append(f"  Sharpe {sh:.2f} -> Mukemmel risk-getiri dengesi")
    elif sh >= 1:
        lines.append(f"  Sharpe {sh:.2f} -> Iyi, iyilestirme potansiyeli var")
    elif sh >= 0:
        lines.append(f"  Sharpe {sh:.2f} -> Orta, strateji optimize edilmeli")
    else:
        lines.append(f"  Sharpe {sh:.2f} -> Negatif, strateji para kaybettiriyor")

    wr = r["win_rate_pct"]
    if wr >= 60:
        lines.append(f"  Win Rate %{wr:.1f} -> Yuksek basari orani")
    elif wr >= 45:
        lines.append(f"  Win Rate %{wr:.1f} -> Kabul edilebilir, PF'ye dikkat et")
    elif r["total_trades"] == 0:
        lines.append("  Win Rate -> Henuz kapanmis islem yok")
    else:
        lines.append(f"  Win Rate %{wr:.1f} -> Dusuk, strateji filtresi guclendirilmeli")

    dd = r["max_drawdown_pct"]
    if dd > -5:
        lines.append(f"  Max DD %{dd:.1f} -> Cok dusuk, sermaye koruyor")
    elif dd > -15:
        lines.append(f"  Max DD %{dd:.1f} -> Normal aralik")
    else:
        lines.append(f"  Max DD %{dd:.1f} -> Yuksek, risk yonetimi gozden gecirilmeli")

    pf = r["profit_factor"]
    if pf == float("inf"):
        lines.append("  Profit Factor -> Inf (hic kayipli islem yok)")
    elif pf >= 1.5:
        lines.append(f"  Profit Factor {pf:.2f} -> Saglikli")
    elif pf >= 1.0:
        lines.append(f"  Profit Factor {pf:.2f} -> Kirin uzerinde, gelistirilebilir")
    elif r["total_trades"] == 0:
        pass
    else:
        lines.append(f"  Profit Factor {pf:.2f} -> 1'in altinda, strateji para kaybediyor")

    return "\n".join(lines)


def print_report(r: dict) -> None:
    """Terminale raporu yazdirir."""
    print(format_txt(r))
    print(format_txt_interpretation(r))


def save_txt(r: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = format_txt(r) + "\n" + format_txt_interpretation(r)
    path.write_text(content, encoding="utf-8")
    print(f"TXT raporu: {path}")


def save_json(r: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # float("inf") JSON'da desteklenmez — str'ye cevir
    safe = {}
    for k, v in r.items():
        safe[k] = "Inf" if v == float("inf") else v
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON raporu: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────────────────────────────────────

def run(output: str = "auto", save_txt_flag: bool = True, save_json_flag: bool = True) -> dict:
    if not STATE_FILE.exists():
        print("HATA: data/bot_state.json bulunamadi. Botu once calistirin.")
        sys.exit(1)

    with open(STATE_FILE, encoding="utf-8") as f:
        state = json.load(f)

    report = build_report(state)
    print_report(report)

    # Dosya adi
    ts = datetime.now(TZ_TR).strftime("%Y%m%d_%H%M")
    base_name = output if output != "auto" else f"report_{ts}"

    if save_txt_flag:
        save_txt(report, REPORTS_DIR / f"{base_name}.txt")
    if save_json_flag:
        save_json(report, REPORTS_DIR / f"{base_name}.json")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otomatik Performans Raporu")
    parser.add_argument("--output", default="auto", help="Dosya adi (uzantisiz)")
    parser.add_argument("--txt",  action="store_true", help="Sadece TXT kaydet")
    parser.add_argument("--json", action="store_true", help="Sadece JSON kaydet")
    args = parser.parse_args()

    save_txt_flag  = True
    save_json_flag = True
    if args.txt and not args.json:
        save_json_flag = False
    if args.json and not args.txt:
        save_txt_flag = False

    run(output=args.output, save_txt_flag=save_txt_flag, save_json_flag=save_json_flag)
