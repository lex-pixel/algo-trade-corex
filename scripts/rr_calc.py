"""
scripts/rr_calc.py
===================
AMACI:
    Risk:Reward hesaplama modulu.
    - Giris fiyati, Stop Loss, Take Profit girerek R:R orani hesaplar
    - Sermaye ve risk yuzdesi ile lot/miktar hesaplar
    - Birden fazla TP hedefi (R1, R2, R3) destekler
    - Terminalde tablo + HTML gorsel uretir

CALISTIRMAK ICIN:
    python scripts/rr_calc.py --entry 70000 --sl 68000 --tp 75000
    python scripts/rr_calc.py --entry 70000 --sl 68000 --tp 75000 --capital 10000 --risk 2
    python scripts/rr_calc.py --entry 70000 --sl 68000 --tp1 73000 --tp2 76000 --tp3 80000
    python scripts/rr_calc.py  # Interaktif mod

CIKTI:
    - R:R orani (ornek: 1:2.5)
    - USD risk miktari
    - USD kazanc miktari
    - Lot/miktar (BTC veya herhangi bir varlik)
    - Kaskade TP: %50 R1'de, %30 R2'de, %20 R3'de
    - HTML gorsel (reports/rr_TARIH.html)
"""

from __future__ import annotations
import argparse
import json
import sys
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR  = PROJECT_ROOT / "reports"
STATE_FILE   = PROJECT_ROOT / "data" / "bot_state.json"

TZ_TR = timezone(timedelta(hours=3))


# ─────────────────────────────────────────────────────────────────────────────
# HESAPLAMA MOTORU
# ─────────────────────────────────────────────────────────────────────────────

def calc_rr(
    entry: float,
    sl: float,
    tp: float,
    direction: str = "LONG",
) -> dict:
    """
    Tek TP icin R:R hesaplar.

    Args:
        entry    : Giris fiyati
        sl       : Stop Loss fiyati
        tp       : Take Profit fiyati
        direction: 'LONG' veya 'SHORT'

    Returns:
        dict: risk_pct, reward_pct, rr_ratio, valid
    """
    if direction == "LONG":
        risk_pts   = entry - sl
        reward_pts = tp - entry
    else:  # SHORT
        risk_pts   = sl - entry
        reward_pts = entry - tp

    if risk_pts <= 0:
        return {"valid": False, "error": "SL giris fiyatinin yanlis tarafinda"}
    if reward_pts <= 0:
        return {"valid": False, "error": "TP giris fiyatinin yanlis tarafinda"}

    risk_pct   = risk_pts / entry * 100
    reward_pct = reward_pts / entry * 100
    rr_ratio   = reward_pts / risk_pts

    return {
        "valid"      : True,
        "direction"  : direction,
        "entry"      : entry,
        "sl"         : sl,
        "tp"         : tp,
        "risk_pts"   : round(risk_pts, 4),
        "reward_pts" : round(reward_pts, 4),
        "risk_pct"   : round(risk_pct, 4),
        "reward_pct" : round(reward_pct, 4),
        "rr_ratio"   : round(rr_ratio, 4),
        "rr_str"     : f"1:{rr_ratio:.2f}",
    }


def calc_position_size(
    capital: float,
    risk_pct: float,   # 0-100 arasi, ornek: 2.0 = %2
    entry: float,
    sl: float,
    direction: str = "LONG",
    min_qty: float = 0.001,
) -> dict:
    """
    Lot/miktar hesaplar.

    Args:
        capital  : Toplam sermaye (USD)
        risk_pct : Riskedilecek sermaye yuzdesi (0-100)
        entry    : Giris fiyati
        sl       : Stop Loss fiyati
        direction: 'LONG' veya 'SHORT'
        min_qty  : Minimum lot buyuklugu

    Returns:
        dict: qty, risk_usd, notional, risk_pct_actual
    """
    risk_usd      = capital * (risk_pct / 100)
    stop_distance = abs(entry - sl)
    if stop_distance == 0:
        return {"valid": False, "error": "SL ile giris ayni fiyatta"}

    qty         = risk_usd / stop_distance
    qty         = max(qty, min_qty)
    notional    = qty * entry
    actual_risk = qty * stop_distance
    actual_risk_pct = actual_risk / capital * 100 if capital > 0 else 0.0

    return {
        "valid"           : True,
        "qty"             : round(qty, 6),
        "risk_usd"        : round(risk_usd, 2),
        "actual_risk_usd" : round(actual_risk, 2),
        "actual_risk_pct" : round(actual_risk_pct, 4),
        "notional_usd"    : round(notional, 2),
        "capital_usage_pct": round(notional / capital * 100 if capital > 0 else 0, 2),
    }


def calc_cascade_tp(
    entry: float,
    sl: float,
    tp1: float,
    tp2: float | None = None,
    tp3: float | None = None,
    qty: float = 1.0,
    direction: str = "LONG",
    weights: tuple = (0.5, 0.3, 0.2),
) -> list[dict]:
    """
    Kaskade TP hesaplar.
    Varsayilan: %50 TP1'de, %30 TP2'de, %20 TP3'de kapat.

    Returns:
        list of dict: Her TP hedefi icin PnL hesabi
    """
    targets = [tp1]
    if tp2: targets.append(tp2)
    if tp3: targets.append(tp3)

    # Agirlik normalize et
    w = list(weights[:len(targets)])
    total_w = sum(w)
    w = [x / total_w for x in w]

    results = []
    risk_pts = abs(entry - sl)

    for i, (tp, weight) in enumerate(zip(targets, w), start=1):
        partial_qty = qty * weight
        if direction == "LONG":
            reward_pts = tp - entry
        else:
            reward_pts = entry - tp

        pnl_usd  = reward_pts * partial_qty
        rr       = reward_pts / risk_pts if risk_pts > 0 else 0

        results.append({
            "tp_level"  : i,
            "price"     : tp,
            "weight_pct": round(weight * 100, 1),
            "qty"       : round(partial_qty, 6),
            "reward_pts": round(reward_pts, 4),
            "rr"        : round(rr, 2),
            "rr_str"    : f"1:{rr:.2f}",
            "pnl_usd"   : round(pnl_usd, 4),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL CIKTI
# ─────────────────────────────────────────────────────────────────────────────

def print_rr_report(
    rr: dict,
    pos: dict | None = None,
    cascade: list | None = None,
    capital: float | None = None,
) -> None:
    """Terminal raporu."""
    if not rr.get("valid"):
        print(f"[HATA] {rr.get('error', 'Gecersiz parametreler')}")
        return

    print("\n" + "=" * 60)
    print(f"  RISK:REWARD HESAPLAYICI — {rr['direction']}")
    print("=" * 60)
    print(f"  Giris Fiyati : ${rr['entry']:>12,.4f}")
    print(f"  Stop Loss    : ${rr['sl']:>12,.4f}  (-%{rr['risk_pct']:.2f})")
    print(f"  Take Profit  : ${rr['tp']:>12,.4f}  (+%{rr['reward_pct']:.2f})")
    print(f"  Risk (puan)  : {rr['risk_pts']:>13,.4f}")
    print(f"  Reward (puan): {rr['reward_pts']:>13,.4f}")
    print(f"  R:R Orani    : {rr['rr_str']:>13}")

    # R:R degerlendirmesi
    rr_val = rr["rr_ratio"]
    if rr_val >= 3:
        grade = "Mukemmel (>=1:3)"
    elif rr_val >= 2:
        grade = "Iyi (>=1:2)"
    elif rr_val >= 1.5:
        grade = "Kabul edilebilir (>=1:1.5)"
    elif rr_val >= 1:
        grade = "Minimum (1:1)"
    else:
        grade = "Yetersiz (<1:1) — GIRILMEMELI"

    print(f"  Degerlendirme: {grade}")

    if pos and pos.get("valid"):
        print(f"\n  POZİSYON BOYUTU")
        print(f"  {'-' * 40}")
        if capital:
            print(f"  Sermaye      : ${capital:>12,.2f}")
        print(f"  Lot/Miktar   : {pos['qty']:>13,.6f}")
        print(f"  Notional     : ${pos['notional_usd']:>12,.2f}")
        print(f"  Risk (USD)   : ${pos['actual_risk_usd']:>12,.2f}  (%{pos['actual_risk_pct']:.2f})")
        if capital:
            reward_usd = pos['qty'] * rr['reward_pts']
            print(f"  Kazanc (USD) : ${reward_usd:>+12,.2f}")
        print(f"  Sermaye Kull.: %{pos['capital_usage_pct']:.2f}")

    if cascade:
        print(f"\n  KASKADE TP PLANI")
        print(f"  {'-' * 40}")
        print(f"  {'TP':>4}  {'Fiyat':>12}  {'Agirlik':>8}  {'Lot':>10}  {'R:R':>6}  {'PnL ($)':>10}")
        print(f"  {'-' * 60}")
        total_pnl = 0.0
        for c in cascade:
            print(
                f"  TP{c['tp_level']:>1}  "
                f"${c['price']:>11,.4f}  "
                f"%{c['weight_pct']:>6.1f}  "
                f"{c['qty']:>10,.6f}  "
                f"{c['rr_str']:>6}  "
                f"${c['pnl_usd']:>+9,.2f}"
            )
            total_pnl += c["pnl_usd"]
        print(f"  {'-' * 60}")
        print(f"  {'Toplam':>46}  ${total_pnl:>+9,.2f}")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# HTML GORSEL
# ─────────────────────────────────────────────────────────────────────────────

def save_html(
    rr: dict,
    pos: dict | None,
    cascade: list | None,
    capital: float | None,
    output_path: Path,
) -> None:
    """Plotly ile gorsel R:R HTML raporu uretir."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly yuklu degil — HTML uretilmedi.")
        return

    if not rr.get("valid"):
        return

    entry = rr["entry"]
    sl    = rr["sl"]
    tp    = rr["tp"]

    # Fiyat ekseni icin aralik
    all_prices = [entry, sl, tp]
    if cascade:
        all_prices += [c["price"] for c in cascade]
    price_min = min(all_prices) * 0.995
    price_max = max(all_prices) * 1.005

    rows = 1 if not cascade else 2
    specs_list = [[{"type": "scatter"}]]
    if cascade:
        specs_list.append([{"type": "bar"}])

    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=(
            [f"R:R Gorsel — {rr['direction']}  {rr['rr_str']}"] +
            (["Kaskade TP PnL"] if cascade else [])
        ),
        specs=specs_list,
        row_heights=[0.6, 0.4] if cascade else [1.0],
    )

    # Fiyat seviyesi yatay cizgiler
    # Giris
    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                  y0=entry, y1=entry, yref="y",
                  line=dict(color="#f0a500", width=2, dash="solid"),
                  row=1, col=1)
    fig.add_annotation(x=1.01, y=entry, xref="paper", yref="y",
                       text=f"GIRIS ${entry:,.2f}",
                       font=dict(color="#f0a500", size=11), showarrow=False,
                       row=1, col=1)

    # Stop Loss (kirmizi)
    fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                  y0=sl, y1=sl, yref="y",
                  line=dict(color="#ff4444", width=2, dash="dash"),
                  row=1, col=1)
    fig.add_annotation(x=1.01, y=sl, xref="paper", yref="y",
                       text=f"SL ${sl:,.2f}  (-%{rr['risk_pct']:.2f})",
                       font=dict(color="#ff4444", size=11), showarrow=False,
                       row=1, col=1)

    # Take Profit seviyeleri (yesil tonlari)
    tp_colors = ["#00ff88", "#00cc66", "#009944"]
    if cascade:
        for i, c in enumerate(cascade):
            color = tp_colors[i % len(tp_colors)]
            fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                          y0=c["price"], y1=c["price"], yref="y",
                          line=dict(color=color, width=1.5, dash="dot"),
                          row=1, col=1)
            fig.add_annotation(x=1.01, y=c["price"], xref="paper", yref="y",
                               text=f"TP{c['tp_level']} ${c['price']:,.2f}  {c['rr_str']}",
                               font=dict(color=color, size=11), showarrow=False,
                               row=1, col=1)
    else:
        fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                      y0=tp, y1=tp, yref="y",
                      line=dict(color="#00ff88", width=2, dash="dot"),
                      row=1, col=1)
        fig.add_annotation(x=1.01, y=tp, xref="paper", yref="y",
                           text=f"TP ${tp:,.2f}  {rr['rr_str']}",
                           font=dict(color="#00ff88", size=11), showarrow=False,
                           row=1, col=1)

    # Risk/Reward bolgeleri (renkli dikdortgen)
    if rr["direction"] == "LONG":
        # Risk bolge (kirmizi)
        fig.add_shape(type="rect", x0=0, x1=1, xref="paper",
                      y0=sl, y1=entry, yref="y",
                      fillcolor="rgba(255,68,68,0.12)", line=dict(width=0),
                      row=1, col=1)
        # Reward bolge (yesil)
        fig.add_shape(type="rect", x0=0, x1=1, xref="paper",
                      y0=entry, y1=tp, yref="y",
                      fillcolor="rgba(0,255,136,0.12)", line=dict(width=0),
                      row=1, col=1)
    else:
        fig.add_shape(type="rect", x0=0, x1=1, xref="paper",
                      y0=entry, y1=sl, yref="y",
                      fillcolor="rgba(255,68,68,0.12)", line=dict(width=0),
                      row=1, col=1)
        fig.add_shape(type="rect", x0=0, x1=1, xref="paper",
                      y0=tp, y1=entry, yref="y",
                      fillcolor="rgba(0,255,136,0.12)", line=dict(width=0),
                      row=1, col=1)

    # Placeholder scatter (bos grafik icin gerekli)
    fig.add_trace(go.Scatter(
        x=[0.5], y=[entry],
        mode="markers",
        marker=dict(color="#f0a500", size=12, symbol="diamond"),
        name=f"Giris ${entry:,.2f}",
    ), row=1, col=1)

    fig.update_yaxes(range=[price_min, price_max], row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=1)

    # Kaskade TP bar grafigi
    if cascade and rows == 2:
        tp_labels = [f"TP{c['tp_level']} ({c['rr_str']})" for c in cascade]
        tp_pnls   = [c["pnl_usd"] for c in cascade]
        fig.add_trace(go.Bar(
            x=tp_labels, y=tp_pnls,
            marker_color=tp_colors[:len(cascade)],
            name="PnL ($)",
            text=[f"${p:+.2f}" for p in tp_pnls],
            textposition="outside",
        ), row=2, col=1)

    # Bilgi kutusu
    info_lines = [
        f"<b>R:R Orani: {rr['rr_str']}</b>",
        f"Giris: ${entry:,.4f}",
        f"SL: ${sl:,.4f}  (-%{rr['risk_pct']:.2f})",
        f"TP: ${tp:,.4f}  (+%{rr['reward_pct']:.2f})",
    ]
    if pos and pos.get("valid"):
        info_lines += [
            f"Lot: {pos['qty']:.6f}",
            f"Risk: ${pos['actual_risk_usd']:,.2f}",
        ]
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text="<br>".join(info_lines),
        align="left", showarrow=False,
        bgcolor="#1e1e3a", bordercolor="#444",
        font=dict(color="white", size=12),
    )

    fig.update_layout(
        title=f"R:R Hesaplayici | {rr['direction']} | {rr['rr_str']} | "
              f"{datetime.now(TZ_TR).strftime('%Y-%m-%d %H:%M')} (UTC+3)",
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#151525",
        font=dict(color="#ccccdd"),
        height=700 if cascade else 500,
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#222233")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"HTML gorsel: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# INTERAKTIF MOD
# ─────────────────────────────────────────────────────────────────────────────

def interactive_mode() -> None:
    """Kullanicidan parametreleri interaktif olarak alir."""
    print("\n  R:R HESAPLAYICI — Interaktif Mod")
    print("  (Cikis icin Ctrl+C)\n")

    try:
        direction = input("  Yon [LONG/SHORT, varsayilan LONG]: ").strip().upper() or "LONG"
        if direction not in ("LONG", "SHORT"):
            direction = "LONG"

        entry = float(input("  Giris Fiyati: $"))
        sl    = float(input("  Stop Loss  : $"))
        tp1   = float(input("  Take Profit 1: $"))
        tp2_s = input("  Take Profit 2 (bos birak=yok): $").strip()
        tp3_s = input("  Take Profit 3 (bos birak=yok): $").strip()
        cap_s = input("  Sermaye (bos birak=yok): $").strip()
        rsk_s = input("  Risk Yuzdesi [varsayilan 2]: %").strip() or "2"

        tp2    = float(tp2_s) if tp2_s else None
        tp3    = float(tp3_s) if tp3_s else None
        capital = float(cap_s) if cap_s else None
        risk_pct = float(rsk_s)

    except (ValueError, KeyboardInterrupt):
        print("\nIptal edildi.")
        return

    main_tp = tp1
    rr   = calc_rr(entry, sl, main_tp, direction)
    pos  = calc_position_size(capital, risk_pct, entry, sl, direction) if capital else None
    qty  = pos["qty"] if pos and pos.get("valid") else 1.0

    cascade = None
    if tp2 or tp3:
        cascade = calc_cascade_tp(entry, sl, tp1, tp2, tp3, qty=qty, direction=direction)

    print_rr_report(rr, pos, cascade, capital)

    save_q = input("  HTML gorsel kaydet? [e/H]: ").strip().lower()
    if save_q == "e":
        ts   = datetime.now(TZ_TR).strftime("%Y%m%d_%H%M")
        path = REPORTS_DIR / f"rr_{ts}.html"
        save_html(rr, pos, cascade, capital, path)


# ─────────────────────────────────────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────────────────────────────────────

def run_cli(args) -> None:
    """CLI modunda calistirir."""
    direction = (args.direction or "LONG").upper()
    entry     = args.entry
    sl        = args.sl
    tp1       = args.tp or args.tp1
    tp2       = getattr(args, "tp2", None)
    tp3       = getattr(args, "tp3", None)
    capital   = getattr(args, "capital", None)
    risk_pct  = getattr(args, "risk", 2.0) or 2.0

    if not all([entry, sl, tp1]):
        print("HATA: --entry, --sl, --tp gerekli.")
        sys.exit(1)

    rr   = calc_rr(entry, sl, tp1, direction)
    pos  = calc_position_size(capital, risk_pct, entry, sl, direction) if capital else None
    qty  = pos["qty"] if pos and pos.get("valid") else 1.0

    cascade = None
    if tp2 or tp3:
        cascade = calc_cascade_tp(entry, sl, tp1, tp2, tp3, qty=qty, direction=direction)

    print_rr_report(rr, pos, cascade, capital)

    if not args.no_html and rr.get("valid"):
        ts   = datetime.now(TZ_TR).strftime("%Y%m%d_%H%M")
        name = args.output or f"rr_{ts}"
        path = REPORTS_DIR / f"{name}.html"
        save_html(rr, pos, cascade, capital, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R:R Hesaplayici")
    parser.add_argument("--direction", default="LONG",  help="LONG veya SHORT")
    parser.add_argument("--entry",  type=float, help="Giris fiyati")
    parser.add_argument("--sl",     type=float, help="Stop Loss")
    parser.add_argument("--tp",     type=float, help="Take Profit (tek TP)")
    parser.add_argument("--tp1",    type=float, help="Take Profit 1 (kaskade)")
    parser.add_argument("--tp2",    type=float, help="Take Profit 2 (kaskade)")
    parser.add_argument("--tp3",    type=float, help="Take Profit 3 (kaskade)")
    parser.add_argument("--capital",type=float, help="Sermaye (USD)")
    parser.add_argument("--risk",   type=float, default=2.0, help="Risk yuzdesi (varsayilan: 2)")
    parser.add_argument("--output", default=None, help="Cikti dosya adi (uzantisiz)")
    parser.add_argument("--no-html", action="store_true", help="HTML uretme")
    args = parser.parse_args()

    if args.entry is None:
        # Hicbir arguman verilmediyse interaktif mod
        interactive_mode()
    else:
        run_cli(args)
