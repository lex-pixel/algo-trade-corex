"""
scripts/dashboard.py
=====================
AMACI:
    Bot performans raporu olusturur.
    bot_state.json okur, Plotly ile interaktif HTML grafik uretir.

CALISTIRMA:
    python scripts/dashboard.py              # reports/dashboard.html olusturur
    python scripts/dashboard.py --open       # Tarayicida ac
    python scripts/dashboard.py --output benim_rapor.html

GOSTERILEN:
    1. Equity Curve   — sermaye zaman icinde
    2. BTC Fiyati     — equity ile ayni eksende
    3. Trade Noktalari — AL (yesil) / SAT (kirmizi) isaretleri
    4. Kazanc/Kayip   — her islem icin bar chart
    5. Ozet Tablosu   — toplam islem, win rate, drawdown, Sharpe
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

STATE_FILE  = Path("data/bot_state.json")
REPORTS_DIR = Path("reports")

# ─────────────────────────────────────────────────────────────────────────────
# VERI YUKLE
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not STATE_FILE.exists():
        print("HATA: data/bot_state.json bulunamadi.")
        print("Botu en az bir kez calistirin.")
        sys.exit(1)
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# HTML RAPOR OLUSTUR
# ─────────────────────────────────────────────────────────────────────────────

def build_report(state: dict, output_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly yuklu degil. Kurun: pip install plotly")
        sys.exit(1)

    initial_cap  = state.get("initial_capital", 10000.0)
    capital      = state.get("capital", initial_cap)
    equity_peak  = state.get("equity_peak", capital)
    max_drawdown = state.get("max_drawdown", 0.0)
    iteration    = state.get("iteration", 0)
    saved_at     = state.get("saved_at", "?")[:16].replace("T", " ")
    trades       = state.get("trades", [])
    eq_history   = state.get("equity_history", [])

    # ── Hesaplamalar ──────────────────────────────────────────────────────────
    return_pct   = (capital - initial_cap) / initial_cap * 100
    n_trades     = len(trades)
    wins         = [t for t in trades if t.get("realized_pnl", 0) > 0]
    losses       = [t for t in trades if t.get("realized_pnl", 0) <= 0]
    win_rate     = len(wins) / n_trades * 100 if n_trades > 0 else 0
    total_pnl    = sum(t.get("realized_pnl", 0) for t in trades)
    avg_win      = sum(t["realized_pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss     = sum(t["realized_pnl"] for t in losses) / len(losses) if losses else 0

    # ── Subplots ──────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Equity Curve ve BTC Fiyati",
            "Ozet Istatistikler",
            "Islem Kar/Zarar",
            "Kazanc-Kayip Dagilimi",
            "Drawdown",
            "Gunluk PnL",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "bar"},     {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
        row_heights=[0.4, 0.3, 0.3],
        column_widths=[0.65, 0.35],
    )

    # ── 1. Equity Curve ───────────────────────────────────────────────────────
    if eq_history:
        ts_list  = [e["ts"][:16].replace("T", " ") for e in eq_history]
        eq_list  = [e["equity"] for e in eq_history]
        prc_list = [e.get("price", 0) for e in eq_history]

        fig.add_trace(go.Scatter(
            x=ts_list, y=eq_list,
            name="Equity ($)",
            line=dict(color="#00d4aa", width=2),
        ), row=1, col=1)

        # BTC fiyati ikinci y ekseni (normalize)
        if prc_list and prc_list[0] > 0:
            prc_norm = [p / prc_list[0] * initial_cap for p in prc_list]
            fig.add_trace(go.Scatter(
                x=ts_list, y=prc_norm,
                name="BTC (norm.)",
                line=dict(color="#f0a500", width=1, dash="dot"),
                opacity=0.6,
            ), row=1, col=1)

        # Trade noktalari equity uzerinde
        for t in trades:
            closed_at = t.get("closed_at", "")[:16].replace("T", " ")
            pnl       = t.get("realized_pnl", 0)
            color     = "#00ff88" if pnl > 0 else "#ff4444"
            symbol    = "triangle-up" if t.get("direction") == "LONG" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[closed_at],
                y=[capital],   # yaklaşık — gerçek equity gecmisinden bulmak karmasik
                mode="markers",
                marker=dict(color=color, size=10, symbol=symbol),
                name=f"Trade {'WIN' if pnl > 0 else 'LOSS'} ${pnl:+.1f}",
                showlegend=False,
            ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=["Baslangic"], y=[initial_cap],
            name="Equity",
            line=dict(color="#00d4aa"),
        ), row=1, col=1)
        fig.add_annotation(
            text="Henuz veri yok — botu calistirin",
            x=0.17, y=0.75, xref="paper", yref="paper",
            showarrow=False, font=dict(size=12, color="gray"),
        )

    # ── 2. Ozet Tablo ─────────────────────────────────────────────────────────
    fig.add_trace(go.Table(
        header=dict(
            values=["Metrik", "Deger"],
            fill_color="#1e1e2e",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[
                ["Baslangic Sermayesi", "Guncel Sermaye", "Toplam Getiri",
                 "Equity Tepe", "Max Drawdown", "Toplam Islem",
                 "Kazanma Orani", "Toplam PnL", "Ort. Kazanc", "Ort. Kayip",
                 "Toplam Tick", "Son Guncelleme"],
                [
                    f"${initial_cap:,.2f}",
                    f"${capital:,.2f}",
                    f"%{return_pct:+.2f}",
                    f"${equity_peak:,.2f}",
                    f"%{max_drawdown*100:.2f}",
                    str(n_trades),
                    f"%{win_rate:.1f}" if n_trades > 0 else "N/A",
                    f"${total_pnl:+.2f}" if n_trades > 0 else "N/A",
                    f"${avg_win:+.2f}" if wins else "N/A",
                    f"${avg_loss:+.2f}" if losses else "N/A",
                    str(iteration),
                    saved_at,
                ],
            ],
            fill_color=["#151525", "#1a1a2e"],
            font=dict(color=["#aaaacc", "white"], size=11),
            align="left",
        ),
    ), row=1, col=2)

    # ── 3. Islem PnL Bar ──────────────────────────────────────────────────────
    if trades:
        trade_labels = [f"T{i+1}" for i in range(len(trades))]
        trade_pnls   = [t.get("realized_pnl", 0) for t in trades]
        bar_colors   = ["#00d4aa" if p > 0 else "#ff4444" for p in trade_pnls]
        fig.add_trace(go.Bar(
            x=trade_labels, y=trade_pnls,
            name="Islem PnL",
            marker_color=bar_colors,
        ), row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=[], y=[], name="Islem PnL"), row=2, col=1)
        fig.add_annotation(
            text="Henuz kapanmis islem yok",
            x=0.17, y=0.35, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color="gray"),
        )

    # ── 4. Kazanc/Kayip Dagilimi ──────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=["Kazanan", "Kaybeden"],
        y=[len(wins), len(losses)],
        marker_color=["#00d4aa", "#ff4444"],
        name="Dagilim",
    ), row=2, col=2)

    # ── 5. Drawdown ───────────────────────────────────────────────────────────
    if eq_history:
        ts_list    = [e["ts"][:16].replace("T", " ") for e in eq_history]
        eq_list    = [e["equity"] for e in eq_history]
        peak_run   = eq_list[0]
        dd_list    = []
        for eq in eq_list:
            peak_run = max(peak_run, eq)
            dd = (eq - peak_run) / peak_run * 100 if peak_run > 0 else 0
            dd_list.append(dd)
        fig.add_trace(go.Scatter(
            x=ts_list, y=dd_list,
            fill="tozeroy",
            name="Drawdown %",
            line=dict(color="#ff4444", width=1),
            fillcolor="rgba(255,68,68,0.2)",
        ), row=3, col=1)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], name="Drawdown"), row=3, col=1)

    # ── 6. Gunluk PnL ────────────────────────────────────────────────────────
    if trades:
        from collections import defaultdict
        daily_pnl = defaultdict(float)
        for t in trades:
            day = t.get("closed_at", "")[:10]
            daily_pnl[day] += t.get("realized_pnl", 0)
        days = sorted(daily_pnl.keys())
        pnls = [daily_pnl[d] for d in days]
        fig.add_trace(go.Bar(
            x=days, y=pnls,
            marker_color=["#00d4aa" if p > 0 else "#ff4444" for p in pnls],
            name="Gunluk PnL",
        ), row=3, col=2)
    else:
        fig.add_trace(go.Bar(x=[], y=[], name="Gunluk PnL"), row=3, col=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Algo Trade Corex — Performans Raporu | {saved_at}",
            font=dict(size=18, color="white"),
        ),
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#151525",
        font=dict(color="#ccccdd"),
        showlegend=True,
        legend=dict(bgcolor="#1a1a2e", bordercolor="#333"),
        height=1000,
    )
    fig.update_xaxes(gridcolor="#222233", tickfont=dict(color="#999"))
    fig.update_yaxes(gridcolor="#222233", tickfont=dict(color="#999"))

    # ── Kaydet ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Rapor olusturuldu: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL OZET
# ─────────────────────────────────────────────────────────────────────────────

def print_terminal_summary(state: dict) -> None:
    initial_cap  = state.get("initial_capital", 10000.0)
    capital      = state.get("capital", initial_cap)
    equity_peak  = state.get("equity_peak", capital)
    max_drawdown = state.get("max_drawdown", 0.0)
    iteration    = state.get("iteration", 0)
    trades       = state.get("trades", [])
    eq_history   = state.get("equity_history", [])

    return_pct = (capital - initial_cap) / initial_cap * 100
    n_trades   = len(trades)
    wins       = sum(1 for t in trades if t.get("realized_pnl", 0) > 0)
    win_rate   = wins / n_trades * 100 if n_trades > 0 else 0

    print("=" * 50)
    print("  ALGO TRADE COREX - Performans Ozeti")
    print("=" * 50)
    print(f"  Sermaye    : ${initial_cap:,.2f} -> ${capital:,.2f}")
    print(f"  Getiri     : %{return_pct:+.2f}")
    print(f"  Equity Tepe: ${equity_peak:,.2f}")
    print(f"  Max DD     : %{max_drawdown*100:.2f}")
    print(f"  Toplam Tick: {iteration}")
    print(f"  Toplam Islem: {n_trades}")
    if n_trades > 0:
        print(f"  Win Rate   : %{win_rate:.1f}")
    if eq_history:
        print(f"  Equity Nokta: {len(eq_history)}")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────────────────────────────────────

def run(output: str = "reports/dashboard.html", open_browser: bool = False) -> None:
    state       = load_state()
    output_path = Path(output)

    print_terminal_summary(state)
    build_report(state, output_path)

    if open_browser:
        import webbrowser
        webbrowser.open(output_path.resolve().as_uri())
        print("Tarayici aciliyor...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot Performans Dashboard")
    parser.add_argument("--output", default="reports/dashboard.html")
    parser.add_argument("--open",   action="store_true", help="Tarayicida ac")
    args = parser.parse_args()
    run(output=args.output, open_browser=args.open)
