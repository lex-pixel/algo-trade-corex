"""
scripts/live_chart.py
=====================
AMACI:
    TradingView benzeri canli mum grafigi.
    - Binance Testnet'ten gercek OHLCV verisi (ccxt)
    - Islem giris/cikis isgaret noktalari (mum uzerinde)
    - Acik pozisyon SL/TP yatay cizgileri
    - RSI alt paneli
    - 30sn otomatik yenileme
    - Kisa ozet kutu (kapital, PnL, iteration)

KULLANIM:
    python scripts/live_chart.py           -> reports/live_chart.html olusturur
    python scripts/live_chart.py --open    -> HTML'i tarayicida acar
    python scripts/live_chart.py --bars 200 --open
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- Proje koku sys.path'e ekle ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly yuklu degil: pip install plotly")
    sys.exit(1)

try:
    import ccxt
    CCXT_OK = True
except ImportError:
    CCXT_OK = False

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
BOT_STATE   = ROOT / "data" / "bot_state.json"
REPORTS_DIR = ROOT / "reports"
OUTPUT_FILE = REPORTS_DIR / "live_chart.html"
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "1h"
TZ_OFFSET   = timedelta(hours=3)   # UTC+3


def _utc3(dt_utc: datetime) -> datetime:
    """UTC datetime -> UTC+3"""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(timezone(TZ_OFFSET))


def _now_tr() -> str:
    return _utc3(datetime.now(timezone.utc)).strftime("%d.%m.%Y %H:%M:%S")


# ---------------------------------------------------------------------------
# Veri Cekme
# ---------------------------------------------------------------------------

def fetch_ohlcv(bars: int = 150) -> pd.DataFrame:
    """Binance Testnet'ten OHLCV verisi cek."""
    if not CCXT_OK:
        return _fake_ohlcv(bars)

    try:
        exchange = ccxt.binance({
            "urls": {"api": {"public": "https://testnet.binance.vision/api"}},
            "options": {"defaultType": "spot"},
        })
        raw = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=bars)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["ts"] = df["ts"].apply(lambda x: _utc3(x))
        df.set_index("ts", inplace=True)
        return df
    except Exception as e:
        print(f"[UYARI] OHLCV cekilemedi: {e} — sahte veri kullaniliyor")
        return _fake_ohlcv(bars)


def _fake_ohlcv(bars: int) -> pd.DataFrame:
    """Test icin rastgele OHLCV uret."""
    rng = np.random.default_rng(42)
    now = datetime.now(timezone(TZ_OFFSET)).replace(minute=0, second=0, microsecond=0)
    times = [now - timedelta(hours=i) for i in reversed(range(bars))]
    close = 70000 + np.cumsum(rng.normal(0, 200, bars))
    high  = close + rng.uniform(50, 300, bars)
    low   = close - rng.uniform(50, 300, bars)
    open_ = close - rng.normal(0, 150, bars)
    vol   = rng.uniform(10, 500, bars)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol
    }, index=pd.DatetimeIndex(times, name="ts"))
    return df


# ---------------------------------------------------------------------------
# Bot State Okuma
# ---------------------------------------------------------------------------

def load_bot_state() -> dict:
    if not BOT_STATE.exists():
        return {}
    try:
        with open(BOT_STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def extract_trades(state: dict) -> pd.DataFrame:
    """Kapali islemleri DataFrame'e donustur."""
    trades = state.get("trades", [])
    if not trades:
        return pd.DataFrame(columns=["open_time", "close_time", "direction", "entry", "exit_price", "pnl"])
    rows = []
    for t in trades:
        rows.append({
            "open_time" : t.get("open_time", ""),
            "close_time": t.get("close_time", ""),
            "direction" : t.get("direction", "LONG"),
            "entry"     : float(t.get("entry_price", 0)),
            "exit_price": float(t.get("exit_price", 0)),
            "pnl"       : float(t.get("pnl", 0)),
        })
    df = pd.DataFrame(rows)
    for col in ["open_time", "close_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = df[col].apply(lambda x: _utc3(x) if pd.notna(x) else x)
    return df


def extract_open_position(state: dict) -> dict | None:
    """Acik pozisyon bilgisi."""
    pos = state.get("position", {})
    if not pos:
        return None
    # Yeni format: list
    if isinstance(pos, list):
        pos = pos[0] if pos else {}
    return pos if pos.get("status") == "OPEN" else None

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Grafik Olusturma
# ---------------------------------------------------------------------------

def build_chart(bars: int = 150) -> str:
    df    = fetch_ohlcv(bars)
    state = load_bot_state()
    trades_df = extract_trades(state)
    open_pos  = extract_open_position(state)

    capital   = state.get("current_capital", state.get("initial_capital", 10000))
    iteration = state.get("iteration", 0)
    equity_h  = state.get("equity_history", [])
    current_eq = equity_h[-1] if equity_h else capital
    pnl_total  = current_eq - state.get("initial_capital", capital)
    pnl_pct    = (pnl_total / state.get("initial_capital", capital)) * 100 if capital else 0

    # RSI
    rsi = _calc_rsi(df["close"])

    # --- Subplots ---
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.15, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("", "", "RSI (14)"),
    )

    # ----- 1. Mum grafigi -----
    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df["open"],
        high  = df["high"],
        low   = df["low"],
        close = df["close"],
        name  = "BTC/USDT",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ----- 2. Hacim -----
    colors_vol = ["#26a69a" if c >= o else "#ef5350"
                  for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(
        x      = df.index,
        y      = df["volume"],
        name   = "Hacim",
        marker_color=colors_vol,
        opacity=0.7,
        showlegend=False,
    ), row=2, col=1)

    # ----- 3. RSI -----
    fig.add_trace(go.Scatter(
        x    = df.index,
        y    = rsi,
        name = "RSI",
        line = dict(color="#7b68ee", width=1.5),
    ), row=3, col=1)

    # RSI seviye cizgileri
    for lvl, clr, dash in [(70, "#ef5350", "dash"), (50, "#888", "dot"), (30, "#26a69a", "dash")]:
        fig.add_hline(y=lvl, line_dash=dash, line_color=clr, line_width=1,
                      opacity=0.5, row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(100,100,100,0.05)",
                  layer="below", line_width=0, row=3, col=1)

    # ----- Islem isaretleri -----
    if not trades_df.empty and "open_time" in trades_df.columns:
        # Giris isaretleri
        for _, tr in trades_df.iterrows():
            ot = tr["open_time"]
            ct = tr["close_time"]
            direction = tr["direction"]
            entry = tr["entry"]
            exit_p = tr["exit_price"]
            pnl = tr["pnl"]
            is_win = pnl >= 0

            arrow_color_entry = "#26a69a" if direction == "LONG" else "#ef5350"
            arrow_color_exit  = "#26a69a" if is_win else "#ef5350"

            # Giris
            if pd.notna(ot):
                fig.add_trace(go.Scatter(
                    x=[ot], y=[entry],
                    mode="markers+text",
                    marker=dict(
                        symbol="triangle-up" if direction == "LONG" else "triangle-down",
                        size=12, color=arrow_color_entry,
                        line=dict(width=1, color="white"),
                    ),
                    text=[f"{'AL' if direction == 'LONG' else 'SAT'}<br>{entry:,.0f}"],
                    textposition="top center" if direction == "LONG" else "bottom center",
                    textfont=dict(size=9, color=arrow_color_entry),
                    name=f"{direction} Giris",
                    showlegend=False,
                    hovertemplate=f"<b>{direction} Giris</b><br>Fiyat: {entry:,.2f}<br>Zaman: {ot}<extra></extra>",
                ), row=1, col=1)

            # Cikis
            if pd.notna(ct):
                fig.add_trace(go.Scatter(
                    x=[ct], y=[exit_p],
                    mode="markers+text",
                    marker=dict(
                        symbol="x", size=10, color=arrow_color_exit,
                        line=dict(width=2, color=arrow_color_exit),
                    ),
                    text=[f"{'+'if pnl>=0 else ''}{pnl:.1f}$"],
                    textposition="top center",
                    textfont=dict(size=9, color=arrow_color_exit),
                    name="Cikis",
                    showlegend=False,
                    hovertemplate=f"<b>Cikis</b><br>Fiyat: {exit_p:,.2f}<br>PnL: {pnl:+.2f}$<extra></extra>",
                ), row=1, col=1)

    # ----- Acik pozisyon SL/TP cizgileri -----
    if open_pos:
        ep  = float(open_pos.get("entry_price", 0))
        sl  = float(open_pos.get("sl", 0))
        tp  = float(open_pos.get("tp", 0))
        qty = float(open_pos.get("quantity", 0))
        direction = open_pos.get("direction", "LONG")
        x0, x1 = df.index[0], df.index[-1]

        if ep:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=ep, y1=ep,
                          line=dict(color="#f0c040", width=1.5, dash="dot"),
                          row=1, col=1)
            fig.add_annotation(x=x1, y=ep, text=f"  Giris {ep:,.0f}",
                               showarrow=False, font=dict(color="#f0c040", size=10),
                               xanchor="left", row=1, col=1)
        if sl:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=sl, y1=sl,
                          line=dict(color="#ef5350", width=1.5, dash="dash"),
                          row=1, col=1)
            fig.add_annotation(x=x1, y=sl, text=f"  SL {sl:,.0f}",
                               showarrow=False, font=dict(color="#ef5350", size=10),
                               xanchor="left", row=1, col=1)
        if tp:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=tp, y1=tp,
                          line=dict(color="#26a69a", width=1.5, dash="dash"),
                          row=1, col=1)
            fig.add_annotation(x=x1, y=tp, text=f"  TP {tp:,.0f}",
                               showarrow=False, font=dict(color="#26a69a", size=10),
                               xanchor="left", row=1, col=1)

        # Risk bolgesi shading
        if ep and sl:
            fig.add_hrect(
                y0=min(ep, sl), y1=max(ep, sl),
                fillcolor="rgba(239,83,80,0.08)",
                layer="below", line_width=0, row=1, col=1,
            )
        if ep and tp:
            fig.add_hrect(
                y0=min(ep, tp), y1=max(ep, tp),
                fillcolor="rgba(38,166,154,0.08)",
                layer="below", line_width=0, row=1, col=1,
            )

    # ----- Ozet kutu (annotation) -----
    win_trades  = [t for t in state.get("trades", []) if float(t.get("pnl", 0)) >= 0]
    all_trades  = state.get("trades", [])
    wr = len(win_trades) / len(all_trades) * 100 if all_trades else 0
    pos_text = f"ACIK: {open_pos.get('direction','?')} @ {float(open_pos.get('entry_price',0)):,.0f}" if open_pos else "Pozisyon: YOK"
    pnl_color = "#26a69a" if pnl_total >= 0 else "#ef5350"

    summary_text = (
        f"<b>BTC/USDT 1H  |  {_now_tr()} (UTC+3)</b><br>"
        f"Kapital: <b>{current_eq:,.2f} USDT</b>  |  "
        f"PnL: <span style='color:{pnl_color}'><b>{pnl_total:+.2f}$ ({pnl_pct:+.2f}%)</b></span><br>"
        f"Islem: {len(all_trades)}  |  WR: {wr:.1f}%  |  Iter: {iteration}<br>"
        f"{pos_text}"
    )

    # --- Layout ---
    fig.update_layout(
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", size=11, family="Trebuchet MS"),
        title=dict(
            text=f"Cloud-Algo — BTC/USDT 1H Canli Grafik  |  {_now_tr()} (UTC+3)",
            font=dict(size=14, color="#d1d4dc"),
            x=0.02,
        ),
        height=800,
        margin=dict(l=60, r=80, t=60, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        annotations=[dict(
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            text=summary_text,
            showarrow=False,
            font=dict(size=11, color="#d1d4dc"),
            bgcolor="rgba(19,23,34,0.8)",
            bordercolor="#2a2e39",
            borderwidth=1,
            borderpad=8,
            align="left",
        )],
    )

    # Eksen renkleri
    axis_style = dict(
        gridcolor="#1e222d",
        zerolinecolor="#2a2e39",
        tickfont=dict(color="#787b86"),
        linecolor="#2a2e39",
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Hacim", row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100], title_font=dict(size=10))

    # RSI renk bolgesi etiketler
    fig.add_annotation(x=df.index[-1], y=72, text="Asiri Al.", showarrow=False,
                       font=dict(size=8, color="#ef5350"), xanchor="right", row=3, col=1)
    fig.add_annotation(x=df.index[-1], y=28, text="Asiri Sat.", showarrow=False,
                       font=dict(size=8, color="#26a69a"), xanchor="right", row=3, col=1)

    # HTML export
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")

    # 30sn auto-refresh meta tag
    html_str = html_str.replace(
        "<head>",
        '<head>\n  <meta http-equiv="refresh" content="30">'
    )
    # Sayfa basligini guncelle
    html_str = html_str.replace(
        "<title>Plotly</title>",
        "<title>Cloud-Algo | BTC/USDT Live Chart</title>"
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_str)

    return str(OUTPUT_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud-Algo Canli Mum Grafigi")
    parser.add_argument("--bars",   type=int, default=150, help="Gosterilecek mum sayisi (varsayilan 150)")
    parser.add_argument("--open",   action="store_true",   help="HTML'i tarayicida ac")
    args = parser.parse_args()

    print(f"[live_chart] OHLCV cekiliyor ({args.bars} bar)...")
    out = build_chart(bars=args.bars)
    print(f"[live_chart] Grafik kaydedildi -> {out}")
    print(f"[live_chart] Otomatik yenileme: 30sn")

    if args.open:
        webbrowser.open(f"file:///{out.replace(chr(92), '/')}")
        print("[live_chart] Tarayici aciliyor...")
