"""
dashboard.py
=============
AMACI:
    Rich kütüphanesi ile terminalde canlı sinyal paneli gösterir.
    Her N saniyede bir Binance'ten veri çekip sinyalleri günceller.

ÇALIŞTIRMAK İÇİN:
    python dashboard.py            # 60 saniyede bir günceller
    python dashboard.py --once     # Bir kez göster, çık
    python dashboard.py --interval 30  # 30 saniyede bir güncelle
"""

import argparse
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box

from config.loader import get_config
from data.fetcher import BinanceFetcher
from data.cleaner import OHLCVCleaner
from strategies.rsi_strategy import RSIStrategy
from strategies.pa_range_strategy import PARangeStrategy
from strategies.indicators import IndicatorSet
from strategies.regime_detector import MarketRegimeDetector, Regime
from utils.logger import get_logger

logger  = get_logger(__name__)
console = Console()


def build_dashboard(df, cfg) -> Layout:
    """
    Tüm widget'ları içeren dashboard layout'unu oluşturur.
    """
    rsi_cfg = cfg.strategies.rsi
    pa_cfg  = cfg.strategies.pa_range

    # ── İndikatörler ─────────────────────────────────────────────────────────
    ind     = IndicatorSet(df)
    v       = ind.values
    detector = MarketRegimeDetector()
    regime   = detector.detect(df)

    # ── Stratejiler ───────────────────────────────────────────────────────────
    rsi_strategy = RSIStrategy(
        symbol=cfg.general.symbol, timeframe=cfg.general.timeframe,
        rsi_period=rsi_cfg.rsi_period, oversold=rsi_cfg.oversold,
        overbought=rsi_cfg.overbought, stop_pct=rsi_cfg.stop_pct,
        tp_pct=rsi_cfg.tp_pct,
    )
    pa_strategy = PARangeStrategy(
        symbol=cfg.general.symbol, timeframe=cfg.general.timeframe,
        lookback=pa_cfg.lookback, rsi_period=pa_cfg.rsi_period,
        rsi_oversold=pa_cfg.rsi_oversold, rsi_overbought=pa_cfg.rsi_overbought,
        proximity_pct=pa_cfg.proximity_pct, stop_pct=pa_cfg.stop_pct,
        tp_pct=pa_cfg.tp_pct, use_regime_filter=pa_cfg.use_regime_filter,
    )

    rsi_signal = rsi_strategy.generate_signal(df)
    pa_signal  = pa_strategy.generate_signal(df)
    levels     = pa_strategy.get_levels(df)

    # ── Fiyat değişimi ────────────────────────────────────────────────────────
    price      = v.current_price or 0.0
    prev_price = v.prev_price    or price
    pct_change = ((price - prev_price) / prev_price * 100) if prev_price else 0.0
    price_color  = "green" if pct_change >= 0 else "red"
    price_arrow  = "+" if pct_change >= 0 else ""

    # ── 1. Fiyat Paneli ───────────────────────────────────────────────────────
    price_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    price_table.add_column(style="dim", width=14)
    price_table.add_column(width=22)

    price_table.add_row(
        "Fiyat",
        Text(f"${price:,.2f}  [{price_arrow}{pct_change:.2f}%]", style=price_color)
    )
    price_table.add_row(
        "Sembol",
        Text(f"{cfg.general.symbol}  {cfg.general.timeframe}", style="bold white")
    )
    price_table.add_row(
        "Piyasa",
        _regime_text(regime)
    )
    price_table.add_row(
        "Mod",
        Text("TESTNET  DRY-RUN", style="yellow")
    )

    # ── 2. İndikatörler Paneli ────────────────────────────────────────────────
    ind_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    ind_table.add_column(style="dim", width=10)
    ind_table.add_column(width=12)
    ind_table.add_column(style="dim", width=10)
    ind_table.add_column(width=12)

    rsi_color = _rsi_color(v.rsi)
    adx_color = "red" if v.adx and v.adx > 25 else "green"

    ind_table.add_row(
        "RSI",   Text(f"{v.rsi:.1f}"   if v.rsi  else "N/A", style=rsi_color),
        "ADX",   Text(f"{v.adx:.1f}"   if v.adx  else "N/A", style=adx_color),
    )
    ind_table.add_row(
        "ATR",   Text(f"{v.atr:.1f}"   if v.atr  else "N/A", style="cyan"),
        "MACD",  Text(f"{v.macd_hist:.2f}" if v.macd_hist else "N/A",
                      style="green" if v.macd_hist and v.macd_hist > 0 else "red"),
    )
    ind_table.add_row(
        "BB Ust", Text(f"${v.bb_upper:,.0f}" if v.bb_upper else "N/A", style="dim"),
        "BB Alt", Text(f"${v.bb_lower:,.0f}" if v.bb_lower else "N/A", style="dim"),
    )

    # ── 3. Sinyal Paneli ──────────────────────────────────────────────────────
    sig_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    sig_table.add_column(style="dim", width=12)
    sig_table.add_column(width=14)
    sig_table.add_column(style="dim", width=8)
    sig_table.add_column(width=8)

    sig_table.add_row(
        "RSI",      _signal_text(rsi_signal.action),
        "Guven",    Text(f"{rsi_signal.confidence:.2f}", style="white"),
    )
    sig_table.add_row(
        "PA Range", _signal_text(pa_signal.action),
        "Guven",    Text(f"{pa_signal.confidence:.2f}", style="white"),
    )
    if rsi_signal.stop_loss:
        sig_table.add_row(
            "Stop-Loss",  Text(f"${rsi_signal.stop_loss:,.0f}", style="red"),
            "TP",         Text(f"${rsi_signal.take_profit:,.0f}" if rsi_signal.take_profit else "N/A", style="green"),
        )

    # ── 4. Seviyeler Paneli ───────────────────────────────────────────────────
    lvl_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    lvl_table.add_column(style="dim", width=12)
    lvl_table.add_column(width=14)

    if levels:
        support    = levels["support"]
        resistance = levels["resistance"]
        dist_sup   = (price - support)    / price * 100
        dist_res   = (resistance - price) / price * 100
        lvl_table.add_row("Destek",  Text(f"${support:,.0f}   (-%{dist_sup:.1f})", style="green"))
        lvl_table.add_row("Direnc",  Text(f"${resistance:,.0f}  (+%{dist_res:.1f})", style="red"))
        lvl_table.add_row("Range",   Text(f"%{levels['range_pct']:.1f}", style="cyan"))
    else:
        lvl_table.add_row("Seviyeler", Text("Yetersiz veri", style="dim"))

    # ── 5. Son mumlar ─────────────────────────────────────────────────────────
    candle_table = Table(
        "Zaman", "Acilis", "Yuksek", "Dusuk", "Kapanis", "Hacim",
        box=box.SIMPLE, show_header=True, header_style="bold dim",
        padding=(0, 1)
    )
    for _, row in df.tail(5).iterrows():
        c = "green" if row["close"] >= row["open"] else "red"
        candle_table.add_row(
            str(row.name)[:16],
            f"${row['open']:,.0f}",
            f"${row['high']:,.0f}",
            f"${row['low']:,.0f}",
            Text(f"${row['close']:,.0f}", style=c),
            f"{row['volume']:.1f}",
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="candles", size=9),
        Layout(name="footer", size=1),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="price"),
        Layout(name="indicators"),
    )
    layout["right"].split_column(
        Layout(name="signals"),
        Layout(name="levels"),
    )

    layout["header"].update(
        Panel(Text("  ALGO TRADE CODEX  —  Canli Sinyal Paneli", style="bold white", justify="center"),
              style="blue")
    )
    layout["price"].update(Panel(price_table, title="[bold]Fiyat", border_style="blue"))
    layout["indicators"].update(Panel(ind_table, title="[bold]Indikatorler", border_style="cyan"))
    layout["signals"].update(Panel(sig_table, title="[bold]Sinyaller", border_style="yellow"))
    layout["levels"].update(Panel(lvl_table, title="[bold]Destek / Direnc", border_style="green"))
    layout["candles"].update(Panel(candle_table, title="[bold]Son 5 Mum", border_style="dim"))
    layout["footer"].update(Text(
        f"  Son guncelleme: {now}  |  {len(df)} mum yuklendi  |  Ctrl+C ile cikis",
        style="dim", justify="center"
    ))

    return layout


def _regime_text(regime: Regime) -> Text:
    mapping = {
        Regime.RANGE:       ("RANGE      (RSI/PA aktif)", "green"),
        Regime.TREND_UP:    ("TREND_UP   (yukselis)", "bright_green"),
        Regime.TREND_DOWN:  ("TREND_DOWN (dusus)", "red"),
        Regime.TRANSITION:  ("TRANSITION (gecis)", "yellow"),
    }
    label, color = mapping.get(regime, (str(regime), "white"))
    return Text(label, style=color)


def _signal_text(action: str) -> Text:
    colors = {"AL": "bold green", "SAT": "bold red", "BEKLE": "dim white"}
    return Text(action, style=colors.get(action, "white"))


def _rsi_color(rsi) -> str:
    if rsi is None:
        return "white"
    if rsi < 30:
        return "green"
    if rsi > 70:
        return "red"
    return "yellow"


def run(interval: int = 60, once: bool = False):
    """Dashboard'u çalıştırır."""
    cfg = get_config()

    fetcher = BinanceFetcher(
        testnet   = cfg.general.testnet,
        symbol    = cfg.general.symbol,
        timeframe = cfg.general.timeframe,
    )
    cleaner = OHLCVCleaner()

    def fetch_data():
        try:
            df_raw = fetcher.fetch_ohlcv(limit=200)
            if df_raw.empty:
                raise ValueError("Bos veri")
            return cleaner.clean(df_raw)
        except Exception as e:
            logger.warning(f"Veri alinamadi: {e}")
            return None

    if once:
        df = fetch_data()
        if df is not None:
            console.print(build_dashboard(df, cfg))
        return

    # Canlı güncelleme modu
    console.print(f"[dim]Dashboard basliyor — her {interval}sn guncellenir. Cikis: Ctrl+C[/dim]\n")

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                df = fetch_data()
                if df is not None:
                    live.update(build_dashboard(df, cfg))
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard kapatildi.[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algo Trade Codex - Canli Dashboard")
    parser.add_argument("--interval", type=int, default=60, help="Guncelleme suresi (saniye)")
    parser.add_argument("--once",     action="store_true",   help="Bir kez goster, cik")
    args = parser.parse_args()
    run(interval=args.interval, once=args.once)
