"""
backtesting/run_backtest.py
============================
AMACI:
    RSI ve PA Range stratejilerini geçmiş BTC/USDT verisiyle test eder.
    Her iki stratejiyi yan yana karşılaştırır.

ÇALIŞTIRMAK İÇİN:
    python -m backtesting.run_backtest

ÇIKTI:
    - Her stratejinin performans özeti
    - Karşılaştırma tablosu
    - data/raw/ klasörüne backtest_btc_usdt_1h.parquet kaydedilir
"""

import sys
from pathlib import Path

# Proje kökünü Python path'ine ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config.loader import get_config
from data.fetcher import BinanceFetcher
from data.cleaner import OHLCVCleaner
from strategies.rsi_strategy import RSIStrategy
from strategies.pa_range_strategy import PARangeStrategy
from backtesting.engine import BacktestEngine
from backtesting.metrics import PerformanceMetrics
from utils.logger import get_logger

logger = get_logger(__name__)


def run():
    print("=" * 60)
    print("  ALGO TRADE CODEX — Backtest Motoru")
    print("=" * 60)

    cfg = get_config()

    # ── 1. Veri Çek ───────────────────────────────────────────────────────────
    df = _get_data(cfg)
    print(f"\nVeri: {len(df)} mum | {df.index[0].date()} -> {df.index[-1].date()}")

    # ── 2. Stratejileri Tanımla ───────────────────────────────────────────────
    rsi_cfg = cfg.strategies.rsi
    pa_cfg  = cfg.strategies.pa_range

    strategies = []

    if rsi_cfg.enabled:
        strategies.append(RSIStrategy(
            symbol     = cfg.general.symbol,
            timeframe  = cfg.general.timeframe,
            rsi_period = rsi_cfg.rsi_period,
            oversold   = rsi_cfg.oversold,
            overbought = rsi_cfg.overbought,
            stop_pct   = rsi_cfg.stop_pct,
            tp_pct     = rsi_cfg.tp_pct,
        ))

    if pa_cfg.enabled:
        strategies.append(PARangeStrategy(
            symbol            = cfg.general.symbol,
            timeframe         = cfg.general.timeframe,
            lookback          = pa_cfg.lookback,
            rsi_period        = pa_cfg.rsi_period,
            rsi_oversold      = pa_cfg.rsi_oversold,
            rsi_overbought    = pa_cfg.rsi_overbought,
            proximity_pct     = pa_cfg.proximity_pct,
            stop_pct          = pa_cfg.stop_pct,
            tp_pct            = pa_cfg.tp_pct,
            use_regime_filter = pa_cfg.use_regime_filter,
        ))

    # ── 3. Her Stratejiyi Backtest Et ─────────────────────────────────────────
    engine  = BacktestEngine(
        initial_capital    = 10_000.0,
        commission         = 0.001,    # %0.1 Binance Spot
        slippage           = 0.0005,   # %0.05
        max_risk_per_trade = 0.02,     # %2 risk per trade
    )

    results = []
    for strategy in strategies:
        print(f"\n[{strategy.name}] backtest calistirilıyor...")
        result = engine.run(df, strategy, warmup_bars=50)
        result.print_summary()
        results.append(result)

    # ── 4. Karşılaştırma Tablosu ──────────────────────────────────────────────
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("  STRATEJ KARSILASTIRMASI")
        print("=" * 60)
        comparison = PerformanceMetrics.compare(results)
        print(comparison.to_string())
        print("=" * 60)

    # ── 5. En İyi Strateji ────────────────────────────────────────────────────
    if results:
        best = max(results, key=lambda r: r.summary().get("sharpe_ratio", -999))
        print(f"\nEn iyi Sharpe oranina sahip strateji: {best.strategy_name}")
        print(f"Sharpe: {best.summary()['sharpe_ratio']:.3f}")

    print("\nBacktest tamamlandi!")


def _get_data(cfg) -> pd.DataFrame:
    """
    Önce kaydedilmiş Parquet'i kontrol eder, yoksa Binance'ten çeker.
    Bu sayede her çalıştırmada API'ye istek atmaz.
    """
    parquet_path = Path("data/raw/backtest_btc_usdt_1h.parquet")

    if parquet_path.exists():
        logger.info(f"Kayitli veri yukleniyor: {parquet_path}")
        df = BinanceFetcher.load_parquet(parquet_path)
        logger.info(f"Yuklendi: {len(df)} mum")
        return df

    logger.info("Veri yok, Binance'ten cekiliyor (90 gunluk)...")
    try:
        fetcher = BinanceFetcher(
            testnet   = cfg.general.testnet,
            symbol    = cfg.general.symbol,
            timeframe = cfg.general.timeframe,
        )
        # 90 günlük veri — batch fetch ile
        df_raw  = fetcher.fetch_since(since_days=90, batch_size=500)

        if df_raw.empty:
            raise ValueError("Borsadan bos veri")

        cleaner  = OHLCVCleaner()
        df_clean = cleaner.clean(df_raw)

        # Parquet'e kaydet
        fetcher.save_parquet(df_clean, filename="backtest_btc_usdt_1h.parquet")
        return df_clean

    except Exception as e:
        logger.warning(f"API verisi alinamadi: {e} — sahte veri kullaniliyor")
        return _make_fake_data(cfg)


def _make_fake_data(cfg) -> pd.DataFrame:
    """Fallback: range + trend karışık piyasa simüle eder."""
    np.random.seed(42)
    n = 500
    closes = [50000.0]
    for i in range(n - 1):
        # İlk 250: range piyasa
        # Son 250: trend piyasa (gerçekçi test için)
        if i < 250:
            change = np.random.uniform(-0.006, 0.006)
        else:
            change = np.random.uniform(-0.004, 0.008)  # Hafif yükseliş trendi
        closes.append(closes[-1] * (1 + change))

    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(n)],
    }, index=pd.date_range("2024-01-01", periods=n, freq=cfg.general.timeframe, tz="UTC"))


if __name__ == "__main__":
    run()
