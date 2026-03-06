"""
main.py
========
AMACI:
    Projenin tek başlangıç noktası.
    Config'i yükler, stratejiyi başlatır, sahte veri üzerinde çalıştırır.

    Phase 2'de bu dosya gerçek borsa verisini çekecek.
    Phase 6'da canlı trading loop'una dönüşecek.

ÇALIŞTIRMAK İÇİN:
    python main.py
"""

import numpy as np
import pandas as pd
from config.loader import get_config
from strategies.rsi_strategy import RSIStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # ── 1. Konfigürasyonu Yükle ──────────────────────────────────────────────
    cfg = get_config()

    logger.info("=" * 50)
    logger.info("ALGO TRADE CODEX basliyor")
    logger.info(f"Sembol   : {cfg.general.symbol}")
    logger.info(f"Timeframe: {cfg.general.timeframe}")
    logger.info(f"Testnet  : {cfg.general.testnet}")
    logger.info(f"Dry-run  : {cfg.general.dry_run}  (emir gonderilmez)")
    logger.info("=" * 50)

    # ── 2. Stratejiyi YAML'dan Gelen Parametrelerle Başlat ───────────────────
    rsi_cfg = cfg.strategies.rsi

    if not rsi_cfg.enabled:
        logger.warning("RSI stratejisi devre disi (settings.yaml -> strategies.rsi.enabled: false)")
        return

    strategy = RSIStrategy(
        symbol     = cfg.general.symbol,
        timeframe  = cfg.general.timeframe,
        rsi_period = rsi_cfg.rsi_period,
        oversold   = rsi_cfg.oversold,
        overbought = rsi_cfg.overbought,
        stop_pct   = rsi_cfg.stop_pct,
        tp_pct     = rsi_cfg.tp_pct,
    )

    # ── 3. Sahte Veri (Phase 2'de gerçek borsa verisiyle değişecek) ──────────
    logger.info("Sahte OHLCV verisi olusturuluyor (Phase 2'de gercek veri gelecek)...")
    np.random.seed(42)
    n = 100
    closes = [50000.0]
    for i in range(n - 1):
        change = np.random.uniform(-0.008, 0.008)
        closes.append(closes[-1] * (1 + change))

    df = pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(n)],
    }, index=pd.date_range("2024-01-01", periods=n, freq=cfg.general.timeframe))

    # ── 4. Son Sinyali Üret ───────────────────────────────────────────────────
    signal = strategy.run(df)

    logger.info("-" * 50)
    logger.info(f"Son Sinyal  : {signal.action}")
    logger.info(f"Guven Skoru : {signal.confidence:.2f}")
    logger.info(f"Sebep       : {signal.reason}")

    if signal.stop_loss:
        logger.info(f"Stop-Loss   : {signal.stop_loss:,.2f} USDT")
    if signal.take_profit:
        logger.info(f"Take-Profit : {signal.take_profit:,.2f} USDT")

    # ── 5. Dry-Run Kontrolü ───────────────────────────────────────────────────
    if signal.is_tradeable(min_confidence=rsi_cfg.min_confidence):
        if cfg.general.dry_run:
            logger.info(f"DRY-RUN: '{signal.action}' sinyali isleme alindi ama emir gonderilmedi.")
        else:
            logger.warning("CANLI MOD: Bu noktada emir gonderilecek (Phase 6'da implement edilecek)")
    else:
        logger.info(f"Sinyal islem icin yeterince guclu degil (guven < {rsi_cfg.min_confidence})")

    logger.info("=" * 50)
    logger.info("ALGO TRADE CODEX tamamlandi")


if __name__ == "__main__":
    main()
