"""
main.py
========
AMACI:
    Projenin tek başlangıç noktası.
    Config'i yükler, stratejileri başlatır, Binance Testnet'ten veri çeker,
    sinyalleri üretir ve kombine eder.

ÇALIŞTIRMAK İÇİN:
    python main.py
"""

import numpy as np
import pandas as pd
from config.loader import get_config
from strategies.rsi_strategy import RSIStrategy
from strategies.pa_range_strategy import PARangeStrategy
from strategies.regime_detector import MarketRegimeDetector
from data.fetcher import BinanceFetcher
from data.cleaner import OHLCVCleaner
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

    # ── 2. Veri Çek ───────────────────────────────────────────────────────────
    df = _get_data(cfg)
    if df.empty:
        logger.error("Veri alinamadi, program durduruluyor")
        return

    # ── 3. Piyasa Rejimini Tespit Et ─────────────────────────────────────────
    detector = MarketRegimeDetector()
    regime   = detector.detect(df)
    logger.info(f"Piyasa rejimi: {regime.value}")

    # ── 4. Stratejileri Başlat ve Sinyal Üret ────────────────────────────────
    signals = []

    # RSI Stratejisi
    rsi_cfg = cfg.strategies.rsi
    if rsi_cfg.enabled:
        rsi_strategy = RSIStrategy(
            symbol     = cfg.general.symbol,
            timeframe  = cfg.general.timeframe,
            rsi_period = rsi_cfg.rsi_period,
            oversold   = rsi_cfg.oversold,
            overbought = rsi_cfg.overbought,
            stop_pct   = rsi_cfg.stop_pct,
            tp_pct     = rsi_cfg.tp_pct,
        )
        rsi_signal = rsi_strategy.run(df)
        signals.append(("RSI", rsi_signal, rsi_cfg.min_confidence))
        logger.info(f"RSI Sinyali    : {rsi_signal.action} (guven: {rsi_signal.confidence:.2f})")

    # PA Range Stratejisi
    pa_cfg = cfg.strategies.pa_range
    if pa_cfg.enabled:
        pa_strategy = PARangeStrategy(
            symbol             = cfg.general.symbol,
            timeframe          = cfg.general.timeframe,
            lookback           = pa_cfg.lookback,
            rsi_period         = pa_cfg.rsi_period,
            rsi_oversold       = pa_cfg.rsi_oversold,
            rsi_overbought     = pa_cfg.rsi_overbought,
            proximity_pct      = pa_cfg.proximity_pct,
            stop_pct           = pa_cfg.stop_pct,
            tp_pct             = pa_cfg.tp_pct,
            use_regime_filter  = pa_cfg.use_regime_filter,
        )
        pa_signal = pa_strategy.run(df)

        # Destek/direnç seviyelerini logla
        levels = pa_strategy.get_levels(df)
        if levels:
            logger.info(
                f"PA Range Seviyeleri | "
                f"Destek: {levels['support']:,.0f} | "
                f"Direnc: {levels['resistance']:,.0f} | "
                f"Range: %{levels['range_pct']:.1f}"
            )
        signals.append(("PA_RANGE", pa_signal, pa_cfg.min_confidence))
        logger.info(f"PA Range Sinyali: {pa_signal.action} (guven: {pa_signal.confidence:.2f})")

    # ── 5. Sinyal Kombinasyonu ────────────────────────────────────────────────
    final_signal = _combine_signals(signals)

    logger.info("-" * 50)
    logger.info(f"FINAL SINYAL : {final_signal.action}")
    logger.info(f"Guven Skoru  : {final_signal.confidence:.2f}")
    logger.info(f"Sebep        : {final_signal.reason}")
    if final_signal.stop_loss:
        logger.info(f"Stop-Loss    : {final_signal.stop_loss:,.2f} USDT")
    if final_signal.take_profit:
        logger.info(f"Take-Profit  : {final_signal.take_profit:,.2f} USDT")

    # ── 6. Dry-Run Kontrolü ───────────────────────────────────────────────────
    min_conf = cfg.strategies.rsi.min_confidence
    if final_signal.is_tradeable(min_confidence=min_conf):
        if cfg.general.dry_run:
            logger.info(f"DRY-RUN: '{final_signal.action}' sinyali isleme alindi ama emir gonderilmedi.")
        else:
            logger.warning("CANLI MOD: Bu noktada emir gonderilecek (Phase 6'da implement edilecek)")
    else:
        logger.info(f"Sinyal islem icin yeterince guclu degil (guven < {min_conf})")

    logger.info("=" * 50)
    logger.info("ALGO TRADE CODEX tamamlandi")


def _combine_signals(signals: list) -> object:
    """
    Birden fazla stratejinin sinyalini kombine eder.

    Kural:
        - İki strateji aynı yönde → güven skoru ortalaması alınır (güçlü sinyal)
        - Stratejiler çelişiyorsa → BEKLE (belirsizlik var)
        - Sadece biri AL/SAT → o sinyali kullan (diğeri BEKLE'yse sorun yok)
    """
    from strategies.base_strategy import Signal

    if not signals:
        return Signal(action="BEKLE", confidence=0.0, reason="Hic strateji aktif degil")

    # Sadece işlem yapılabilir sinyalleri al
    tradeable = [
        (name, sig) for name, sig, min_conf in signals
        if sig.is_tradeable(min_confidence=min_conf)
    ]

    if not tradeable:
        # Hepsi BEKLE — en son sinyali döndür
        last_name, last_sig, _ = signals[-1]
        return Signal(
            action="BEKLE", confidence=0.0,
            reason=f"Tum stratejiler BEKLE | Son: {last_sig.reason}"
        )

    # Yönleri kontrol et
    actions = [sig.action for _, sig in tradeable]
    al_count  = actions.count("AL")
    sat_count = actions.count("SAT")

    if al_count > 0 and sat_count > 0:
        # Çelişki var → BEKLE
        return Signal(
            action="BEKLE", confidence=0.0,
            reason=f"Strateji celiskisi: {al_count} AL vs {sat_count} SAT"
        )

    # Tüm sinyaller aynı yönde
    dominant_action = "AL" if al_count > 0 else "SAT"
    avg_confidence  = sum(sig.confidence for _, sig in tradeable) / len(tradeable)
    combined_reason = " | ".join(f"{n}: {s.reason}" for n, s in tradeable)

    # Stop/TP için ilk sinyali kullan
    first_sig = tradeable[0][1]
    return Signal(
        action=dominant_action,
        confidence=round(avg_confidence, 3),
        stop_loss=first_sig.stop_loss,
        take_profit=first_sig.take_profit,
        reason=f"[KOMBINE] {combined_reason}"
    )


def _get_data(cfg) -> pd.DataFrame:
    """Binance Testnet'ten veri çekmeye çalışır, hata varsa sahte veriye döner."""
    try:
        fetcher  = BinanceFetcher(
            testnet   = cfg.general.testnet,
            symbol    = cfg.general.symbol,
            timeframe = cfg.general.timeframe,
        )
        df_raw   = fetcher.fetch_ohlcv(limit=200)
        if df_raw.empty:
            raise ValueError("Borsadan bos veri dondu")

        cleaner  = OHLCVCleaner()
        df_clean = cleaner.clean(df_raw)
        report   = cleaner.validate(df_clean)
        logger.info(
            f"Gercek veri kullaniliyor | "
            f"{report['rows']} mum | {report['date_start']} -> {report['date_end']}"
        )
        return df_clean

    except Exception as e:
        logger.warning(f"Borsa verisi alinamadi ({type(e).__name__}: {e})")
        logger.warning("Sahte veriyle devam ediliyor (fallback)")
        return _make_fake_data(cfg)


def _make_fake_data(cfg) -> pd.DataFrame:
    """Fallback: sahte OHLCV verisi üretir."""
    np.random.seed(42)
    n = 200
    closes = [50000.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + np.random.uniform(-0.008, 0.008)))
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(n)],
    }, index=pd.date_range("2024-01-01", periods=n, freq=cfg.general.timeframe))


if __name__ == "__main__":
    main()
