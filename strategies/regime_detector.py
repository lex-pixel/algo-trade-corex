"""
strategies/regime_detector.py
==============================
AMACI:
    Piyasanın hangi "rejimde" olduğunu tespit eder:
        - RANGE    : Fiyat yatay seyrediyor, destek/direnç arası gidip geliyor
        - TREND_UP : Güçlü yukarı trend
        - TREND_DOWN: Güçlü aşağı trend
        - TRANSITION: Rejim değişiyor, henüz belli değil

NEDEN ÖNEMLİ?
    RSI ve PA Range stratejileri RANGE piyasada çalışır.
    Trend piyasada RSI yanlış sinyal üretir:
        - Güçlü yükselişte RSI hep 70+ olur → SAT sinyali → ama fiyat çıkmaya devam eder → ZARAR
    Bu sınıf, strateji seçimini otomatikleştirir.

KULLANIM:
    from strategies.regime_detector import MarketRegimeDetector, Regime

    detector = MarketRegimeDetector()
    regime   = detector.detect(df)

    if regime == Regime.RANGE:
        # RSI / PA Range stratejisini çalıştır
    elif regime == Regime.TREND_UP:
        # Trend stratejisi çalıştır (Phase 5'te eklenecek)
"""

from enum import Enum
import pandas as pd
import pandas_ta as ta
from strategies.indicators import IndicatorSet, ADX_TREND_THRESHOLD
from utils.logger import get_logger

logger = get_logger(__name__)


class Regime(str, Enum):
    """
    Piyasa rejimlerini temsil eden enum.
    str'den miras aldığı için doğrudan string olarak kullanılabilir:
        regime == "RANGE"  →  True (Regime.RANGE == "RANGE")
    """
    RANGE       = "RANGE"        # Yatay piyasa — RSI/PA iyi çalışır
    TREND_UP    = "TREND_UP"     # Yukarı trend
    TREND_DOWN  = "TREND_DOWN"   # Aşağı trend
    TRANSITION  = "TRANSITION"   # Geçiş dönemi, sinyal üretme


class MarketRegimeDetector:
    """
    ADX tabanlı piyasa rejim dedektörü.

    Parametreler:
        adx_period        : ADX periyodu (varsayılan 14)
        adx_trend_thresh  : Bu değerin üstü = trend (varsayılan 25)
        adx_range_thresh  : Bu değerin altı = range (varsayılan 20)
                            20-25 arası = TRANSITION

    Neden iki eşik?
        Tek eşik (25) kullanılırsa:
            ADX 24.9 → RANGE → RSI sinyali üret
            ADX 25.1 → TREND → RSI sinyali üretme
        Bu "bant genişliği" sayesinde sürekli rejim değişimi önlenir.
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_thresh: float = 25.0,
        adx_range_thresh: float = 20.0,
    ):
        self.adx_period       = adx_period
        self.adx_trend_thresh = adx_trend_thresh
        self.adx_range_thresh = adx_range_thresh

    def detect(self, df: pd.DataFrame) -> Regime:
        """
        DataFrame'i alır, piyasa rejimini döndürür.

        Args:
            df: OHLCV DataFrame, en az 30 satır önerilir

        Returns:
            Regime: RANGE | TREND_UP | TREND_DOWN | TRANSITION
        """
        if len(df) < self.adx_period + 2:
            logger.warning(f"Rejim tespiti icin yetersiz veri ({len(df)} satir)")
            return Regime.TRANSITION

        ind    = IndicatorSet(df, adx_period=self.adx_period)
        values = ind.values

        if values.adx is None:
            logger.warning("ADX hesaplanamadi, TRANSITION donuluyor")
            return Regime.TRANSITION

        adx = values.adx
        dmp = values.dmp   # +DI (yukarı yön gücü)
        dmn = values.dmn   # -DI (aşağı yön gücü)

        # Rejim kararı
        if adx < self.adx_range_thresh:
            # Zayıf trend = kesin range
            regime = Regime.RANGE

        elif adx < self.adx_trend_thresh:
            # Belirsiz bölge — TRANSITION
            regime = Regime.TRANSITION

        else:
            # Güçlü trend — yön tespiti için +DI / -DI karşılaştır
            if dmp is not None and dmn is not None:
                regime = Regime.TREND_UP if dmp > dmn else Regime.TREND_DOWN
            else:
                regime = Regime.TRANSITION

        dmp_str = f"{dmp:.1f}" if dmp is not None else "N/A"
        dmn_str = f"{dmn:.1f}" if dmn is not None else "N/A"
        logger.debug(
            f"Piyasa rejimi: {regime.value} | "
            f"ADX: {adx:.1f} | "
            f"+DI: {dmp_str} | "
            f"-DI: {dmn_str}"
        )

        return regime

    def is_suitable_for_mean_reversion(self, df: pd.DataFrame) -> bool:
        """
        RSI / PA Range gibi mean-reversion stratejileri için
        piyasa uygun mu?

        Returns:
            True → RANGE veya TRANSITION rejiminde (sinyal üretilebilir)
            False → TREND rejiminde (sinyal üretme)
        """
        regime = self.detect(df)
        suitable = regime in (Regime.RANGE, Regime.TRANSITION)
        if not suitable:
            logger.info(f"Mean-reversion icin uygun degil: {regime.value}")
        return suitable


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOGU
# python -m strategies.regime_detector
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    print("=" * 60)
    print("  ALGO TRADE CODEX - MarketRegimeDetector Test")
    print("=" * 60)

    np.random.seed(5)
    detector = MarketRegimeDetector()

    # Senaryo 1: Range piyasa (küçük rastgele hareketler)
    print("\n[SENARYO 1] Range piyasa (kucuk dalgalanma):")
    closes = [50000.0]
    for _ in range(79):
        closes.append(closes[-1] * (1 + np.random.uniform(-0.004, 0.004)))
    df_range = pd.DataFrame({
        "open": [c * 0.999 for c in closes], "high": [c * 1.001 for c in closes],
        "low":  [c * 0.999 for c in closes], "close": closes,
        "volume": [300.0] * 80,
    }, index=pd.date_range("2024-01-01", periods=80, freq="1h"))
    regime = detector.detect(df_range)
    print(f"  Tespit edilen rejim: {regime.value}")
    print(f"  Mean-reversion icin uygun: {detector.is_suitable_for_mean_reversion(df_range)}")

    # Senaryo 2: Trend piyasa (sürekli yükseliş)
    print("\n[SENARYO 2] Trend piyasa (surekli yukselis):")
    closes2 = [50000.0]
    for _ in range(79):
        closes2.append(closes2[-1] * (1 + np.random.uniform(0.005, 0.015)))
    df_trend = pd.DataFrame({
        "open": [c * 0.999 for c in closes2], "high": [c * 1.003 for c in closes2],
        "low":  [c * 0.997 for c in closes2], "close": closes2,
        "volume": [300.0] * 80,
    }, index=pd.date_range("2024-04-01", periods=80, freq="1h"))
    regime2 = detector.detect(df_trend)
    print(f"  Tespit edilen rejim: {regime2.value}")
    print(f"  Mean-reversion icin uygun: {detector.is_suitable_for_mean_reversion(df_trend)}")

    print("\nBASARI: MarketRegimeDetector calisiyor!")
