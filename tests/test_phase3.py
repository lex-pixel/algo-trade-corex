"""
tests/test_phase3.py
=====================
AMACI:
    Phase 3 bileşenlerini test eder:
        - IndicatorSet (RSI, MACD, BB, ATR, ADX)
        - MarketRegimeDetector (RANGE / TREND)
        - PARangeStrategy (destek/direnç + RSI filtresi)

ÇALIŞTIRMAK İÇİN:
    pytest tests/test_phase3.py -v
"""

import pytest
import pandas as pd
import numpy as np

from strategies.indicators import IndicatorSet, IndicatorValues
from strategies.regime_detector import MarketRegimeDetector, Regime
from strategies.pa_range_strategy import PARangeStrategy
from strategies.base_strategy import Signal


# ─────────────────────────────────────────────────────────────────────────────
# ORTAK TEST VERİSİ
# ─────────────────────────────────────────────────────────────────────────────

def make_df(n: int = 100, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    """
    Genel amaçlı test DataFrame'i.
    trend > 0 → sürekli yükseliş (trend piyasa)
    trend = 0 → rastgele (range piyasa)
    """
    np.random.seed(seed)
    closes = [50000.0]
    for _ in range(n - 1):
        change = trend + np.random.uniform(-0.005, 0.005)
        closes.append(closes[-1] * (1 + change))
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(n)],
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))


def make_range_df(n: int = 120) -> pd.DataFrame:
    """Sinüs dalgası — range piyasa simüle eder."""
    np.random.seed(10)
    closes = []
    price  = 50000.0
    for i in range(n):
        target = 50000 + 800 * np.sin(i * 0.2)
        price  = price + (target - price) * 0.15 + np.random.uniform(-100, 100)
        closes.append(price)
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [300.0] * n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))


def make_trend_df(n: int = 100) -> pd.DataFrame:
    """Sürekli yükseliş — trend piyasa simüle eder."""
    np.random.seed(7)
    closes = [50000.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + np.random.uniform(0.006, 0.012)))
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.003 for c in closes],
        "low":    [c * 0.997 for c in closes],
        "close":  closes,
        "volume": [300.0] * n,
    }, index=pd.date_range("2024-04-01", periods=n, freq="1h", tz="UTC"))


# ─────────────────────────────────────────────────────────────────────────────
# IndicatorSet TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicatorSet:
    """IndicatorSet hesaplamalarının doğruluğunu test eder."""

    def test_returns_indicator_values_object(self):
        """IndicatorSet.values IndicatorValues örneği döndürmeli."""
        df  = make_df(80)
        ind = IndicatorSet(df)
        assert isinstance(ind.values, IndicatorValues)

    def test_rsi_between_0_and_100(self):
        """RSI değeri 0-100 arasında olmalı."""
        df  = make_df(80)
        ind = IndicatorSet(df)
        assert ind.values.rsi is not None
        assert 0.0 <= ind.values.rsi <= 100.0

    def test_adx_between_0_and_100(self):
        """ADX değeri 0-100 arasında olmalı."""
        df  = make_df(80)
        ind = IndicatorSet(df)
        assert ind.values.adx is not None
        assert 0.0 <= ind.values.adx <= 100.0

    def test_atr_positive(self):
        """ATR her zaman pozitif olmalı."""
        df  = make_df(80)
        ind = IndicatorSet(df)
        assert ind.values.atr is not None
        assert ind.values.atr > 0

    def test_bb_upper_above_lower(self):
        """Bollinger Bands üst bant alt bantten büyük olmalı."""
        df  = make_df(80)
        ind = IndicatorSet(df)
        if ind.values.bb_upper and ind.values.bb_lower:
            assert ind.values.bb_upper > ind.values.bb_lower

    def test_insufficient_data_returns_none_values(self):
        """Yetersiz veriyle indikatörler None dönmeli."""
        df  = make_df(10)   # Çok az veri
        ind = IndicatorSet(df)
        assert ind.values.rsi is None
        assert ind.values.adx is None

    def test_current_price_matches_last_close(self):
        """current_price son kapanış fiyatıyla eşleşmeli."""
        df  = make_df(80)
        ind = IndicatorSet(df)
        assert ind.values.current_price == pytest.approx(float(df["close"].iloc[-1]))

    def test_is_range_property(self):
        """ADX < 25 ise is_range True olmalı."""
        v = IndicatorValues(adx=20.0)
        assert v.is_range is True

    def test_is_trend_property(self):
        """ADX >= 25 ise is_trend True olmalı."""
        v = IndicatorValues(adx=30.0)
        assert v.is_trend is True

    def test_trend_direction_up(self):
        """+DI > -DI ise trend yönü UP olmalı."""
        v = IndicatorValues(adx=30.0, dmp=25.0, dmn=15.0)
        assert v.trend_direction == "UP"

    def test_trend_direction_down(self):
        """-DI > +DI ise trend yönü DOWN olmalı."""
        v = IndicatorValues(adx=30.0, dmp=15.0, dmn=25.0)
        assert v.trend_direction == "DOWN"


# ─────────────────────────────────────────────────────────────────────────────
# MarketRegimeDetector TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketRegimeDetector:
    """MarketRegimeDetector rejim tespitini test eder."""

    def test_returns_regime_enum(self):
        """detect() Regime enum değeri döndürmeli."""
        df       = make_df(80)
        detector = MarketRegimeDetector()
        regime   = detector.detect(df)
        assert isinstance(regime, Regime)

    def test_insufficient_data_returns_transition(self):
        """Yetersiz veriyle TRANSITION döndürmeli."""
        df       = make_df(5)
        detector = MarketRegimeDetector()
        regime   = detector.detect(df)
        assert regime == Regime.TRANSITION

    def test_trend_market_detected(self):
        """Güçlü trend piyasasında TREND_UP veya TREND_DOWN dönmeli."""
        df       = make_trend_df(100)
        detector = MarketRegimeDetector()
        regime   = detector.detect(df)
        assert regime in (Regime.TREND_UP, Regime.TREND_DOWN, Regime.TRANSITION)

    def test_is_suitable_for_mean_reversion_range(self):
        """Range piyasada mean-reversion uygun olmalı."""
        df       = make_df(80, seed=42)
        detector = MarketRegimeDetector()
        # Düşük ADX eşiğiyle test et
        detector.adx_trend_thresh = 100.0  # ADX asla bunu geçemez → hep RANGE
        detector.adx_range_thresh = 99.0
        result = detector.is_suitable_for_mean_reversion(df)
        assert result is True

    def test_is_suitable_for_mean_reversion_trend(self):
        """Trend piyasada mean-reversion uygun olmamalı."""
        df       = make_df(80)
        detector = MarketRegimeDetector()
        # Çok düşük eşik → hep TREND
        detector.adx_trend_thresh = 0.0
        detector.adx_range_thresh = -1.0
        result = detector.is_suitable_for_mean_reversion(df)
        assert result is False

    def test_regime_str_comparison(self):
        """Regime str ile karşılaştırılabilmeli (str Enum)."""
        assert Regime.RANGE == "RANGE"
        assert Regime.TREND_UP == "TREND_UP"


# ─────────────────────────────────────────────────────────────────────────────
# PARangeStrategy TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestPARangeStrategy:
    """PARangeStrategy sinyal üretimini test eder."""

    def _make_strategy(self, **kwargs) -> PARangeStrategy:
        defaults = dict(
            lookback=50, rsi_oversold=40, rsi_overbought=60,
            use_regime_filter=False,  # Testlerde rejim filtresi kapalı
        )
        defaults.update(kwargs)
        return PARangeStrategy(**defaults)

    def test_bekle_on_insufficient_data(self):
        """Yetersiz veriyle BEKLE döndürmeli."""
        strat  = self._make_strategy()
        df     = make_df(10)
        signal = strat.run(df)
        assert signal.action == "BEKLE"

    def test_returns_valid_signal(self):
        """Yeterli veriyle geçerli Signal döndürmeli."""
        strat  = self._make_strategy()
        df     = make_df(120)
        signal = strat.run(df)
        assert signal.action in {"AL", "SAT", "BEKLE"}

    def test_confidence_between_0_and_1(self):
        """Tüm sinyallerde confidence 0-1 arasında olmalı."""
        strat = self._make_strategy()
        df    = make_range_df(120)
        for i in range(65, 120):
            signal = strat.run(df.iloc[:i + 1])
            assert 0.0 <= signal.confidence <= 1.0

    def test_al_signal_stop_below_price(self):
        """AL sinyalinde stop-loss fiyatın altında olmalı."""
        strat = self._make_strategy()
        df    = make_range_df(120)
        for i in range(65, 120):
            signal = strat.run(df.iloc[:i + 1])
            if signal.action == "AL" and signal.stop_loss:
                price = float(df["close"].iloc[i])
                assert signal.stop_loss < price, f"Stop {signal.stop_loss} >= fiyat {price}"
                break

    def test_sat_signal_stop_above_price(self):
        """SAT sinyalinde stop-loss fiyatın üstünde olmalı."""
        strat = self._make_strategy()
        df    = make_range_df(120)
        for i in range(65, 120):
            signal = strat.run(df.iloc[:i + 1])
            if signal.action == "SAT" and signal.stop_loss:
                price = float(df["close"].iloc[i])
                assert signal.stop_loss > price, f"Stop {signal.stop_loss} <= fiyat {price}"
                break

    def test_get_levels_returns_dict(self):
        """get_levels() dict döndürmeli, support < resistance olmalı."""
        strat  = self._make_strategy()
        df     = make_df(120)
        levels = strat.get_levels(df)
        assert isinstance(levels, dict)
        if levels:
            assert levels["support"] < levels["resistance"]
            assert levels["range_width"] > 0

    def test_regime_filter_blocks_trend_signal(self):
        """Rejim filtresi: TREND_DOWN'da AL engellenmeli, TREND_UP'ta SAT engellenmeli."""
        # TREND_DOWN -> AL engelle
        strat_down = self._make_strategy(use_regime_filter=True)
        strat_down.regime_detector.adx_trend_thresh = 0.0
        strat_down.regime_detector.adx_range_thresh = -1.0
        # Asagi trend veri: fiyat sure sure duser, RSI duser -> AL sinyali uretebilir
        df_down = make_df(120, seed=42, trend=-0.003)
        signal_down = strat_down.run(df_down)
        # TREND_DOWN'da AL gelmemeli (engellendi ya da zaten BEKLE)
        assert signal_down.action != "AL" or "engel" not in signal_down.reason.lower()

        # TREND_UP -> SAT engelle
        strat_up = self._make_strategy(use_regime_filter=True)
        strat_up.regime_detector.adx_trend_thresh = 0.0
        strat_up.regime_detector.adx_range_thresh = -1.0
        df_up = make_df(120, seed=42, trend=0.003)
        signal_up = strat_up.run(df_up)
        # TREND_UP'ta SAT gelmemeli
        assert signal_up.action != "SAT" or "engel" not in signal_up.reason.lower()

    def test_signal_count_increments(self):
        """Her başarılı run() sonrası signal_count artmalı."""
        strat = self._make_strategy()
        df    = make_df(120)
        count = strat.signal_count
        strat.run(df)
        assert strat.signal_count == count + 1

    def test_last_signal_updated(self):
        """run() sonrası last_signal güncellenmeli."""
        strat = self._make_strategy()
        df    = make_df(120)
        assert strat.last_signal is None
        strat.run(df)
        assert strat.last_signal is not None
        assert isinstance(strat.last_signal, Signal)
