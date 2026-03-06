"""
tests/test_strategies.py
=========================
AMACI:
    BaseStrategy, Signal, HelloStrategy ve RSIStrategy sınıflarının
    doğru çalıştığını otomatik olarak doğrular.

ÇALIŞTIRMAK İÇİN:
    pytest tests/test_strategies.py -v

    -v flag'i her testin adını ve sonucunu gösterir.
    Hepsi yeşil (PASSED) ise sistem sağlıklı demektir.

NEDEN TEST YAZIYORUZ?
    İleride RSIStrategy'yi değiştirirken yanlışlıkla bir şeyi bozarsak,
    testler anında haber verir. Manuel kontrol yerine otomatik güvenlik ağı.
"""

import pytest
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, Signal
from strategies.hello_strategy import HelloStrategy
from strategies.rsi_strategy import RSIStrategy


# ─────────────────────────────────────────────────────────────────────────────
# ORTAK TEST VERİSİ (Fixture)
# pytest fixture: her test fonksiyonuna otomatik parametre olarak gönderilir
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df_small():
    """5 satırlık minimal OHLCV DataFrame — veri yetersizliği testleri için."""
    return pd.DataFrame({
        "open":   [100, 101, 102, 103, 104],
        "high":   [105, 106, 107, 108, 109],
        "low":    [95,  96,  97,  98,  99],
        "close":  [101, 102, 103, 104, 105],
        "volume": [1000, 1100, 900, 1200, 800],
    }, index=pd.date_range("2024-01-01", periods=5, freq="1h"))


@pytest.fixture
def sample_df_large():
    """100 satırlık OHLCV DataFrame — RSI testleri için yeterli geçmiş."""
    np.random.seed(42)
    closes = [50000.0]
    for _ in range(99):
        closes.append(closes[-1] * (1 + np.random.uniform(-0.005, 0.005)))
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(100)],
    }, index=pd.date_range("2024-01-01", periods=100, freq="1h"))


@pytest.fixture
def oversold_df():
    """RSI'nin oversold bölgesine girip çıkacağı sahte veri — AL sinyali beklenir."""
    np.random.seed(1)
    closes = [50000.0]
    # 40 mum düşüş (RSI oversold'a girer)
    for _ in range(40):
        closes.append(closes[-1] * (1 + np.random.uniform(-0.015, 0.001)))
    # 20 mum yükseliş (RSI oversold'dan çıkar → AL sinyali)
    for _ in range(20):
        closes.append(closes[-1] * (1 + np.random.uniform(0.005, 0.020)))
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.003 for c in closes],
        "low":    [c * 0.997 for c in closes],
        "close":  closes,
        "volume": [500.0] * 61,
    }, index=pd.date_range("2024-01-01", periods=61, freq="1h"))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SINIFI TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestSignal:
    """Signal dataclass'ının doğru çalışıp çalışmadığını test eder."""

    def test_valid_signal_al(self):
        """AL sinyali sorunsuz oluşturulabilmeli."""
        s = Signal(action="AL", confidence=0.8, stop_loss=49000, take_profit=52000)
        assert s.action == "AL"
        assert s.confidence == 0.8
        assert s.stop_loss == 49000
        assert s.take_profit == 52000

    def test_valid_signal_bekle(self):
        """BEKLE sinyali (stop/tp olmadan) oluşturulabilmeli."""
        s = Signal(action="BEKLE", confidence=0.0)
        assert s.action == "BEKLE"
        assert s.stop_loss is None
        assert s.take_profit is None

    def test_invalid_action_raises(self):
        """Geçersiz action değeri ValueError fırlatmalı."""
        with pytest.raises(ValueError, match="Geçersiz action"):
            Signal(action="YUKSEL", confidence=0.5)   # 'YUKSEL' geçerli değil

    def test_confidence_out_of_range_raises(self):
        """Confidence 0-1 dışında ValueError fırlatmalı."""
        with pytest.raises(ValueError, match="Confidence"):
            Signal(action="AL", confidence=1.5)        # 1.5 > 1.0

    def test_is_tradeable_above_threshold(self):
        """Yüksek güvenli AL sinyali işlem yapılabilir olmalı."""
        s = Signal(action="AL", confidence=0.75)
        assert s.is_tradeable(min_confidence=0.6) is True

    def test_bekle_not_tradeable(self):
        """BEKLE sinyali asla işlem yapılabilir olmamalı."""
        s = Signal(action="BEKLE", confidence=1.0)
        assert s.is_tradeable() is False

    def test_low_confidence_not_tradeable(self):
        """Düşük güvenli sinyal işlem yapılabilir olmamalı."""
        s = Signal(action="AL", confidence=0.3)
        assert s.is_tradeable(min_confidence=0.6) is False


# ─────────────────────────────────────────────────────────────────────────────
# BASE STRATEGY TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseStrategy:
    """BaseStrategy'nin doğrulama ve çalışma mantığını test eder."""

    def test_validate_data_valid(self, sample_df_small):
        """Geçerli DataFrame doğrulamayı geçmeli."""
        strat = HelloStrategy()
        assert strat.validate_data(sample_df_small) is True

    def test_validate_data_missing_column(self, sample_df_small):
        """Eksik sütun varsa doğrulama False döndürmeli."""
        strat = HelloStrategy()
        df_bad = sample_df_small.drop(columns=["volume"])
        assert strat.validate_data(df_bad) is False

    def test_validate_data_too_short(self):
        """Tek satırlık veri doğrulamayı geçmemeli."""
        strat = HelloStrategy()
        df_one = pd.DataFrame({
            "open": [100], "high": [105], "low": [95],
            "close": [101], "volume": [1000]
        })
        assert strat.validate_data(df_one) is False

    def test_run_returns_bekle_on_bad_data(self):
        """Hatalı veri gelince run() çökmemeli, BEKLE döndürmeli."""
        strat = HelloStrategy()
        df_bad = pd.DataFrame({"open": [1], "high": [2], "low": [0], "close": [1], "volume": [100]})
        signal = strat.run(df_bad)
        assert signal.action == "BEKLE"

    def test_signal_count_increments(self, sample_df_large):
        """Her başarılı run() sonrası signal_count artmalı."""
        strat = HelloStrategy()
        initial_count = strat.signal_count
        strat.run(sample_df_large)
        assert strat.signal_count == initial_count + 1

    def test_last_signal_updated(self, sample_df_large):
        """run() sonrası last_signal güncellenmeli."""
        strat = HelloStrategy()
        assert strat.last_signal is None
        strat.run(sample_df_large)
        assert strat.last_signal is not None
        assert isinstance(strat.last_signal, Signal)


# ─────────────────────────────────────────────────────────────────────────────
# HELLO STRATEGY TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestHelloStrategy:
    """HelloStrategy'nin sinyal üretme mantığını test eder."""

    def _make_df(self, closes: list) -> pd.DataFrame:
        """Verilen kapanış fiyatlarından test DataFrame'i oluşturur."""
        return pd.DataFrame({
            "open":   [c * 0.999 for c in closes],
            "high":   [c * 1.002 for c in closes],
            "low":    [c * 0.998 for c in closes],
            "close":  closes,
            "volume": [1000.0] * len(closes),
        }, index=pd.date_range("2024-01-01", periods=len(closes), freq="1h"))

    def test_al_signal_on_rise(self):
        """Fiyat yeterince yükseldiyse AL sinyali üretilmeli."""
        strat = HelloStrategy(threshold=0.001)
        # Fiyat 100'den 101'e çıktı → %1 artış > %0.1 eşik
        df = self._make_df([100.0, 101.0])
        signal = strat.run(df)
        assert signal.action == "AL"
        assert signal.confidence > 0

    def test_sat_signal_on_drop(self):
        """Fiyat yeterince düştüyse SAT sinyali üretilmeli."""
        strat = HelloStrategy(threshold=0.001)
        df = self._make_df([100.0, 98.0])
        signal = strat.run(df)
        assert signal.action == "SAT"

    def test_bekle_signal_on_flat(self):
        """Fiyat eşik altında değişti ise BEKLE sinyali üretilmeli."""
        strat = HelloStrategy(threshold=0.01)    # %1 eşik
        df = self._make_df([100.0, 100.05])      # sadece %0.05 artış
        signal = strat.run(df)
        assert signal.action == "BEKLE"

    def test_stop_loss_below_price_on_al(self):
        """AL sinyalinde stop-loss, giriş fiyatının altında olmalı."""
        strat = HelloStrategy()
        df = self._make_df([100.0, 101.0])
        signal = strat.run(df)
        assert signal.action == "AL"
        assert signal.stop_loss < 101.0

    def test_take_profit_above_price_on_al(self):
        """AL sinyalinde take-profit, giriş fiyatının üstünde olmalı."""
        strat = HelloStrategy()
        df = self._make_df([100.0, 101.0])
        signal = strat.run(df)
        assert signal.take_profit > 101.0


# ─────────────────────────────────────────────────────────────────────────────
# RSI STRATEGY TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestRSIStrategy:
    """RSIStrategy'nin RSI hesaplama ve sinyal mantığını test eder."""

    def test_bekle_on_insufficient_data(self, sample_df_small):
        """5 satırlık veriyle RSI hesaplanamaz → BEKLE döndürmeli."""
        strat = RSIStrategy(rsi_period=14)
        signal = strat.run(sample_df_small)
        assert signal.action == "BEKLE"

    def test_returns_signal_on_enough_data(self, sample_df_large):
        """100 satırlık veriyle RSI hesaplanabilmeli → geçerli Signal döndürmeli."""
        strat = RSIStrategy(rsi_period=14)
        signal = strat.run(sample_df_large)
        assert signal.action in {"AL", "SAT", "BEKLE"}
        assert isinstance(signal.confidence, float)

    def test_al_signal_on_oversold(self, oversold_df):
        """Oversold'dan çıkış senaryosunda AL sinyali üretilmeli."""
        strat = RSIStrategy(rsi_period=14, oversold=30)
        al_found = False
        for i in range(15, len(oversold_df)):
            signal = strat.run(oversold_df.iloc[:i + 1])
            if signal.action == "AL":
                al_found = True
                break
        assert al_found, "Oversold senaryosunda AL sinyali bekleniyor"

    def test_get_current_rsi_returns_float(self, sample_df_large):
        """get_current_rsi() float döndürmeli ve 0-100 arasında olmalı."""
        strat = RSIStrategy(rsi_period=14)
        rsi = strat.get_current_rsi(sample_df_large)
        assert rsi is not None
        assert 0.0 <= rsi <= 100.0

    def test_get_current_rsi_none_on_small_data(self, sample_df_small):
        """Yetersiz veride get_current_rsi() None döndürmeli."""
        strat = RSIStrategy(rsi_period=14)
        assert strat.get_current_rsi(sample_df_small) is None

    def test_confidence_between_0_and_1(self, oversold_df):
        """Üretilen tüm sinyallerin confidence değeri 0-1 arasında olmalı."""
        strat = RSIStrategy(rsi_period=14)
        for i in range(15, len(oversold_df)):
            signal = strat.run(oversold_df.iloc[:i + 1])
            assert 0.0 <= signal.confidence <= 1.0
