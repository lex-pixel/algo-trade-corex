"""
tests/test_fetcher.py
======================
AMACI:
    BinanceFetcher ve OHLCVCleaner sınıflarının doğru çalıştığını test eder.
    Borsaya bağlanmadan (mock ile) çalışır — internet gerektirmez.

ÇALIŞTIRMAK İÇİN:
    pytest tests/test_fetcher.py -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from data.fetcher import BinanceFetcher
from data.cleaner import OHLCVCleaner


# ─────────────────────────────────────────────────────────────────────────────
# ORTAK YARDIMCILAR
# ─────────────────────────────────────────────────────────────────────────────

def make_raw_ccxt_data(n: int = 50, start_ts_ms: int = 1704067200000) -> list:
    """
    CCXT'nin döndürdüğü formatta sahte ham veri üretir.
    Format: [[timestamp_ms, open, high, low, close, volume], ...]
    """
    np.random.seed(42)
    price = 50000.0
    data  = []
    ts    = start_ts_ms
    for _ in range(n):
        price *= (1 + np.random.uniform(-0.005, 0.005))
        data.append([
            ts,
            price * 0.999,   # open
            price * 1.002,   # high
            price * 0.997,   # low
            price,           # close
            np.random.uniform(100, 500),  # volume
        ])
        ts += 3_600_000   # +1 saat (ms cinsinden)
    return data


def make_clean_df(n: int = 50) -> pd.DataFrame:
    """Temiz OHLCV DataFrame üretir (DatetimeIndex UTC)."""
    raw = make_raw_ccxt_data(n)
    return BinanceFetcher._to_dataframe(raw)


# ─────────────────────────────────────────────────────────────────────────────
# BinanceFetcher — SINIF OLUŞTURMA TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestBinanceFetcherInit:
    """BinanceFetcher oluşturma ve doğrulama testleri."""

    @patch("data.fetcher.ccxt.binance")
    def test_valid_init(self, mock_binance):
        """Geçerli parametrelerle BinanceFetcher oluşturulabilmeli."""
        mock_binance.return_value = MagicMock()
        fetcher = BinanceFetcher(testnet=True, symbol="BTC/USDT", timeframe="1h")
        assert fetcher.symbol    == "BTC/USDT"
        assert fetcher.timeframe == "1h"
        assert fetcher.testnet   is True

    @patch("data.fetcher.ccxt.binance")
    def test_invalid_timeframe_raises(self, mock_binance):
        """Geçersiz timeframe ValueError fırlatmalı."""
        mock_binance.return_value = MagicMock()
        with pytest.raises(ValueError, match="Geçersiz timeframe"):
            BinanceFetcher(timeframe="2h")   # 2h geçerli değil

    @patch("data.fetcher.ccxt.binance")
    def test_all_valid_timeframes_accepted(self, mock_binance):
        """Tüm geçerli timeframe değerleri kabul edilmeli."""
        mock_binance.return_value = MagicMock()
        for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            f = BinanceFetcher(timeframe=tf)
            assert f.timeframe == tf


# ─────────────────────────────────────────────────────────────────────────────
# BinanceFetcher — VERİ DÖNÜŞTÜRME TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestToDataFrame:
    """_to_dataframe() statik metodunun testleri."""

    def test_correct_columns(self):
        """DataFrame doğru sütunlara sahip olmalı."""
        raw = make_raw_ccxt_data(10)
        df  = BinanceFetcher._to_dataframe(raw)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_correct_index_type(self):
        """Index DatetimeIndex olmalı."""
        raw = make_raw_ccxt_data(10)
        df  = BinanceFetcher._to_dataframe(raw)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_is_utc(self):
        """Index UTC timezone'unda olmalı."""
        raw = make_raw_ccxt_data(10)
        df  = BinanceFetcher._to_dataframe(raw)
        assert str(df.index.tz) == "UTC"

    def test_row_count_matches(self):
        """Çıktı satır sayısı giriş satır sayısıyla eşleşmeli."""
        raw = make_raw_ccxt_data(25)
        df  = BinanceFetcher._to_dataframe(raw)
        assert len(df) == 25

    def test_all_numeric_dtypes(self):
        """Tüm fiyat/hacim sütunları float64 olmalı."""
        raw = make_raw_ccxt_data(10)
        df  = BinanceFetcher._to_dataframe(raw)
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# BinanceFetcher — FETCH MOCK TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchOHLCV:
    """fetch_ohlcv() metodunun mock üzerinden testleri."""

    def _make_fetcher(self):
        """Mock exchange ile BinanceFetcher oluşturur."""
        with patch("data.fetcher.ccxt.binance") as mock_cls:
            mock_exchange = MagicMock()
            mock_cls.return_value = mock_exchange
            fetcher = BinanceFetcher(testnet=True)
            fetcher.exchange = mock_exchange
            return fetcher, mock_exchange

    def test_returns_dataframe_on_success(self):
        """Başarılı fetch DataFrame döndürmeli."""
        fetcher, mock_ex = self._make_fetcher()
        mock_ex.fetch_ohlcv.return_value = make_raw_ccxt_data(50)
        df = fetcher.fetch_ohlcv(limit=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    def test_returns_empty_on_empty_response(self):
        """Borsa boş liste döndürürse boş DataFrame dönmeli."""
        fetcher, mock_ex = self._make_fetcher()
        mock_ex.fetch_ohlcv.return_value = []
        df = fetcher.fetch_ohlcv(limit=50)
        assert df.empty

    def test_network_error_raises(self):
        """NetworkError fırlatılınca yukarıya iletilmeli."""
        import ccxt
        fetcher, mock_ex = self._make_fetcher()
        mock_ex.fetch_ohlcv.side_effect = ccxt.NetworkError("timeout")
        with pytest.raises(ccxt.NetworkError):
            fetcher.fetch_ohlcv()


# ─────────────────────────────────────────────────────────────────────────────
# OHLCVCleaner TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestOHLCVCleaner:
    """OHLCVCleaner temizleme ve doğrulama testleri."""

    def test_clean_returns_copy(self):
        """clean() orijinal DataFrame'i değiştirmemeli."""
        df      = make_clean_df(30)
        df_orig = df.copy()
        cleaner = OHLCVCleaner()
        cleaner.clean(df)
        pd.testing.assert_frame_equal(df, df_orig)

    def test_clean_removes_nan_rows(self):
        """NaN içeren satırlar temizlenmeli."""
        df = make_clean_df(30)
        df.iloc[5, df.columns.get_loc("close")] = float("nan")
        cleaner = OHLCVCleaner()
        df_clean = cleaner.clean(df)
        assert df_clean["close"].isna().sum() == 0
        assert len(df_clean) == 29

    def test_clean_removes_duplicates(self):
        """Duplikat timestamp'ler kaldırılmalı."""
        df      = make_clean_df(30)
        df_dup  = pd.concat([df, df.iloc[[10]]])
        cleaner = OHLCVCleaner()
        df_clean = cleaner.clean(df_dup)
        assert len(df_clean) == 30
        assert not df_clean.index.duplicated().any()

    def test_clean_sorts_chronologically(self):
        """Karışık sıradaki veri kronolojik sıralanmalı."""
        df        = make_clean_df(30)
        df_shuffled = df.iloc[np.random.permutation(len(df))]
        cleaner   = OHLCVCleaner()
        df_clean  = cleaner.clean(df_shuffled)
        assert df_clean.index.is_monotonic_increasing

    def test_clean_removes_negative_price(self):
        """Negatif fiyatlı satırlar kaldırılmalı."""
        df = make_clean_df(30)
        df.iloc[3, df.columns.get_loc("close")] = -100.0
        cleaner  = OHLCVCleaner()
        df_clean = cleaner.clean(df)
        assert (df_clean["close"] > 0).all()

    def test_missing_column_raises(self):
        """Gerekli sütun eksikse ValueError fırlatmalı."""
        df      = make_clean_df(10).drop(columns=["volume"])
        cleaner = OHLCVCleaner()
        with pytest.raises(ValueError, match="Eksik sutunlar"):
            cleaner.clean(df)

    def test_empty_dataframe_returns_empty(self):
        """Boş DataFrame temizlenince yine boş dönmeli."""
        cleaner = OHLCVCleaner()
        result  = cleaner.clean(pd.DataFrame())
        assert result.empty

    def test_validate_clean_data_passes(self):
        """Temiz veri doğrulamayı geçmeli."""
        df      = make_clean_df(50)
        cleaner = OHLCVCleaner()
        report  = cleaner.validate(df)
        assert report["is_clean"] is True
        assert report["nan_count"] == 0
        assert report["duplicate_ts"] == 0

    def test_validate_returns_correct_row_count(self):
        """Rapordaki satır sayısı gerçek satır sayısıyla eşleşmeli."""
        df      = make_clean_df(40)
        cleaner = OHLCVCleaner()
        report  = cleaner.validate(df)
        assert report["rows"] == 40

    def test_validate_detects_nan(self):
        """NaN içeren veri doğrulamada is_clean=False dönmeli."""
        df = make_clean_df(20)
        df.iloc[2, df.columns.get_loc("close")] = float("nan")
        cleaner = OHLCVCleaner()
        report  = cleaner.validate(df)
        assert report["is_clean"] is False
        assert report["nan_count"] > 0
