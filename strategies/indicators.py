"""
strategies/indicators.py
==========================
AMACI:
    Tüm teknik indikatörleri tek bir yerde hesaplar.
    Stratejiler bu modülden hazır indikatör değerlerini alır —
    her strateji kendi başına hesaplamaz.

NEDEN MERKEZİ?
    RSIStrategy, PARangeStrategy ve ilerideki stratejiler aynı veriyi kullanır.
    Merkezi hesaplama = tek seferlik işlem, tutarlı değerler.

İÇERİKLER:
    RSI   — Momentum osilatörü (aşırı alım/satım)
    MACD  — Trend takipçisi (momentum değişimi)
    BB    — Bollinger Bands (volatilite + fiyat kanalı)
    ATR   — Average True Range (volatilite ölçer, stop-loss için)
    ADX   — Trend gücü ölçer (Range mı Trend mi?)

KULLANIM:
    from strategies.indicators import IndicatorSet

    ind = IndicatorSet(df)
    print(ind.rsi)        # son RSI değeri (float)
    print(ind.adx)        # son ADX değeri (float)
    print(ind.is_range)   # ADX < 25 ise True
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────
ADX_TREND_THRESHOLD  = 25.0   # ADX > 25 → trend piyasası
ADX_STRONG_THRESHOLD = 40.0   # ADX > 40 → çok güçlü trend
MIN_ROWS_REQUIRED    = 30     # Güvenilir hesap için minimum mum sayısı


@dataclass
class IndicatorValues:
    """
    Hesaplanan indikatör değerlerini tutan veri sınıfı.
    Her alan son mumun değerini temsil eder.
    None = hesaplanamadı (yetersiz veri)
    """
    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi: float | None = None          # 0-100 arası

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd: float | None = None         # MACD çizgisi
    macd_signal: float | None = None  # Sinyal çizgisi
    macd_hist: float | None = None    # Histogram (macd - signal)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_upper: float | None = None     # Üst bant
    bb_middle: float | None = None    # Orta bant (20 periyot SMA)
    bb_lower: float | None = None     # Alt bant
    bb_width: float | None = None     # Bant genişliği (volatilite ölçer)
    bb_pct: float | None = None       # Fiyatın bant içindeki yüzde konumu (0-1)

    # ── ATR ──────────────────────────────────────────────────────────────────
    atr: float | None = None          # Average True Range (volatilite)

    # ── ADX ──────────────────────────────────────────────────────────────────
    adx: float | None = None          # Trend gücü (0-100)
    dmp: float | None = None          # +DI (yukarı trend bileşeni)
    dmn: float | None = None          # -DI (aşağı trend bileşeni)

    # ── Hesaplama Metadatası ──────────────────────────────────────────────────
    current_price: float | None = None
    prev_price: float | None = None
    rows_used: int = 0

    # ── Türetilmiş Özellikler (property olarak) ────────────────────────────
    @property
    def is_range(self) -> bool:
        """ADX < 25 ise piyasa yatay (range) demektir."""
        if self.adx is None:
            return True   # ADX yoksa güvenli taraf: range kabul et
        return self.adx < ADX_TREND_THRESHOLD

    @property
    def is_trend(self) -> bool:
        """ADX >= 25 ise piyasa trend modunda."""
        return not self.is_range

    @property
    def trend_direction(self) -> str:
        """
        Trend yönünü döndürür: 'UP', 'DOWN', 'NONE'
        +DI > -DI ise yukarı trend, tersi aşağı trend.
        """
        if self.dmp is None or self.dmn is None:
            return "NONE"
        if self.dmp > self.dmn:
            return "UP"
        elif self.dmn > self.dmp:
            return "DOWN"
        return "NONE"

    @property
    def macd_bullish(self) -> bool:
        """MACD histogramı pozitif ve büyüyorsa yükseliş momentumu var."""
        return self.macd_hist is not None and self.macd_hist > 0

    @property
    def macd_bearish(self) -> bool:
        """MACD histogramı negatif ve küçülüyorsa düşüş momentumu var."""
        return self.macd_hist is not None and self.macd_hist < 0


class IndicatorSet:
    """
    DataFrame üzerinden tüm indikatörleri hesaplayan sınıf.

    Kullanım:
        ind = IndicatorSet(df, rsi_period=14, adx_period=14)
        if ind.values.is_range:
            # RSI veya PA Range stratejisini kullan
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        adx_period: int = 14,
    ):
        self.df              = df
        self.rsi_period      = rsi_period
        self.macd_fast       = macd_fast
        self.macd_slow       = macd_slow
        self.macd_signal_p   = macd_signal_period
        self.bb_period       = bb_period
        self.bb_std          = bb_std
        self.atr_period      = atr_period
        self.adx_period      = adx_period

        self.values = self._calculate()

    def _calculate(self) -> IndicatorValues:
        """
        Tüm indikatörleri hesaplar ve IndicatorValues döndürür.
        Yeterli veri yoksa ilgili alan None kalır.
        """
        vals = IndicatorValues(rows_used=len(self.df))

        if len(self.df) < MIN_ROWS_REQUIRED:
            logger.warning(f"Yetersiz veri: {len(self.df)} satir, en az {MIN_ROWS_REQUIRED} gerekli")
            return vals

        if len(self.df) >= 1:
            vals.current_price = float(self.df["close"].iloc[-1])
        if len(self.df) >= 2:
            vals.prev_price = float(self.df["close"].iloc[-2])

        # Her indikatörü ayrı try/except ile hesapla
        # Biri başarısız olursa diğerleri çalışmaya devam eder
        vals.rsi      = self._calc_rsi()
        vals.macd, vals.macd_signal, vals.macd_hist = self._calc_macd()
        vals.bb_upper, vals.bb_middle, vals.bb_lower, vals.bb_width, vals.bb_pct = self._calc_bb()
        vals.atr      = self._calc_atr()
        vals.adx, vals.dmp, vals.dmn = self._calc_adx()

        rsi_str = f"{vals.rsi:.1f}" if vals.rsi is not None else "N/A"
        adx_str = f"{vals.adx:.1f}" if vals.adx is not None else "N/A"
        logger.debug(
            f"Indikatorler hesaplandi | "
            f"RSI: {rsi_str} | "
            f"ADX: {adx_str} | "
            f"Rejim: {'RANGE' if vals.is_range else 'TREND'}"
        )

        return vals

    # ── RSI ──────────────────────────────────────────────────────────────────

    def _calc_rsi(self) -> float | None:
        try:
            series = ta.rsi(self.df["close"], length=self.rsi_period)
            val = series.iloc[-1]
            return float(val) if not pd.isna(val) else None
        except Exception as e:
            logger.warning(f"RSI hesaplanamadi: {e}")
            return None

    # ── MACD ─────────────────────────────────────────────────────────────────

    def _calc_macd(self) -> tuple[float | None, float | None, float | None]:
        try:
            macd_df = ta.macd(
                self.df["close"],
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal_p,
            )
            if macd_df is None or macd_df.empty:
                return None, None, None

            # pandas-ta MACD sütun adları: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            macd_col   = [c for c in macd_df.columns if c.startswith("MACD_")]
            signal_col = [c for c in macd_df.columns if c.startswith("MACDs_")]
            hist_col   = [c for c in macd_df.columns if c.startswith("MACDh_")]

            macd_val   = float(macd_df[macd_col[0]].iloc[-1])   if macd_col   else None
            signal_val = float(macd_df[signal_col[0]].iloc[-1]) if signal_col else None
            hist_val   = float(macd_df[hist_col[0]].iloc[-1])   if hist_col   else None

            macd_val   = None if macd_val   is not None and pd.isna(macd_val)   else macd_val
            signal_val = None if signal_val is not None and pd.isna(signal_val) else signal_val
            hist_val   = None if hist_val   is not None and pd.isna(hist_val)   else hist_val

            return macd_val, signal_val, hist_val
        except Exception as e:
            logger.warning(f"MACD hesaplanamadi: {e}")
            return None, None, None

    # ── Bollinger Bands ───────────────────────────────────────────────────────

    def _calc_bb(self) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        try:
            bb = ta.bbands(self.df["close"], length=self.bb_period, std=self.bb_std)
            if bb is None or bb.empty:
                return None, None, None, None, None

            upper_col  = [c for c in bb.columns if "BBU" in c]
            mid_col    = [c for c in bb.columns if "BBM" in c]
            lower_col  = [c for c in bb.columns if "BBL" in c]
            bwidth_col = [c for c in bb.columns if "BBB" in c]
            bpct_col   = [c for c in bb.columns if "BBP" in c]

            def _safe(col_list):
                if not col_list:
                    return None
                v = bb[col_list[0]].iloc[-1]
                return float(v) if not pd.isna(v) else None

            return _safe(upper_col), _safe(mid_col), _safe(lower_col), _safe(bwidth_col), _safe(bpct_col)
        except Exception as e:
            logger.warning(f"Bollinger Bands hesaplanamadi: {e}")
            return None, None, None, None, None

    # ── ATR ──────────────────────────────────────────────────────────────────

    def _calc_atr(self) -> float | None:
        try:
            series = ta.atr(
                self.df["high"], self.df["low"], self.df["close"],
                length=self.atr_period,
            )
            val = series.iloc[-1]
            return float(val) if not pd.isna(val) else None
        except Exception as e:
            logger.warning(f"ATR hesaplanamadi: {e}")
            return None

    # ── ADX ──────────────────────────────────────────────────────────────────

    def _calc_adx(self) -> tuple[float | None, float | None, float | None]:
        try:
            adx_df = ta.adx(
                self.df["high"], self.df["low"], self.df["close"],
                length=self.adx_period,
            )
            if adx_df is None or adx_df.empty:
                return None, None, None

            adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
            dmp_col = [c for c in adx_df.columns if c.startswith("DMP_")]
            dmn_col = [c for c in adx_df.columns if c.startswith("DMN_")]

            def _safe(col_list):
                if not col_list:
                    return None
                v = adx_df[col_list[0]].iloc[-1]
                return float(v) if not pd.isna(v) else None

            return _safe(adx_col), _safe(dmp_col), _safe(dmn_col)
        except Exception as e:
            logger.warning(f"ADX hesaplanamadi: {e}")
            return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOGU
# python -m strategies.indicators
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ALGO TRADE CODEX - IndicatorSet Test")
    print("=" * 60)

    np.random.seed(42)
    closes = [50000.0]
    for _ in range(99):
        closes.append(closes[-1] * (1 + np.random.uniform(-0.008, 0.008)))

    df = pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(100)],
    }, index=pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC"))

    ind = IndicatorSet(df)
    v   = ind.values

    print(f"\nFiyat       : {v.current_price:,.2f} USDT")
    print(f"RSI         : {v.rsi:.2f}" if v.rsi else "RSI         : N/A")
    print(f"MACD        : {v.macd:.4f} | Signal: {v.macd_signal:.4f} | Hist: {v.macd_hist:.4f}"
          if v.macd else "MACD        : N/A")
    print(f"BB Upper    : {v.bb_upper:,.2f}" if v.bb_upper else "BB Upper    : N/A")
    print(f"BB Lower    : {v.bb_lower:,.2f}" if v.bb_lower else "BB Lower    : N/A")
    print(f"BB %        : {v.bb_pct:.3f}" if v.bb_pct is not None else "BB %        : N/A")
    print(f"ATR         : {v.atr:.2f}" if v.atr else "ATR         : N/A")
    print(f"ADX         : {v.adx:.2f}" if v.adx else "ADX         : N/A")
    print(f"Rejim       : {'RANGE (RSI/PA calisiyor)' if v.is_range else 'TREND (dikkatli ol)'}")
    print(f"Trend Yonu  : {v.trend_direction}")
    print(f"MACD Bullish: {v.macd_bullish}")

    print("\nBASARI: IndicatorSet calisiyor!")
