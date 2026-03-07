"""
ml/feature_engineering.py
==========================
AMACI:
    Ham OHLCV verisinden XGBoost modeli icin ozellik (feature) matrisi uretir.
    Lookahead bias (gelecek bilgisi sizintisi) onlenmistir:
    - Tum indiktorler sadece o anda mevcut verileri kullanir
    - Hedef (label) gelecekteki kapanisa gore uretilir — gecmis verideki
      gelecek fiyat hedefe bakarak etiketlenirse bu yasalmali bir hataydi;
      bu kod bunu dogru shiftleyerek yapar.

OZELLIK GRUPLARI (50+):
    1. Fiyat oran ozellikleri     — open/close orani, high/low orani vs.
    2. RSI                        — 7, 14, 21 periyot
    3. MACD                       — histogram, sinyalle fark
    4. Bollinger Bands            — bb_pct (bant icindeki konum), bb_width
    5. ATR                        — 7, 14 periyot, normalize edilmis
    6. ADX + DI'lar               — trend gucunu olcer
    7. Hacim ozellikleri          — normalize hacim, hacim momentum
    8. Momentum / Return          — 1, 2, 3, 5, 10, 20 bar getirisi
    9. Rolling istatistikler      — std, skew, ortalama (5, 10, 20 window)
   10. Lag ozellikleri            — 1, 2, 3 geciktirmis kapanislar
   11. Mum ozellikleri            — mum govdesi, uzun/kisa golgeler

HEDEF (TARGET):
    forward_return_pct = (close t+n - close t) / close t * 100
    n = horizon_bars (varsayilan 3 bar)

    Siniflar:
        AL   (2) = return > +threshold_pct
        SAT  (0) = return < -threshold_pct
        BEKLE(1) = geri kalan

KULLANIM:
    from ml.feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    X, y, feature_names = fe.build(df)
    # X: pd.DataFrame (ozellikler), y: pd.Series (etiketler)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pandas_ta as ta
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────
LABEL_BUY    = 2     # AL
LABEL_HOLD   = 1     # BEKLE
LABEL_SELL   = 0     # SAT


class FeatureEngineer:
    """
    Ham OHLCV DataFrame'inden ML icin ozellik matrisi uretir.

    Parametreler:
        horizon_bars    : kac bar sonrasinin getirisini tahmin etmek istiyoruz
        threshold_pct   : bu yuzde esigi asinca AL/SAT siniflandirilir
        min_rows        : feature uretimi icin minimum satir sayisi
    """

    def __init__(
        self,
        horizon_bars: int   = 3,
        threshold_pct: float = 0.3,
        min_rows: int        = 100,
    ):
        self.horizon_bars  = horizon_bars
        self.threshold_pct = threshold_pct
        self.min_rows      = min_rows

    # ── Ana Metod ─────────────────────────────────────────────────────────────

    def build(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """
        Ozellik matrisi ve etiket serisi uretir.

        Returns:
            X            : ozellik DataFrame'i (NaN icermez, hizalanmis)
            y            : hedef etiket Series (LABEL_BUY/HOLD/SELL)
            feature_names: X kolonlarinin listesi
        """
        if len(df) < self.min_rows:
            raise ValueError(
                f"Yetersiz veri: {len(df)} satir, en az {self.min_rows} gerekli"
            )

        df = df.copy()

        # ── Ozellik gruplarini hesapla ────────────────────────────────────────
        feat = pd.DataFrame(index=df.index)

        feat = self._add_price_features(feat, df)
        feat = self._add_rsi_features(feat, df)
        feat = self._add_macd_features(feat, df)
        feat = self._add_bb_features(feat, df)
        feat = self._add_atr_features(feat, df)
        feat = self._add_adx_features(feat, df)
        feat = self._add_volume_features(feat, df)
        feat = self._add_momentum_features(feat, df)
        feat = self._add_rolling_features(feat, df)
        feat = self._add_lag_features(feat, df)
        feat = self._add_candle_features(feat, df)

        # ── Hedef etiketi uret ────────────────────────────────────────────────
        forward_ret = df["close"].pct_change(self.horizon_bars).shift(-self.horizon_bars) * 100
        y_raw = pd.Series(LABEL_HOLD, index=df.index, dtype=int)
        y_raw[forward_ret >  self.threshold_pct]  = LABEL_BUY
        y_raw[forward_ret < -self.threshold_pct]  = LABEL_SELL

        # Son horizon_bars satir icin gelecek bilgisi yok — at
        valid_mask = ~forward_ret.isna()

        feat_clean = feat[valid_mask].copy()
        y_clean    = y_raw[valid_mask].copy()

        # NaN iceren satirlari temizle (indiktorlerin ilk barlari nan olabilir)
        nan_mask   = feat_clean.isna().any(axis=1)
        feat_clean = feat_clean[~nan_mask]
        y_clean    = y_clean[~nan_mask]

        feature_names = list(feat_clean.columns)

        logger.info(
            f"Feature engineering tamamlandi: "
            f"{len(feat_clean)} satir, {len(feature_names)} ozellik | "
            f"AL:{(y_clean==LABEL_BUY).sum()} "
            f"SAT:{(y_clean==LABEL_SELL).sum()} "
            f"BEKLE:{(y_clean==LABEL_HOLD).sum()}"
        )

        return feat_clean, y_clean, feature_names

    # ── Ozellik Hesaplama Metodlari ───────────────────────────────────────────

    def _add_price_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Fiyat oran ozellikleri — mumun sekli ve gunluk araliklar."""
        close = df["close"]
        open_ = df["open"]
        high  = df["high"]
        low   = df["low"]

        # Kapanma fiyati / acilis fiyati orani
        feat["price_oc_ratio"]  = (close - open_) / open_
        # Gunluk range / kapanma
        feat["price_hl_range"]  = (high - low) / close
        # Kapanisin gunluk aralik icindeki konumu (0=dip, 1=zirve)
        hl = (high - low).replace(0, np.nan)
        feat["price_close_pos"] = (close - low) / hl
        # Kapanma ile 20 periyot SMA fark
        sma20 = close.rolling(20).mean()
        feat["price_vs_sma20"]  = (close - sma20) / sma20
        # Kapanma ile 50 periyot SMA fark
        sma50 = close.rolling(50).mean()
        feat["price_vs_sma50"]  = (close - sma50) / sma50
        # SMA20 vs SMA50 (kisa/uzun vade)
        feat["sma20_vs_sma50"]  = (sma20 - sma50) / sma50.replace(0, np.nan)

        return feat

    def _add_rsi_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """RSI 7, 14, 21 periyot."""
        for period in [7, 14, 21]:
            rsi = ta.rsi(df["close"], length=period)
            feat[f"rsi_{period}"] = rsi / 100.0   # 0-1 araligina normalize et
        # RSI 14 degisimi (momentum)
        feat["rsi_14_change"] = feat["rsi_14"].diff()
        return feat

    def _add_macd_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """MACD histogrami ve normalized degerler."""
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            hist_col   = [c for c in macd_df.columns if c.startswith("MACDh_")]
            macd_col   = [c for c in macd_df.columns if c.startswith("MACD_")]
            signal_col = [c for c in macd_df.columns if c.startswith("MACDs_")]

            if hist_col:
                feat["macd_hist"]        = macd_df[hist_col[0]]
                feat["macd_hist_change"] = macd_df[hist_col[0]].diff()
            if macd_col and signal_col:
                feat["macd_vs_signal"] = (
                    macd_df[macd_col[0]] - macd_df[signal_col[0]]
                )
        return feat

    def _add_bb_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands — bant icindeki konum ve genislik."""
        bb = ta.bbands(df["close"], length=20, std=2.0)
        if bb is not None and not bb.empty:
            upper_col  = [c for c in bb.columns if "BBU" in c]
            lower_col  = [c for c in bb.columns if "BBL" in c]
            bwidth_col = [c for c in bb.columns if "BBB" in c]
            bpct_col   = [c for c in bb.columns if "BBP" in c]

            if bpct_col:
                feat["bb_pct"]   = bb[bpct_col[0]]
            if bwidth_col:
                feat["bb_width"] = bb[bwidth_col[0]]
            if upper_col and lower_col:
                bb_range = (bb[upper_col[0]] - bb[lower_col[0]]).replace(0, np.nan)
                feat["bb_squeeze"] = bb_range / df["close"]  # dar bant = squeeze

        # Daha dar bant periyodu (10)
        bb10 = ta.bbands(df["close"], length=10, std=2.0)
        if bb10 is not None and not bb10.empty:
            bpct10 = [c for c in bb10.columns if "BBP" in c]
            if bpct10:
                feat["bb10_pct"] = bb10[bpct10[0]]

        return feat

    def _add_atr_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """ATR volatilite ozellikleri — normalize edilmis."""
        for period in [7, 14]:
            atr = ta.atr(df["high"], df["low"], df["close"], length=period)
            # ATR'yi kapanma fiyatina bolarak normalize et
            feat[f"atr_{period}_norm"] = atr / df["close"]

        # ATR degisimi — volatilite artisi/azalisi
        atr14 = ta.atr(df["high"], df["low"], df["close"], length=14)
        feat["atr_14_change"] = atr14.pct_change(fill_method=None)

        return feat

    def _add_adx_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """ADX trend gucu ve DI yonu."""
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
            dmp_col = [c for c in adx_df.columns if c.startswith("DMP_")]
            dmn_col = [c for c in adx_df.columns if c.startswith("DMN_")]

            if adx_col:
                feat["adx"]        = adx_df[adx_col[0]] / 100.0  # 0-1
                feat["adx_change"] = adx_df[adx_col[0]].diff()
            if dmp_col and dmn_col:
                feat["di_diff"] = (
                    adx_df[dmp_col[0]] - adx_df[dmn_col[0]]
                ) / 100.0   # +DI - -DI: pozitif = yukselis baskilisi
        return feat

    def _add_volume_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Hacim momentum ve normalize degerler."""
        vol    = df["volume"]
        # Hacim / 20 periyot ortalama hacim
        vol_ma = vol.rolling(20).mean()
        feat["vol_ratio"]   = vol / vol_ma.replace(0, np.nan)
        # Hacim degisimi
        feat["vol_change"]  = vol.pct_change(fill_method=None)
        # OBV (On Balance Volume) momentum
        obv = ta.obv(df["close"], vol)
        if obv is not None:
            # OBV'yi normalize et (son 20 bar stanart sapmasiyla)
            obv_std = obv.rolling(20).std().replace(0, np.nan)
            obv_ma  = obv.rolling(20).mean()
            feat["obv_norm"] = (obv - obv_ma) / obv_std

        # Fiyat * hacim momentumu
        feat["price_vol_trend"] = (df["close"] - df["close"].shift(1)) * vol

        return feat

    def _add_momentum_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """N bar getirileri — kisa ve orta vade momentum."""
        close = df["close"]
        for n in [1, 2, 3, 5, 10, 20]:
            feat[f"ret_{n}"] = close.pct_change(n, fill_method=None)

        # Momentum osilalatoru: 10 gunluk vs 3 gunluk getiri farki
        feat["mom_ratio"] = (
            close.pct_change(10, fill_method=None) -
            close.pct_change(3, fill_method=None)
        )
        return feat

    def _add_rolling_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling istatistikler — std, skewness, volatilite."""
        close_ret = df["close"].pct_change(fill_method=None)
        for window in [5, 10, 20]:
            feat[f"roll_std_{window}"]  = close_ret.rolling(window).std()
            feat[f"roll_mean_{window}"] = close_ret.rolling(window).mean()
        # Fiyat volatilitesi (standart sapma / ortalama)
        feat["cv_20"] = (
            df["close"].rolling(20).std() /
            df["close"].rolling(20).mean().replace(0, np.nan)
        )
        return feat

    def _add_lag_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Lag (geciktirmis) kapanma fiyati getirileri."""
        close = df["close"]
        for lag in [1, 2, 3]:
            feat[f"lag_ret_{lag}"] = close.pct_change(fill_method=None).shift(lag)
        return feat

    def _add_candle_features(self, feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Mum sekli ozellikleri — govde, ust/alt golge."""
        high  = df["high"]
        low   = df["low"]
        open_ = df["open"]
        close = df["close"]

        hl    = (high - low).replace(0, np.nan)
        # Mum govdesi buyuklugu (|close - open| / range)
        feat["candle_body"] = (close - open_).abs() / hl
        # Ust golge orani — yuksek noktadan kapanisa uzaklik
        feat["candle_upper_shadow"] = (high - close.clip(lower=open_)) / hl
        # Alt golge orani — kapanisten dip noktaya uzaklik
        feat["candle_lower_shadow"] = (close.clip(upper=open_) - low) / hl
        # Artis mi dusus mu (binary)
        feat["candle_bullish"] = (close > open_).astype(int)

        return feat

    # ── Yardimci: Label Isimleri ──────────────────────────────────────────────

    @staticmethod
    def label_name(code: int) -> str:
        """Etiket kodundan okunabilir isim dondurur."""
        return {LABEL_BUY: "AL", LABEL_HOLD: "BEKLE", LABEL_SELL: "SAT"}.get(code, "?")

    @staticmethod
    def get_label_map() -> dict[int, str]:
        return {LABEL_BUY: "AL", LABEL_HOLD: "BEKLE", LABEL_SELL: "SAT"}
