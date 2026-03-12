"""
strategies/pa_range_strategy.py
=================================
AMACI:
    Price Action Range (PA Range) stratejisi.
    Piyasanın destek ve direnç bölgelerini tespit eder,
    RSI filtresiyle AL/SAT sinyali üretir.

ÇALIŞMA MANTIĞI:
    1. Son N mumun en yükseği → Direnç (Resistance)
    2. Son N mumun en düşüğü  → Destek (Support)
    3. Fiyat desteğe yaklaştı + RSI düşükse → AL
    4. Fiyat dirence yaklaştı + RSI yüksekse → SAT
    5. ADX filtresi → trend varsa sinyal üretme

GÖRSEL:
    Direnç ─────────────────────────── 51,000
                ↑ "yakınlık eşiği" (%2)
    Fiyat       ................................ 50,200 ← SAT bölgesi
                        ...
    Fiyat       ................................ 49,800 ← AL bölgesi
                ↓ "yakınlık eşiği" (%2)
    Destek ─────────────────────────── 49,000

PARAMETRELER (settings.yaml'dan gelir):
    lookback      : Destek/direnç için kaç mum geriye bakılır (50)
    rsi_period    : RSI periyodu (14)
    rsi_oversold  : AL için RSI eşiği (40 — RSI <30 yerine daha geniş)
    rsi_overbought: SAT için RSI eşiği (60)
    proximity_pct : Direnç/desteğe ne kadar yakın sayılır? (0.02 = %2)
"""

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, Signal
from strategies.indicators import IndicatorSet
from strategies.regime_detector import MarketRegimeDetector, Regime
from utils.logger import get_logger

logger = get_logger(__name__)


class PARangeStrategy(BaseStrategy):
    """
    Price Action Range Stratejisi.

    RSI Mean Reversion'dan farkı:
        - RSI: Sadece RSI seviyesine bakar
        - PA Range: Fiyatın destek/dirençe olan KONUMUNA bakar + RSI filtresi
        - Rejim filtresi: Trend varsa (ADX > 25) sinyal üretmez

    Bu kombinasyon daha az ama daha güvenilir sinyal üretir.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        lookback: int = 50,
        rsi_period: int = 14,
        rsi_oversold: float = 40.0,
        rsi_overbought: float = 60.0,
        proximity_pct: float = 0.02,
        stop_pct: float = 0.015,
        tp_pct: float = 0.030,
        use_regime_filter: bool = True,
        volume_confirm_mult: float = 1.5,   # hacim onaylama: cari hacim bu kat ortalama uzerinde olmali
        fakeout_filter: bool = True,        # kapanisa gore kirilim dogrulama
        rsi_divergence: bool = True,        # RSI uyumsuzlugu tespiti
    ):
        """
        Args:
            lookback       : Destek/direnç için geriye bakılan mum sayısı
            rsi_oversold   : RSI bu değerin altındaysa AL bölgesinde sayılır
            rsi_overbought : RSI bu değerin üstündeyse SAT bölgesinde sayılır
            proximity_pct  : Fiyat desteğe/dirence bu yüzde kadar yakınsa "yakın" sayılır
            use_regime_filter: True ise trend piyasada sinyal üretmez
        """
        super().__init__(name="PARangeStrategy", symbol=symbol, timeframe=timeframe)
        self.lookback             = lookback
        self.rsi_period           = rsi_period
        self.rsi_oversold         = rsi_oversold
        self.rsi_overbought       = rsi_overbought
        self.proximity_pct        = proximity_pct
        self.stop_pct             = stop_pct
        self.tp_pct               = tp_pct
        self.use_regime_filter    = use_regime_filter
        self.volume_confirm_mult  = volume_confirm_mult
        self.fakeout_filter       = fakeout_filter
        self.rsi_divergence       = rsi_divergence

        self.regime_detector = MarketRegimeDetector() if use_regime_filter else None

        logger.info(
            f"PARangeStrategy baslatildi | {symbol} {timeframe} | "
            f"Lookback: {lookback} | RSI OS/OB: {rsi_oversold}/{rsi_overbought} | "
            f"Proximity: %{proximity_pct*100:.1f} | Rejim filtresi: {use_regime_filter}"
        )

    # ── Ana Sinyal Üretimi ────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        PA Range sinyali üretir.

        Adımlar:
        1. Yeterli veri var mı?
        2. Rejim filtresi (ADX) → trend mi range mi?
        3. Destek/Direnç seviyelerini hesapla
        4. Fiyat konumunu belirle
        5. RSI hesapla
        6. Karar ver: AL / SAT / BEKLE
        """
        min_rows = self.lookback + self.rsi_period
        if len(df) < min_rows:
            return Signal(
                action="BEKLE", confidence=0.0,
                reason=f"Yetersiz veri: {len(df)}/{min_rows}"
            )

        # ── Adım 1: Rejim Filtresi ────────────────────────────────────────
        # TREND_DOWN: AL sinyali engellenir (trende karsi gitme)
        # TREND_UP:   SAT sinyali engellenir (yukari trende karsi satma)
        # Her iki durumda da trend YONUNDE islem acilabilir
        _blocked_action: str | None = None
        if self.use_regime_filter and self.regime_detector:
            regime = self.regime_detector.detect(df)
            if regime == Regime.TREND_DOWN:
                _blocked_action = "AL"   # asagi trendde AL engelle
            elif regime == Regime.TREND_UP:
                _blocked_action = "SAT"  # yukari trendde SAT engelle

        # ── Adım 2: İndikatörleri Hesapla ────────────────────────────────
        ind = IndicatorSet(df, rsi_period=self.rsi_period)
        v   = ind.values

        if v.rsi is None or v.current_price is None:
            return Signal(action="BEKLE", confidence=0.0, reason="Indikatör hesaplanamadi")

        # ── Adım 3: Destek / Direnç Seviyelerini Bul ─────────────────────
        # Son 'lookback' mumun fiyat aralığı (son mumu dahil etmiyoruz — önyargısız)
        window = df["close"].iloc[-(self.lookback + 1):-1]
        support    = float(window.min())    # En düşük = destek
        resistance = float(window.max())   # En yüksek = direnç
        price      = v.current_price
        rsi        = v.rsi

        # Range genişliği
        range_width = resistance - support
        if range_width <= 0:
            return Signal(action="BEKLE", confidence=0.0, reason="Range genisligi sifir")

        # ── Adım 4: Fiyat Konumu ──────────────────────────────────────────
        # Fiyat range içinde nerede? 0.0 = destek, 1.0 = direnç
        price_position = (price - support) / range_width

        near_support    = price <= support    + (range_width * self.proximity_pct)
        near_resistance = price >= resistance - (range_width * self.proximity_pct)

        logger.debug(
            f"PA Range | Fiyat: {price:,.0f} | "
            f"Destek: {support:,.0f} | Direnc: {resistance:,.0f} | "
            f"Konum: %{price_position*100:.1f} | RSI: {rsi:.1f}"
        )

        # ── Adım 5: Volume Confirmation ───────────────────────────────────
        # Hacim ortalamasi: son 20 mum (son mum haric — bias onleme)
        vol_ma_20 = float(df["volume"].iloc[-21:-1].mean()) if len(df) >= 22 else None
        current_vol = float(df["volume"].iloc[-1])
        volume_ok = True
        if vol_ma_20 and vol_ma_20 > 0:
            vol_ratio = current_vol / vol_ma_20
            volume_ok = vol_ratio >= self.volume_confirm_mult
            if not volume_ok:
                logger.debug(
                    f"Hacim filtresi: {vol_ratio:.2f}x < {self.volume_confirm_mult}x "
                    f"(gerekli onay yok)"
                )

        # ── Adım 6: Fakeout Filter ────────────────────────────────────────
        # Kapanış fiyatı desteğe/dirence yakın olmalı (sadece gölge yok)
        last_close = float(df["close"].iloc[-1])

        def fakeout_al_ok() -> bool:
            """LONG için: kapaniş yakın = destek bölgesinde kapanmali"""
            if not self.fakeout_filter:
                return True
            # Kapanisin desteğe yakınlığı: kapanış <= destek + prox_range ise gerçek
            prox_range = range_width * self.proximity_pct
            return last_close <= support + prox_range

        def fakeout_sat_ok() -> bool:
            """SHORT için: kapaniş yakın = direnc bölgesinde kapanmali"""
            if not self.fakeout_filter:
                return True
            prox_range = range_width * self.proximity_pct
            return last_close >= resistance - prox_range

        # ── Adım 7: RSI Divergence Bonus ──────────────────────────────────
        # Bullish divergence: fiyat yeni dip yaparken RSI yapmiyorsa — AL güclendirir
        # Bearish divergence: fiyat yeni zirve yaparken RSI yapmiyorsa — SAT güçlendirir
        divergence_bonus = 0.0
        if self.rsi_divergence and len(df) >= self.lookback + self.rsi_period + 5:
            try:
                import pandas_ta as _ta
                rsi_series = _ta.rsi(df["close"], length=self.rsi_period)
                if rsi_series is not None and len(rsi_series) >= 10:
                    # Son 5 bar vs onceki 5 bar karsilastir
                    close_now = float(df["close"].iloc[-3:].min())
                    close_prev = float(df["close"].iloc[-8:-3].min())
                    rsi_now   = float(rsi_series.iloc[-3:].min())
                    rsi_prev  = float(rsi_series.iloc[-8:-3].min())
                    # Bullish divergence: fiyat dip yapti, RSI yapmadi
                    if close_now < close_prev and rsi_now > rsi_prev:
                        divergence_bonus = 0.10
                        logger.debug(f"Bullish RSI divergence tespit edildi, bonus: +{divergence_bonus}")
                    # Bearish divergence: fiyat zirve yapti, RSI yapmadi
                    close_peak_now  = float(df["close"].iloc[-3:].max())
                    close_peak_prev = float(df["close"].iloc[-8:-3].max())
                    rsi_peak_now    = float(rsi_series.iloc[-3:].max())
                    rsi_peak_prev   = float(rsi_series.iloc[-8:-3].max())
                    if close_peak_now > close_peak_prev and rsi_peak_now < rsi_peak_prev:
                        divergence_bonus = 0.10
                        logger.debug(f"Bearish RSI divergence tespit edildi, bonus: +{divergence_bonus}")
            except Exception as _e:
                logger.debug(f"RSI divergence hesaplanamadi: {_e}")

        # ── Adım 8: Sinyal Kararı ─────────────────────────────────────────

        # AL: Fiyat desteğe yakın VE RSI aşırı satım bölgesinde
        if near_support and rsi < self.rsi_oversold:
            # TREND_DOWN'da AL engelle (trende karsi gitme)
            if _blocked_action == "AL":
                logger.debug(f"AL engellendi: TREND_DOWN rejimi | RSI: {rsi:.1f}")
                return Signal(action="BEKLE", confidence=0.0, reason="TREND_DOWN: AL engellendi")

            # Fakeout kontrolu
            if not fakeout_al_ok():
                logger.debug(
                    f"AL engellendi: fakeout filtresi | kapaniş {last_close:,.0f} "
                    f"destek bölgesi disinda (destek: {support:,.0f})"
                )
                return Signal(action="BEKLE", confidence=0.0,
                              reason="Fakeout: kapaniş destek bölgesi disinda")

            confidence = self._calc_confidence_al(price, support, resistance, rsi)
            # Volume onay bonusu
            if volume_ok and vol_ma_20:
                confidence = min(1.0, confidence + 0.08)
            # RSI divergence bonusu
            confidence = min(1.0, confidence + divergence_bonus)

            signal = Signal(
                action="AL",
                confidence=round(confidence, 3),
                stop_loss=support * (1 - self.stop_pct),
                take_profit=price + (range_width * 0.5),
                reason=(
                    f"Destege yakin: {price:,.0f} ~ {support:,.0f} | "
                    f"RSI: {rsi:.1f} < {self.rsi_oversold} | "
                    f"Vol: {'OK' if volume_ok else 'ZAYIF'}"
                )
            )
            logger.info(
                f"SINYAL AL | {self.symbol} | Fiyat: {price:,.2f} | "
                f"Destek: {support:,.2f} | RSI: {rsi:.1f} | Guven: {confidence:.2f} | "
                f"Vol: {'OK' if volume_ok else 'ZAYIF'}"
            )
            return signal

        # SAT: Fiyat dirence yakın VE RSI aşırı alım bölgesinde
        elif near_resistance and rsi > self.rsi_overbought:
            # TREND_UP'ta SAT engelle (yukari trende karsi satma)
            if _blocked_action == "SAT":
                logger.debug(f"SAT engellendi: TREND_UP rejimi | RSI: {rsi:.1f}")
                return Signal(action="BEKLE", confidence=0.0, reason="TREND_UP: SAT engellendi")

            # Fakeout kontrolu
            if not fakeout_sat_ok():
                logger.debug(
                    f"SAT engellendi: fakeout filtresi | kapaniş {last_close:,.0f} "
                    f"direnc bölgesi disinda (direnc: {resistance:,.0f})"
                )
                return Signal(action="BEKLE", confidence=0.0,
                              reason="Fakeout: kapaniş direnc bölgesi disinda")

            confidence = self._calc_confidence_sat(price, support, resistance, rsi)
            # Volume onay bonusu
            if volume_ok and vol_ma_20:
                confidence = min(1.0, confidence + 0.08)
            # RSI divergence bonusu
            confidence = min(1.0, confidence + divergence_bonus)

            signal = Signal(
                action="SAT",
                confidence=round(confidence, 3),
                stop_loss=resistance * (1 + self.stop_pct),
                take_profit=price - (range_width * 0.5),
                reason=(
                    f"Direnca yakin: {price:,.0f} ~ {resistance:,.0f} | "
                    f"RSI: {rsi:.1f} > {self.rsi_overbought} | "
                    f"Vol: {'OK' if volume_ok else 'ZAYIF'}"
                )
            )
            logger.info(
                f"SINYAL SAT | {self.symbol} | Fiyat: {price:,.2f} | "
                f"Direnc: {resistance:,.2f} | RSI: {rsi:.1f} | Guven: {confidence:.2f} | "
                f"Vol: {'OK' if volume_ok else 'ZAYIF'}"
            )
            return signal

        # BEKLE: Ne destek ne direnç bölgesinde
        else:
            logger.debug(
                f"BEKLE | Fiyat ortada (%{price_position*100:.1f}) | RSI: {rsi:.1f}"
            )
            return Signal(
                action="BEKLE", confidence=0.0,
                reason=f"Fiyat range ortasinda: konum %{price_position*100:.1f} | RSI: {rsi:.1f}"
            )

    # ── Yardımcı Metodlar ────────────────────────────────────────────────────

    def _calc_confidence_al(
        self, price: float, support: float, resistance: float, rsi: float
    ) -> float:
        """
        AL sinyali güven skoru hesaplar.

        İki faktör:
        1. Fiyat desteğe ne kadar yakın? (yakınsa yüksek güven)
        2. RSI ne kadar düşük? (düşükse yüksek güven)
        """
        range_width = resistance - support

        # Fiyat faktörü: destek ile fiyat arasındaki mesafe
        price_factor = max(0.0, 1.0 - (price - support) / (range_width * self.proximity_pct))

        # RSI faktörü: RSI ne kadar düşükse o kadar güçlü sinyal
        # rsi_oversold=40 → RSI=40'ta 0.5, RSI=20'de 1.0
        rsi_factor = max(0.0, min(1.0, (self.rsi_oversold - rsi) / self.rsi_oversold))

        # İkisinin ortalaması
        confidence = round((price_factor * 0.5 + rsi_factor * 0.5), 3)
        return min(confidence, 1.0)

    def _calc_confidence_sat(
        self, price: float, support: float, resistance: float, rsi: float
    ) -> float:
        """SAT sinyali güven skoru hesaplar."""
        range_width = resistance - support

        price_factor = max(0.0, 1.0 - (resistance - price) / (range_width * self.proximity_pct))
        rsi_factor   = max(0.0, min(1.0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)))

        confidence = round((price_factor * 0.5 + rsi_factor * 0.5), 3)
        return min(confidence, 1.0)

    def get_levels(self, df: pd.DataFrame) -> dict:
        """
        Mevcut destek/direnç seviyelerini döndürür.
        Dashboard veya debug için kullanılır.
        """
        if len(df) < self.lookback + 1:
            return {}
        window = df["close"].iloc[-(self.lookback + 1):-1]
        support    = float(window.min())
        resistance = float(window.max())
        return {
            "support"   : support,
            "resistance": resistance,
            "range_width": resistance - support,
            "range_pct" : (resistance - support) / support * 100,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOGU
# python -m strategies.pa_range_strategy
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ALGO TRADE CODEX - PARangeStrategy Test")
    print("=" * 60)

    np.random.seed(10)

    # Range piyasa: 49,000 - 51,000 arası gidip geliyor
    closes = []
    price  = 50000.0
    for i in range(120):
        # Rastgele ama range'de tutan bir hareket
        target = 50000 + 1000 * np.sin(i * 0.15)   # sinüs dalgası = range
        price  = price + (target - price) * 0.1 + np.random.uniform(-200, 200)
        closes.append(price)

    df = pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(120)],
    }, index=pd.date_range("2024-01-01", periods=120, freq="1h"))

    strategy = PARangeStrategy(
        symbol="BTC/USDT", timeframe="1h",
        lookback=50, rsi_oversold=40, rsi_overbought=60,
        use_regime_filter=False,   # Test için rejim filtresi kapalı
    )

    # Seviyeleri göster
    levels = strategy.get_levels(df)
    print(f"\nDestek    : {levels.get('support', 0):,.0f} USDT")
    print(f"Direnc    : {levels.get('resistance', 0):,.0f} USDT")
    print(f"Range     : {levels.get('range_width', 0):,.0f} USDT (%{levels.get('range_pct', 0):.1f})")

    # Tüm mumları tara
    print("\nUretilen AL/SAT sinyalleri:")
    print("-" * 60)
    al_count = sat_count = 0
    for i in range(65, 120):
        df_slice = df.iloc[:i + 1]
        signal   = strategy.run(df_slice)
        if signal.action != "BEKLE":
            ind = IndicatorSet(df_slice)
            print(
                f"  Mum {i+1:3d} | Fiyat: {df_slice['close'].iloc[-1]:,.0f} | "
                f"RSI: {ind.values.rsi:.1f if ind.values.rsi else 'N/A'} | "
                f"{signal.action} | Guven: {signal.confidence:.2f} | {signal.reason}"
            )
            if signal.action == "AL":
                al_count += 1
            else:
                sat_count += 1

    print("-" * 60)
    print(f"\nToplam AL : {al_count}")
    print(f"Toplam SAT: {sat_count}")
    print("\nBASARI: PARangeStrategy calisiyor!")
