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

        # PA-1/2/3/4/5 parametreleri
        self.eq_bonus       = 0.10   # EQ bölgesi confidence bonusu
        self.dev_bonus      = 0.15   # Deviasyon confirmation bonusu
        self.ob_bonus       = 0.12   # Order Block bonusu
        self.ob_lookback    = 10     # Order Block tespiti için kaç muma bakılır
        self.ms_swing_n     = 5      # Market yapısı: swing high/low tespiti için kaç mum her yanda
        self.ote_bonus      = 0.15   # OTE Fibonacci bölgesi bonusu
        self.imbalance_bars = 30     # Imbalance tespiti için kaç muma bakılır
        self.key_level_pct  = 0.005  # Key level yakınlık eşiği (%0.5)
        self.key_bonus      = 0.10   # Key level yakınlık bonusu
        self.amd_bars       = 24     # Power of 3 AMD analizi için kaç mum (24h = 1 gün)
        self.sfp_bonus      = 0.13   # SFP (Swing Failure Pattern) bonusu

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

        # ── Adım 6b: PA-1 EQ (Equilibrium) Seviyesi Bonusu ──────────────
        # Range ortasi = EQ = (support + resistance) / 2
        # Fiyat EQ altinda = ucuz bolge → AL bonusu
        # Fiyat EQ ustunde = pahali bolge → SAT bonusu
        eq_al_bonus  = self.eq_bonus if price_position < 0.5 else 0.0
        eq_sat_bonus = self.eq_bonus if price_position > 0.5 else 0.0

        # ── Adım 6c: PA-2 Deviasyon Tespiti ──────────────────────────────
        # Son mumun high/low range disina cikip kapanisi icerde mi?
        dev_al_bonus, dev_sat_bonus = self._detect_deviation(df, support, resistance)

        # ── Adım 6d: PA-3 Order Block Tespiti ────────────────────────────
        ob_al_bonus, ob_sat_bonus = self._find_order_block(
            df, support, resistance, range_width
        )

        # ── Adım 6e: PA-4 Market Yapısı (CHoCH/BOS) ──────────────────────
        # Bullish yapı → AL tercih (SAT sinyalini filtrele)
        # Bearish yapı → SAT tercih (AL sinyalini filtrele)
        market_structure = self._detect_market_structure(df)

        # ── Adım 6f: PA-5 OTE (Optimal Trade Entry) Fibonacci ────────────
        # Swing Low → Swing High arasına 0.618-0.786 bölgesi = ideal AL bölgesi
        # Swing High → Swing Low arasına 0.618-0.786 bölgesi = ideal SAT bölgesi
        ote_al_bonus, ote_sat_bonus = self._calc_ote_zone(df, price)

        # ── Adım 6g: PA-8 Key Levels (Kurumsal Seviyeler) ─────────────────
        # Daily/Weekly open yakınında sinyal daha güvenilir
        key_al_bonus, key_sat_bonus = self._check_key_levels(df, price)

        # ── Adım 6h: PA-9 Power of 3 (AMD) Yapısı ────────────────────────
        # Accumulation → Manipulation → Distribution aşaması tespiti
        amd_phase = self._detect_amd_phase(df)

        # ── Adım 6i: PA-10 SFP (Swing Failure Pattern) ───────────────────
        # Swing kırıldı ama kapanış içeride → güçlü ters sinyal
        sfp_al_bonus, sfp_sat_bonus = self._detect_sfp(df, support, resistance)

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
            # PA-4: Bearish market yapısı varsa AL engelle (CHoCH/BOS aşağı)
            if market_structure == "bearish":
                logger.debug(f"AL engellendi: PA-4 bearish market yapisi (CHoCH/BOS asagi)")
                return Signal(action="BEKLE", confidence=0.0, reason="PA-4: Bearish market yapisi, AL engellendi")

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
            # PA-1 EQ bonusu (ucuz bolge)
            confidence = min(1.0, confidence + eq_al_bonus)
            # PA-2 Deviasyon bonusu (sahte kirilim geri donusu)
            confidence = min(1.0, confidence + dev_al_bonus)
            # PA-3 Order Block bonusu (kurumsal alim bolgesi)
            confidence = min(1.0, confidence + ob_al_bonus)
            # PA-5 OTE Fibonacci bonusu (optimal giris bolgesi)
            confidence = min(1.0, confidence + ote_al_bonus)
            # PA-8 Key Level bonusu (kurumsal seviye yakinligi)
            confidence = min(1.0, confidence + key_al_bonus)
            # PA-9 AMD: Distribution aşamasında AL engelle, Accumulation'da bonus
            if amd_phase == "distribution":
                return Signal(action="BEKLE", confidence=0.0,
                              reason="PA-9: AMD Distribution asamasi, AL engellendi")
            if amd_phase == "accumulation":
                confidence = min(1.0, confidence + 0.08)
            # PA-10 SFP bonusu (swing failure pattern)
            confidence = min(1.0, confidence + sfp_al_bonus)

            # PA-6/7: TP hedefi — Imbalance > Likidite > Klasik (öncelik sırası)
            imb_tp = self._find_imbalance_tp(df, price, direction="AL")
            liq_tp = self._find_liquidity_target(df, price, direction="AL")
            al_tp  = imb_tp or liq_tp or (price + range_width * 0.5)

            signal = Signal(
                action="AL",
                confidence=round(confidence, 3),
                stop_loss=support * (1 - self.stop_pct),
                take_profit=al_tp,
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
            # PA-4: Bullish market yapısı varsa SAT engelle (CHoCH/BOS yukarı)
            if market_structure == "bullish":
                logger.debug(f"SAT engellendi: PA-4 bullish market yapisi (CHoCH/BOS yukari)")
                return Signal(action="BEKLE", confidence=0.0, reason="PA-4: Bullish market yapisi, SAT engellendi")

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
            # PA-1 EQ bonusu (pahali bolge)
            confidence = min(1.0, confidence + eq_sat_bonus)
            # PA-2 Deviasyon bonusu (sahte kirilim geri donusu)
            confidence = min(1.0, confidence + dev_sat_bonus)
            # PA-3 Order Block bonusu (kurumsal satis bolgesi)
            confidence = min(1.0, confidence + ob_sat_bonus)
            # PA-5 OTE Fibonacci bonusu (optimal giris bolgesi)
            confidence = min(1.0, confidence + ote_sat_bonus)
            # PA-8 Key Level bonusu (kurumsal seviye yakinligi)
            confidence = min(1.0, confidence + key_sat_bonus)
            # PA-9 AMD: Accumulation aşamasında SAT engelle, Distribution'da bonus
            if amd_phase == "accumulation":
                return Signal(action="BEKLE", confidence=0.0,
                              reason="PA-9: AMD Accumulation asamasi, SAT engellendi")
            if amd_phase == "distribution":
                confidence = min(1.0, confidence + 0.08)
            # PA-10 SFP bonusu (swing failure pattern)
            confidence = min(1.0, confidence + sfp_sat_bonus)

            # PA-6/7: TP hedefi — Imbalance > Likidite > Klasik (öncelik sırası)
            imb_tp  = self._find_imbalance_tp(df, price, direction="SAT")
            liq_tp  = self._find_liquidity_target(df, price, direction="SAT")
            sat_tp  = imb_tp or liq_tp or (price - range_width * 0.5)

            signal = Signal(
                action="SAT",
                confidence=round(confidence, 3),
                stop_loss=resistance * (1 + self.stop_pct),
                take_profit=sat_tp,
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

    # ── PA-7: Likidite Tabanlı TP Hedefi ─────────────────────────────────────────

    def _find_liquidity_target(
        self, df: pd.DataFrame, price: float, direction: str
    ) -> float | None:
        """
        PA-7: Likidite tabanlı TP hedefi.

        Likidite = Swing High/Low noktaları (stop avcılığı bölgesi).
        Fiyat bu noktalara doğru hareket eder, oradaki stop'ları toplar.

        AL pozisyonu için: fiyatın ÜSTÜNDEKI en yakın Swing High = TP
        SAT pozisyonu için: fiyatın ALTINDAKİ en yakın Swing Low = TP

        "Equal High/Low" (aynı seviyede birden fazla dokunuş) = çok güçlü likidite.

        Returns:
            TP hedefi fiyatı veya None
        """
        n = self.ms_swing_n
        if len(df) < n * 2 + 4:
            return None

        highs = df["high"].values
        lows  = df["low"].values
        size  = len(highs)
        look  = min(size, 60)

        swing_highs = []
        swing_lows  = []

        for i in range(n, look - n):
            idx = size - look + i
            if all(highs[idx] > highs[idx - j] for j in range(1, n + 1)) and \
               all(highs[idx] > highs[idx + j] for j in range(1, n + 1)):
                swing_highs.append(float(highs[idx]))
            if all(lows[idx] < lows[idx - j] for j in range(1, n + 1)) and \
               all(lows[idx] < lows[idx + j] for j in range(1, n + 1)):
                swing_lows.append(float(lows[idx]))

        if direction == "AL" and swing_highs:
            # Fiyatın üstündeki en yakın Swing High = likidite hedefi
            above = [sh for sh in swing_highs if sh > price]
            if above:
                target = min(above)
                logger.debug(f"PA-7 Likidite TP (AL): {target:,.0f}")
                return target

        elif direction == "SAT" and swing_lows:
            # Fiyatın altındaki en yakın Swing Low = likidite hedefi
            below = [sl for sl in swing_lows if sl < price]
            if below:
                target = max(below)
                logger.debug(f"PA-7 Likidite TP (SAT): {target:,.0f}")
                return target

        return None

    # ── PA-6: Imbalance / GAP Tespiti — TP Hedefi ───────────────────────────────

    def _find_imbalance_tp(
        self, df: pd.DataFrame, price: float, direction: str
    ) -> float | None:
        """
        PA-6: Imbalance (dengesizlik bölgesi) tespiti ve TP hedefi.

        Imbalance: Hızlı hareketle oluşan boşluk.
            - Mum[i-1] high ile Mum[i+1] low arasında gap varsa → Bullish Imbalance
            - Mum[i-1] low  ile Mum[i+1] high arasında gap varsa → Bearish Imbalance

        Kural:
            - AL pozisyonu için: fiyatın ÜSTÜNDEKI en yakın bearish imbalance = TP hedefi
            - SAT pozisyonu için: fiyatın ALTINDAKİ en yakın bullish imbalance = TP hedefi
            - İmbalance doldurulmuşsa (fiyat oradan geçtiyse) görmezden gel

        Returns:
            TP hedefi fiyatı veya None (imbalance bulunamazsa)
        """
        n = self.imbalance_bars
        if len(df) < n + 2:
            return None

        window = df.iloc[-(n + 2):-1]  # Son N+2 mum, son mum hariç
        candidates = []

        for i in range(1, len(window) - 1):
            prev = window.iloc[i - 1]
            curr = window.iloc[i]
            nxt  = window.iloc[i + 1]

            prev_high = float(prev["high"])
            prev_low  = float(prev["low"])
            nxt_high  = float(nxt["high"])
            nxt_low   = float(nxt["low"])

            # Bullish Imbalance: prev_low > nxt_high (yukarı boşluk)
            # Ortası = boşluğun orta noktası
            if prev_low > nxt_high:
                mid = (prev_low + nxt_high) / 2
                candidates.append(("bullish", mid, prev_low, nxt_high))

            # Bearish Imbalance: prev_high < nxt_low (aşağı boşluk)
            if prev_high < nxt_low:
                mid = (prev_high + nxt_low) / 2
                candidates.append(("bearish", mid, prev_high, nxt_low))

        if not candidates:
            return None

        if direction == "AL":
            # Fiyatın üstündeki en yakın bearish imbalance (fiyatın gideceği yer)
            above = [(t, mid, lo, hi) for t, mid, lo, hi in candidates
                     if t == "bearish" and mid > price]
            if above:
                closest = min(above, key=lambda x: x[1] - price)
                logger.debug(
                    f"PA-6 Bearish imbalance TP: {closest[1]:,.0f} "
                    f"(zone={closest[2]:,.0f}-{closest[3]:,.0f})"
                )
                return closest[1]

        elif direction == "SAT":
            # Fiyatın altındaki en yakın bullish imbalance
            below = [(t, mid, lo, hi) for t, mid, lo, hi in candidates
                     if t == "bullish" and mid < price]
            if below:
                closest = max(below, key=lambda x: x[1])
                logger.debug(
                    f"PA-6 Bullish imbalance TP: {closest[1]:,.0f} "
                    f"(zone={closest[2]:,.0f}-{closest[3]:,.0f})"
                )
                return closest[1]

        return None

    # ── PA-5: OTE (Optimal Trade Entry) — Fibonacci 0.618 / 0.705 / 0.786 ─────

    def _calc_ote_zone(self, df: pd.DataFrame, price: float) -> tuple[float, float]:
        """
        PA-5: OTE (Optimal Trade Entry) bölgesi tespiti.

        Bullish OTE (AL için):
            Son belirgin Swing Low → Swing High arasına Fibonacci çekiliyor.
            0.618 - 0.786 retrace bölgesi = optimal long giriş.
            Fiyat bu bölgedeyse → AL sinyali güçlenir (+ote_bonus)

        Bearish OTE (SAT için):
            Son belirgin Swing High → Swing Low arasına Fibonacci çekiliyor.
            0.618 - 0.786 retrace bölgesi = optimal short giriş.
            Fiyat bu bölgedeyse → SAT sinyali güçlenir (+ote_bonus)

        0.705 = en sık kullanılan "sweet spot" OTE seviyesi (DD Finance)

        Returns:
            (al_bonus, sat_bonus): float tuple
        """
        n = self.ms_swing_n
        min_bars = n * 2 + 10
        if len(df) < min_bars:
            return 0.0, 0.0

        highs = df["high"].values
        lows  = df["low"].values
        size  = len(highs)
        look  = min(size, 80)

        # Swing noktalarını bul
        swing_highs = []
        swing_lows  = []
        for i in range(n, look - n):
            idx = size - look + i
            if all(highs[idx] > highs[idx - j] for j in range(1, n + 1)) and \
               all(highs[idx] > highs[idx + j] for j in range(1, n + 1)):
                swing_highs.append((idx, float(highs[idx])))
            if all(lows[idx] < lows[idx - j] for j in range(1, n + 1)) and \
               all(lows[idx] < lows[idx + j] for j in range(1, n + 1)):
                swing_lows.append((idx, float(lows[idx])))

        if not swing_highs or not swing_lows:
            return 0.0, 0.0

        # OTE Fibonacci seviyeleri
        FIB_LOW  = 0.618
        FIB_HIGH = 0.786

        al_bonus  = 0.0
        sat_bonus = 0.0

        # Bullish OTE: Son Swing Low'dan son Swing High'a çekilen Fibonacci
        # Swing Low sonra Swing High olmalı (yukarı hareket)
        last_sl_idx, last_sl = swing_lows[-1]
        last_sh_idx, last_sh = swing_highs[-1]

        if last_sl_idx < last_sh_idx and last_sh > last_sl:
            # Fibonacci seviyeleri (retrace = geri çekilme)
            move      = last_sh - last_sl
            ote_low   = last_sh - move * FIB_HIGH   # 0.786 retrace
            ote_high  = last_sh - move * FIB_LOW    # 0.618 retrace
            if ote_low <= price <= ote_high:
                al_bonus = self.ote_bonus
                logger.debug(
                    f"PA-5 Bullish OTE: SL={last_sl:,.0f} SH={last_sh:,.0f} | "
                    f"OTE zone={ote_low:,.0f}-{ote_high:,.0f} | fiyat={price:,.0f} | "
                    f"AL bonus: +{al_bonus}"
                )

        # Bearish OTE: Son Swing High'dan son Swing Low'a çekilen Fibonacci
        # Swing High sonra Swing Low olmalı (aşağı hareket)
        if last_sh_idx < last_sl_idx and last_sl < last_sh:
            move      = last_sh - last_sl
            ote_low   = last_sl + move * FIB_LOW    # 0.618 retrace yukarı
            ote_high  = last_sl + move * FIB_HIGH   # 0.786 retrace yukarı
            if ote_low <= price <= ote_high:
                sat_bonus = self.ote_bonus
                logger.debug(
                    f"PA-5 Bearish OTE: SH={last_sh:,.0f} SL={last_sl:,.0f} | "
                    f"OTE zone={ote_low:,.0f}-{ote_high:,.0f} | fiyat={price:,.0f} | "
                    f"SAT bonus: +{sat_bonus}"
                )

        return al_bonus, sat_bonus

    # ── PA-4: Market Yapısı (CHoCH / BOS) ────────────────────────────────────

    def _detect_market_structure(self, df: pd.DataFrame) -> str:
        """
        PA-4: Market yapısı tespiti — Swing High/Low + CHoCH/BOS analizi.

        Swing High: Her iki yanında ms_swing_n kadar daha düşük high olan tepe
        Swing Low : Her iki yanında ms_swing_n kadar daha yüksek low olan dip

        BOS   (Break of Structure): Önceki swing kırıldı → trend teyidi
        CHoCH (Change of Character): Zıt yönde swing kırıldı → trend değişimi

        Karar mantığı:
          Son 2 Swing High her seferinde yükseldiyse VE son BOS yukarıysa → "bullish"
          Son 2 Swing Low her seferinde düştüyse VE son BOS aşağıysa   → "bearish"
          Aksi halde                                                     → "neutral"

        Returns:
            "bullish" | "bearish" | "neutral"
        """
        n = self.ms_swing_n
        min_bars = n * 2 + 4
        if len(df) < min_bars:
            return "neutral"

        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        size   = len(highs)

        # Swing High/Low tespiti (son 60 mum yeterli)
        look = min(size, 60)
        swing_highs = []
        swing_lows  = []

        for i in range(n, look - n):
            idx = size - look + i
            # Swing High: her iki yanda n mum daha alçak
            if all(highs[idx] > highs[idx - j] for j in range(1, n + 1)) and \
               all(highs[idx] > highs[idx + j] for j in range(1, n + 1)):
                swing_highs.append((idx, highs[idx]))
            # Swing Low: her iki yanda n mum daha yüksek
            if all(lows[idx] < lows[idx - j] for j in range(1, n + 1)) and \
               all(lows[idx] < lows[idx + j] for j in range(1, n + 1)):
                swing_lows.append((idx, lows[idx]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "neutral"

        # Son iki swing karsilastir
        last_sh,  prev_sh  = swing_highs[-1][1], swing_highs[-2][1]
        last_sl,  prev_sl  = swing_lows[-1][1],  swing_lows[-2][1]
        last_close = float(closes[-1])

        # Bullish yapı: Higher High + Higher Low
        hh = last_sh > prev_sh   # Higher High
        hl = last_sl > prev_sl   # Higher Low
        # Bearish yapı: Lower High + Lower Low
        lh = last_sh < prev_sh   # Lower High
        ll = last_sl < prev_sl   # Lower Low

        # BOS/CHoCH teyidi: fiyat son swing High/Low'u aştı mı?
        bos_bullish = last_close > last_sh  # Fiyat son Swing High'ı kırdı
        bos_bearish = last_close < last_sl  # Fiyat son Swing Low'u kırdı

        if hh and hl:
            structure = "bullish"
        elif lh and ll:
            structure = "bearish"
        else:
            structure = "neutral"

        # BOS teyidi varsa daha net sonuç
        if bos_bullish and structure != "bearish":
            structure = "bullish"
        elif bos_bearish and structure != "bullish":
            structure = "bearish"

        logger.debug(
            f"PA-4 Market yapisi: {structure} | "
            f"SH: {prev_sh:,.0f}->{last_sh:,.0f} ({'HH' if hh else 'LH'}) | "
            f"SL: {prev_sl:,.0f}->{last_sl:,.0f} ({'HL' if hl else 'LL'}) | "
            f"BOS_bull={bos_bullish} BOS_bear={bos_bearish}"
        )
        return structure

    # ── PA-2: Deviasyon Tespiti ───────────────────────────────────────────────

    def _detect_deviation(
        self, df: pd.DataFrame, support: float, resistance: float
    ) -> tuple[float, float]:
        """
        PA-2: Deviasyon (sahte kirilim) tespiti.

        Bullish deviasyon: Son mumun low < support AMA close >= support
            → Fiyat desteği kırdı görüntüsü verdi, ama kapandı içeride
            → Güçlü AL sinyali (+dev_bonus)

        Bearish deviasyon: Son mumun high > resistance AMA close <= resistance
            → Fiyat direnci kırdı görüntüsü verdi, ama kapandı içeride
            → Güçlü SAT sinyali (+dev_bonus)

        Returns:
            (al_bonus, sat_bonus): float tuple
        """
        if len(df) < 2:
            return 0.0, 0.0

        last = df.iloc[-1]
        al_bonus  = 0.0
        sat_bonus = 0.0

        # Bullish deviasyon: low support altina indi, close support uzerinde kapandi
        if float(last["low"]) < support and float(last["close"]) >= support:
            al_bonus = self.dev_bonus
            logger.debug(
                f"PA-2 Bullish deviasyon: low={last['low']:,.0f} < support={support:,.0f}, "
                f"close={last['close']:,.0f} >= support | AL bonus: +{al_bonus}"
            )

        # Bearish deviasyon: high resistance ustune cikti, close resistance altinda kapandi
        if float(last["high"]) > resistance and float(last["close"]) <= resistance:
            sat_bonus = self.dev_bonus
            logger.debug(
                f"PA-2 Bearish deviasyon: high={last['high']:,.0f} > resistance={resistance:,.0f}, "
                f"close={last['close']:,.0f} <= resistance | SAT bonus: +{sat_bonus}"
            )

        return al_bonus, sat_bonus

    # ── PA-3: Order Block Tespiti ─────────────────────────────────────────────

    def _find_order_block(
        self,
        df: pd.DataFrame,
        support: float,
        resistance: float,
        range_width: float,
    ) -> tuple[float, float]:
        """
        PA-3: Order Block (OB) tespiti.

        Bullish OB: Kırmızı mum ardından büyük yeşil mum (önceki düşüşü kapatan)
            + OB destek bölgesinde (support yakını) ise → AL bonusu
        Bearish OB: Yeşil mum ardından büyük kırmızı mum
            + OB direnç bölgesinde (resistance yakını) ise → SAT bonusu

        "Büyük" tanımı: Gövdesi ortalama gövdenin 1.5 katı üzerinde

        Returns:
            (al_bonus, sat_bonus): float tuple
        """
        n = self.ob_lookback
        if len(df) < n + 2:
            return 0.0, 0.0

        window = df.iloc[-(n + 1):-1]  # Son N mum (son mumu dahil etme)
        prox   = range_width * self.proximity_pct

        # Ortalama mum gövdesi (bias onleme: son mum dahil edilmiyor)
        bodies = abs(window["close"] - window["open"])
        avg_body = float(bodies.mean()) if len(bodies) > 0 else 1.0

        al_bonus  = 0.0
        sat_bonus = 0.0

        for i in range(len(window) - 1):
            prev = window.iloc[i]
            curr = window.iloc[i + 1]

            prev_bearish = float(prev["close"]) < float(prev["open"])  # Kırmızı mum
            curr_bullish = float(curr["close"]) > float(curr["open"])  # Yeşil mum
            curr_body    = abs(float(curr["close"]) - float(curr["open"]))

            # Bullish OB: kirmizi → buyuk yesil
            if (
                prev_bearish
                and curr_bullish
                and curr_body >= avg_body * 1.5
                and float(curr["close"]) >= float(prev["open"])  # onceki dususu kapatiyor
            ):
                # OB destek bolgesinde mi?
                ob_level = float(prev["low"])  # OB seviyesi = kirmizi mumun dibi
                if ob_level <= support + prox:
                    al_bonus = self.ob_bonus
                    logger.debug(
                        f"PA-3 Bullish OB tespit edildi: ob_level={ob_level:,.0f} "
                        f"destek={support:,.0f} | AL bonus: +{al_bonus}"
                    )
                    break

            prev_bullish = float(prev["close"]) > float(prev["open"])  # Yeşil mum
            curr_bearish = float(curr["close"]) < float(curr["open"])  # Kırmızı mum
            curr_body2   = abs(float(curr["close"]) - float(curr["open"]))

            # Bearish OB: yesil → buyuk kirmizi
            if (
                prev_bullish
                and curr_bearish
                and curr_body2 >= avg_body * 1.5
                and float(curr["close"]) <= float(prev["open"])  # onceki yukselisi kapatiyor
            ):
                # OB direnc bolgesinde mi?
                ob_level = float(prev["high"])  # OB seviyesi = yesil mumun tepesi
                if ob_level >= resistance - prox:
                    sat_bonus = self.ob_bonus
                    logger.debug(
                        f"PA-3 Bearish OB tespit edildi: ob_level={ob_level:,.0f} "
                        f"direnc={resistance:,.0f} | SAT bonus: +{sat_bonus}"
                    )
                    break

        return al_bonus, sat_bonus

    # ── PA-8: Key Levels (Kurumsal Seviyeler) ────────────────────────────────

    def _check_key_levels(
        self, df: pd.DataFrame, price: float
    ) -> tuple[float, float]:
        """
        PA-8: Kurumsal key level yakınlığı tespiti.

        Daily Open   : Günün ilk kapanış fiyatı (00:00 UTC mumu)
        Weekly Open  : Haftanın ilk mumu (Pazartesi 00:00)
        Monday High/Low: Pazar/Pazartesi çerçevesi

        Bu seviyelere yakın sinyal daha güvenilir çünkü kurumlar bu seviyeleri
        referans alarak işlem yapar.

        Returns:
            (al_bonus, sat_bonus): float tuple
        """
        if len(df) < 2 or not hasattr(df.index, 'day_of_week'):
            return 0.0, 0.0

        try:
            prox = price * self.key_level_pct  # %0.5 yakınlık

            key_levels = []

            # Daily Open: Bugünün ilk mumunun open değeri
            today = df.index[-1].date()
            today_candles = df[df.index.date == today]
            if len(today_candles) > 0:
                daily_open = float(today_candles.iloc[0]["open"])
                key_levels.append(daily_open)

            # Weekly Open: Bu haftanın ilk mumunun open değeri
            week_start = df.index[-1] - pd.Timedelta(days=df.index[-1].day_of_week)
            week_candles = df[df.index >= week_start.normalize()]
            if len(week_candles) > 0:
                weekly_open = float(week_candles.iloc[0]["open"])
                key_levels.append(weekly_open)

            # Monday High/Low
            monday_candles = df[df.index.day_of_week == 0]
            if len(monday_candles) > 0:
                monday_high = float(monday_candles["high"].iloc[-1])
                monday_low  = float(monday_candles["low"].iloc[-1])
                key_levels.extend([monday_high, monday_low])

            al_bonus  = 0.0
            sat_bonus = 0.0

            for level in key_levels:
                if abs(price - level) <= prox:
                    # Fiyat key level altındaysa ve yakınsa → AL güçlü
                    if price <= level:
                        al_bonus = max(al_bonus, self.key_bonus)
                        logger.debug(f"PA-8 Key level AL: fiyat={price:,.0f} <= level={level:,.0f}")
                    # Fiyat key level üstündeyse ve yakınsa → SAT güçlü
                    else:
                        sat_bonus = max(sat_bonus, self.key_bonus)
                        logger.debug(f"PA-8 Key level SAT: fiyat={price:,.0f} >= level={level:,.0f}")

            return al_bonus, sat_bonus

        except Exception as e:
            logger.debug(f"PA-8 Key levels hesaplanamadi: {e}")
            return 0.0, 0.0

    # ── PA-9: Power of 3 (AMD — Accumulation / Manipulation / Distribution) ──

    def _detect_amd_phase(self, df: pd.DataFrame) -> str:
        """
        PA-9: Power of 3 (AMD) yapısı tespiti.

        ICT'nin Power of 3 konsepti:
          Accumulation  : Range içinde fiyat sıkışıyor (düşük volatilite, dar range)
          Manipulation  : Fiyat bir yöne sert hareket eder (stop avı), sahte kırılım
          Distribution  : Asıl yön ortaya çıkar, güçlü trend hareketi

        Basit tespit:
          - Son amd_bars mumun range'i ortalamaya kıyasla dar → Accumulation
          - Önceki mumlara göre ani yüksek hacim + büyük mum → Manipulation
          - Son mum büyük + kapanış net bir yönde → Distribution

        Returns:
            "accumulation" | "manipulation" | "distribution" | "neutral"
        """
        n = self.amd_bars
        if len(df) < n + 5:
            return "neutral"

        try:
            window  = df.iloc[-n:]
            prev    = df.iloc[-(n + 5):-n]

            # Range hesabı
            curr_range = float((window["high"] - window["low"]).mean())
            prev_range = float((prev["high"] - prev["low"]).mean()) if len(prev) > 0 else curr_range

            # Hacim hesabı
            curr_vol  = float(window["volume"].mean())
            prev_vol  = float(prev["volume"].mean()) if len(prev) > 0 else curr_vol

            # Son mum gövdesi
            last_body = abs(float(df["close"].iloc[-1]) - float(df["open"].iloc[-1]))
            avg_body  = float(abs(window["close"] - window["open"]).mean())

            # Accumulation: range daralıyor, hacim düşük
            if curr_range < prev_range * 0.7 and curr_vol < prev_vol * 0.8:
                return "accumulation"

            # Distribution: son mum büyük, hacim yüksek
            if last_body > avg_body * 2.0 and float(df["volume"].iloc[-1]) > curr_vol * 1.5:
                return "distribution"

            # Manipulation: hacim ani spike, range genişliyor ama kapanış ortada
            last_high  = float(df["high"].iloc[-1])
            last_low   = float(df["low"].iloc[-1])
            last_close = float(df["close"].iloc[-1])
            last_range = last_high - last_low
            close_pos  = (last_close - last_low) / last_range if last_range > 0 else 0.5

            if (float(df["volume"].iloc[-1]) > curr_vol * 2.0
                    and 0.3 < close_pos < 0.7):  # Kapanış ortada = manipulation
                return "manipulation"

            return "neutral"

        except Exception as e:
            logger.debug(f"PA-9 AMD tespiti hatasi: {e}")
            return "neutral"

    # ── PA-10: SFP (Swing Failure Pattern) ───────────────────────────────────

    def _detect_sfp(
        self, df: pd.DataFrame, support: float, resistance: float
    ) -> tuple[float, float]:
        """
        PA-10: Swing Failure Pattern (SFP) tespiti.

        SFP: Önceki swing high/low kırıldı AMA kapanış içeride kaldı.
            → Sahte kırılım + güçlü geri dönüş sinyali

        Bullish SFP: Son mum bir önceki swing low'u kırdı (low < prev_swing_low)
                     AMA close > prev_swing_low → AL sinyali güçlü
        Bearish SFP: Son mum bir önceki swing high'ı kırdı (high > prev_swing_high)
                     AMA close < prev_swing_high → SAT sinyali güçlü

        Deviasyon ile farkı: SFP önceki swing noktasını referans alır,
        Deviasyon ise mevcut range support/resistance'ını referans alır.

        Returns:
            (al_bonus, sat_bonus): float tuple
        """
        n = self.ms_swing_n
        if len(df) < n * 2 + 6:
            return 0.0, 0.0

        try:
            highs = df["high"].values
            lows  = df["low"].values
            size  = len(highs)
            look  = min(size - 1, 40)  # Son 40 mumda swing ara

            swing_highs = []
            swing_lows  = []

            for i in range(n, look - n):
                idx = size - 1 - look + i  # Son mumu hariç tut
                if idx <= 0 or idx >= size - 1:
                    continue
                if all(highs[idx] > highs[idx - j] for j in range(1, n + 1)) and \
                   all(highs[idx] > highs[idx + j] for j in range(1, n + 1)):
                    swing_highs.append(float(highs[idx]))
                if all(lows[idx] < lows[idx - j] for j in range(1, n + 1)) and \
                   all(lows[idx] < lows[idx + j] for j in range(1, n + 1)):
                    swing_lows.append(float(lows[idx]))

            last      = df.iloc[-1]
            last_high = float(last["high"])
            last_low  = float(last["low"])
            last_close = float(last["close"])

            al_bonus  = 0.0
            sat_bonus = 0.0

            # Bullish SFP: son mum swing low kırdı ama içeride kapandı
            if swing_lows:
                prev_sl = swing_lows[-1]
                if last_low < prev_sl and last_close >= prev_sl:
                    al_bonus = self.sfp_bonus
                    logger.debug(
                        f"PA-10 Bullish SFP: low={last_low:,.0f} < swing_low={prev_sl:,.0f}, "
                        f"close={last_close:,.0f} >= swing_low | AL bonus: +{al_bonus}"
                    )

            # Bearish SFP: son mum swing high kırdı ama içeride kapandı
            if swing_highs:
                prev_sh = swing_highs[-1]
                if last_high > prev_sh and last_close <= prev_sh:
                    sat_bonus = self.sfp_bonus
                    logger.debug(
                        f"PA-10 Bearish SFP: high={last_high:,.0f} > swing_high={prev_sh:,.0f}, "
                        f"close={last_close:,.0f} <= swing_high | SAT bonus: +{sat_bonus}"
                    )

            return al_bonus, sat_bonus

        except Exception as e:
            logger.debug(f"PA-10 SFP tespiti hatasi: {e}")
            return 0.0, 0.0

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
