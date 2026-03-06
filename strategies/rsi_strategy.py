"""
strategies/rsi_strategy.py
============================
AMACI:
    RSI (Relative Strength Index) tabanlı Mean Reversion stratejisi.
    "Mean Reversion" = Fiyat aşırıya gidince ortaya döner mantığı.

ÇALIŞMA MANTIĞI:
    RSI 0-100 arası bir göstergedir:
        RSI < 30  → Aşırı satım (oversold)  → Fiyat çok düştü, geri çıkabilir → AL
        RSI > 70  → Aşırı alım (overbought) → Fiyat çok yükseldi, düşebilir  → SAT
        30-70 arası → Normal bölge → BEKLE

KULLANDIĞI ARAÇ:
    pandas-ta kütüphanesi → df.ta.rsi() ile tek satırda RSI hesaplar
    Manuel hesaplamaya göre çok daha hızlı ve güvenilir.

ÇALIŞTIRMAK İÇİN:
    python -m strategies.rsi_strategy
"""

import pandas as pd
import numpy as np
import pandas_ta as ta                              # teknik indikatör kütüphanesi
from strategies.base_strategy import BaseStrategy, Signal
from utils.logger import get_logger

logger = get_logger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI Mean Reversion Stratejisi.

    Parametreler:
        rsi_period  : RSI hesaplama periyodu (kaç mum geriye bakılır) — varsayılan 14
        oversold    : Bu değerin altı → AL sinyali  — varsayılan 30
        overbought  : Bu değerin üstü → SAT sinyali — varsayılan 70
        stop_pct    : Stop-loss yüzdesi              — varsayılan %1.5
        tp_pct      : Take-profit yüzdesi            — varsayılan %3.0

    Neden 14 periyot?
        RSI'nin mucidi Welles Wilder tarafından önerilen standart değer.
        Daha düşük (7) → daha fazla ama gürültülü sinyal.
        Daha yüksek (21) → daha az ama güvenilir sinyal.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        stop_pct: float = 0.015,
        tp_pct: float = 0.030,
    ):
        super().__init__(name="RSIStrategy", symbol=symbol, timeframe=timeframe)
        self.rsi_period  = rsi_period
        self.oversold    = oversold
        self.overbought  = overbought
        self.stop_pct    = stop_pct
        self.tp_pct      = tp_pct

        logger.info(
            f"RSIStrategy baslatildi | {symbol} {timeframe} | "
            f"Periyot: {rsi_period} | Oversold: {oversold} | Overbought: {overbought}"
        )

    # ── Ana Sinyal Üretimi ────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        DataFrame'i alır, RSI hesaplar, sinyal üretir.

        df: OHLCV verisi — en az (rsi_period + 1) satır olmalı
        """
        # Yeterli veri var mı?
        min_rows = self.rsi_period + 1
        if len(df) < min_rows:
            logger.warning(f"Yetersiz veri: {len(df)} satir, en az {min_rows} gerekli")
            return Signal(action="BEKLE", confidence=0.0,
                          reason=f"Yetersiz veri: {len(df)}/{min_rows}")

        # ── RSI Hesapla ───────────────────────────────────────────────────────
        # pandas-ta ile tek satır: df["close"] kolonuna bakarak RSI serisi üretir
        # Sonuç: her satır için bir RSI değeri (ilk rsi_period satır NaN olur)
        rsi_series = ta.rsi(df["close"], length=self.rsi_period)

        # Son iki RSI değerini al
        current_rsi = rsi_series.iloc[-1]    # şu anki RSI
        prev_rsi    = rsi_series.iloc[-2]    # bir önceki RSI

        # RSI NaN ise (veri yetersiz veya hata) BEKLE döndür
        if pd.isna(current_rsi):
            return Signal(action="BEKLE", confidence=0.0, reason="RSI hesaplanamadi")

        current_price = df["close"].iloc[-1]

        logger.debug(
            f"RSI hesaplandi | Onceki: {prev_rsi:.1f} | Simdiki: {current_rsi:.1f} | "
            f"Fiyat: {current_price:,.2f}"
        )

        # ── Sinyal Mantığı ────────────────────────────────────────────────────

        # AL sinyali: RSI oversold bölgesini YUKARI kesiyor
        # Sadece RSI < oversold değil, çünkü RSI 20'de iken de düşmeye devam edebilir.
        # "Yukarı kesiyor" = önceki RSI oversold altında, şimdiki üstüne çıktı → dönüş sinyali
        if prev_rsi < self.oversold and current_rsi >= self.oversold:
            confidence = self._calc_confidence(prev_rsi, mode="AL")   # önceki RSI ne kadar derindeydi?
            signal = Signal(
                action="AL",
                confidence=confidence,
                stop_loss=current_price * (1 - self.stop_pct),
                take_profit=current_price * (1 + self.tp_pct),
                reason=f"RSI oversold'dan cikti: {prev_rsi:.1f} -> {current_rsi:.1f} (esik: {self.oversold})"
            )
            logger.info(
                f"SINYAL AL | {self.symbol} | Fiyat: {current_price:,.2f} | "
                f"RSI: {current_rsi:.1f} | Stop: {signal.stop_loss:,.2f} | TP: {signal.take_profit:,.2f}"
            )
            return signal

        # SAT sinyali: RSI overbought bölgesini AŞAĞI kesiyor
        elif prev_rsi > self.overbought and current_rsi <= self.overbought:
            confidence = self._calc_confidence(prev_rsi, mode="SAT")   # önceki RSI ne kadar yüksekteydi?
            signal = Signal(
                action="SAT",
                confidence=confidence,
                stop_loss=current_price * (1 + self.stop_pct),
                take_profit=current_price * (1 - self.tp_pct),
                reason=f"RSI overbought'dan indi: {prev_rsi:.1f} -> {current_rsi:.1f} (esik: {self.overbought})"
            )
            logger.info(
                f"SINYAL SAT | {self.symbol} | Fiyat: {current_price:,.2f} | "
                f"RSI: {current_rsi:.1f} | Stop: {signal.stop_loss:,.2f} | TP: {signal.take_profit:,.2f}"
            )
            return signal

        # BEKLE: RSI normal bölgede
        else:
            logger.debug(f"BEKLE | RSI: {current_rsi:.1f} (normal bolge {self.oversold}-{self.overbought})")
            return Signal(
                action="BEKLE",
                confidence=0.0,
                reason=f"RSI normal bolgede: {current_rsi:.1f}"
            )

    # ── Yardımcı Metod ────────────────────────────────────────────────────────

    def _calc_confidence(self, rsi: float, mode: str) -> float:
        """
        RSI değerine göre sinyal güven skoru hesaplar (0.0 - 1.0).

        AL modunda: RSI ne kadar düşükse güven o kadar yüksek
            RSI=30 → güven 0.50 (eşikte)
            RSI=20 → güven 0.75
            RSI=10 → güven 1.00

        SAT modunda: RSI ne kadar yüksekse güven o kadar yüksek
            RSI=70 → güven 0.50
            RSI=80 → güven 0.75
            RSI=90 → güven 1.00
        """
        if mode == "AL":
            # Oversold eşiğinden (30) ne kadar uzaksa (aşağı) o kadar güçlü
            distance = max(0, self.oversold - rsi)      # örn: 30 - 18 = 12
            confidence = min(distance / self.oversold, 1.0)  # 0.0 - 1.0
        else:
            # Overbought eşiğinden (70) ne kadar uzaksa (yukarı) o kadar güçlü
            distance = max(0, rsi - self.overbought)     # örn: 85 - 70 = 15
            confidence = min(distance / (100 - self.overbought), 1.0)
        return round(confidence, 3)

    def get_current_rsi(self, df: pd.DataFrame) -> float | None:
        """
        Dışarıdan mevcut RSI değerini sorgulamak için yardımcı metod.
        Dashboard veya loglama için kullanılabilir.
        """
        if len(df) < self.rsi_period + 1:
            return None
        rsi_series = ta.rsi(df["close"], length=self.rsi_period)
        val = rsi_series.iloc[-1]
        return float(val) if not pd.isna(val) else None


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOGU
# python -m strategies.rsi_strategy
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ALGO TRADE CODEX — RSIStrategy Test")
    print("=" * 60)

    np.random.seed(7)
    n = 100     # RSI için 100 mum lazım (14 periyot + yeterli geçmiş)

    # Senaryolu sahte fiyat verisi:
    # 1. Önce düşüş (RSI oversold bölgesine girer → AL sinyali beklenir)
    # 2. Sonra yükseliş (RSI overbought bölgesine girer → SAT sinyali beklenir)
    closes = [50000]
    for i in range(n - 1):
        if i < 30:
            change = np.random.uniform(-0.012, 0.002)   # Ağırlıklı düşüş
        elif i < 60:
            change = np.random.uniform(-0.002, 0.015)   # Ağırlıklı yükseliş
        else:
            change = np.random.uniform(-0.005, 0.005)   # Yatay
        closes.append(closes[-1] * (1 + change))

    df = pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.003 for c in closes],
        "low":    [c * 0.997 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(50, 300) for _ in range(n)],
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))

    print(f"\nSahte veri: {n} mum | BTC/USDT 1h")
    print(f"Fiyat araligi: {min(closes):,.0f} - {max(closes):,.0f} USDT\n")

    strategy = RSIStrategy(
        symbol="BTC/USDT", timeframe="1h",
        rsi_period=14, oversold=30, overbought=70
    )
    print(f"Strateji: {strategy}\n")

    # Tüm mumları tara, sadece AL/SAT sinyallerini göster
    print("Uretilen AL/SAT sinyalleri:")
    print("-" * 60)
    al_count = sat_count = 0
    for i in range(15, n):          # İlk 15 mumu atla (RSI için yeterli geçmiş yok)
        df_slice = df.iloc[:i + 1]
        signal = strategy.run(df_slice)
        if signal.action != "BEKLE":
            rsi_val = strategy.get_current_rsi(df_slice)
            print(
                f"  Mum {i+1:3d} | Fiyat: {df_slice['close'].iloc[-1]:,.0f} | "
                f"RSI: {rsi_val:.1f} | {signal.action} | Guven: {signal.confidence:.2f} | {signal.reason}"
            )
            if signal.action == "AL":
                al_count += 1
            else:
                sat_count += 1

    print("-" * 60)
    print(f"\nToplam AL sinyali : {al_count}")
    print(f"Toplam SAT sinyali: {sat_count}")
    print(f"Toplam sinyal      : {strategy.signal_count}")
    print(f"\nSon RSI degeri: {strategy.get_current_rsi(df):.1f}")
    print("\nBASARI: RSIStrategy calisiyor!")
