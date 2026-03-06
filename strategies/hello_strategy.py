"""
strategies/hello_strategy.py
==============================
AMACI:
    Phase 1'in test stratejisi — "Hello World" eşdeğeri.
    Gerçek bir trading stratejisi DEĞİL, sadece:
        - BaseStrategy'nin nasıl kullanıldığını gösterir
        - Sistemin çalıştığını doğrular
        - Gelecekteki stratejiler için şablon görevi görür

ÇALIŞMA MANTIĞI (Basit Momentum):
    - Son mumun kapanışı önceki mumun kapanışından yüksekse -> AL
    - Son mumun kapanışı önceki mumun kapanışından düşükse  -> SAT
    - Fark %0.1'den azsa                                   -> BEKLE

ÇALIŞTIRMAK İÇİN:
    python strategies/hello_strategy.py
"""

import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, Signal
from utils.logger import get_logger

logger = get_logger(__name__)   # __name__ = "strategies.hello_strategy"


class HelloStrategy(BaseStrategy):
    """
    Basit momentum stratejisi — eğitim ve test amaçlı.

    Parametre:
        threshold: Sinyal üretmek için minimum fiyat değişim yüzdesi (varsayılan %0.1)
    """

    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h", threshold: float = 0.001):
        """
        threshold = 0.001 → fiyat en az %0.1 değişmeli ki sinyal üretsin
        """
        super().__init__(name="HelloStrategy", symbol=symbol, timeframe=timeframe)
        self.threshold = threshold

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Son iki mumu karşılaştırır, yön ve güç hesaplar.

        df: En az 2 satır OHLCV DataFrame (son satır = son kapanan mum)
        """
        # Son iki mumun kapanış fiyatını al
        last_close = df["close"].iloc[-1]      # En son mum
        prev_close = df["close"].iloc[-2]      # Bir önceki mum

        # Yüzde değişim hesapla
        pct_change = (last_close - prev_close) / prev_close  # örn: 0.0023 = %0.23 artış

        # Güven skoru: değişim ne kadar büyükse sinyal o kadar güçlü (max 1.0)
        confidence = min(abs(pct_change) / 0.01, 1.0)  # %1 değişim = 1.0 güven

        # Stop loss ve take profit hesapla (ATR tabanlı değil, basit sabit yüzde)
        stop_loss_pct = 0.005    # %0.5 stop loss
        take_profit_pct = 0.01  # %1.0 take profit

        if pct_change > self.threshold:
            # Fiyat yükseliyor → AL
            signal = Signal(
                action="AL",
                confidence=confidence,
                stop_loss=last_close * (1 - stop_loss_pct),
                take_profit=last_close * (1 + take_profit_pct),
                reason=f"Fiyat %{pct_change*100:.3f} artti (esik: %{self.threshold*100:.1f})"
            )
            logger.info(f"Sinyal: AL | {self.symbol} | Fiyat: {last_close:,.2f} | Guven: {confidence:.2f}")
            return signal
        elif pct_change < -self.threshold:
            # Fiyat düşüyor → SAT
            signal = Signal(
                action="SAT",
                confidence=confidence,
                stop_loss=last_close * (1 + stop_loss_pct),
                take_profit=last_close * (1 - take_profit_pct),
                reason=f"Fiyat %{abs(pct_change)*100:.3f} dustu (esik: %{self.threshold*100:.1f})"
            )
            logger.info(f"Sinyal: SAT | {self.symbol} | Fiyat: {last_close:,.2f} | Guven: {confidence:.2f}")
            return signal
        else:
            # Değişim çok küçük → BEKLE
            logger.debug(f"Sinyal: BEKLE | Degisim %{abs(pct_change)*100:.3f} < esik %{self.threshold*100:.1f}")
            return Signal(
                action="BEKLE",
                confidence=0.0,
                reason=f"Degisim esik altinda: %{abs(pct_change)*100:.3f} < %{self.threshold*100:.1f}"
            )


# ============================================================
# TEST BLOGU — Bu dosyayı direkt çalıştırınca burası koşar
# python strategies/hello_strategy.py
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  ALGO TRADE CODEX — HelloStrategy Test")
    print("=" * 55)

    # Sahte OHLCV verisi oluştur (gerçek borsa bağlantısı yok)
    np.random.seed(42)
    n = 20  # 20 mum

    # BTC fiyatı 60,000'den başlayan sahte fiyatlar
    closes = [60000]
    for _ in range(n - 1):
        change = np.random.uniform(-0.005, 0.005)   # ±%0.5 rastgele değişim
        closes.append(closes[-1] * (1 + change))

    fake_df = pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(n)]
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))

    print(f"\nSahte veri: {n} mum, BTC/USDT 1h")
    print(f"Fiyat araligi: {min(closes):.0f} - {max(closes):.0f} USDT\n")

    # Stratejiyi başlat ve çalıştır
    strategy = HelloStrategy(symbol="BTC/USDT", timeframe="1h", threshold=0.001)
    print(f"Strateji: {strategy}\n")

    # Son 5 mum için sinyal üret
    print("Son 5 mum için sinyaller:")
    print("-" * 55)
    for i in range(n - 5, n):
        df_slice = fake_df.iloc[:i + 1]          # i+1 muma kadar olan veri
        signal = strategy.run(df_slice)
        print(f"  Mum {i+1:2d} | Kapanış: {df_slice['close'].iloc[-1]:,.0f} | {signal}")

    print("-" * 55)
    print(f"\nToplam üretilen sinyal sayisi: {strategy.signal_count}")
    print(f"Son sinyal: {strategy.last_signal}")
    print("\nBASARI: HelloStrategy çalisiyor!")
