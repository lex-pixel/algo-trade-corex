"""
strategies/base_strategy.py
============================
AMACI:
    Tüm stratejilerin uyması gereken ortak şablonu (sözleşmeyi) tanımlar.
    Her yeni strateji bu sınıftan türer ve generate_signal() metodunu
    kendi kurallarına göre doldurur.

NEDEN ABC (Abstract Base Class)?
    ABC, generate_signal() metodunu "soyut" yapar.
    Yani bu sınıftan türeyen her alt sınıf bu metodu MUTLAKA yazmak zorundadır.
    Yazmadan çalıştırmaya kalkışırsa Python hata fırlatır — bu bir güvenlik ağıdır.

KULLANIM:
    from strategies.base_strategy import BaseStrategy

    class BenimStratejim(BaseStrategy):
        def generate_signal(self, df):
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    """
    Bir strateji sinyalini temsil eder.

    Alanlar:
        action      : 'AL', 'SAT' veya 'BEKLE'
        confidence  : 0.0 ile 1.0 arası güven skoru (1.0 = çok güçlü sinyal)
        stop_loss   : Zararı kes fiyatı (None ise henüz hesaplanmadı)
        take_profit : Kar al fiyatı (None ise henüz hesaplanmadı)
        timestamp   : Sinyalin üretildiği zaman
        strategy    : Sinyali üreten strateji adı
        reason      : Sinyalin sebebi (log için okunabilir açıklama)
    """
    action: str                        # 'AL' | 'SAT' | 'BEKLE'
    confidence: float                  # 0.0 - 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy: str = "unknown"
    reason: str = ""

    def __post_init__(self):
        # Geçerli action değeri mi kontrol et
        valid_actions = {"AL", "SAT", "BEKLE"}
        if self.action not in valid_actions:
            raise ValueError(f"Geçersiz action: '{self.action}'. Olması gereken: {valid_actions}")

        # Confidence 0-1 arasında mı?
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence 0.0-1.0 arasında olmalı, verilen: {self.confidence}")

    def is_tradeable(self, min_confidence: float = 0.6) -> bool:
        """
        Sinyal işlem açmak için yeterince güçlü mü?
        min_confidence: Bu eşiğin altındaki sinyaller görmezden gelinir.
        """
        return self.action != "BEKLE" and self.confidence >= min_confidence

    def __repr__(self) -> str:
        return (
            f"Signal(action={self.action}, confidence={self.confidence:.2f}, "
            f"strategy={self.strategy}, reason='{self.reason}')"
        )


class BaseStrategy(ABC):
    """
    Tüm trading stratejilerinin türediği soyut temel sınıf.

    Bir strateji yazmak için:
        1. Bu sınıftan türet: class BenimStratejim(BaseStrategy)
        2. generate_signal(df) metodunu yaz
        3. Opsiyonel: validate_data() ve __str__() metodlarını override et
    """

    def __init__(self, name: str, symbol: str = "BTC/USDT", timeframe: str = "1h"):
        """
        Args:
            name      : Strateji adı (loglarda görünür)
            symbol    : İşlem çifti, örn: 'BTC/USDT', 'ETH/USDT'
            timeframe : Zaman dilimi: '1m', '5m', '15m', '1h', '4h', '1d'
        """
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.signal_count = 0          # Kaç sinyal üretildi (istatistik için)
        self.last_signal: Optional[Signal] = None

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        DataFrame'i alır, analiz eder, Signal döndürür.

        Args:
            df: OHLCV verisi içeren pandas DataFrame.
                Sütunlar: open, high, low, close, volume
                Index  : datetime

        Returns:
            Signal objesi (action, confidence, stop_loss, take_profit)

        NOT: Bu metod her alt sınıfta MUTLAKA implement edilmeli.
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        DataFrame'in gerekli sütunları içerip içermediğini kontrol eder.
        generate_signal() çağırmadan önce bu metodu çağırabilirsin.

        Returns:
            True  : Veri geçerli, devam edebilirsin
            False : Eksik sütun var, işlem yapma
        """
        required_columns = {"open", "high", "low", "close", "volume"}
        missing = required_columns - set(df.columns)
        if missing:
            print(f"[{self.name}] HATA: Eksik sütunlar: {missing}")
            return False
        if len(df) < 2:
            print(f"[{self.name}] HATA: En az 2 satır veri gerekli, gelen: {len(df)}")
            return False
        return True

    def run(self, df: pd.DataFrame) -> Signal:
        """
        Stratejiyi güvenli şekilde çalıştıran wrapper metod.
        validate_data() kontrolü yapar, sonra generate_signal() çağırır.
        Hata olursa BEKLE sinyali döndürür — sistem çökmez.
        """
        if not self.validate_data(df):
            return Signal(action="BEKLE", confidence=0.0, strategy=self.name,
                          reason="Veri doğrulama hatası")
        try:
            signal = self.generate_signal(df)
            signal.strategy = self.name
            self.last_signal = signal
            self.signal_count += 1
            return signal
        except Exception as e:
            print(f"[{self.name}] generate_signal() hata: {e}")
            return Signal(action="BEKLE", confidence=0.0, strategy=self.name,
                          reason=f"Strateji hatası: {str(e)}")

    def __str__(self) -> str:
        return f"{self.name} | {self.symbol} | {self.timeframe}"

    def __repr__(self) -> str:
        return f"BaseStrategy(name={self.name!r}, symbol={self.symbol!r}, timeframe={self.timeframe!r})"
