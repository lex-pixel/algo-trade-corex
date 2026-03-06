"""
data/fetcher.py
================
AMACI:
    Binance Testnet'ten gerçek OHLCV (fiyat/hacim) verisi çekmek.
    CCXT kütüphanesi üzerinden çalışır — borsa değişirse sadece bu dosya güncellenir.

NE YAPAR:
    1. .env dosyasından API key'leri okur (python-dotenv)
    2. Binance Testnet'e bağlanır (ccxt)
    3. İstenen sembol + timeframe için tarihsel mum verisi çeker
    4. Temiz pandas DataFrame döndürür
    5. İsteğe bağlı: Parquet dosyasına kaydeder (data/raw/)

CCXT NEDİR?
    100+ borsayı tek API ile konuşturan kütüphane.
    Binance, Bybit, OKX... hepsi aynı fonksiyonlarla çalışır.
    `exchange.fetch_ohlcv()` → [timestamp, open, high, low, close, volume]

ÇALIŞTIRMAK İÇİN:
    python -m data.fetcher
"""

import os
import time
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import pandas as pd
from dotenv import load_dotenv
from utils.logger import get_logger

# .env dosyasını yükle (proje kökündeki .env)
load_dotenv(Path(__file__).parent.parent / ".env")

logger = get_logger(__name__)

# Ham veri kayıt klasörü
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


class BinanceFetcher:
    """
    Binance (veya Binance Testnet) üzerinden OHLCV verisi çeken sınıf.

    Parametreler:
        testnet  : True → Testnet (sahte para), False → Gerçek borsa
        symbol   : İşlem çifti, örn. "BTC/USDT"
        timeframe: Mum boyutu, örn. "1h"
    """

    # CCXT'nin kabul ettiği timeframe değerleri
    VALID_TIMEFRAMES = {"1m", "5m", "15m", "1h", "4h", "1d"}

    def __init__(
        self,
        testnet: bool = True,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ):
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Geçersiz timeframe: '{timeframe}'. Olması gereken: {self.VALID_TIMEFRAMES}")

        self.symbol    = symbol
        self.timeframe = timeframe
        self.testnet   = testnet

        # Borsaya bağlan
        self.exchange = self._create_exchange()
        logger.info(
            f"BinanceFetcher hazir | {'TESTNET' if testnet else 'CANLI'} | "
            f"{symbol} {timeframe}"
        )

    # ── Borsa Bağlantısı ─────────────────────────────────────────────────────

    def _create_exchange(self) -> ccxt.binance:
        """
        CCXT ile Binance bağlantısı oluşturur.

        Testnet=True ise API istekleri testnet.binance.vision'a gider.
        API key olmadan da public endpoint'ler (fiyat, mum) çalışır.
        API key sadece emir göndermek için gerekli.
        """
        api_key    = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        exchange = ccxt.binance({
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {
                "defaultType": "spot",   # spot | future
            },
            "enableRateLimit": True,     # Binance rate limit'e uymak için otomatik bekleme
        })

        if self.testnet:
            # Testnet URL'ini override et
            exchange.set_sandbox_mode(True)

        return exchange

    # ── Veri Çekme ────────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        limit: int = 500,
        since_days: int | None = None,
    ) -> pd.DataFrame:
        """
        Binance'ten OHLCV verisi çeker ve DataFrame döndürür.

        Args:
            limit     : Kaç mum çekilsin (max 1000, Binance limiti)
            since_days: Son kaç günün verisi? (None ise 'limit' kadar son mum)

        Returns:
            pd.DataFrame: open, high, low, close, volume sütunlarıyla, DatetimeIndex

        Raises:
            ccxt.NetworkError  : Bağlantı hatası
            ccxt.ExchangeError : Borsa hatası
        """
        since_ms = None
        if since_days is not None:
            # Kaç ms öncesinden başlayacağız?
            since_ms = int((time.time() - since_days * 86400) * 1000)

        logger.info(
            f"Veri cekiliyor | {self.symbol} {self.timeframe} | "
            f"Limit: {limit} | Since: {since_days} gun"
        )

        try:
            raw = self.exchange.fetch_ohlcv(
                symbol    = self.symbol,
                timeframe = self.timeframe,
                since     = since_ms,
                limit     = limit,
            )
        except ccxt.NetworkError as e:
            logger.error(f"Ag hatasi: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Borsa hatasi: {e}")
            raise

        if not raw:
            logger.warning("Bos veri dondu — borsadan hic mum gelmedi")
            return pd.DataFrame()

        df = self._to_dataframe(raw)
        logger.info(f"Cekilen veri: {len(df)} mum | {df.index[0]} -> {df.index[-1]}")
        return df

    def fetch_since(self, since_days: int = 90, batch_size: int = 500) -> pd.DataFrame:
        """
        Belirli bir günden bu yana TÜM mumları çeker.
        Binance tek seferde max ~1000 mum döndürür; bu fonksiyon birden fazla
        istek atarak istenen tüm geçmişi toplar (pagination).

        Args:
            since_days: Kaç günlük geçmiş çekilsin
            batch_size: Her istekte kaç mum (max 1000)

        Returns:
            pd.DataFrame: Tüm geçmiş birleştirilmiş
        """
        since_ms = int((time.time() - since_days * 86400) * 1000)
        all_candles = []
        fetched = 0

        logger.info(f"{since_days} gunluk veri cekiliyor | {self.symbol} {self.timeframe}")

        while True:
            try:
                batch = self.exchange.fetch_ohlcv(
                    symbol    = self.symbol,
                    timeframe = self.timeframe,
                    since     = since_ms,
                    limit     = batch_size,
                )
            except ccxt.NetworkError as e:
                logger.error(f"Ag hatasi (batch): {e}")
                break

            if not batch:
                break

            all_candles.extend(batch)
            fetched += len(batch)

            # Son mumun timestamp'inden devam et
            since_ms = batch[-1][0] + 1

            logger.debug(f"Batch: {len(batch)} mum | Toplam: {fetched}")

            # Son batch'ten az veri geldiyse bitti demektir
            if len(batch) < batch_size:
                break

            # Rate limit için küçük bekleme
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_candles:
            logger.warning("Hic mum cekilemedi")
            return pd.DataFrame()

        df = self._to_dataframe(all_candles)

        # Duplikat timestamp varsa kaldır (iki batch arasında çakışma olabilir)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        logger.info(f"Toplam cekilen: {len(df)} mum | {df.index[0]} -> {df.index[-1]}")
        return df

    # ── Parquet Kaydetme ──────────────────────────────────────────────────────

    def save_parquet(self, df: pd.DataFrame, filename: str | None = None) -> Path:
        """
        DataFrame'i Parquet formatında data/raw/ klasörüne kaydeder.

        Parquet nedir?
            CSV'den 10x daha küçük, 100x daha hızlı okunur.
            Veri tipleri korunur (CSV'de her şey string olurdu).

        Args:
            df      : Kaydedilecek DataFrame
            filename: Dosya adı (None ise otomatik: BTC_USDT_1h_20240101.parquet)

        Returns:
            Path: Kaydedilen dosyanın yolu
        """
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        if filename is None:
            symbol_safe = self.symbol.replace("/", "_")
            date_str    = datetime.now(timezone.utc).strftime("%Y%m%d")
            filename    = f"{symbol_safe}_{self.timeframe}_{date_str}.parquet"

        path = RAW_DATA_DIR / filename
        df.to_parquet(path, engine="pyarrow")
        logger.info(f"Parquet kaydedildi: {path} ({len(df)} satir)")
        return path

    @staticmethod
    def load_parquet(path: str | Path) -> pd.DataFrame:
        """
        Parquet dosyasını okuyup DataFrame döndürür.

        Args:
            path: Dosya yolu

        Returns:
            pd.DataFrame
        """
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Parquet yuklendi: {path} ({len(df)} satir)")
        return df

    # ── Yardımcı ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(raw: list) -> pd.DataFrame:
        """
        CCXT'nin döndürdüğü ham listeyi pandas DataFrame'e çevirir.

        CCXT formatı: [[timestamp_ms, open, high, low, close, volume], ...]
        """
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Timestamp: milisaniye → UTC datetime → index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"

        # Float'a çevir (CCXT bazen string döndürebilir)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOGU
# python -m data.fetcher
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ALGO TRADE CODEX — BinanceFetcher Test")
    print("=" * 60)

    fetcher = BinanceFetcher(testnet=True, symbol="BTC/USDT", timeframe="1h")

    # Son 100 mumu çek
    print("\n[TEST 1] Son 100 mum cekiliyor...")
    df = fetcher.fetch_ohlcv(limit=100)

    if df.empty:
        print("HATA: Veri cekelemedi!")
    else:
        print(f"Basarili! {len(df)} mum cekild.")
        print(f"Tarih araligi: {df.index[0]} → {df.index[-1]}")
        print(f"\nIlk 3 mum:\n{df.head(3)}")
        print(f"\nSon 3 mum:\n{df.tail(3)}")
        print(f"\nVeri tipleri:\n{df.dtypes}")

        # Parquet kaydet
        print("\n[TEST 2] Parquet kaydi...")
        path = fetcher.save_parquet(df, filename="test_btc_usdt_1h.parquet")
        print(f"Kaydedildi: {path}")

        # Tekrar oku
        print("\n[TEST 3] Parquet okuma...")
        df2 = BinanceFetcher.load_parquet(path)
        print(f"Okundu: {len(df2)} satir — ilk satir: {df2.index[0]}")

    print("\n" + "=" * 60)
    print("Test tamamlandi!")
