"""
ml/external_data.py
====================
AMACI:
    Piyasa ile ilgili dis kaynaklardan veri ceker:
    - Fear & Greed Index (alternative.me, API anahtari gerektirmez)
    - BTC Dominance (CoinGecko, API anahtari gerektirmez)
    - Funding Rate (Binance Futures, API anahtari gerektirmez)
    - Open Interest (Binance Futures, API anahtari gerektirmez)

    Hata durumunda None doner, bot calismasini engellemez.
    Bu veriler feature_engineering.py'de opsiyonel olarak kullanilir.

KULLANIM:
    from ml.external_data import ExternalDataFetcher

    fetcher = ExternalDataFetcher()
    fg  = fetcher.get_fear_greed()       # 0-100, 50 = neutral
    dom = fetcher.get_btc_dominance()    # 0-100, orn. 52.3
    fr  = fetcher.get_funding_rate()     # float, orn. 0.0001 = %0.01
    oi  = fetcher.get_open_interest()    # float (USD), orn. 8.5e9
"""

from __future__ import annotations
import time
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Onbellekleme suresi (saniye) — API'yi gerekcesiz sarsmamak icin
_CACHE_TTL = 3600  # 1 saat


class ExternalDataFetcher:
    """
    Dis kaynaklardan piyasa verisini ceker.

    Tum metodlar hata durumunda None doner — bot calismasini engellemez.
    Sonuclar onbelleklenir (varsayilan: 1 saat), gereksiz API cagrisi olmaz.
    """

    def __init__(self, cache_ttl: int = _CACHE_TTL, timeout: int = 5):
        self.timeout   = timeout    # HTTP istek zaman asimi (sn)
        self.cache_ttl = cache_ttl

        # Onbellek: {key: (deger, timestamp)}
        self._cache: dict[str, tuple] = {}

    # ── Onbellek Yardimcisi ───────────────────────────────────────────────────

    def _cached(self, key: str):
        """Onbellekte taze veri var mi? Varsa dondur, yoksa None."""
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return value
        return None

    def _set_cache(self, key: str, value) -> None:
        self._cache[key] = (value, time.time())

    # ── Fear & Greed Index ────────────────────────────────────────────────────

    def get_fear_greed(self) -> Optional[float]:
        """
        Fear & Greed Index degerini ceker (alternative.me).

        Deger aralik: 0-100
            0-24  = Extreme Fear (asiri korku)
            25-49 = Fear (korku)
            50    = Neutral (notral)
            51-74 = Greed (acgozluluk)
            75-100 = Extreme Greed (asiri acgozluluk)

        Returns:
            float (0-100) veya None (API hatasi)
        """
        key = "fear_greed"
        cached = self._cached(key)
        if cached is not None:
            return cached

        try:
            import urllib.request
            import json
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            value = float(data["data"][0]["value"])
            self._set_cache(key, value)
            logger.debug(f"Fear & Greed: {value:.0f}")
            return value
        except Exception as e:
            logger.debug(f"Fear & Greed alinamadi: {e}")
            return None

    # ── BTC Dominance ─────────────────────────────────────────────────────────

    def get_btc_dominance(self) -> Optional[float]:
        """
        BTC piyasa hakimiyetini ceker (CoinGecko /global endpoint).

        Deger aralik: 0-100 (yuzde)
            50+ = BTC hakim, altcoinler zayif
            40-  = altseason, altcoinler guclu

        Returns:
            float (0-100) veya None (API hatasi)
        """
        key = "btc_dominance"
        cached = self._cached(key)
        if cached is not None:
            return cached

        try:
            import urllib.request
            import json
            url = "https://api.coingecko.com/api/v3/global"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AlgoTradeCodex/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            dominance = float(
                data["data"]["market_cap_percentage"].get("btc", 50.0)
            )
            self._set_cache(key, dominance)
            logger.debug(f"BTC Dominance: {dominance:.1f}%")
            return dominance
        except Exception as e:
            logger.debug(f"BTC Dominance alinamadi: {e}")
            return None

    # ── Funding Rate (Binance Futures) ───────────────────────────────────────

    def get_funding_rate(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """
        Binance Futures funding rate degerini ceker.

        Her 8 saatte bir guncellenir.
        Deger aralik: genellikle -0.003 ile +0.003 arasi
            > +0.001 = asiri long, dusus riski
            < -0.001 = asiri short, yukselis riski
            ~ 0      = dengeli piyasa

        API: https://fapi.binance.com/fapi/v1/fundingRate
        API anahtari gerektirmez.

        Returns:
            float (orn. 0.0001) veya None (API hatasi)
        """
        key = f"funding_rate_{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached

        try:
            import urllib.request
            import json
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            req = urllib.request.Request(
                url, headers={"User-Agent": "AlgoTradeCodex/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            if data and len(data) > 0:
                value = float(data[0]["fundingRate"])
                self._set_cache(key, value)
                logger.debug(f"Funding Rate ({symbol}): {value:.6f}")
                return value
        except Exception as e:
            logger.debug(f"Funding Rate alinamadi: {e}")
        return None

    # ── Open Interest (Binance Futures) ──────────────────────────────────────

    def get_open_interest(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """
        Binance Futures open interest (toplam acik pozisyon) degerini ceker.

        Deger: USD cinsinden toplam pozisyon buyuklugu
            Fiyat yukseliyor + OI yukseliyor = guclu trend (para giriyor)
            Fiyat yukseliyor + OI dusuyor   = zayif hareket (kapanmalar var)
            Fiyat dusuyor   + OI yukseliyor = guclu bearish basinci
            Fiyat dusuyor   + OI dusuyor    = short kapanmalar (dip yakin olabilir)

        API: https://fapi.binance.com/fapi/v1/openInterest
        API anahtari gerektirmez.

        Returns:
            float (USD) veya None (API hatasi)
        """
        key = f"open_interest_{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached

        try:
            import urllib.request
            import json
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            req = urllib.request.Request(
                url, headers={"User-Agent": "AlgoTradeCodex/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            # openInterest = kontrat sayisi, * fiyat = USD degeri
            oi_contracts = float(data["openInterest"])
            # Fiyati da cekip USD'ye cevirelim
            price_url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
            price_req = urllib.request.Request(
                price_url, headers={"User-Agent": "AlgoTradeCodex/1.0"}
            )
            with urllib.request.urlopen(price_req, timeout=self.timeout) as resp2:
                price_data = json.loads(resp2.read().decode())
            price = float(price_data["price"])
            oi_usd = oi_contracts * price
            self._set_cache(key, oi_usd)
            logger.debug(f"Open Interest ({symbol}): ${oi_usd/1e9:.2f}B")
            return oi_usd
        except Exception as e:
            logger.debug(f"Open Interest alinamadi: {e}")
        return None

    # ── Tum Dis Verileri ──────────────────────────────────────────────────────

    def fetch_all(self) -> dict:
        """
        Tum dis verileri cekip dict olarak dondurur.
        Hata olan degerler None olur — bot devam eder.

        Returns:
            {
                "fear_greed"    : float | None,  # 0-100
                "btc_dominance" : float | None,  # 0-100
                "funding_rate"  : float | None,  # orn. 0.0001
                "open_interest" : float | None,  # USD
            }
        """
        return {
            "fear_greed"    : self.get_fear_greed(),
            "btc_dominance" : self.get_btc_dominance(),
            "funding_rate"  : self.get_funding_rate(),
            "open_interest" : self.get_open_interest(),
        }


# ── Modul seviyesinde tek ornek (singleton) ───────────────────────────────────
_instance: Optional[ExternalDataFetcher] = None


def get_external_fetcher() -> ExternalDataFetcher:
    """Singleton external data fetcher dondurur."""
    global _instance
    if _instance is None:
        _instance = ExternalDataFetcher()
    return _instance
