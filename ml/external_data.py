"""
ml/external_data.py
====================
AMACI:
    Piyasa ile ilgili dis kaynaklardan veri ceker:
    - Fear & Greed Index (alternative.me, API anahtari gerektirmez)
    - BTC Dominance (CoinGecko, API anahtari gerektirmez)

    Hata durumunda None doner, bot calismasini engellemez.
    Bu veriler feature_engineering.py'de opsiyonel olarak kullanilir.

KULLANIM:
    from ml.external_data import ExternalDataFetcher

    fetcher = ExternalDataFetcher()
    fg = fetcher.get_fear_greed()       # 0-100, 50 = neutral
    dom = fetcher.get_btc_dominance()   # 0-100, orn. 52.3
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

    # ── Tum Dis Verileri ──────────────────────────────────────────────────────

    def fetch_all(self) -> dict:
        """
        Tum dis verileri cekip dict olarak dondurur.
        Hata olan degerler None olur — bot devam eder.

        Returns:
            {
                "fear_greed"    : float | None,  # 0-100
                "btc_dominance" : float | None,  # 0-100
            }
        """
        return {
            "fear_greed"    : self.get_fear_greed(),
            "btc_dominance" : self.get_btc_dominance(),
        }


# ── Modul seviyesinde tek ornek (singleton) ───────────────────────────────────
_instance: Optional[ExternalDataFetcher] = None


def get_external_fetcher() -> ExternalDataFetcher:
    """Singleton external data fetcher dondurur."""
    global _instance
    if _instance is None:
        _instance = ExternalDataFetcher()
    return _instance
