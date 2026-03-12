"""
config/loader.py
=================
AMACI:
    settings.yaml dosyasını okur, pydantic ile doğrular ve
    projenin her yerine tip-güvenli (type-safe) ayarlar sağlar.

NEDEN PYDANTIC?
    YAML'ı düz okursak string/int karışıklığı, eksik alan,
    yanlış değer gibi hatalar runtime'da patlar.
    Pydantic, dosyayı okur okumaz her alanı kontrol eder:
        - rsi_period negatifse → ValueError
        - oversold > overbought ise → ValueError
        - enabled alanı eksikse → varsayılan değer kullan

KULLANIM (herhangi bir dosyadan):
    from config.loader import get_config

    cfg = get_config()
    print(cfg.general.symbol)          # "BTC/USDT"
    print(cfg.strategies.rsi.oversold) # 30
    print(cfg.risk.max_daily_loss)     # 0.05
"""

import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from utils.logger import get_logger

logger = get_logger(__name__)

# settings.yaml'ın konumu
CONFIG_PATH = Path(__file__).parent / "settings.yaml"
# Optuna optimizer ciktisi (varsa settings.yaml uzerine uygulanir)
OPTIMIZED_PARAMS_PATH = Path(__file__).parent / "optimized_params.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC ŞEMA TANIMLARI
# Her sınıf, YAML'ın bir bölümünü temsil eder.
# Field(...) = zorunlu alan | Field(default=X) = varsayılan değerli alan
# ─────────────────────────────────────────────────────────────────────────────

class GeneralConfig(BaseModel):
    symbol:    str  = Field(default="BTC/USDT")
    timeframe: str  = Field(default="1h")
    exchange:  str  = Field(default="binance")
    testnet:   bool = Field(default=True)
    dry_run:   bool = Field(default=True)

    @field_validator("timeframe")
    @classmethod
    def valid_timeframe(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if v not in allowed:
            raise ValueError(f"Geçersiz timeframe: '{v}'. Olması gereken: {allowed}")
        return v

    @field_validator("exchange")
    @classmethod
    def valid_exchange(cls, v: str) -> str:
        allowed = {"binance", "bybit"}
        if v.lower() not in allowed:
            raise ValueError(f"Geçersiz exchange: '{v}'. Desteklenen: {allowed}")
        return v.lower()


class RSIConfig(BaseModel):
    enabled:        bool  = True
    rsi_period:     int   = Field(default=14, ge=2, le=100)   # ge=min, le=max
    oversold:       float = Field(default=30.0, ge=0, le=100)
    overbought:     float = Field(default=70.0, ge=0, le=100)
    stop_pct:       float = Field(default=0.015, gt=0, le=0.1)
    tp_pct:         float = Field(default=0.030, gt=0, le=0.5)
    min_confidence: float = Field(default=0.05, ge=0, le=1.0)

    @model_validator(mode="after")
    def oversold_below_overbought(self) -> "RSIConfig":
        if self.oversold >= self.overbought:
            raise ValueError(
                f"oversold ({self.oversold}) < overbought ({self.overbought}) olmalı"
            )
        return self


class PARangeConfig(BaseModel):
    enabled:              bool  = True
    lookback:             int   = Field(default=50, ge=10, le=500)
    rsi_period:           int   = Field(default=14, ge=2, le=100)
    rsi_oversold:         float = Field(default=40.0, ge=0, le=100)
    rsi_overbought:       float = Field(default=60.0, ge=0, le=100)
    proximity_pct:        float = Field(default=0.02, gt=0, le=0.2)
    stop_pct:             float = Field(default=0.015, gt=0, le=0.1)
    tp_pct:               float = Field(default=0.030, gt=0, le=0.5)
    min_confidence:       float = Field(default=0.05, ge=0, le=1.0)
    use_regime_filter:    bool  = True
    volume_confirm_mult:  float = Field(default=1.5, ge=1.0, le=5.0)  # hacim onaylama katsayisi
    fakeout_filter:       bool  = True    # kapanisa gore kirilim dogrulama
    rsi_divergence:       bool  = True    # RSI uyumsuzlugu tespiti


class StrategiesConfig(BaseModel):
    rsi:      RSIConfig      = Field(default_factory=RSIConfig)
    pa_range: PARangeConfig  = Field(default_factory=PARangeConfig)


class RiskConfig(BaseModel):
    max_position_risk:  float = Field(default=0.02, gt=0, le=0.5)
    max_daily_loss:     float = Field(default=0.05, gt=0, le=1.0)
    max_drawdown:       float = Field(default=0.15, gt=0, le=1.0)
    max_open_positions: int   = Field(default=2, ge=1, le=20)   # 2: LONG+SHORT ayni anda


class TrailingStopConfig(BaseModel):
    enabled:           bool  = True
    breakeven_pct:     float = Field(default=0.015, gt=0, le=0.5)  # %1.5 karda breakeven
    partial_close_pct: float = Field(default=0.030, gt=0, le=0.5)  # %3 karda yarisini kapat
    trail_sl_pct:      float = Field(default=0.015, gt=0, le=0.5)  # kalan yarim SL mesafesi


class MTFConfig(BaseModel):
    penalty_pct:           float = Field(default=0.30, ge=0.0, le=1.0)
    entry_15m_overbought:  float = Field(default=78.0, ge=50, le=100)
    entry_15m_oversold:    float = Field(default=22.0, ge=0, le=50)
    retrain_every:         int   = Field(default=720, ge=1)


class DataConfig(BaseModel):
    history_days: int = Field(default=90, ge=1, le=3650)
    cache_size:   int = Field(default=500, ge=10, le=10000)
    parquet_dir:  str = Field(default="data/raw")


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")

    @field_validator("level")
    @classmethod
    def valid_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Geçersiz log seviyesi: '{v}'. Olması gereken: {allowed}")
        return v.upper()


class AppConfig(BaseModel):
    """Tüm konfigürasyonu bir arada tutan ana sınıf."""
    general:       GeneralConfig       = Field(default_factory=GeneralConfig)
    strategies:    StrategiesConfig    = Field(default_factory=StrategiesConfig)
    risk:          RiskConfig          = Field(default_factory=RiskConfig)
    trailing_stop: TrailingStopConfig  = Field(default_factory=TrailingStopConfig)
    mtf:           MTFConfig           = Field(default_factory=MTFConfig)
    data:          DataConfig          = Field(default_factory=DataConfig)
    logging:       LoggingConfig       = Field(default_factory=LoggingConfig)


# ─────────────────────────────────────────────────────────────────────────────
# YÜKLEME FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────

_config_cache: AppConfig | None = None   # Bir kez yükle, hep aynı objeyi döndür


def _apply_optimized_params(raw: dict) -> dict:
    """
    config/optimized_params.yaml varsa, icindeki degerleri
    settings.yaml uzerine yazar (override).

    Ornek optimized_params.yaml:
        rsi:
          oversold: 29
          overbought: 60
          rsi_period: 23

    Bu fonksiyon raw["strategies"]["rsi"] icine bu degerleri ekler.
    Sadece var olan anahtarlar guncellenir, yeni anahtar eklenmez.
    """
    if not OPTIMIZED_PARAMS_PATH.exists():
        return raw

    with open(OPTIMIZED_PARAMS_PATH, encoding="utf-8") as f:
        opt = yaml.safe_load(f) or {}

    if not isinstance(opt, dict):
        return raw

    # strategies altindaki her section icin merge et
    strategies_raw = raw.get("strategies", {})
    for section, params in opt.items():
        if not isinstance(params, dict):
            continue
        if section in strategies_raw:
            strategies_raw[section].update(params)
            logger.info(
                f"Optimize parametreler uygulandi: strategies.{section} | "
                f"Anahtarlar: {list(params.keys())}"
            )

    raw["strategies"] = strategies_raw
    return raw


def get_config(path: Path = CONFIG_PATH) -> AppConfig:
    """
    settings.yaml'ı okur, doğrular ve AppConfig objesi döndürür.
    İkinci çağrıda dosyayı tekrar okumaz (önbellekten döner).

    Args:
        path: YAML dosyasının yolu (varsayılan: config/settings.yaml)

    Returns:
        AppConfig: Doğrulanmış konfigürasyon objesi

    Raises:
        FileNotFoundError : YAML dosyası bulunamazsa
        ValueError        : Geçersiz değer varsa (pydantic hatası)
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not path.exists():
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Optuna optimize parametrelerini uygula (varsa)
    raw = _apply_optimized_params(raw)

    try:
        _config_cache = AppConfig(**raw)
        logger.info(
            f"Konfigurasyon yuklendi | "
            f"Sembol: {_config_cache.general.symbol} | "
            f"Timeframe: {_config_cache.general.timeframe} | "
            f"Testnet: {_config_cache.general.testnet}"
        )
        return _config_cache
    except Exception as e:
        logger.error(f"Konfigurasyon hatasi: {e}")
        raise


def reload_config(path: Path = CONFIG_PATH) -> AppConfig:
    """
    Önbelleği temizler ve config'i yeniden yükler.
    settings.yaml'ı değiştirdikten sonra uygulamayı yeniden başlatmadan
    değişikliği almak için kullanılır.
    """
    global _config_cache
    _config_cache = None
    return get_config(path)
