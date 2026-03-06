"""
utils/logger.py
================
AMACI:
    Projenin her yerinden kullanılabilen merkezi log sistemi.
    Tüm modüller bu dosyadan logger'ı import eder.

NEDEN LOGURU?
    Python'un yerleşik logging modülü karmaşık ve verbose.
    Loguru tek satırda kurulum, renkli terminal çıktısı,
    otomatik dosya rotasyonu ve exception traceback desteği sağlar.

KULLANIM (herhangi bir dosyadan):
    from utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Sinyal üretildi: AL")
    logger.warning("API gecikmesi yüksek: 850ms")
    logger.error("Bağlantı kesildi")
    logger.debug("DataFrame şekli: (1000, 6)")   # DEBUG modunda görünür

LOG SEVİYELERİ (düşükten yükseğe):
    DEBUG    → Geliştirme sırasında detaylı bilgi
    INFO     → Normal sistem olayları (sinyal, emir, bağlantı)
    WARNING  → Dikkat gerektiren ama sistem çökmüyor
    ERROR    → Bir şeyler ters gitti ama devam edebilir
    CRITICAL → Sistem durabilir, acil müdahale gerekir
"""

import sys
from pathlib import Path
from loguru import logger as _logger


# ── Sabitler ──────────────────────────────────────────────────────────────────

# Log dosyalarının yazılacağı klasör
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)   # logs/ klasörü yoksa oluştur

# Log dosyası yolları
MAIN_LOG    = LOG_DIR / "algo_trade.log"       # Tüm loglar
TRADE_LOG   = LOG_DIR / "trades.log"           # Sadece işlem logları
ERROR_LOG   = LOG_DIR / "errors.log"           # Sadece hata logları


# ── Logger Kurulumu ───────────────────────────────────────────────────────────

def setup_logger(level: str = "INFO") -> None:
    """
    Logger'ı yapılandırır. Program başında bir kez çağrılır.

    Args:
        level: Log seviyesi — "DEBUG", "INFO", "WARNING", "ERROR"
               .env'den okumak için: os.getenv("LOG_LEVEL", "INFO")
    """
    # Varsayılan handler'ı temizle (loguru'nun kendi handler'ı)
    _logger.remove()

    # ── 1. TERMİNAL ÇIKTISI ─────────────────────────────────────────────────
    # Renkli, okunabilir format
    # {time}     → 2024-01-15 14:23:01
    # {level}    → INFO / WARNING / ERROR (renkli)
    # {name}     → hangi modülden geldi (örn: strategies.rsi_strategy)
    # {function} → hangi fonksiyondan
    # {message}  → log mesajı
    _logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # ── 2. ANA LOG DOSYASI ───────────────────────────────────────────────────
    # Tüm seviyeler buraya yazılır
    # rotation: Dosya 10MB olunca yeni dosya açılır (algo_trade.log.1, .2...)
    # retention: 30 günden eski loglar silinir
    # compression: Eski loglar zip'lenir (disk tasarrufu)
    _logger.add(
        MAIN_LOG,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    # ── 3. HATA LOG DOSYASI ──────────────────────────────────────────────────
    # Sadece ERROR ve CRITICAL seviyesi buraya yazılır
    # backtrace=True  → hata olunca tam stack trace göster
    # diagnose=True   → değişken değerlerini de göster (geliştirme için)
    _logger.add(
        ERROR_LOG,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="5 MB",
        retention="60 days",
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )

    # ── 4. İŞLEM LOG DOSYASI ────────────────────────────────────────────────
    # Sadece "TRADE" filtresiyle işaretlenen loglar buraya düşer
    # Kullanım: logger.bind(trade=True).info("AL | BTC/USDT | 68,500")
    _logger.add(
        TRADE_LOG,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        filter=lambda record: record["extra"].get("trade", False),
        rotation="10 MB",
        retention="90 days",   # İşlem logları daha uzun tutulur
        encoding="utf-8",
    )

    _logger.info(f"Logger kuruldu | Seviye: {level} | Log dizini: {LOG_DIR}")


def get_logger(name: str):
    """
    Modüle özel logger döndürür.

    Kullanım:
        from utils.logger import get_logger
        logger = get_logger(__name__)   # __name__ = "strategies.rsi_strategy"

    Args:
        name: Genellikle __name__ geçilir (modül adını otomatik alır)

    Returns:
        loguru logger instance (bind ile name eklenmiş)
    """
    return _logger.bind(module=name)


# ── İşlem Logu Kısayolu ───────────────────────────────────────────────────────

def log_trade(action: str, symbol: str, price: float, size: float,
              pnl: float = None, strategy: str = "unknown") -> None:
    """
    Bir işlemi trade log dosyasına yazar.
    Telegram bildirim sistemiyle de entegre edilecek (Phase 6).

    Args:
        action   : 'AL' veya 'SAT'
        symbol   : 'BTC/USDT'
        price    : İşlem fiyatı
        size     : İşlem miktarı (BTC cinsinden)
        pnl      : Gerçekleşen kâr/zarar (pozisyon kapanınca)
        strategy : Sinyali üreten strateji adı
    """
    pnl_str = f" | P&L: {pnl:+.2f} USDT" if pnl is not None else ""
    message = (
        f"{action} | {symbol} | Fiyat: {price:,.2f} | "
        f"Miktar: {size:.4f} | Strateji: {strategy}{pnl_str}"
    )
    _logger.bind(trade=True).info(message)


# ── Otomatik Kurulum ──────────────────────────────────────────────────────────
# Bu dosya import edildiğinde logger otomatik kurulur
# Seviyeyi .env'den okumak için python-dotenv gerekirken,
# henüz kurulmadığı için varsayılan INFO kullanıyoruz.
setup_logger(level="INFO")
