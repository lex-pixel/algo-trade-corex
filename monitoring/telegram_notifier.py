"""
monitoring/telegram_notifier.py
=================================
AMACI:
    Telegram Bot API uzerinden islem bildirimleri gonderir.
    Bot olmadan da calisir (dry_run=True veya token/chat_id bos ise sessizce gec).

KURULUM:
    1. @BotFather'dan bot olustur -> TOKEN al
    2. Bot ile bir mesaj at -> CHAT_ID al
    3. .env dosyasina ekle:
       TELEGRAM_TOKEN=1234567890:ABC...
       TELEGRAM_CHAT_ID=-100123456789

BILDIRIM TURLERI:
    - AL/SAT sinyali
    - Pozisyon acildi / kapatildi
    - Stop-loss veya take-profit tetiklendi
    - Bot baslatildi / durduruldu
    - Hata bildirimi

KULLANIM:
    from monitoring.telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier()
    notifier.send_signal("AL", "BTC/USDT", price=95000, confidence=0.75)
    notifier.send_position_opened(direction="LONG", price=95000, qty=0.001, sl=93000, tp=98000)
    notifier.send_position_closed(direction="LONG", entry=95000, exit=97000, pnl=+20.5)
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from typing import Optional

try:
    import urllib.request
    import urllib.error
    _URL_LIB_AVAILABLE = True
except ImportError:
    _URL_LIB_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)

# Telegram API base URL
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """
    Telegram uzerinden trading bildirimleri gonderir.

    Parametreler:
        token    : Telegram Bot Token (@BotFather'dan)
        chat_id  : Mesajin gonderilecegi chat/kanal ID'si
        dry_run  : True = gercek mesaj gonderme, sadece logla
        symbol   : Islem cifti (mesajlarda gozukur)

    NOT: Token veya chat_id bos ise dry_run=True gibi davranir.
    """

    def __init__(
        self,
        token: str | None   = None,
        chat_id: str | None = None,
        dry_run: bool       = False,
        symbol: str         = "BTC/USDT",
    ):
        # .env'den oku (veya constructor parametresi)
        self.token   = token   or os.getenv("TELEGRAM_TOKEN",   "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.symbol  = symbol

        # Token yoksa dry_run moduna gec
        if not self.token or not self.chat_id:
            dry_run = True
            logger.info(
                "Telegram token/chat_id bulunamadi. "
                "Dry-run modunda calisiliyor (mesajlar gonderilmeyecek)."
            )

        self.dry_run    = dry_run
        self._sent      = 0    # Gonderilen mesaj sayisi
        self._failed    = 0    # Basarisiz mesaj sayisi

        mode = "DRY-RUN" if dry_run else "AKTIF"
        logger.info(f"TelegramNotifier baslatildi | Mod: {mode}")

    # ── Bildirim Metodlari ────────────────────────────────────────────────────

    def send_signal(
        self,
        action: str,
        symbol: str | None = None,
        price: float       = 0.0,
        confidence: float  = 0.0,
        rsi_signal: str    = "",
        pa_signal: str     = "",
    ) -> bool:
        """Yeni sinyal bildirimi gonderir."""
        symbol = symbol or self.symbol
        emoji  = {"AL": "YUKSELIS", "SAT": "DUSUS", "BEKLE": "BEKLE"}.get(action, action)
        now    = datetime.now(timezone.utc).strftime("%H:%M UTC")

        text = (
            f"[{emoji}] *{action}* SINYALI\n"
            f"Symbol: `{symbol}`\n"
            f"Fiyat: `${price:,.2f}`\n"
            f"Guven: `{confidence:.0%}`\n"
            f"RSI: `{rsi_signal}` | PA: `{pa_signal}`\n"
            f"Zaman: {now}"
        )
        return self._send(text)

    def send_position_opened(
        self,
        direction: str,
        price: float,
        quantity: float,
        stop_loss: float | None   = None,
        take_profit: float | None = None,
        strategy: str             = "",
        symbol: str | None        = None,
    ) -> bool:
        """Pozisyon acildi bildirimi."""
        symbol = symbol or self.symbol
        emoji  = "YUKSELIS" if direction == "LONG" else "DUSUS"
        sl_str = f"${stop_loss:,.2f}" if stop_loss else "yok"
        tp_str = f"${take_profit:,.2f}" if take_profit else "yok"
        now    = datetime.now(timezone.utc).strftime("%H:%M UTC")

        text = (
            f"[POZISYON ACILDI] [{emoji}] *{direction}*\n"
            f"Symbol: `{symbol}`\n"
            f"Giris: `${price:,.2f}`\n"
            f"Miktar: `{quantity:.6f} BTC`\n"
            f"Notional: `${price * quantity:,.2f} USDT`\n"
            f"Stop-Loss: `{sl_str}`\n"
            f"Take-Profit: `{tp_str}`\n"
            f"Strateji: {strategy}\n"
            f"Zaman: {now}"
        )
        return self._send(text)

    def send_position_closed(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        realized_pnl: float,
        realized_pct: float,
        exit_reason: str    = "SIGNAL",
        symbol: str | None  = None,
    ) -> bool:
        """Pozisyon kapatildi bildirimi."""
        symbol   = symbol or self.symbol
        pnl_sign = "+" if realized_pnl >= 0 else ""
        result   = "KAR" if realized_pnl >= 0 else "ZARAR"
        now      = datetime.now(timezone.utc).strftime("%H:%M UTC")

        text = (
            f"[POZISYON KAPANDI] [{result}]\n"
            f"Symbol: `{symbol}` | {direction}\n"
            f"Giris: `${entry_price:,.2f}` -> Cikis: `${exit_price:,.2f}`\n"
            f"Miktar: `{quantity:.6f} BTC`\n"
            f"PnL: `{pnl_sign}{realized_pnl:.2f} USDT "
            f"({pnl_sign}{realized_pct:.2f}%)`\n"
            f"Sebep: {exit_reason}\n"
            f"Zaman: {now}"
        )
        return self._send(text)

    def send_stop_loss(
        self,
        symbol: str | None = None,
        price: float       = 0.0,
        sl_price: float    = 0.0,
        pnl: float         = 0.0,
    ) -> bool:
        """Stop-loss tetiklendi bildirimi."""
        symbol = symbol or self.symbol
        text = (
            f"[UYARI] STOP-LOSS TETIKLENDI\n"
            f"Symbol: `{symbol}`\n"
            f"Fiyat: `${price:,.2f}` (SL: `${sl_price:,.2f}`)\n"
            f"Zarar: `{pnl:.2f} USDT`\n"
            f"Zaman: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send(text)

    def send_take_profit(
        self,
        symbol: str | None = None,
        price: float       = 0.0,
        tp_price: float    = 0.0,
        pnl: float         = 0.0,
    ) -> bool:
        """Take-profit bildirimi."""
        symbol = symbol or self.symbol
        text = (
            f"[BASARI] TAKE-PROFIT!\n"
            f"Symbol: `{symbol}`\n"
            f"Fiyat: `${price:,.2f}` (TP: `${tp_price:,.2f}`)\n"
            f"Kar: `+{pnl:.2f} USDT`\n"
            f"Zaman: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send(text)

    def send_bot_started(self, capital: float, paper: bool = True) -> bool:
        """Bot baslatildi bildirimi."""
        mode = "PAPER TRADING" if paper else "LIVE TRADING"
        text = (
            f"[BOT BASLATILDI]\n"
            f"Mod: *{mode}*\n"
            f"Sermaye: `${capital:,.2f} USDT`\n"
            f"Zaman: {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')}"
        )
        return self._send(text)

    def send_bot_stopped(self, total_pnl: float, win_rate: float) -> bool:
        """Bot durduruldu bildirimi."""
        sign = "+" if total_pnl >= 0 else ""
        text = (
            f"[BOT DURDURULDU]\n"
            f"Toplam PnL: `{sign}{total_pnl:.2f} USDT`\n"
            f"Win Rate: `{win_rate:.1f}%`\n"
            f"Zaman: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send(text)

    def send_error(self, error_msg: str, context: str = "") -> bool:
        """Hata bildirimi."""
        text = (
            f"[HATA]\n"
            f"Mesaj: `{error_msg[:200]}`\n"  # Max 200 karakter
            f"Baglam: {context}\n"
            f"Zaman: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send(text)

    def send_daily_summary(
        self,
        trades: int,
        pnl: float,
        win_rate: float,
        capital: float,
    ) -> bool:
        """Gunluk ozet bildirimi."""
        sign = "+" if pnl >= 0 else ""
        text = (
            f"[GUNLUK OZET]\n"
            f"Islem: `{trades}`\n"
            f"PnL: `{sign}{pnl:.2f} USDT`\n"
            f"Win Rate: `{win_rate:.1f}%`\n"
            f"Sermaye: `${capital:,.2f}`\n"
            f"Tarih: {datetime.now(timezone.utc).strftime('%d.%m.%Y')}"
        )
        return self._send(text)

    def send_raw(self, text: str) -> bool:
        """Ham metin gonderir (test icin)."""
        return self._send(text)

    # ── Gonderim Altyapisi ────────────────────────────────────────────────────

    def _send(self, text: str) -> bool:
        """
        Telegram API'ye mesaj gonderir.
        dry_run=True ise sadece loglar.

        Returns:
            True = basarili, False = basarisiz
        """
        if self.dry_run:
            logger.info(f"[TELEGRAM DRY-RUN] {text[:80]}...")
            self._sent += 1
            return True

        if not _URL_LIB_AVAILABLE:
            logger.warning("urllib.request mevcut degil, mesaj gonderilemedi.")
            return False

        url     = TELEGRAM_API.format(token=self.token)
        payload = json.dumps({
            "chat_id"    : self.chat_id,
            "text"       : text,
            "parse_mode" : "Markdown",
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    self._sent += 1
                    logger.debug("Telegram mesaji gonderildi.")
                    return True
                else:
                    self._failed += 1
                    logger.warning(f"Telegram API hata kodu: {resp.status}")
                    return False
        except urllib.error.URLError as e:
            self._failed += 1
            logger.warning(f"Telegram baglanti hatasi: {e}")
            return False
        except Exception as e:
            self._failed += 1
            logger.warning(f"Telegram mesaj hatasi: {e}")
            return False

    # ── Istatistikler ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Gonderim istatistikleri."""
        return {
            "sent"   : self._sent,
            "failed" : self._failed,
            "dry_run": self.dry_run,
        }
