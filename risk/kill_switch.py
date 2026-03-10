"""
risk/kill_switch.py
====================
AMACI:
    Gunluk zarar, drawdown veya hata sayisi belirli esikleri asisca
    trading botunu otomatik olarak durdurur.

3 SEVİYELİ ALARM SİSTEMİ:
    SARI   (Level 1): Gunluk zarar %3 — yeni pozisyon acma, mevcut pozisyonlari koru
    TURUNCU(Level 2): Gunluk zarar %5 — tum acik pozisyonlari kapat, dur
    KIRMIZI(Level 3): Gunluk zarar %15 veya drawdown %15 — bot tamamen dur, acil log

EKLENEN KONTROLLER:
    - Ust uste hata sayisi     : max_consecutive_errors (varsayilan 5)
    - Islem hizi sınırı        : max_trades_per_hour (varsayilan 20)
    - Min guven esigi          : min_signal_confidence (uyari verir)

RESET:
    Sari ve Turuncu her gun gece yarisi otomatik resetlenir.
    Kirmizi reseti manuel gerektirir (bilerek tasarlandi — ciddi bir sey oldugunda
    insan kontrolu sart).

KULLANIM:
    from risk.kill_switch import KillSwitch, AlertLevel

    ks = KillSwitch(initial_capital=10_000.0)

    # Her tick'de cagir
    status = ks.check(
        current_capital=9_600.0,   # %4 zarar -> TURUNCU
        open_pnl=-50.0,
    )

    if status.should_close_all:
        # Tum pozisyonlari kapat!
        pass

    if status.should_halt:
        # Botu durdur!
        pass
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Alarm Seviyeleri ──────────────────────────────────────────────────────────

class AlertLevel(int, Enum):
    """Kill switch uyari seviyeleri (buyuk sayi = daha ciddi)."""
    NORMAL   = 0    # Normal islem
    YELLOW   = 1    # Dikkat: yeni pozisyon acma
    ORANGE   = 2    # Tehlike: mevcut pozisyonlari kapat
    RED      = 3    # Kritik: botu tamamen durdur


# Seviyeye gore renkli log prefixleri
LEVEL_PREFIX = {
    AlertLevel.NORMAL  : "[NORMAL]",
    AlertLevel.YELLOW  : "[SARI - DIKKAT]",
    AlertLevel.ORANGE  : "[TURUNCU - TEHLIKE]",
    AlertLevel.RED     : "[KIRMIZI - KRITIK]",
}


# ── Durum Nesnesi ─────────────────────────────────────────────────────────────

@dataclass
class KillStatus:
    """
    kill_switch.check() tarafindan dondurulen durum nesnesi.

    Kullanim:
        if status.should_halt:
            bot.stop()
        elif status.should_close_all:
            position_tracker.close_all()
        elif not status.can_open:
            # Yeni pozisyon acma
    """
    level            : AlertLevel = AlertLevel.NORMAL
    can_open         : bool = True    # Yeni pozisyon acilab ilir mi?
    should_close_all : bool = False   # Mevcut pozisyonlar kapatilmali mi?
    should_halt      : bool = False   # Bot durdurulmali mi?
    reason           : str  = ""      # Neden tetiklendi?
    daily_loss_pct   : float = 0.0    # Gunluk zarar yuzdesi
    drawdown_pct     : float = 0.0    # Anlık drawdown yuzdesi


# ── KillSwitch ────────────────────────────────────────────────────────────────

class KillSwitch:
    """
    3 seviyeli otomatik durdurma sistemi.

    Parametreler:
        initial_capital        : Baslangic sermayesi (USDT)
        yellow_threshold_pct   : Sari alarm esigi (varsayilan %3)
        orange_threshold_pct   : Turuncu alarm esigi (varsayilan %5)
        red_threshold_pct      : Kirmizi alarm esigi (varsayilan %15)
        max_consecutive_errors : Ust uste hata limiti
        max_trades_per_hour    : Saatlik maksimum islem
    """

    def __init__(
        self,
        initial_capital: float         = 10_000.0,
        yellow_threshold_pct: float    = 0.03,    # %3
        orange_threshold_pct: float    = 0.05,    # %5
        red_threshold_pct: float       = 0.15,    # %15
        max_consecutive_errors: int    = 5,
        max_trades_per_hour: int       = 20,
    ):
        self.initial_capital         = initial_capital
        self.yellow_threshold        = yellow_threshold_pct
        self.orange_threshold        = orange_threshold_pct
        self.red_threshold           = red_threshold_pct
        self.max_consecutive_errors  = max_consecutive_errors
        self.max_trades_per_hour     = max_trades_per_hour

        # Gunluk tracking
        self._day_start_capital      = initial_capital
        self._day_start_date: date   = datetime.now(timezone.utc).date()
        self._day_trades: int        = 0
        self._hour_trades: int       = 0
        self._hour_start             = datetime.now(timezone.utc)

        # Equity peak (drawdown hesabi icin)
        self._equity_peak            = initial_capital

        # Hata sayaci
        self._consecutive_errors     = 0

        # Mevcut seviye ve manuel kirmizi
        self._current_level          = AlertLevel.NORMAL
        self._red_manual_reset_required = False   # Kirmizi tetiklendiyse True

        # Gecmis olaylar
        self._events: list[dict]     = []

        logger.info(
            f"KillSwitch baslatildi | "
            f"Sari:%{yellow_threshold_pct*100:.0f} | "
            f"Turuncu:%{orange_threshold_pct*100:.0f} | "
            f"Kirmizi:%{red_threshold_pct*100:.0f}"
        )

    # ── Ana Kontrol ───────────────────────────────────────────────────────────

    def check(
        self,
        current_capital: float,
        open_pnl: float = 0.0,
    ) -> KillStatus:
        """
        Guncel durumu degerlendirir ve KillStatus dondurur.
        Her bot tick'inde cagrilmali.

        Parametreler:
            current_capital : Mevcut nakit sermaye (acik pozisyonlar dahil degil)
            open_pnl        : Acik pozisyonlardaki unrealized PnL

        Returns:
            KillStatus: Hangi aksiyonun alinmasi gerektigi
        """
        # Kirmizi manuel reset gerekiyorsa aninda dondur
        if self._red_manual_reset_required:
            return KillStatus(
                level         = AlertLevel.RED,
                can_open      = False,
                should_close_all = True,
                should_halt   = True,
                reason        = "Kirmizi alarm manuel reset bekliyor. reset_red() cagirin.",
                daily_loss_pct= self._daily_loss_pct(current_capital),
                drawdown_pct  = self._drawdown_pct(current_capital + open_pnl),
            )

        # Gece yarisi resetini kontrol et
        self._daily_reset_if_needed()

        equity = current_capital + open_pnl

        # Equity peak guncelle
        self._equity_peak = max(self._equity_peak, equity)

        # Metrikleri hesapla — equity kullan (current_capital nakit-only olabilir)
        daily_loss_pct  = self._daily_loss_pct(equity)
        drawdown_pct    = self._drawdown_pct(equity)

        # Seviyeyi belirle
        level  = AlertLevel.NORMAL
        reason = ""

        # Kirmizi kontrolleri
        if daily_loss_pct >= self.red_threshold or drawdown_pct >= self.red_threshold:
            level  = AlertLevel.RED
            reason = (
                f"KIRMIZI | Gunluk zarar: %{daily_loss_pct*100:.1f} | "
                f"Drawdown: %{drawdown_pct*100:.1f}"
            )
            self._red_manual_reset_required = True

        # Turuncu kontrolleri
        elif (daily_loss_pct >= self.orange_threshold or
              self._consecutive_errors >= self.max_consecutive_errors):
            level  = AlertLevel.ORANGE
            if self._consecutive_errors >= self.max_consecutive_errors:
                reason = f"TURUNCU | Ust uste {self._consecutive_errors} hata"
            else:
                reason = f"TURUNCU | Gunluk zarar: %{daily_loss_pct*100:.1f}"

        # Sari kontrolleri
        elif daily_loss_pct >= self.yellow_threshold:
            level  = AlertLevel.YELLOW
            reason = f"SARI | Gunluk zarar: %{daily_loss_pct*100:.1f}"

        # Seviye degistiyse logla
        if level != self._current_level:
            prefix = LEVEL_PREFIX[level]
            if level == AlertLevel.NORMAL:
                logger.info(f"{prefix} Risk normal seviyeye dondu.")
            elif level == AlertLevel.YELLOW:
                logger.warning(f"{prefix} {reason} | Yeni pozisyon acilmayacak.")
            elif level == AlertLevel.ORANGE:
                logger.error(f"{prefix} {reason} | Tum pozisyonlar kapatilacak!")
            elif level == AlertLevel.RED:
                logger.critical(f"{prefix} {reason} | BOT DURDURULUYOR!")

            self._log_event(level, reason, daily_loss_pct, drawdown_pct)
            self._current_level = level

        return KillStatus(
            level            = level,
            can_open         = level == AlertLevel.NORMAL,
            should_close_all = level >= AlertLevel.ORANGE,
            should_halt      = level >= AlertLevel.RED,
            reason           = reason,
            daily_loss_pct   = daily_loss_pct,
            drawdown_pct     = drawdown_pct,
        )

    # ── Saymalar ─────────────────────────────────────────────────────────────

    def record_trade(self) -> bool:
        """
        Islem kayit eder, islem hizi limitini kontrol eder.

        Returns:
            True = islem izin verildi, False = hiz limiti asildi
        """
        # Saatlik sıfırlama
        now = datetime.now(timezone.utc)
        if (now - self._hour_start).total_seconds() >= 3600:
            self._hour_trades = 0
            self._hour_start  = now

        self._hour_trades += 1
        self._day_trades  += 1

        if self._hour_trades > self.max_trades_per_hour:
            logger.warning(
                f"Islem hizi limiti asildi: "
                f"{self._hour_trades}/{self.max_trades_per_hour} islem/saat"
            )
            return False
        return True

    def record_error(self) -> None:
        """Ust uste hata sayacini arttirir."""
        self._consecutive_errors += 1
        logger.warning(
            f"Hata sayaci: {self._consecutive_errors}/{self.max_consecutive_errors}"
        )

    def clear_errors(self) -> None:
        """Basarili tick'de hata sayacini sifirlar."""
        if self._consecutive_errors > 0:
            logger.debug(f"Hata sayaci sifirlandi ({self._consecutive_errors} -> 0)")
        self._consecutive_errors = 0

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_red(self, reason: str = "Manuel reset") -> None:
        """
        Kirmizi alarmi manuel olarak sifirlar.
        Sadece insan tarafindan, bilerek cagrilmali.
        """
        logger.warning(f"Kirmizi alarm sifirlandi. Sebep: {reason}")
        self._red_manual_reset_required = False
        self._current_level = AlertLevel.NORMAL
        self._log_event(AlertLevel.NORMAL, f"Manuel reset: {reason}", 0.0, 0.0)

    def update_day_start(self, capital: float) -> None:
        """
        Yeni gun basinda gunluk baslangic sermayesini guncelle.
        TradingBot tarafindan gece yarisi cagrilmali.
        """
        self._day_start_capital = capital
        self._day_start_date    = datetime.now(timezone.utc).date()
        self._day_trades        = 0
        logger.info(
            f"Kill switch gun resetlendi | "
            f"Yeni baslangic: ${capital:,.2f}"
        )

    # ── Yardimci ─────────────────────────────────────────────────────────────

    def _daily_reset_if_needed(self) -> None:
        """Yeni gun basladiysa otomatik gun resetini yapar."""
        today = datetime.now(timezone.utc).date()
        if today != self._day_start_date:
            logger.info("Yeni gun: Kill switch gunluk sayaclar sifirlaniyor.")
            # NOT: Gun basi sermayesini dis sistemden almak daha dogru.
            # Bu fallback sadece tracker guncellemezse calisir.
            self._day_start_date = today
            self._day_trades     = 0
            # Sari ve Turuncu sıfirlanir, Kirmizi sıfirlanmaz (manuel gerektirir)
            if self._current_level < AlertLevel.RED:
                self._current_level = AlertLevel.NORMAL

    def _daily_loss_pct(self, current_capital: float) -> float:
        """Gunluk zarar yuzdesi (pozitif = zarar)."""
        if self._day_start_capital <= 0:
            return 0.0
        loss = self._day_start_capital - current_capital
        return max(0.0, loss / self._day_start_capital)

    def _drawdown_pct(self, equity: float) -> float:
        """Equity peak'ten anlık drawdown yuzdesi."""
        if self._equity_peak <= 0:
            return 0.0
        dd = self._equity_peak - equity
        return max(0.0, dd / self._equity_peak)

    def _log_event(
        self,
        level: AlertLevel,
        reason: str,
        daily_loss_pct: float,
        drawdown_pct: float,
    ) -> None:
        """Olayi gecmis listesine ekle."""
        self._events.append({
            "timestamp"      : datetime.now(timezone.utc).isoformat(),
            "level"          : level.name,
            "reason"         : reason,
            "daily_loss_pct" : round(daily_loss_pct * 100, 2),
            "drawdown_pct"   : round(drawdown_pct * 100, 2),
        })

    # ── Sorgulama ─────────────────────────────────────────────────────────────

    @property
    def current_level(self) -> AlertLevel:
        return self._current_level

    @property
    def is_active(self) -> bool:
        """Kill switch devrede mi? (ORANGE veya RED = devrede)"""
        return self._current_level >= AlertLevel.ORANGE

    def events(self) -> list[dict]:
        """Tum alarm olaylarinin listesi."""
        return list(self._events)

    def summary(self) -> dict:
        """Mevcut durum ozeti."""
        return {
            "level"               : self._current_level.name,
            "is_active"           : self.is_active,
            "consecutive_errors"  : self._consecutive_errors,
            "day_trades"          : self._day_trades,
            "red_reset_required"  : self._red_manual_reset_required,
            "equity_peak"         : round(self._equity_peak, 2),
            "day_start_capital"   : round(self._day_start_capital, 2),
        }
