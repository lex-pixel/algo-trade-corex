"""
risk/risk_manager.py
======================
AMACI:
    PositionSizer + KillSwitch + ek kurallari birlestiren ust seviye risk yoneticisi.
    Trading botu bu sinifi kullanir — ayri ayri risk modullerine bakmak zorunda kalmaz.

SORUMLULUKLAR:
    1. Islem acilabilir mi? (kill switch, pozisyon limiti, guven esigi)
    2. Kac BTC alinmali? (position sizer)
    3. Stop-loss / take-profit nerede olmali?
    4. Pozisyon kapanmali mi? (SL/TP + kill switch)
    5. Tum risk kararlarini audit log'a yazar

KULLANIM:
    from risk.risk_manager import RiskManager

    rm = RiskManager(initial_capital=10_000.0)

    # Sinyal geldi — islem yapabilir miyiz?
    decision = rm.evaluate_signal(
        action="AL",
        confidence=0.72,
        current_capital=9_800.0,
        open_pnl=50.0,
        price=95000.0,
        atr=450.0,
    )

    if decision.approved:
        # Emri gonder
        qty = decision.quantity
        ...

    # Her tick: pozisyon kontrol
    exits = rm.check_positions(
        current_price=93000.0,
        current_capital=9_800.0,
        open_pnl=-120.0,
    )
    for pid, reason in exits:
        position_tracker.close_position(pid, reason)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from risk.position_sizer import PositionSizer
from risk.kill_switch import KillSwitch, AlertLevel, KillStatus
from trading.position_tracker import PositionTracker
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum sinyal guveni (bu altinda islem acilmaz)
DEFAULT_MIN_CONFIDENCE = 0.55


# ── Karar Nesnesi ─────────────────────────────────────────────────────────────

@dataclass
class TradeDecision:
    """
    RiskManager.evaluate_signal() tarafindan dondurulen karar.

    Alanlar:
        approved    : True = islem ac, False = atla
        quantity    : BTC miktari (approved=True ise gecerli)
        stop_loss   : Onerilen stop-loss fiyati
        take_profit : Onerilen take-profit fiyati
        reason      : Onay veya red sebebi
        kill_status : Anlık kill switch durumu
    """
    approved    : bool
    quantity    : float = 0.0
    stop_loss   : Optional[float] = None
    take_profit : Optional[float] = None
    reason      : str   = ""
    kill_status : Optional[KillStatus] = None

    def __str__(self) -> str:
        status = "ONAYLANDI" if self.approved else "REDDEDILDI"
        return f"TradeDecision({status} | qty={self.quantity:.6f} | {self.reason})"


# ── RiskManager ───────────────────────────────────────────────────────────────

class RiskManager:
    """
    Merkezi risk yonetim sistemi.

    PositionSizer ve KillSwitch'i entegre eder.
    TradingBot bu sinifle konusur, diger risk modullerini dogrudan cagirmaz.

    Parametreler:
        initial_capital     : Baslangic sermayesi
        max_risk_pct        : Tek islemde maks risk (%2 = 0.02)
        max_open_positions  : Ayni anda maks acik pozisyon
        min_confidence      : Islem icin minimum sinyal guveni
        sl_atr_mult         : Stop-loss = sl_atr_mult * ATR
        tp_atr_mult         : Take-profit = tp_atr_mult * ATR
        yellow_pct          : Sari alarm esigi
        orange_pct          : Turuncu alarm esigi
        red_pct             : Kirmizi alarm esigi
    """

    def __init__(
        self,
        initial_capital: float      = 10_000.0,
        max_risk_pct: float         = 0.02,
        max_open_positions: int     = 1,
        min_confidence: float       = DEFAULT_MIN_CONFIDENCE,
        sl_atr_mult: float          = 2.0,
        tp_atr_mult: float          = 3.0,
        yellow_pct: float           = 0.03,
        orange_pct: float           = 0.05,
        red_pct: float              = 0.15,
    ):
        self.initial_capital     = initial_capital
        self.max_open_positions  = max_open_positions
        self.min_confidence      = min_confidence
        self.sl_atr_mult         = sl_atr_mult
        self.tp_atr_mult         = tp_atr_mult

        self.sizer = PositionSizer(
            max_risk_pct    = max_risk_pct,
            max_capital_pct = 0.10,
            max_qty         = 0.10,
        )

        self.kill_switch = KillSwitch(
            initial_capital        = initial_capital,
            yellow_threshold_pct   = yellow_pct,
            orange_threshold_pct   = orange_pct,
            red_threshold_pct      = red_pct,
        )

        # Audit log: her karar kaydedilir
        self._audit: list[dict] = []

        logger.info(
            f"RiskManager baslatildi | "
            f"Sermaye: ${initial_capital:,.0f} | "
            f"max_risk=%{max_risk_pct*100:.0f} | "
            f"min_confidence={min_confidence}"
        )

    # ── Ana Karar Metodlari ───────────────────────────────────────────────────

    def evaluate_signal(
        self,
        action: str,
        confidence: float,
        current_capital: float,
        open_pnl: float,
        price: float,
        atr: float | None           = None,
        open_positions_count: int   = 0,
    ) -> TradeDecision:
        """
        Gelen sinyali risk kurallarina gore degerlendirir.

        Parametreler:
            action               : "AL" veya "SAT"
            confidence           : Sinyal guven skoru (0-1)
            current_capital      : Mevcut nakit (USDT)
            open_pnl             : Acik pozisyonlarin unrealized PnL
            price                : Guncel fiyat
            atr                  : Average True Range (None ise sabit stop kullanilir)
            open_positions_count : Kac tane acik pozisyon var?

        Returns:
            TradeDecision
        """
        # 1. Kill switch kontrol
        ks_status = self.kill_switch.check(current_capital, open_pnl)

        if ks_status.should_halt:
            return self._reject("Kill switch KIRMIZI: bot durduruldu", ks_status)

        if ks_status.should_close_all:
            return self._reject("Kill switch TURUNCU: yeni islem yok", ks_status)

        if not ks_status.can_open:
            return self._reject(
                f"Kill switch {ks_status.level.name}: yeni pozisyon acilmiyor",
                ks_status
            )

        # 2. BEKLE sinyali — hic birsey yapma
        if action == "BEKLE":
            return self._reject("Sinyal BEKLE", ks_status)

        # 3. Guven esigi
        if confidence < self.min_confidence:
            return self._reject(
                f"Guven dusuk: {confidence:.2f} < {self.min_confidence}",
                ks_status,
            )

        # 4. Maks pozisyon limiti
        if open_positions_count >= self.max_open_positions:
            return self._reject(
                f"Maks pozisyon asildi: {open_positions_count}/{self.max_open_positions}",
                ks_status,
            )

        # 5. Yeterli sermaye var mi?
        if current_capital < 50:  # Minimum USDT
            return self._reject(f"Yetersiz sermaye: ${current_capital:.2f}", ks_status)

        # 6. Pozisyon boyutunu hesapla
        if atr and atr > 0:
            qty = self.sizer.atr_based(current_capital, price, atr)
        else:
            qty = self.sizer.fixed_fraction(current_capital, price)

        if qty <= 0:
            return self._reject("Pozisyon boyutu sifir hesaplandi", ks_status)

        # 7. SL / TP hesapla
        direction = "LONG" if action == "AL" else "SHORT"
        stop_loss, take_profit = self._calc_sl_tp(direction, price, atr)

        risk = self.sizer.risk_summary(current_capital, price, qty)

        reason = (
            f"Onaylandi | {action} | qty={qty:.6f} BTC | "
            f"risk=%{risk['risk_pct']:.2f} | "
            f"guven={confidence:.2f}"
        )
        logger.info(reason)

        decision = TradeDecision(
            approved    = True,
            quantity    = qty,
            stop_loss   = stop_loss,
            take_profit = take_profit,
            reason      = reason,
            kill_status = ks_status,
        )
        self._audit_log("ONAY", action, confidence, decision)
        return decision

    def check_exit_conditions(
        self,
        position_tracker: PositionTracker,
        current_price: float,
        current_capital: float,
        open_pnl: float,
    ) -> list[tuple[str, str]]:
        """
        Kill switch + pozisyon SL/TP kontrolu yapar.

        Returns:
            list of (position_id, reason): Kapatilmasi gereken pozisyonlar
        """
        exits = []

        # Kill switch kontrol
        ks_status = self.kill_switch.check(current_capital, open_pnl)

        if ks_status.should_close_all or ks_status.should_halt:
            # Tum pozisyonlari kapat
            for pos in position_tracker.open_positions():
                exits.append((pos.position_id, f"KILL_SWITCH_{ks_status.level.name}"))
            if exits:
                logger.warning(
                    f"Kill switch {ks_status.level.name}: "
                    f"{len(exits)} pozisyon kapatiliyor."
                )
            return exits

        # Normal SL/TP kontrol
        sl_tp_exits = position_tracker.check_exit_conditions(current_price)
        exits.extend(sl_tp_exits)

        return exits

    def record_trade_executed(self) -> bool:
        """Gerceklesen islem kaydeder, hiz limitini kontrol eder."""
        return self.kill_switch.record_trade()

    def record_error(self) -> None:
        """Hata kaydeder."""
        self.kill_switch.record_error()

    def clear_errors(self) -> None:
        """Hatalarisifirlar."""
        self.kill_switch.clear_errors()

    def update_day_start(self, capital: float) -> None:
        """Yeni gun baslangic sermayesini gunceller."""
        self.kill_switch.update_day_start(capital)

    def reset_red_alert(self, reason: str = "Manuel") -> None:
        """Kirmizi alarmi manuel olarak sifirlar."""
        self.kill_switch.reset_red(reason)

    # ── Yardimci ─────────────────────────────────────────────────────────────

    def _calc_sl_tp(
        self,
        direction: str,
        price: float,
        atr: float | None,
    ) -> tuple[float | None, float | None]:
        """ATR tabanli SL/TP hesapla."""
        if not atr or atr <= 0:
            # Fallback: sabit yuzde
            if direction == "LONG":
                return price * 0.98, price * 1.03
            else:
                return price * 1.02, price * 0.97

        if direction == "LONG":
            stop_loss   = price - self.sl_atr_mult * atr
            take_profit = price + self.tp_atr_mult * atr
        else:
            stop_loss   = price + self.sl_atr_mult * atr
            take_profit = price - self.tp_atr_mult * atr

        return round(stop_loss, 2), round(take_profit, 2)

    def _reject(
        self, reason: str, ks_status: KillStatus | None = None
    ) -> TradeDecision:
        """Red kararı uret ve logla."""
        logger.debug(f"Islem reddedildi: {reason}")
        self._audit_log("RED", "", 0.0, None, reason)
        return TradeDecision(
            approved    = False,
            reason      = reason,
            kill_status = ks_status,
        )

    def _audit_log(
        self,
        decision: str,
        action: str,
        confidence: float,
        trade_decision: TradeDecision | None,
        reason: str = "",
    ) -> None:
        """Her karari audit loguna yaz."""
        from datetime import datetime, timezone
        self._audit.append({
            "timestamp" : datetime.now(timezone.utc).isoformat(),
            "decision"  : decision,
            "action"    : action,
            "confidence": confidence,
            "quantity"  : trade_decision.quantity if trade_decision else 0,
            "reason"    : reason or (trade_decision.reason if trade_decision else ""),
        })

    # ── Sorgulama ─────────────────────────────────────────────────────────────

    def audit_log(self) -> list[dict]:
        """Tum risk kararlarinin audit logu."""
        return list(self._audit)

    def status(self) -> dict:
        """Genel risk durumu ozeti."""
        ks = self.kill_switch.summary()
        return {
            "kill_switch_level": ks["level"],
            "kill_switch_active": ks["is_active"],
            "consecutive_errors": ks["consecutive_errors"],
            "day_trades"        : ks["day_trades"],
            "total_decisions"   : len(self._audit),
            "approved_count"    : sum(1 for a in self._audit if a["decision"] == "ONAY"),
        }

    def print_status(self) -> None:
        """Durumu terminale yazdirir."""
        s = self.status()
        ks = self.kill_switch.summary()
        print(f"\n{'='*55}")
        print(f"  RiskManager Durumu")
        print(f"{'='*55}")
        print(f"  Kill Switch       : {s['kill_switch_level']}")
        print(f"  Aktif mi          : {'EVET' if s['kill_switch_active'] else 'HAYIR'}")
        print(f"  Ust Uste Hata     : {s['consecutive_errors']}")
        print(f"  Bugunki Islemler  : {s['day_trades']}")
        print(f"  Toplam Karar      : {s['total_decisions']}")
        print(f"  Onaylanan         : {s['approved_count']}")
        print(f"  Equity Peak       : ${ks['equity_peak']:,.2f}")
        print(f"{'='*55}\n")
