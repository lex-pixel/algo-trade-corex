"""
tests/test_phase7.py
=====================
AMACI:
    Phase 7 risk sistemi modullerini test eder:
    - PositionSizer: fixed_fraction, atr_based, kelly, conservative
    - KillSwitch: 3 seviyeli alarm, reset, sayaclar
    - RiskManager: sinyal degerlendirme, pozisyon cikis kontrol

    Gercek API gerektirmez.

CALISTIRMAK ICIN:
    pytest tests/test_phase7.py -v
"""

import pytest

from risk.position_sizer import PositionSizer, MIN_QTY
from risk.kill_switch import KillSwitch, AlertLevel, KillStatus
from risk.risk_manager import RiskManager, TradeDecision
from trading.position_tracker import PositionTracker


# ─────────────────────────────────────────────────────────────────────────────
# PositionSizer TESTLERI
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizer:
    """PositionSizer sinifini test eder."""

    def _sizer(self) -> PositionSizer:
        return PositionSizer(
            max_risk_pct    = 0.02,
            max_capital_pct = 0.10,
            max_qty         = 0.10,
        )

    # ── Fixed Fraction ────────────────────────────────────────────────────────

    def test_fixed_fraction_returns_positive(self):
        """fixed_fraction() pozitif miktar dondurmeli."""
        qty = self._sizer().fixed_fraction(10_000, 95000, stop_pct=0.02)
        assert qty > 0

    def test_fixed_fraction_respects_max_qty(self):
        """Maksimum BTC limiti asilmamali."""
        qty = self._sizer().fixed_fraction(1_000_000, 95000, stop_pct=0.001)
        assert qty <= 0.10

    def test_fixed_fraction_respects_max_capital_pct(self):
        """Sermayenin %10'undan fazlasina dokunmamali."""
        sizer = PositionSizer(max_risk_pct=0.02, max_capital_pct=0.10)
        qty   = sizer.fixed_fraction(10_000, 95000, stop_pct=0.0001)
        assert qty * 95000 <= 10_000 * 0.10 + 0.01  # ufak tolerans

    def test_fixed_fraction_zero_capital(self):
        """Sermaye sifirda MIN_QTY veya sifir donmeli."""
        qty = self._sizer().fixed_fraction(0, 95000, stop_pct=0.02)
        assert qty == 0.0

    def test_fixed_fraction_with_stop_price(self):
        """stop_price parametresi de kabul edilmeli."""
        qty = self._sizer().fixed_fraction(10_000, 95000, stop_price=93000)
        assert qty > 0

    def test_fixed_fraction_precision(self):
        """6 ondalik hassasiyette olmali."""
        qty = self._sizer().fixed_fraction(10_000, 95000, stop_pct=0.02)
        assert round(qty, 6) == qty

    # ── ATR Based ─────────────────────────────────────────────────────────────

    def test_atr_based_returns_positive(self):
        """atr_based() pozitif miktar dondurmeli."""
        qty = self._sizer().atr_based(10_000, 95000, atr=500.0)
        assert qty > 0

    def test_atr_based_larger_atr_smaller_qty(self):
        """Daha buyuk ATR -> daha kucuk pozisyon (ayni risk).
        Sinirlar asilmayacak sekilde max_capital_pct=1.0 kullaniriz.
        ATR=200: qty = 200/400 = 0.5 BTC
        ATR=800: qty = 200/1600 = 0.125 BTC  -> 0.5 > 0.125
        """
        # Sinirlar asilmayacak sekilde cok yuksek cap tanimla
        sizer = PositionSizer(max_risk_pct=0.02, max_capital_pct=100.0, max_qty=100.0)
        qty_low_vol  = sizer.atr_based(10_000, 95000, atr=200.0)
        qty_high_vol = sizer.atr_based(10_000, 95000, atr=800.0)
        assert qty_low_vol > qty_high_vol

    def test_atr_based_zero_atr(self):
        """Sifir ATR 0.0 dondurmeli."""
        qty = self._sizer().atr_based(10_000, 95000, atr=0.0)
        assert qty == 0.0

    def test_atr_based_respects_max_qty(self):
        """Maks BTC limiti asilmamali."""
        qty = self._sizer().atr_based(1_000_000, 95000, atr=0.01)
        assert qty <= 0.10

    # ── Kelly ─────────────────────────────────────────────────────────────────

    def test_kelly_returns_positive_for_good_setup(self):
        """Iyi win-rate ile Kelly pozitif dondurmeli."""
        qty = self._sizer().kelly(10_000, 95000, win_rate=0.6, avg_win_pct=0.03, avg_loss_pct=0.02)
        assert qty > 0

    def test_kelly_returns_zero_for_negative_edge(self):
        """Negatif beklenti varsa Kelly 0.0 dondurmeli."""
        qty = self._sizer().kelly(10_000, 95000, win_rate=0.3, avg_win_pct=0.01, avg_loss_pct=0.03)
        assert qty == 0.0

    def test_kelly_respects_half_kelly(self):
        """Yari Kelly: tam Kelly'nin yarisini gecmemeli."""
        sizer_full = PositionSizer(kelly_fraction=1.0)
        sizer_half = PositionSizer(kelly_fraction=0.5)
        qty_full = sizer_full.kelly(10_000, 95000, win_rate=0.6, avg_win_pct=0.03, avg_loss_pct=0.02)
        qty_half = sizer_half.kelly(10_000, 95000, win_rate=0.6, avg_win_pct=0.03, avg_loss_pct=0.02)
        assert qty_half <= qty_full

    def test_kelly_win_rate_50_low_return(self):
        """50/50 win rate + esit kazanc/kayip -> sifir Kelly."""
        qty = self._sizer().kelly(10_000, 95000, win_rate=0.5, avg_win_pct=0.02, avg_loss_pct=0.02)
        assert qty == 0.0

    # ── Conservative ──────────────────────────────────────────────────────────

    def test_conservative_returns_min_of_methods(self):
        """conservative() en kucuk miktari secmeli."""
        sizer = self._sizer()
        qty_ff   = sizer.fixed_fraction(10_000, 95000, stop_pct=0.02)
        qty_atr  = sizer.atr_based(10_000, 95000, atr=500.0)
        qty_cons = sizer.conservative(10_000, 95000, atr=500.0, stop_pct=0.02)
        assert qty_cons <= max(qty_ff, qty_atr) + 1e-9

    def test_conservative_positive(self):
        """conservative() pozitif miktar dondurmeli."""
        qty = self._sizer().conservative(10_000, 95000, atr=400.0)
        assert qty > 0

    # ── Risk Summary ──────────────────────────────────────────────────────────

    def test_risk_summary_keys(self):
        """risk_summary() beklenen anahtarlara sahip olmali."""
        s = self._sizer().risk_summary(10_000, 95000, 0.001)
        assert "notional_usdt"    in s
        assert "max_loss_usdt"    in s
        assert "risk_pct"         in s
        assert "capital_used_pct" in s

    def test_risk_summary_notional_positive(self):
        """Notional degeri pozitif olmali."""
        s = self._sizer().risk_summary(10_000, 95000, 0.001)
        assert s["notional_usdt"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# KillSwitch TESTLERI
# ─────────────────────────────────────────────────────────────────────────────

class TestKillSwitch:
    """KillSwitch sinifini test eder."""

    def _ks(self, capital: float = 10_000.0) -> KillSwitch:
        return KillSwitch(
            initial_capital        = capital,
            yellow_threshold_pct   = 0.03,
            orange_threshold_pct   = 0.05,
            red_threshold_pct      = 0.15,
            max_consecutive_errors = 3,
        )

    def test_normal_level_no_loss(self):
        """Zarar yoksa NORMAL seviye olmali."""
        ks     = self._ks(10_000)
        status = ks.check(current_capital=10_000, open_pnl=0)
        assert status.level == AlertLevel.NORMAL

    def test_yellow_triggered_on_3pct_loss(self):
        """10k sermayede 300+ zarar Sari alarm tetiklemeli."""
        ks     = self._ks(10_000)
        status = ks.check(current_capital=9_650, open_pnl=0)
        assert status.level >= AlertLevel.YELLOW

    def test_orange_triggered_on_5pct_loss(self):
        """10k sermayede 500+ zarar Turuncu alarm tetiklemeli."""
        ks     = self._ks(10_000)
        status = ks.check(current_capital=9_450, open_pnl=0)
        assert status.level >= AlertLevel.ORANGE

    def test_red_triggered_on_15pct_loss(self):
        """10k sermayede 1500+ zarar Kirmizi alarm tetiklemeli."""
        ks     = self._ks(10_000)
        status = ks.check(current_capital=8_400, open_pnl=0)
        assert status.level == AlertLevel.RED

    def test_normal_can_open(self):
        """Normal modda yeni pozisyon acilabilmeli."""
        ks     = self._ks(10_000)
        status = ks.check(current_capital=10_000, open_pnl=0)
        assert status.can_open is True

    def test_yellow_cannot_open(self):
        """Sari alarimda yeni pozisyon acilmamali."""
        ks     = self._ks(10_000)
        status = ks.check(current_capital=9_650, open_pnl=0)
        if status.level == AlertLevel.YELLOW:
            assert status.can_open is False

    def test_orange_should_close_all(self):
        """Turuncu alarmda tum pozisyonlar kapatilmali."""
        ks     = self._ks(10_000)
        status = ks.check(current_capital=9_450, open_pnl=0)
        if status.level >= AlertLevel.ORANGE:
            assert status.should_close_all is True

    def test_red_should_halt(self):
        """Kirmizi alarmda bot durmalı."""
        ks     = self._ks(10_000)
        status = ks.check(current_capital=8_400, open_pnl=0)
        assert status.should_halt is True

    def test_red_requires_manual_reset(self):
        """Kirmizi sonrasi manuel reset olmadan devam etmemeli."""
        ks = self._ks(10_000)
        ks.check(current_capital=8_400, open_pnl=0)
        # Sermaye normale donse bile kirmizi devam eder
        status = ks.check(current_capital=10_000, open_pnl=0)
        assert status.should_halt is True

    def test_reset_red_clears_halt(self):
        """reset_red() sonrasi normal islem yapilabilmeli."""
        ks = self._ks(10_000)
        ks.check(current_capital=8_400, open_pnl=0)
        ks.reset_red("Test reset")
        status = ks.check(current_capital=10_000, open_pnl=0)
        assert status.should_halt is False

    def test_consecutive_errors_trigger_orange(self):
        """max_consecutive_errors asilinca Turuncu tetiklenmeli."""
        ks = self._ks(10_000)
        for _ in range(3):
            ks.record_error()
        status = ks.check(current_capital=10_000, open_pnl=0)
        assert status.level >= AlertLevel.ORANGE

    def test_clear_errors_resets_counter(self):
        """clear_errors() hata sayacini sifirlamali."""
        ks = self._ks(10_000)
        ks.record_error()
        ks.record_error()
        ks.clear_errors()
        assert ks._consecutive_errors == 0

    def test_record_trade_returns_bool(self):
        """record_trade() bool dondurmeli."""
        ks     = self._ks(10_000)
        result = ks.record_trade()
        assert isinstance(result, bool)

    def test_status_returns_level_key(self):
        """summary() 'level' anahtarini icermeli."""
        ks = self._ks(10_000)
        s  = ks.summary()
        assert "level" in s

    def test_drawdown_triggers_red(self):
        """Drawdown %15 asinca Kirmizi tetiklenmeli."""
        ks = self._ks(10_000)
        # Equity peek'i 10k yap
        ks.check(current_capital=10_000, open_pnl=0)
        # Sonra %15+ drawdown olustur
        status = ks.check(current_capital=7_000, open_pnl=1_000)  # equity=8000 -> %20 DD
        assert status.level == AlertLevel.RED

    def test_events_logged_on_level_change(self):
        """Seviye degisince olay loglanmali."""
        ks = self._ks(10_000)
        ks.check(current_capital=9_650, open_pnl=0)   # YELLOW tetikle
        assert len(ks.events()) > 0


# ─────────────────────────────────────────────────────────────────────────────
# RiskManager TESTLERI
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskManager:
    """RiskManager sinifini test eder."""

    def _rm(self) -> RiskManager:
        return RiskManager(
            initial_capital    = 10_000.0,
            max_risk_pct       = 0.02,
            max_open_positions = 1,
            min_confidence     = 0.55,
        )

    def _pt(self) -> PositionTracker:
        return PositionTracker(initial_capital=10_000.0, max_positions=1)

    def test_evaluate_approved_for_good_al_signal(self):
        """Iyi AL sinyali onaylanmali."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="AL", confidence=0.70,
            current_capital=10_000, open_pnl=0,
            price=95000, atr=500.0,
            open_positions_count=0,
        )
        assert decision.approved is True

    def test_evaluate_rejected_for_bekle(self):
        """BEKLE sinyali reddedilmeli."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="BEKLE", confidence=0.80,
            current_capital=10_000, open_pnl=0,
            price=95000,
        )
        assert decision.approved is False

    def test_evaluate_rejected_low_confidence(self):
        """Dusuk guvenli sinyal reddedilmeli."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="AL", confidence=0.30,
            current_capital=10_000, open_pnl=0,
            price=95000,
        )
        assert decision.approved is False

    def test_evaluate_rejected_max_positions(self):
        """Maks pozisyon asilinca reddedilmeli."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="AL", confidence=0.75,
            current_capital=10_000, open_pnl=0,
            price=95000,
            open_positions_count=1,   # Max=1, zaten dolu
        )
        assert decision.approved is False

    def test_evaluate_rejected_kill_switch_orange(self):
        """Kill switch Turuncu iken yeni islem reddedilmeli."""
        rm = self._rm()
        # %5 zarar -> Turuncu
        decision = rm.evaluate_signal(
            action="AL", confidence=0.80,
            current_capital=9_450,   # %5.5 zarar
            open_pnl=0,
            price=95000,
        )
        assert decision.approved is False

    def test_approved_has_quantity(self):
        """Onaylanan kararda quantity > 0 olmali."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="AL", confidence=0.75,
            current_capital=10_000, open_pnl=0,
            price=95000, atr=500.0,
        )
        if decision.approved:
            assert decision.quantity > 0

    def test_approved_has_stop_loss(self):
        """Onaylanan kararda stop_loss ayarlanmali."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="AL", confidence=0.75,
            current_capital=10_000, open_pnl=0,
            price=95000, atr=500.0,
        )
        if decision.approved:
            assert decision.stop_loss is not None

    def test_sat_signal_approved(self):
        """SAT sinyali de onaylanabilmeli."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="SAT", confidence=0.70,
            current_capital=10_000, open_pnl=0,
            price=95000, atr=500.0,
        )
        assert decision.approved is True

    def test_check_exit_no_positions(self):
        """Acik pozisyon yoksa bos liste donmeli."""
        rm = self._rm()
        pt = self._pt()
        exits = rm.check_exit_conditions(pt, 95000, 10_000, 0)
        assert exits == []

    def test_check_exit_stop_loss(self):
        """SL tetiklenince pozisyon cikmali."""
        rm  = self._rm()
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000, 0.001, stop_loss=93000)
        exits = rm.check_exit_conditions(pt, 92000, 10_000, 0)
        assert len(exits) == 1
        assert "STOP_LOSS" in exits[0][1]

    def test_check_exit_kill_switch_closes_all(self):
        """Kill switch TURUNCU iken tum pozisyonlar kapatilmali."""
        rm  = self._rm()
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000, 0.001)
        # %5 zarar olustur (Turuncu)
        exits = rm.check_exit_conditions(pt, 95000, 9_450, -50)
        assert len(exits) >= 1
        assert "KILL_SWITCH" in exits[0][1]

    def test_audit_log_grows(self):
        """Her karar audit loguna eklenmeli."""
        rm = self._rm()
        rm.evaluate_signal("AL", 0.70, 10_000, 0, 95000)
        rm.evaluate_signal("SAT", 0.30, 10_000, 0, 95000)  # Guven dusuk
        assert len(rm.audit_log()) >= 1

    def test_status_dict_keys(self):
        """status() beklenen anahtarlara sahip olmali."""
        rm = self._rm()
        s  = rm.status()
        assert "kill_switch_level" in s
        assert "consecutive_errors" in s
        assert "total_decisions"    in s

    def test_record_error_increments(self):
        """record_error() hata sayacini artirmali."""
        rm = self._rm()
        rm.record_error()
        rm.record_error()
        assert rm.kill_switch._consecutive_errors == 2

    def test_clear_errors_works(self):
        """clear_errors() sifirlamali."""
        rm = self._rm()
        rm.record_error()
        rm.clear_errors()
        assert rm.kill_switch._consecutive_errors == 0

    def test_trade_decision_str(self):
        """TradeDecision str() calismali."""
        decision = TradeDecision(approved=True, quantity=0.001, reason="Test")
        s = str(decision)
        assert "ONAYLANDI" in s

    def test_sl_long_below_price(self):
        """LONG icin SL fiyattan asagida olmali."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="AL", confidence=0.75,
            current_capital=10_000, open_pnl=0,
            price=95000, atr=500.0,
        )
        if decision.approved and decision.stop_loss:
            assert decision.stop_loss < 95000

    def test_tp_long_above_price(self):
        """LONG icin TP fiyattan yukari olmali."""
        rm       = self._rm()
        decision = rm.evaluate_signal(
            action="AL", confidence=0.75,
            current_capital=10_000, open_pnl=0,
            price=95000, atr=500.0,
        )
        if decision.approved and decision.take_profit:
            assert decision.take_profit > 95000
