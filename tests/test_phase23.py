"""
tests/test_phase23.py
======================
Phase 23: Kaldiracli Trading (LeverageManager, leveraged sizing, liquidation) testleri
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from risk.leverage_manager import LeverageManager, LeverageDecision
from risk.position_sizer import PositionSizer


# ── TestLeverageManagerSuggest ─────────────────────────────────────────────

class TestLeverageManagerSuggest:
    """suggest_leverage() testleri."""

    def setup_method(self):
        self.lm = LeverageManager(max_leverage=5)

    def test_range_market_max_5x(self):
        """ADX < 20 range piyasa -> max 5x."""
        dec = self.lm.suggest_leverage(adx=15.0, atr_pct=0.008)
        assert dec.max_leverage == 5
        assert dec.regime == "RANGE"

    def test_mixed_market_max_3x(self):
        """ADX 20-30 karisik -> max 3x."""
        dec = self.lm.suggest_leverage(adx=25.0, atr_pct=0.010)
        assert dec.max_leverage == 3
        assert dec.regime == "MIXED"

    def test_trend_market_max_2x(self):
        """ADX > 30 trend -> max 2x."""
        dec = self.lm.suggest_leverage(adx=35.0, atr_pct=0.010)
        assert dec.max_leverage == 2
        assert dec.regime == "TREND"

    def test_high_volatility_forces_1x(self):
        """ATR > %3 -> 1x (spot)."""
        dec = self.lm.suggest_leverage(adx=15.0, atr_pct=0.035)
        assert dec.leverage == 1
        assert dec.regime == "HIGH_VOL"

    def test_returns_leverage_decision(self):
        """Donus tipi LeverageDecision olmali."""
        dec = self.lm.suggest_leverage(adx=20.0, atr_pct=0.012)
        assert isinstance(dec, LeverageDecision)

    def test_leverage_at_least_1(self):
        """Onerilen kaldirac hic bir zaman 0'a dusmemeli."""
        for adx in [5.0, 20.0, 30.0, 45.0]:
            for atr_pct in [0.005, 0.015, 0.025]:
                dec = self.lm.suggest_leverage(adx=adx, atr_pct=atr_pct)
                assert dec.leverage >= 1

    def test_leverage_not_exceed_max(self):
        """Onerilen kaldirac system max'i asmamali."""
        lm = LeverageManager(max_leverage=3)
        dec = lm.suggest_leverage(adx=10.0, atr_pct=0.005)
        assert dec.leverage <= 3

    def test_high_atr_reduces_leverage(self):
        """Yuksek ATR leverage'i dusturmeli."""
        dec_low_atr  = self.lm.suggest_leverage(adx=15.0, atr_pct=0.008)
        dec_high_atr = self.lm.suggest_leverage(adx=15.0, atr_pct=0.022)
        assert dec_high_atr.leverage <= dec_low_atr.leverage

    def test_reason_not_empty(self):
        """reason alani dolu olmali."""
        dec = self.lm.suggest_leverage(adx=25.0, atr_pct=0.012)
        assert len(dec.reason) > 0


# ── TestLiquidationBuffer ──────────────────────────────────────────────────

class TestLiquidationBuffer:
    """check_liquidation_buffer() testleri."""

    def setup_method(self):
        self.lm = LeverageManager()

    def test_long_safe_buffer(self):
        """LONG: SL liq'den uzaksa OK."""
        ok, msg = self.lm.check_liquidation_buffer(
            direction="LONG",
            entry=70000, sl=68000,
            leverage=3, atr=500,
        )
        assert ok is True

    def test_long_unsafe_buffer(self):
        """LONG: SL liq'e cok yakinsa FAIL."""
        # 3x leverage: liq ~= 70000 * (1 - 1/3 + 0.004) ~= 46947
        # SL=47000, liq~=46947 -> mesafe cok az
        ok, msg = self.lm.check_liquidation_buffer(
            direction="LONG",
            entry=70000, sl=47000,
            leverage=3, atr=500,
        )
        assert ok is False

    def test_short_safe_buffer(self):
        """SHORT: SL liq'den uzaksa OK."""
        ok, msg = self.lm.check_liquidation_buffer(
            direction="SHORT",
            entry=70000, sl=72000,
            leverage=3, atr=500,
        )
        assert ok is True

    def test_spot_always_ok(self):
        """Leverage=1 (spot) -> hic bir zaman likidite riski yok."""
        ok, msg = self.lm.check_liquidation_buffer(
            direction="LONG",
            entry=70000, sl=60000,
            leverage=1, atr=500,
        )
        assert ok is True
        assert "spot" in msg.lower()

    def test_returns_tuple(self):
        """Tuple donmeli: (bool, str)."""
        result = self.lm.check_liquidation_buffer(
            direction="LONG", entry=70000, sl=68000, leverage=3, atr=500
        )
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_msg_not_empty(self):
        """Mesaj bos olmamali."""
        _, msg = self.lm.check_liquidation_buffer(
            direction="LONG", entry=70000, sl=68000, leverage=3, atr=500
        )
        assert len(msg) > 0


# ── TestLiquidationPrice ───────────────────────────────────────────────────

class TestLiquidationPrice:
    """liquidation_price() testleri."""

    def test_long_1x_returns_zero(self):
        """Spot (1x) LONG liq price 0 olmali."""
        liq = LeverageManager.liquidation_price("LONG", 70000, 1)
        assert liq == 0.0

    def test_short_1x_returns_inf(self):
        """Spot (1x) SHORT liq price inf olmali."""
        liq = LeverageManager.liquidation_price("SHORT", 70000, 1)
        assert liq == float("inf")

    def test_long_liq_below_entry(self):
        """LONG liq price giris fiyatinin altinda olmali."""
        liq = LeverageManager.liquidation_price("LONG", 70000, 5)
        assert liq < 70000

    def test_short_liq_above_entry(self):
        """SHORT liq price giris fiyatinin uzerinde olmali."""
        liq = LeverageManager.liquidation_price("SHORT", 70000, 5)
        assert liq > 70000

    def test_higher_leverage_closer_liquidation_long(self):
        """Daha yuksek leverage = daha yakin likidite (LONG)."""
        liq_2x = LeverageManager.liquidation_price("LONG", 70000, 2)
        liq_5x = LeverageManager.liquidation_price("LONG", 70000, 5)
        assert liq_5x > liq_2x  # 5x liq daha yukarda (girise yakin)

    def test_long_3x_approximate(self):
        """LONG 3x: liq ~= entry * (1 - 1/3 + 0.004) ~= entry * 0.671."""
        liq = LeverageManager.liquidation_price("LONG", 70000, 3)
        expected = 70000 * (1 - 1/3 + 0.004)
        assert abs(liq - expected) < 1.0


# ── TestFundingCost ────────────────────────────────────────────────────────

class TestFundingCost:
    """funding_cost() testleri."""

    def setup_method(self):
        self.lm = LeverageManager(funding_rate_8h=0.0001)

    def test_zero_hours(self):
        """0 saat = 0 maliyet."""
        cost = self.lm.funding_cost(notional=10000, leverage=3, hours=0)
        assert cost == 0.0

    def test_8_hours_one_period(self):
        """8 saat = 1 periyot = notional * rate."""
        cost = self.lm.funding_cost(notional=10000, leverage=3, hours=8)
        assert abs(cost - 1.0) < 0.001   # 10000 * 0.0001 = 1.0

    def test_24_hours_three_periods(self):
        """24 saat = 3 periyot."""
        cost = self.lm.funding_cost(notional=10000, leverage=3, hours=24)
        assert abs(cost - 3.0) < 0.001   # 3 * 1.0

    def test_higher_notional_higher_cost(self):
        """Buyuk pozisyon daha fazla funding odemeli."""
        cost_small = self.lm.funding_cost(notional=1000,  leverage=2, hours=8)
        cost_large = self.lm.funding_cost(notional=10000, leverage=2, hours=8)
        assert cost_large > cost_small


# ── TestMarginUsage ────────────────────────────────────────────────────────

class TestMarginUsage:
    """margin_usage() testleri."""

    def test_no_positions(self):
        """Acik pozisyon yok -> 0 kullanim."""
        usage = LeverageManager.margin_usage([], leverage=3, capital=10000)
        assert usage == 0.0

    def test_full_margin(self):
        """Tum sermaye marjin -> 1.0 (veya yakin)."""
        # Notional=30000, leverage=3 -> marjin=10000, capital=10000 -> usage=1.0
        usage = LeverageManager.margin_usage([30000.0], leverage=3, capital=10000)
        assert abs(usage - 1.0) < 0.01

    def test_half_margin(self):
        """Yari marjin -> 0.5."""
        usage = LeverageManager.margin_usage([15000.0], leverage=3, capital=10000)
        assert abs(usage - 0.5) < 0.01


# ── TestPositionSizerLeveraged ─────────────────────────────────────────────

class TestPositionSizerLeveraged:
    """PositionSizer.leveraged() testleri."""

    def setup_method(self):
        self.sizer = PositionSizer(max_risk_pct=0.02)

    def test_returns_positive(self):
        """Sonuc pozitif olmali."""
        qty = self.sizer.leveraged(capital=10000, price=70000, leverage=3)
        assert qty > 0

    def test_higher_leverage_more_quantity(self):
        """Daha yuksek leverage daha fazla miktar vermeli."""
        qty_2x = self.sizer.leveraged(capital=10000, price=70000, leverage=2)
        qty_5x = self.sizer.leveraged(capital=10000, price=70000, leverage=5)
        assert qty_5x > qty_2x

    def test_spot_leverage_1(self):
        """leverage=1 spot ile ayni sonucu vermeli (marjin bazli)."""
        qty = self.sizer.leveraged(capital=10000, price=70000, leverage=1, margin_pct=0.02)
        # marjin=200, notional=200, qty=200/70000=0.00286
        expected = min(200.0 / 70000, 0.1)
        assert abs(qty - expected) < 0.0001

    def test_max_margin_cap(self):
        """max_margin_pct=0.05 ile margin_pct=0.20 verilse 0.05 kullanilmali."""
        qty_capped   = self.sizer.leveraged(
            capital=10000, price=70000, leverage=3,
            margin_pct=0.20, max_margin_pct=0.05
        )
        qty_uncapped = self.sizer.leveraged(
            capital=10000, price=70000, leverage=3,
            margin_pct=0.05, max_margin_pct=0.05
        )
        assert abs(qty_capped - qty_uncapped) < 0.000001

    def test_zero_price_returns_zero(self):
        """Fiyat 0 ise 0 donmeli."""
        qty = self.sizer.leveraged(capital=10000, price=0, leverage=3)
        assert qty == 0.0

    def test_zero_capital_returns_zero(self):
        """Sermaye 0 ise 0 donmeli."""
        qty = self.sizer.leveraged(capital=0, price=70000, leverage=3)
        assert qty == 0.0


# ── TestPositionWithLeverage ───────────────────────────────────────────────

class TestPositionWithLeverage:
    """Position dataclass leverage alanlari testleri."""

    def test_position_default_leverage_1(self):
        """Varsayilan leverage 1 olmali."""
        from trading.position_tracker import Position
        from datetime import datetime, timezone
        pos = Position(
            position_id="test",
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=70000,
            quantity=0.01,
        )
        assert pos.leverage == 1
        assert pos.liquidation_price == 0.0
        assert pos.margin_mode == "ISOLATED"

    def test_position_margin_spot(self):
        """leverage=1 (spot) -> margin == notional."""
        from trading.position_tracker import Position
        pos = Position(
            position_id="test", symbol="BTC/USDT", direction="LONG",
            entry_price=70000, quantity=0.01, leverage=1,
        )
        assert abs(pos.margin - pos.notional) < 0.001

    def test_position_margin_3x(self):
        """leverage=3 -> margin = notional / 3."""
        from trading.position_tracker import Position
        pos = Position(
            position_id="test", symbol="BTC/USDT", direction="LONG",
            entry_price=70000, quantity=0.01, leverage=3,
        )
        assert abs(pos.margin - pos.notional / 3) < 0.001

    def test_position_not_near_liquidation_spot(self):
        """Spot (leverage=1) -> is_near_liquidation her zaman False."""
        from trading.position_tracker import Position
        pos = Position(
            position_id="test", symbol="BTC/USDT", direction="LONG",
            entry_price=70000, quantity=0.01, leverage=1,
        )
        pos.current_price = 50000  # Buyuk dusus
        assert pos.is_near_liquidation is False

    def test_position_near_liquidation_detected(self):
        """LONG 10x: liq yaklaşık entry*0.9, fiyat liq'e yaklasirsa uyari."""
        from trading.position_tracker import Position
        liq = LeverageManager.liquidation_price("LONG", 70000, 10)
        pos = Position(
            position_id="test", symbol="BTC/USDT", direction="LONG",
            entry_price=70000, quantity=0.01, leverage=10,
            liquidation_price=liq,
        )
        # Fiyati liq'in %102'sine cek -> near_liquidation True olmali
        pos.current_price = liq * 1.02
        assert pos.is_near_liquidation is True
