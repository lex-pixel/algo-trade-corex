"""
tests/test_phase6.py
=====================
AMACI:
    Phase 6 canli islem altyapisi modullerini test eder:
    - OrderManager: emir gonderme, iptal, ozet
    - PositionTracker: pozisyon ac/kapat, P&L, SL/TP
    - TelegramNotifier: dry-run mod, bildirim metodlari

    Gercek API gerektirmez — paper mod ve sahte veri kullanilir.

CALISTIRMAK ICIN:
    pytest tests/test_phase6.py -v
"""

import pytest
import uuid
from datetime import datetime, timezone

from trading.order_manager import OrderManager, Order, OrderStatus, OrderSide, OrderType
from trading.position_tracker import PositionTracker, Position, ClosedTrade
from monitoring.telegram_notifier import TelegramNotifier


# ─────────────────────────────────────────────────────────────────────────────
# OrderManager TESTLERI
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderManager:
    """OrderManager sinifini test eder."""

    def _om(self) -> OrderManager:
        return OrderManager(symbol="BTC/USDT", paper=True, commission=0.001)

    def test_paper_market_buy_filled(self):
        """Paper modda market buy emri aninda dolmali."""
        om    = self._om()
        order = om.place_market_order("buy", quantity=0.001, current_price=95000.0)
        assert order.status == OrderStatus.FILLED
        assert order.side   == "buy"
        assert order.paper  is True

    def test_paper_market_sell_filled(self):
        """Paper modda market sell emri aninda dolmali."""
        om    = self._om()
        order = om.place_market_order("sell", quantity=0.001, current_price=95000.0)
        assert order.status == OrderStatus.FILLED
        assert order.side   == "sell"

    def test_filled_price_includes_slippage_buy(self):
        """Buy emrinde doluş fiyatı piyasadan biraz yukari olmali (slipaj)."""
        om    = self._om()
        price = 95000.0
        order = om.place_market_order("buy", 0.001, current_price=price)
        assert order.filled_price > price   # slipaj eklenmeli

    def test_filled_price_includes_slippage_sell(self):
        """Sell emrinde doluş fiyatı piyasadan biraz asagi olmali (slipaj)."""
        om    = self._om()
        price = 95000.0
        order = om.place_market_order("sell", 0.001, current_price=price)
        assert order.filled_price < price   # slipaj dusulmeli

    def test_commission_fee_positive(self):
        """Komisyon sifirdan buyuk olmali."""
        om    = self._om()
        order = om.place_market_order("buy", 0.001, current_price=95000.0)
        assert order.fee > 0

    def test_order_has_unique_id(self):
        """Her emir benzersiz ID'ye sahip olmali."""
        om = self._om()
        o1 = om.place_market_order("buy",  0.001, current_price=95000.0)
        o2 = om.place_market_order("sell", 0.001, current_price=95000.0)
        assert o1.order_id != o2.order_id

    def test_invalid_side_raises(self):
        """Gecersiz side ValueError firlatmali."""
        om = self._om()
        with pytest.raises(ValueError):
            om.place_market_order("alsat", 0.001, current_price=95000.0)

    def test_zero_quantity_raises(self):
        """Sifir miktar ValueError firlatmali."""
        om = self._om()
        with pytest.raises(ValueError):
            om.place_market_order("buy", 0.0, current_price=95000.0)

    def test_limit_order_status_open(self):
        """Limit emir OPEN durumunda olmali."""
        om    = self._om()
        order = om.place_limit_order("buy", 0.001, limit_price=90000.0)
        assert order.status == OrderStatus.OPEN
        assert order.type   == OrderType.LIMIT

    def test_simulate_fill_buy_fills_when_price_drops(self):
        """Fiyat limit altina dusunce buy limit dolmali."""
        om    = self._om()
        order = om.place_limit_order("buy", 0.001, limit_price=90000.0)
        # Fiyat 89000'e dustii — dolmali
        filled = om.simulate_fill(order.order_id, current_price=89000.0)
        assert filled is True
        assert om.get_order(order.order_id).status == OrderStatus.FILLED

    def test_simulate_fill_not_filled_when_price_high(self):
        """Fiyat limit uzerindeyse buy limit dolmamali."""
        om    = self._om()
        order = om.place_limit_order("buy", 0.001, limit_price=90000.0)
        filled = om.simulate_fill(order.order_id, current_price=92000.0)
        assert filled is False

    def test_cancel_open_order(self):
        """Acik emir iptal edilebilmeli."""
        om    = self._om()
        order = om.place_limit_order("buy", 0.001, limit_price=90000.0)
        result = om.cancel_order(order.order_id)
        assert result is True
        assert om.get_order(order.order_id).status == OrderStatus.CANCELLED

    def test_cancel_filled_order_fails(self):
        """Dolu emir iptal edilemez."""
        om    = self._om()
        order = om.place_market_order("buy", 0.001, current_price=95000.0)
        result = om.cancel_order(order.order_id)
        assert result is False   # Zaten dolmuş

    def test_list_open_orders(self):
        """list_open_orders() sadece acik emirleri dondurmeli."""
        om = self._om()
        om.place_market_order("buy",  0.001, current_price=95000.0)  # dolar
        om.place_limit_order("buy",   0.001, limit_price=90000.0)     # acik kalir
        open_orders = om.list_open_orders()
        assert len(open_orders) == 1

    def test_list_filled_orders(self):
        """list_filled_orders() sadece dolan emirleri dondurmeli."""
        om = self._om()
        om.place_market_order("buy",  0.001, current_price=95000.0)
        om.place_market_order("sell", 0.001, current_price=95000.0)
        filled = om.list_filled_orders()
        assert len(filled) == 2

    def test_summary_dict_keys(self):
        """summary_dict() beklenen anahtarlara sahip olmali."""
        om = self._om()
        s  = om.summary_dict()
        assert "open_orders"   in s
        assert "filled_orders" in s
        assert "total_fee_usdt" in s

    def test_cancel_all_open_orders(self):
        """cancel_all_open_orders() tum acik emirleri iptal etmeli."""
        om = self._om()
        om.place_limit_order("buy",  0.001, limit_price=90000.0)
        om.place_limit_order("sell", 0.001, limit_price=100000.0)
        count = om.cancel_all_open_orders()
        assert count == 2
        assert len(om.list_open_orders()) == 0

    def test_order_notional(self):
        """Emir notional deger dogru hesaplanmali."""
        om    = self._om()
        order = om.place_market_order("buy", 0.001, current_price=95000.0)
        # notional = quantity * filled_price (slipajli)
        assert order.notional > 0


# ─────────────────────────────────────────────────────────────────────────────
# PositionTracker TESTLERI
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionTracker:
    """PositionTracker sinifini test eder."""

    def _pt(self, capital: float = 10_000.0) -> PositionTracker:
        return PositionTracker(initial_capital=capital, max_positions=2)

    def test_open_position_returns_position(self):
        """open_position() Position nesnesi dondurmeli."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        assert isinstance(pos, Position)

    def test_open_position_direction_stored(self):
        """Pozisyon yonu dogru kaydedilmeli."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        assert pos.direction == "LONG"

    def test_open_position_reduces_capital(self):
        """Pozisyon acilinca sermaye azalmali."""
        pt = self._pt(10_000.0)
        pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        assert pt.capital < 10_000.0

    def test_max_positions_limit(self):
        """max_positions asiminda None donmeli."""
        pt = PositionTracker(initial_capital=100_000.0, max_positions=1)
        pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        pos2 = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        assert pos2 is None

    def test_insufficient_capital_returns_none(self):
        """Yetersiz sermayede None donmeli."""
        pt  = PositionTracker(initial_capital=10.0, max_positions=3)
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 1.0)  # 95k USDT lazim
        assert pos is None

    def test_update_sets_unrealized_pnl(self):
        """update() sonrasi unrealized PnL hesaplanmali."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        pos.update_price(96000.0)
        # LONG: (96000 - 95000) * 0.001 = +1.0
        assert pos.unrealized_pnl == pytest.approx(1.0, abs=0.01)

    def test_update_short_negative_when_price_rises(self):
        """SHORT pozisyonda fiyat artinca unrealized PnL negatif olmali."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "SHORT", 95000.0, 0.001)
        pos.update_price(97000.0)
        # SHORT: (95000 - 97000) * 0.001 = -2.0
        assert pos.unrealized_pnl == pytest.approx(-2.0, abs=0.01)

    def test_stop_loss_triggered_long(self):
        """LONG pozisyonda fiyat SL altina dusunce tetiklenm ali."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001,
                               stop_loss=93000.0)
        assert pos.should_stop_loss(92000.0) is True
        assert pos.should_stop_loss(96000.0) is False

    def test_take_profit_triggered_long(self):
        """LONG pozisyonda fiyat TP uzerine cikinca tetiklenmeli."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001,
                               take_profit=98000.0)
        assert pos.should_take_profit(99000.0) is True
        assert pos.should_take_profit(97000.0) is False

    def test_stop_loss_triggered_short(self):
        """SHORT pozisyonda fiyat SL uzerine cikinca tetiklenmeli."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "SHORT", 95000.0, 0.001,
                               stop_loss=97000.0)
        assert pos.should_stop_loss(98000.0) is True
        assert pos.should_stop_loss(94000.0) is False

    def test_close_position_returns_closed_trade(self):
        """close_position() ClosedTrade dondurmeli."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        trade = pt.close_position(pos.position_id, exit_price=96000.0)
        assert isinstance(trade, ClosedTrade)

    def test_close_position_updates_capital(self):
        """Pozisyon kapatilinca sermaye guncellenmeli."""
        pt  = self._pt(10_000.0)
        initial_capital = pt.capital
        pos  = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        trade = pt.close_position(pos.position_id, exit_price=96000.0)
        assert pt.capital != initial_capital

    def test_close_position_pnl_positive_for_winner(self):
        """Kazanan islemde PnL pozitif olmali."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.01)
        trade = pt.close_position(pos.position_id, exit_price=97000.0)
        assert trade.realized_pnl > 0

    def test_close_position_pnl_negative_for_loser(self):
        """Kaybeden islemde PnL negatif olmali."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.01)
        trade = pt.close_position(pos.position_id, exit_price=93000.0)
        assert trade.realized_pnl < 0

    def test_position_removed_after_close(self):
        """Pozisyon kapatilinca listeden kalkmali."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        pt.close_position(pos.position_id, exit_price=95000.0)
        assert pt.get_position(pos.position_id) is None

    def test_closed_trade_in_history(self):
        """Kapatilan islem gecmis listesine eklenmeli."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        pt.close_position(pos.position_id, exit_price=96000.0)
        assert len(pt.closed_trades()) == 1

    def test_check_exit_conditions_stop_loss(self):
        """check_exit_conditions() SL tetiklenince position_id ve sebep dondur."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001,
                               stop_loss=93000.0)
        exits = pt.check_exit_conditions(current_price=92000.0)
        assert len(exits) == 1
        assert exits[0][1] == "STOP_LOSS"

    def test_check_exit_conditions_no_exit(self):
        """SL/TP tetiklenmiyorsa bos liste donmeli."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001,
                               stop_loss=93000.0, take_profit=98000.0)
        exits = pt.check_exit_conditions(current_price=95500.0)
        assert len(exits) == 0

    def test_performance_summary_no_trades(self):
        """Islem yokken ozet sifir dondurmeli."""
        pt = self._pt()
        s  = pt.performance_summary()
        assert s["total_trades"] == 0
        assert s["win_rate_pct"] == 0.0

    def test_performance_summary_after_trade(self):
        """Islem sonrasi ozet dogru hesaplanmali."""
        pt  = self._pt()
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.01)
        pt.close_position(pos.position_id, exit_price=97000.0)
        s   = pt.performance_summary()
        assert s["total_trades"] == 1
        assert s["win_rate_pct"] == 100.0

    def test_close_all_positions(self):
        """close_all_positions() tum acik pozisyonlari kapatmali."""
        pt = PositionTracker(initial_capital=100_000.0, max_positions=3)
        pt.open_position("BTC/USDT", "LONG",  95000.0, 0.001)
        pt.open_position("BTC/USDT", "SHORT", 95000.0, 0.001)
        trades = pt.close_all_positions(current_price=95000.0, reason="MANUAL")
        assert len(trades) == 2
        assert len(pt.open_positions()) == 0

    def test_has_open_position(self):
        """has_open_position() dogru boolean dondurmeli."""
        pt  = self._pt()
        assert pt.has_open_position() is False
        pos = pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        assert pt.has_open_position() is True

    def test_can_open_position(self):
        """can_open_position() limiti asinca False dondurmeli."""
        pt = PositionTracker(initial_capital=100_000.0, max_positions=1)
        assert pt.can_open_position() is True
        pt.open_position("BTC/USDT", "LONG", 95000.0, 0.001)
        assert pt.can_open_position() is False


# ─────────────────────────────────────────────────────────────────────────────
# TelegramNotifier TESTLERI
# ─────────────────────────────────────────────────────────────────────────────

class TestTelegramNotifier:
    """TelegramNotifier sinifini test eder (dry-run mod)."""

    def _notifier(self) -> TelegramNotifier:
        return TelegramNotifier(dry_run=True, symbol="BTC/USDT")

    def test_dry_run_signal_returns_true(self):
        """Dry-run modda send_signal() True dondurmeli."""
        n = self._notifier()
        result = n.send_signal("AL", price=95000.0, confidence=0.75)
        assert result is True

    def test_dry_run_position_opened_returns_true(self):
        """Dry-run modda send_position_opened() True dondurmeli."""
        n = self._notifier()
        result = n.send_position_opened(
            direction="LONG", price=95000.0, quantity=0.001
        )
        assert result is True

    def test_dry_run_position_closed_returns_true(self):
        """Dry-run modda send_position_closed() True dondurmeli."""
        n = self._notifier()
        result = n.send_position_closed(
            direction="LONG", entry_price=95000.0, exit_price=97000.0,
            quantity=0.001, realized_pnl=20.0, realized_pct=2.0,
        )
        assert result is True

    def test_dry_run_stop_loss_returns_true(self):
        n = self._notifier()
        result = n.send_stop_loss(price=93000.0, sl_price=93000.0, pnl=-10.0)
        assert result is True

    def test_dry_run_take_profit_returns_true(self):
        n = self._notifier()
        result = n.send_take_profit(price=98000.0, tp_price=98000.0, pnl=30.0)
        assert result is True

    def test_dry_run_bot_started_returns_true(self):
        n = self._notifier()
        assert n.send_bot_started(capital=10_000.0, paper=True) is True

    def test_dry_run_bot_stopped_returns_true(self):
        n = self._notifier()
        assert n.send_bot_stopped(total_pnl=150.0, win_rate=60.0) is True

    def test_dry_run_error_returns_true(self):
        n = self._notifier()
        assert n.send_error("Test hatasi", context="test") is True

    def test_dry_run_daily_summary_returns_true(self):
        n = self._notifier()
        result = n.send_daily_summary(
            trades=5, pnl=80.0, win_rate=60.0, capital=10_200.0
        )
        assert result is True

    def test_sent_counter_increments(self):
        """Her basarili gonderimde sayac artmali."""
        n = self._notifier()
        n.send_signal("AL", price=95000.0, confidence=0.7)
        n.send_signal("SAT", price=95000.0, confidence=0.6)
        assert n.stats()["sent"] == 2

    def test_no_token_activates_dry_run(self):
        """Token olmadan dry_run=True olmali."""
        n = TelegramNotifier(token="", chat_id="")
        assert n.dry_run is True

    def test_send_raw(self):
        """send_raw() dry-run modda True dondurmeli."""
        n = self._notifier()
        assert n.send_raw("Test mesaji") is True

    def test_stats_dict_keys(self):
        """stats() beklenen anahtarlara sahip olmali."""
        n = self._notifier()
        s = n.stats()
        assert "sent"    in s
        assert "failed"  in s
        assert "dry_run" in s
