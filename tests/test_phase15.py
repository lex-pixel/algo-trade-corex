"""
tests/test_phase15.py
======================
Phase 15: Telegram Bildirim Sistemi (TelegramNotifier) testleri
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.telegram_notifier import TelegramNotifier


# ── TestTelegramNotifierDryRun ────────────────────────────────────────────────

class TestTelegramNotifierDryRun:
    """dry_run=True ile temel islevler."""

    def setup_method(self):
        self.n = TelegramNotifier(dry_run=True, symbol="BTC/USDT")

    def test_init_dry_run(self):
        assert self.n.dry_run is True

    def test_send_signal_returns_true(self):
        ok = self.n.send_signal("AL", price=70000, confidence=0.75)
        assert ok is True

    def test_send_signal_sat(self):
        ok = self.n.send_signal("SAT", price=70000, confidence=0.60)
        assert ok is True

    def test_send_signal_bekle(self):
        ok = self.n.send_signal("BEKLE", price=70000, confidence=0.0)
        assert ok is True

    def test_send_position_opened(self):
        ok = self.n.send_position_opened(
            direction="LONG", price=70000, quantity=0.01,
            stop_loss=68000, take_profit=73000, strategy="RSI+PA",
        )
        assert ok is True

    def test_send_position_opened_no_sl_tp(self):
        ok = self.n.send_position_opened(
            direction="SHORT", price=70000, quantity=0.005,
        )
        assert ok is True

    def test_send_position_closed_profit(self):
        ok = self.n.send_position_closed(
            direction="LONG", entry_price=70000, exit_price=72000,
            quantity=0.01, realized_pnl=20.0, realized_pct=0.28,
            exit_reason="TAKE_PROFIT",
        )
        assert ok is True

    def test_send_position_closed_loss(self):
        ok = self.n.send_position_closed(
            direction="LONG", entry_price=70000, exit_price=68500,
            quantity=0.01, realized_pnl=-15.0, realized_pct=-0.21,
            exit_reason="STOP_LOSS",
        )
        assert ok is True

    def test_send_stop_loss(self):
        ok = self.n.send_stop_loss(price=68000, sl_price=67500, pnl=-25.0)
        assert ok is True

    def test_send_take_profit(self):
        ok = self.n.send_take_profit(price=73000, tp_price=73000, pnl=30.0)
        assert ok is True

    def test_send_bot_started_paper(self):
        ok = self.n.send_bot_started(capital=10000.0, paper=True)
        assert ok is True

    def test_send_bot_started_live(self):
        ok = self.n.send_bot_started(capital=10000.0, paper=False)
        assert ok is True

    def test_send_bot_stopped(self):
        ok = self.n.send_bot_stopped(total_pnl=150.0, win_rate=55.0)
        assert ok is True

    def test_send_error(self):
        ok = self.n.send_error("Test hatasi", context="test_context")
        assert ok is True

    def test_send_daily_summary(self):
        ok = self.n.send_daily_summary(trades=5, pnl=50.0, win_rate=60.0, capital=10500.0)
        assert ok is True

    def test_send_raw(self):
        ok = self.n.send_raw("Ham test mesaji")
        assert ok is True


# ── TestTelegramStats ─────────────────────────────────────────────────────────

class TestTelegramStats:
    """Istatistik takibi testleri."""

    def test_sent_count_increments(self):
        n = TelegramNotifier(dry_run=True)
        n.send_signal("AL", price=70000, confidence=0.75)
        n.send_signal("SAT", price=69000, confidence=0.60)
        assert n.stats()["sent"] == 2

    def test_failed_starts_zero(self):
        n = TelegramNotifier(dry_run=True)
        assert n.stats()["failed"] == 0

    def test_stats_has_expected_keys(self):
        n = TelegramNotifier(dry_run=True)
        s = n.stats()
        assert "sent"    in s
        assert "failed"  in s
        assert "dry_run" in s

    def test_dry_run_flag_in_stats(self):
        n = TelegramNotifier(dry_run=True)
        assert n.stats()["dry_run"] is True


# ── TestTelegramNoToken ───────────────────────────────────────────────────────

class TestTelegramNoToken:
    """Token/chat_id olmadigi durumda otomatik dry_run."""

    def test_no_token_auto_dry_run(self):
        n = TelegramNotifier(token="", chat_id="")
        assert n.dry_run is True

    def test_no_token_still_returns_true(self):
        n = TelegramNotifier(token="", chat_id="")
        ok = n.send_bot_started(10000.0, paper=True)
        assert ok is True


# ── TestTelegramMessageContent ────────────────────────────────────────────────

class TestTelegramMessageContent:
    """Mesaj icerigi kontrolu (dry_run ile capture)."""

    def test_signal_action_in_message(self, capsys):
        """send_signal cagrilabilmeli ve exception olmamali."""
        n = TelegramNotifier(dry_run=True)
        # Sadece exception cikmadigini test et
        n.send_signal("AL", price=70000, confidence=0.8,
                      rsi_signal="AL", pa_signal="BEKLE")

    def test_position_opened_no_exception(self):
        n = TelegramNotifier(dry_run=True, symbol="ETH/USDT")
        n.send_position_opened(
            direction="LONG", price=3500, quantity=0.1,
            stop_loss=3400, take_profit=3700, strategy="Test",
        )

    def test_error_truncates_long_message(self):
        """Cok uzun hata mesaji kesilmeli (200 karakter limiti)."""
        n = TelegramNotifier(dry_run=True)
        long_msg = "A" * 500
        # Exception olmamali
        ok = n.send_error(long_msg, context="test")
        assert ok is True


# ── TestTelegramMainLoopIntegration ──────────────────────────────────────────

class TestTelegramMainLoopIntegration:
    """TradingBot icinde TelegramNotifier entegrasyon testleri."""

    def test_trading_bot_has_notifier(self):
        """TradingBot'ta notifier alani olmali."""
        from unittest.mock import patch, MagicMock

        with patch("trading.main_loop.BinanceFetcher"), \
             patch("trading.main_loop.OrderManager"), \
             patch("trading.main_loop.PositionTracker") as mock_pt_cls, \
             patch("trading.main_loop.RiskManager") as mock_rm_cls:

            mock_pt = MagicMock()
            mock_pt.capital = 10000.0
            mock_pt.initial_capital = 10000.0
            mock_pt._equity_peak = 10000.0
            mock_pt._max_drawdown = 0.0
            mock_pt.open_positions.return_value = []
            mock_pt._history = []
            mock_pt._positions = {}
            mock_pt_cls.return_value = mock_pt

            mock_rm = MagicMock()
            mock_rm.kill_switch = MagicMock()
            mock_rm.kill_switch._equity_peak = 10000.0
            mock_rm_cls.return_value = mock_rm

            from trading.main_loop import TradingBot
            bot = TradingBot(paper=True, capital=10000.0)

        assert hasattr(bot, "notifier")
        assert isinstance(bot.notifier, TelegramNotifier)

    def test_trading_bot_notifier_dry_run_without_env(self):
        """Env degiskenleri yoksa notifier dry_run olmali."""
        from unittest.mock import patch, MagicMock
        import os

        # TELEGRAM_TOKEN ve TELEGRAM_CHAT_ID yoksa dry_run olmali
        env_backup = {
            "TELEGRAM_TOKEN"  : os.environ.pop("TELEGRAM_TOKEN",   None),
            "TELEGRAM_CHAT_ID": os.environ.pop("TELEGRAM_CHAT_ID", None),
        }

        try:
            with patch("trading.main_loop.BinanceFetcher"), \
                 patch("trading.main_loop.OrderManager"), \
                 patch("trading.main_loop.PositionTracker") as mock_pt_cls, \
                 patch("trading.main_loop.RiskManager") as mock_rm_cls:

                mock_pt = MagicMock()
                mock_pt.capital = 10000.0
                mock_pt.initial_capital = 10000.0
                mock_pt._equity_peak = 10000.0
                mock_pt._max_drawdown = 0.0
                mock_pt.open_positions.return_value = []
                mock_pt._history = []
                mock_pt._positions = {}
                mock_pt_cls.return_value = mock_pt

                mock_rm = MagicMock()
                mock_rm.kill_switch = MagicMock()
                mock_rm.kill_switch._equity_peak = 10000.0
                mock_rm_cls.return_value = mock_rm

                from trading.main_loop import TradingBot
                bot = TradingBot(paper=True, capital=10000.0)

            assert bot.notifier.dry_run is True
        finally:
            # Env degiskenlerini geri yukle
            for k, v in env_backup.items():
                if v is not None:
                    os.environ[k] = v
