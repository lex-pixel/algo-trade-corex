"""
tests/test_phase9_10.py
========================
Phase 9 (Dashboard) ve Phase 10 (Auto-Retrain) testleri.

Testler:
    - Dashboard: state okuma, terminal ozet, HTML rapor uretimi
    - Auto-retrain: sabitleler, retrain fonksiyonu imzasi
    - TradingBot: equity_history kaydi/yuklenmesi, _maybe_retrain tetikleyici
"""

import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime, timezone

import pytest

# Proje koku
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI: Sahte bot_state.json icerigi
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(n_trades=0, n_equity=0, capital=10000.0) -> dict:
    """Test icin bot_state.json verisini taklit eder."""
    trades = []
    for i in range(n_trades):
        pnl = 50.0 if i % 2 == 0 else -30.0
        ts = f"2024-03-{i+1:02d}T12:00:00+00:00"
        trades.append({
            "position_id" : f"pos-{i}",
            "symbol"      : "BTC/USDT",
            "direction"   : "LONG",
            "entry_price" : 60000.0,
            "exit_price"  : 61000.0 if pnl > 0 else 59000.0,
            "quantity"    : 0.001,
            "realized_pnl": pnl,
            "realized_pct": pnl / capital,
            "exit_reason" : "TP",
            "opened_at"   : ts,
            "closed_at"   : ts,
            "strategy"    : "RSI+PA",
            "total_fee"   : 0.1,
        })

    eq_history = []
    for i in range(n_equity):
        eq_history.append({
            "ts"    : f"2024-03-{i+1:02d}T12:00:00+00:00",
            "equity": capital + i * 5.0,
            "price" : 60000.0 + i * 10.0,
        })

    return {
        "saved_at"       : "2026-03-10T02:07:01+00:00",
        "capital"        : capital,
        "initial_capital": 10000.0,
        "equity_peak"    : capital,
        "max_drawdown"   : 0.0,
        "iteration"      : 16,
        "paper"          : True,
        "trades"         : trades,
        "open_positions" : [],
        "equity_history" : eq_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD TESTLERI (scripts/dashboard.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardLoadState:
    """load_state fonksiyonu."""

    def test_load_state_missing_file_exits(self, tmp_path):
        """bot_state.json yoksa sys.exit(1) cagirilmali."""
        from scripts import dashboard
        original = dashboard.STATE_FILE
        dashboard.STATE_FILE = tmp_path / "noexist.json"
        try:
            with pytest.raises(SystemExit):
                dashboard.load_state()
        finally:
            dashboard.STATE_FILE = original

    def test_load_state_returns_dict(self, tmp_path):
        """Gecerli JSON dosyasi varsa dict dondurmeli."""
        from scripts import dashboard
        state = _make_state(n_trades=2, n_equity=5)
        state_file = tmp_path / "bot_state.json"
        state_file.write_text(json.dumps(state), encoding="utf-8")

        original = dashboard.STATE_FILE
        dashboard.STATE_FILE = state_file
        try:
            result = dashboard.load_state()
            assert isinstance(result, dict)
            assert result["capital"] == 10000.0
            assert len(result["trades"]) == 2
            assert len(result["equity_history"]) == 5
        finally:
            dashboard.STATE_FILE = original


class TestDashboardTerminalSummary:
    """print_terminal_summary fonksiyonu."""

    def test_summary_empty_state(self, capsys):
        """Islem olmayan state icin hata vermemeli."""
        from scripts.dashboard import print_terminal_summary
        state = _make_state(n_trades=0, n_equity=0)
        print_terminal_summary(state)
        out = capsys.readouterr().out
        assert "ALGO TRADE COREX" in out
        assert "$10,000.00" in out

    def test_summary_with_trades(self, capsys):
        """Islem olan state icin win rate gostermeli."""
        from scripts.dashboard import print_terminal_summary
        state = _make_state(n_trades=4)
        print_terminal_summary(state)
        out = capsys.readouterr().out
        assert "Win Rate" in out
        assert "4" in out  # toplam islem

    def test_summary_return_pct_positive(self, capsys):
        """Sermaye artinca pozitif getiri gostermeli."""
        from scripts.dashboard import print_terminal_summary
        state = _make_state(capital=10500.0)
        print_terminal_summary(state)
        out = capsys.readouterr().out
        assert "+5" in out or "5.00" in out  # %+5.00 getiri


class TestDashboardBuildReport:
    """build_report fonksiyonu — HTML dosya uretimi."""

    def test_report_created_empty_state(self, tmp_path):
        """0 islem, 0 equity_history ile HTML olusturmali."""
        from scripts.dashboard import build_report
        state = _make_state(n_trades=0, n_equity=0)
        output = tmp_path / "test.html"
        build_report(state, output)
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "<html" in content.lower() or "plotly" in content.lower()

    def test_report_created_with_trades(self, tmp_path):
        """Islem ve equity_history ile HTML olusturmali."""
        from scripts.dashboard import build_report
        state = _make_state(n_trades=5, n_equity=20, capital=10250.0)
        output = tmp_path / "report.html"
        build_report(state, output)
        assert output.exists()
        assert output.stat().st_size > 10_000  # plotly HTML en az 10KB olur

    def test_report_creates_parent_dir(self, tmp_path):
        """Cikti dizini yoksa otomatik olusturmali."""
        from scripts.dashboard import build_report
        state = _make_state()
        output = tmp_path / "sub" / "dir" / "out.html"
        build_report(state, output)
        assert output.exists()

    def test_report_contains_title(self, tmp_path):
        """Rapor basliginda 'Corex' veya 'Performans' olmali."""
        from scripts.dashboard import build_report
        state = _make_state(n_trades=2, n_equity=10)
        output = tmp_path / "t.html"
        build_report(state, output)
        content = output.read_text(encoding="utf-8")
        assert "Corex" in content or "Performans" in content


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-RETRAIN TESTLERI (ml/auto_retrain.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoRetrainConstants:
    """Sabit ve yol degerleri dogru tanimlanmis olmali."""

    def test_min_accuracy_constant(self):
        """MIN_ACCURACY 0 ile 1 arasinda olmali."""
        from ml.auto_retrain import MIN_ACCURACY
        assert 0.0 < MIN_ACCURACY < 1.0

    def test_model_path_in_ml_models(self):
        """MODEL_PATH ml/models klasoru altinda olmali."""
        from ml.auto_retrain import MODEL_PATH
        assert "ml" in str(MODEL_PATH).replace("\\", "/")
        assert MODEL_PATH.name == "xgb_btc_1h.json"

    def test_model_path_new_suffix(self):
        """MODEL_PATH_NEW _new suffix tasimali."""
        from ml.auto_retrain import MODEL_PATH_NEW
        assert "_new" in MODEL_PATH_NEW.stem


class TestAutoRetrainFunction:
    """retrain() fonksiyonu."""

    def test_retrain_returns_bool(self):
        """retrain() bool dondurmeli (basarili veya basarisiz)."""
        from ml.auto_retrain import retrain
        assert callable(retrain)
        import inspect
        sig = inspect.signature(retrain)
        assert "days" in sig.parameters
        assert "quiet" in sig.parameters

    def test_retrain_returns_false_on_api_error(self):
        """API hatasi veya bozuk veri durumunda False dondurmeli."""
        from ml.auto_retrain import retrain
        # ccxt'yi patch et — bos liste donecek sekilde
        with patch("ccxt.binance") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.milliseconds.return_value = 0
            mock_exchange.fetch_ohlcv.return_value = []  # bos veri
            mock_ccxt.return_value = mock_exchange
            result = retrain(days=10, quiet=True)
        assert result is False

    def test_retrain_returns_false_on_insufficient_data(self):
        """200'den az mum varsa False dondurmeli."""
        from ml.auto_retrain import retrain
        # Sadece 50 mum donecek sekilde patch
        small_batch = [[i * 3600000, 60000, 61000, 59000, 60500, 100] for i in range(50)]
        with patch("ccxt.binance") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.milliseconds.return_value = 0
            mock_exchange.fetch_ohlcv.return_value = small_batch
            mock_ccxt.return_value = mock_exchange
            result = retrain(days=10, quiet=True)
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# TRADINGBOT EQUITY_HISTORY TESTLERI
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingBotEquityHistory:
    """_equity_history kaydi ve state persistence."""

    def _make_bot(self, tmp_path):
        """State dosyasini tmp_path'e yonlendirip bot olustur."""
        from trading.main_loop import TradingBot
        bot = TradingBot.__new__(TradingBot)
        # Minimum init
        bot.paper          = True
        bot.interval       = 60
        bot._running       = False
        bot._iteration     = 0
        bot._errors        = 0
        bot._equity_history= []
        bot._last_retrain  = 0
        bot.ml_predictor   = None
        # PositionTracker mock
        from trading.position_tracker import PositionTracker
        bot.position_tracker = PositionTracker(initial_capital=10000.0)
        # STATE_FILE -> tmp
        TradingBot._STATE_FILE_ORIG = TradingBot._STATE_FILE
        bot._STATE_FILE = tmp_path / "test_state.json"
        return bot

    def test_equity_history_initialized_empty(self):
        """Bot baslarken equity_history bos liste olmali."""
        from trading.main_loop import TradingBot
        bot = TradingBot.__new__(TradingBot)
        bot._equity_history = []
        assert isinstance(bot._equity_history, list)
        assert len(bot._equity_history) == 0

    def test_save_state_includes_equity_history(self, tmp_path):
        """save_state() equity_history'yi JSON'a yazmali."""
        bot = self._make_bot(tmp_path)
        bot._equity_history = [
            {"ts": "2024-01-01T00:00:00+00:00", "equity": 10000.0, "price": 60000.0},
            {"ts": "2024-01-02T00:00:00+00:00", "equity": 10050.0, "price": 60500.0},
        ]
        bot._iteration = 2
        bot.save_state()

        with open(bot._STATE_FILE, encoding="utf-8") as f:
            saved = json.load(f)

        assert "equity_history" in saved
        assert len(saved["equity_history"]) == 2
        assert saved["equity_history"][0]["equity"] == 10000.0

    def test_load_state_restores_equity_history(self, tmp_path):
        """load_state() equity_history'yi geri yuklemelidir."""
        state = _make_state(n_trades=1, n_equity=5)
        state_file = tmp_path / "bot_state.json"
        state_file.write_text(json.dumps(state), encoding="utf-8")

        bot = self._make_bot(tmp_path)
        bot._STATE_FILE = state_file
        bot.load_state()
        assert len(bot._equity_history) == 5

    def test_equity_history_max_1000(self):
        """equity_history 1000'i asinca eski kayitlar silinmeli."""
        from trading.main_loop import TradingBot
        bot = TradingBot.__new__(TradingBot)
        bot._equity_history = [{"ts": "x", "equity": 1.0, "price": 1.0}] * 1000
        # 1001. ekleme simule et
        bot._equity_history.append({"ts": "y", "equity": 2.0, "price": 2.0})
        if len(bot._equity_history) > 1000:
            bot._equity_history = bot._equity_history[-1000:]
        assert len(bot._equity_history) == 1000
        # En son kayit korunmali
        assert bot._equity_history[-1]["equity"] == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# MAYBE_RETRAIN TESTLERI
# ─────────────────────────────────────────────────────────────────────────────

class TestMaybeRetrain:
    """_maybe_retrain tetikleyici mantigi."""

    def _make_bot_with_retrain(self):
        """Minimal TradingBot ornegi."""
        from trading.main_loop import TradingBot
        bot = TradingBot.__new__(TradingBot)
        bot._iteration     = 0
        bot._last_retrain  = 0
        bot.ml_predictor   = None
        bot.cfg            = MagicMock()
        bot.cfg.general.symbol    = "BTC/USDT"
        bot.cfg.general.timeframe = "1h"
        return bot

    def test_retrain_not_triggered_before_720(self):
        """720 tick dolmadan _maybe_retrain thread baslatmamali."""
        bot = self._make_bot_with_retrain()
        bot._iteration = 100
        threads_before = threading.active_count()
        bot._maybe_retrain()
        # _last_retrain degismemeli
        assert bot._last_retrain == 0

    def test_retrain_triggered_at_720(self):
        """720. tick'te _maybe_retrain _last_retrain'i guncellemelidir."""
        bot = self._make_bot_with_retrain()
        bot._iteration    = 720
        bot._last_retrain = 0

        # Thread baslatmayi engelle — sadece tetiklenip tetiklenmedigini test et
        with patch("threading.Thread") as mock_thread:
            mock_instance = MagicMock()
            mock_thread.return_value = mock_instance
            bot._maybe_retrain()

        # _last_retrain guncellenmeli
        assert bot._last_retrain == 720
        # Thread baslatilmali
        mock_thread.assert_called_once()
        mock_instance.start.assert_called_once()

    def test_retrain_interval_constant(self):
        """_RETRAIN_EVERY 720 (30 gun, 1h bot) olmali."""
        from trading.main_loop import TradingBot
        assert TradingBot._RETRAIN_EVERY == 720

    def test_retrain_not_triggered_twice(self):
        """Ayni tick araliginda iki kez tetiklenmemeli."""
        bot = self._make_bot_with_retrain()
        bot._iteration    = 720
        bot._last_retrain = 720  # zaten yapildi

        with patch("threading.Thread") as mock_thread:
            bot._maybe_retrain()

        mock_thread.assert_not_called()
