"""
tests/test_phase20.py
======================
Phase 20: Coklu Coin (MultiCoinBot) testleri
"""

import sys
import json
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.multi_coin_bot import CoinConfig, CoinWorker, MultiCoinBot


# ── Yardimci ──────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 200, start_price: float = 70000.0) -> pd.DataFrame:
    """Test icin sahte OHLCV DataFrame uretir."""
    rng = np.random.default_rng(42)
    closes = start_price + np.cumsum(rng.normal(0, 200, n))
    highs  = closes + rng.uniform(50, 300, n)
    lows   = closes - rng.uniform(50, 300, n)
    opens  = closes + rng.normal(0, 100, n)
    vols   = rng.uniform(100, 500, n)
    idx    = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    }, index=idx)


# ── TestCoinConfig ─────────────────────────────────────────────────────────

class TestCoinConfig:
    """CoinConfig dataclass testleri."""

    def test_default_values(self):
        c = CoinConfig(symbol="BTC/USDT")
        assert c.timeframe  == "1h"
        assert c.alloc_pct  == 1.0
        assert c.model_path is None

    def test_custom_values(self):
        c = CoinConfig(symbol="ETH/USDT", timeframe="4h", alloc_pct=0.3)
        assert c.symbol    == "ETH/USDT"
        assert c.timeframe == "4h"
        assert c.alloc_pct == 0.3

    def test_multiple_configs(self):
        coins = [
            CoinConfig(symbol="BTC/USDT", alloc_pct=0.5),
            CoinConfig(symbol="ETH/USDT", alloc_pct=0.3),
            CoinConfig(symbol="SOL/USDT", alloc_pct=0.2),
        ]
        total = sum(c.alloc_pct for c in coins)
        assert abs(total - 1.0) < 0.001


# ── TestCoinWorkerInit ─────────────────────────────────────────────────────

class TestCoinWorkerInit:
    """CoinWorker baslangic testleri (network gerektirmez)."""

    @patch("trading.multi_coin_bot.BinanceFetcher")
    @patch("trading.multi_coin_bot.OrderManager")
    @patch("trading.multi_coin_bot.PositionTracker")
    @patch("trading.multi_coin_bot.RiskManager")
    def test_worker_init_btc(self, mock_rm, mock_pt, mock_om, mock_bf):
        """BTC/USDT worker olusturulabilmeli."""
        # PositionTracker mock: open_positions bos liste donduruyor
        mock_pt_inst = MagicMock()
        mock_pt_inst.open_positions.return_value = []
        mock_pt_inst._history = []
        mock_pt_inst.capital  = 5000.0
        mock_pt_inst.initial_capital = 5000.0
        mock_pt_inst._equity_peak = 5000.0
        mock_pt_inst._max_drawdown = 0.0
        mock_pt.return_value = mock_pt_inst

        cfg = CoinConfig(symbol="BTC/USDT", alloc_pct=0.5)
        w   = CoinWorker(coin_cfg=cfg, capital=5000.0, paper=True, testnet=True)

        assert w.symbol    == "BTC/USDT"
        assert w.capital   == 5000.0
        assert w.paper     is True
        assert w._iteration == 0

    @patch("trading.multi_coin_bot.BinanceFetcher")
    @patch("trading.multi_coin_bot.OrderManager")
    @patch("trading.multi_coin_bot.PositionTracker")
    @patch("trading.multi_coin_bot.RiskManager")
    def test_worker_state_file_naming(self, mock_rm, mock_pt, mock_om, mock_bf):
        """State dosyasi sembol adina gore olusturulmali."""
        mock_pt_inst = MagicMock()
        mock_pt_inst.open_positions.return_value = []
        mock_pt_inst._history = []
        mock_pt_inst.capital  = 3000.0
        mock_pt_inst.initial_capital = 3000.0
        mock_pt_inst._equity_peak = 3000.0
        mock_pt_inst._max_drawdown = 0.0
        mock_pt.return_value = mock_pt_inst

        cfg = CoinConfig(symbol="ETH/USDT")
        w   = CoinWorker(coin_cfg=cfg, capital=3000.0, paper=True, testnet=True)
        assert "ETH_USDT" in str(w.state_file)

    @patch("trading.multi_coin_bot.BinanceFetcher")
    @patch("trading.multi_coin_bot.OrderManager")
    @patch("trading.multi_coin_bot.PositionTracker")
    @patch("trading.multi_coin_bot.RiskManager")
    def test_worker_sol_state_file(self, mock_rm, mock_pt, mock_om, mock_bf):
        """SOL/USDT icin dosya adi ETH ile karismasin."""
        mock_pt_inst = MagicMock()
        mock_pt_inst.open_positions.return_value = []
        mock_pt_inst._history = []
        mock_pt_inst.capital  = 2000.0
        mock_pt_inst.initial_capital = 2000.0
        mock_pt_inst._equity_peak = 2000.0
        mock_pt_inst._max_drawdown = 0.0
        mock_pt.return_value = mock_pt_inst

        cfg = CoinConfig(symbol="SOL/USDT")
        w   = CoinWorker(coin_cfg=cfg, capital=2000.0, paper=True, testnet=True)
        assert "SOL_USDT" in str(w.state_file)
        assert "ETH_USDT" not in str(w.state_file)


# ── TestCoinWorkerSaveLoad ─────────────────────────────────────────────────

class TestCoinWorkerSaveLoad:
    """State kayit/yukle testleri."""

    def _make_worker(self, symbol="BTC/USDT", capital=5000.0):
        """Mock worker olusturur."""
        with patch("trading.multi_coin_bot.BinanceFetcher"), \
             patch("trading.multi_coin_bot.OrderManager"):

            mock_pt = MagicMock()
            mock_pt.open_positions.return_value = []
            mock_pt._history = []
            mock_pt.capital  = capital
            mock_pt.initial_capital = capital
            mock_pt._equity_peak = capital
            mock_pt._max_drawdown = 0.0

            with patch("trading.multi_coin_bot.PositionTracker", return_value=mock_pt), \
                 patch("trading.multi_coin_bot.RiskManager"):
                cfg = CoinConfig(symbol=symbol)
                w   = CoinWorker(coin_cfg=cfg, capital=capital, paper=True, testnet=True)
                return w

    def test_save_creates_file(self):
        """save_state() JSON dosyasi olusturmali."""
        w = self._make_worker()
        with tempfile.TemporaryDirectory() as tmp:
            w.state_file = Path(tmp) / "test_state.json"
            w.save_state()
            assert w.state_file.exists()

    def test_save_json_structure(self):
        """JSON'da gerekli alanlar olmali."""
        w = self._make_worker()
        with tempfile.TemporaryDirectory() as tmp:
            w.state_file = Path(tmp) / "test_state.json"
            w.save_state()
            with open(w.state_file) as f:
                data = json.load(f)
        assert "symbol"          in data
        assert "capital"         in data
        assert "initial_capital" in data
        assert "trades"          in data
        assert "equity_history"  in data

    def test_save_symbol_in_json(self):
        """JSON'da symbol alani dogru olmali."""
        w = self._make_worker(symbol="ETH/USDT")
        with tempfile.TemporaryDirectory() as tmp:
            w.state_file = Path(tmp) / "test_state.json"
            w.save_state()
            with open(w.state_file) as f:
                data = json.load(f)
        assert data["symbol"] == "ETH/USDT"

    def test_load_state_returns_false_no_file(self):
        """State dosyasi yoksa load_state() False donmeli."""
        w = self._make_worker()
        w.state_file = Path("/nonexistent/path/state.json")
        assert w.load_state() is False

    def test_equity_history_saved(self):
        """equity_history kaydedilmeli."""
        w = self._make_worker()
        w._equity_history = [
            {"ts": "2026-01-01T00:00:00", "equity": 5000.0, "price": 70000.0},
            {"ts": "2026-01-01T01:00:00", "equity": 5050.0, "price": 71000.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            w.state_file = Path(tmp) / "test_state.json"
            w.save_state()
            with open(w.state_file) as f:
                data = json.load(f)
        assert len(data["equity_history"]) == 2


# ── TestMultiCoinBotInit ───────────────────────────────────────────────────

def _make_mock_worker(capital: float = 5000.0, symbol: str = "BTC/USDT") -> MagicMock:
    """Format-safe CoinWorker mock uretir."""
    pt = MagicMock()
    pt._history = []
    pt.open_positions.return_value = []
    pt.capital = capital

    m = MagicMock()
    m.symbol           = symbol
    m.capital          = capital
    m._iteration       = 0
    m.position_tracker = pt
    return m


class TestMultiCoinBotInit:
    """MultiCoinBot portfoy dagitimi testleri."""

    @patch("trading.multi_coin_bot.CoinWorker")
    def test_equal_allocation(self, mock_worker):
        """Esit agirlik: 2 coin -> her biri %50."""
        mock_worker.side_effect = [
            _make_mock_worker(5000.0, "BTC/USDT"),
            _make_mock_worker(5000.0, "ETH/USDT"),
        ]
        coins = [
            CoinConfig(symbol="BTC/USDT", alloc_pct=1.0),
            CoinConfig(symbol="ETH/USDT", alloc_pct=1.0),
        ]
        bot = MultiCoinBot(coins=coins, capital=10_000.0)
        assert len(bot.workers) == 2
        # Her worker $5000 ile cagrilmali
        capitals = [c.kwargs.get("capital") for c in mock_worker.call_args_list]
        assert all(abs(c - 5000.0) < 1.0 for c in capitals)

    @patch("trading.multi_coin_bot.CoinWorker")
    def test_custom_allocation(self, mock_worker):
        """Ozel agirlik: BTC %60, ETH %40."""
        mock_worker.side_effect = [
            _make_mock_worker(6000.0, "BTC/USDT"),
            _make_mock_worker(4000.0, "ETH/USDT"),
        ]
        coins = [
            CoinConfig(symbol="BTC/USDT", alloc_pct=0.6),
            CoinConfig(symbol="ETH/USDT", alloc_pct=0.4),
        ]
        bot = MultiCoinBot(coins=coins, capital=10_000.0)
        capitals = [c.kwargs.get("capital") for c in mock_worker.call_args_list]
        assert abs(capitals[0] - 6000.0) < 1.0
        assert abs(capitals[1] - 4000.0) < 1.0

    @patch("trading.multi_coin_bot.CoinWorker")
    def test_three_coins(self, mock_worker):
        """3 coin ile calisabilmeli."""
        mock_worker.side_effect = [
            _make_mock_worker(5000.0, "BTC/USDT"),
            _make_mock_worker(3000.0, "ETH/USDT"),
            _make_mock_worker(2000.0, "SOL/USDT"),
        ]
        coins = [
            CoinConfig(symbol="BTC/USDT", alloc_pct=0.5),
            CoinConfig(symbol="ETH/USDT", alloc_pct=0.3),
            CoinConfig(symbol="SOL/USDT", alloc_pct=0.2),
        ]
        bot = MultiCoinBot(coins=coins, capital=10_000.0)
        assert len(bot.workers) == 3

    @patch("trading.multi_coin_bot.CoinWorker")
    def test_worker_count_matches_coins(self, mock_worker):
        """Worker sayisi coin sayisina esit olmali."""
        for n in [1, 2, 3]:
            mock_worker.side_effect = [
                _make_mock_worker(10_000.0 / n, f"COIN{i}/USDT")
                for i in range(n)
            ]
            coins = [CoinConfig(symbol=f"COIN{i}/USDT") for i in range(n)]
            bot = MultiCoinBot(coins=coins, capital=10_000.0)
            assert len(bot.workers) == n


# ── TestPortfolioState ─────────────────────────────────────────────────────

class TestPortfolioState:
    """Portfoy durum raporu testleri."""

    @patch("trading.multi_coin_bot.CoinWorker")
    def test_portfolio_state_has_coins(self, mock_worker):
        """portfolio_state() coins listesi icermeli."""
        mock_worker.side_effect = [_make_mock_worker(5000.0, "BTC/USDT")]
        coins = [CoinConfig(symbol="BTC/USDT")]
        bot   = MultiCoinBot(coins=coins, capital=5000.0)
        state = bot.portfolio_state()

        assert "coins"        in state
        assert "total_equity" in state
        assert "generated_at" in state

    @patch("trading.multi_coin_bot.CoinWorker")
    def test_portfolio_total_equity_sum(self, mock_worker):
        """total_equity coin equity'lerinin toplami olmali."""
        w1 = _make_mock_worker(5000.0, "BTC/USDT")
        w2 = _make_mock_worker(3000.0, "ETH/USDT")
        mock_worker.side_effect = [w1, w2]

        coins = [
            CoinConfig(symbol="BTC/USDT", alloc_pct=0.5),
            CoinConfig(symbol="ETH/USDT", alloc_pct=0.5),
        ]
        bot = MultiCoinBot(coins=coins, capital=8000.0)

        state = bot.portfolio_state()
        assert state["total_equity"] == pytest.approx(8000.0)


# ── TestCoinWorkerSignal ────────────────────────────────────────────────────

class TestCoinWorkerSignal:
    """_get_signal() fonksiyon testleri."""

    def _make_worker_for_signal(self):
        """Sinyal testleri icin basit worker mock."""
        with patch("trading.multi_coin_bot.BinanceFetcher"), \
             patch("trading.multi_coin_bot.OrderManager"):
            mock_pt = MagicMock()
            mock_pt.open_positions.return_value = []
            mock_pt._history = []
            mock_pt.capital  = 5000.0
            mock_pt.initial_capital = 5000.0
            mock_pt._equity_peak = 5000.0
            mock_pt._max_drawdown = 0.0
            with patch("trading.multi_coin_bot.PositionTracker", return_value=mock_pt), \
                 patch("trading.multi_coin_bot.RiskManager"):
                cfg = CoinConfig(symbol="BTC/USDT")
                return CoinWorker(cfg, capital=5000.0, paper=True, testnet=True)

    def test_get_signal_returns_tuple(self):
        """_get_signal() 5-tuple donmeli."""
        from strategies.regime_detector import Regime
        w   = self._make_worker_for_signal()
        df  = make_ohlcv(200)
        res = w._get_signal(df, Regime.RANGE, None, None)
        assert isinstance(res, tuple)
        assert len(res) == 5

    def test_get_signal_action_valid(self):
        """Sinyal AL, SAT veya BEKLE olmali."""
        from strategies.regime_detector import Regime
        w      = self._make_worker_for_signal()
        df     = make_ohlcv(200)
        action = w._get_signal(df, Regime.RANGE, None, None)[0]
        assert action in ["AL", "SAT", "BEKLE"]

    def test_get_signal_confidence_range(self):
        """Guven 0-1 araliginda olmali."""
        from strategies.regime_detector import Regime
        w    = self._make_worker_for_signal()
        df   = make_ohlcv(200)
        conf = w._get_signal(df, Regime.RANGE, None, None)[1]
        assert 0.0 <= conf <= 1.0
