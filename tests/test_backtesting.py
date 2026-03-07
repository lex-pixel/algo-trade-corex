"""
tests/test_backtesting.py
==========================
AMACI:
    BacktestEngine ve PerformanceMetrics sınıflarını test eder.
    Gerçek API gerektirmez — sahte veri kullanır.

ÇALIŞTIRMAK İÇİN:
    pytest tests/test_backtesting.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from backtesting.engine import BacktestEngine, Trade, BacktestResult
from backtesting.metrics import PerformanceMetrics
from strategies.rsi_strategy import RSIStrategy
from strategies.pa_range_strategy import PARangeStrategy


# ─────────────────────────────────────────────────────────────────────────────
# ORTAK YARDIMCILAR
# ─────────────────────────────────────────────────────────────────────────────

def make_df(n: int = 200, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    np.random.seed(seed)
    closes = [50000.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + np.random.uniform(-0.006, 0.006)))
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [300.0] * n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))


def make_trade(pnl: float = 100.0, pnl_pct: float = 1.0, winner: bool = True) -> Trade:
    """Test için sahte Trade objesi üretir."""
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return Trade(
        entry_time  = t,
        exit_time   = t,
        entry_price = 50000.0,
        exit_price  = 50000.0 + (500 if winner else -500),
        direction   = "LONG",
        size        = 0.001,
        pnl         = pnl if winner else -abs(pnl),
        pnl_pct     = pnl_pct if winner else -abs(pnl_pct),
        exit_reason = "SIGNAL",
    )


# ─────────────────────────────────────────────────────────────────────────────
# BacktestEngine TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestEngine:
    """BacktestEngine davranışını test eder."""

    def _engine(self) -> BacktestEngine:
        return BacktestEngine(
            initial_capital    = 10_000.0,
            commission         = 0.001,
            slippage           = 0.0005,
            max_risk_per_trade = 0.02,
        )

    def _rsi_strategy(self) -> RSIStrategy:
        return RSIStrategy(symbol="BTC/USDT", timeframe="1h")

    def test_returns_backtest_result(self):
        """run() BacktestResult döndürmeli."""
        engine = self._engine()
        df     = make_df(200)
        result = engine.run(df, self._rsi_strategy())
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length(self):
        """Equity curve warmup_bars sonrası tüm barları kapsamalı."""
        engine    = self._engine()
        df        = make_df(200)
        warmup    = 50
        result    = engine.run(df, self._rsi_strategy(), warmup_bars=warmup)
        expected  = len(df) - warmup
        assert len(result.equity_curve) == expected

    def test_final_capital_positive(self):
        """Final sermaye pozitif olmalı (strateji ne kadar kötü olursa olsun)."""
        engine = self._engine()
        df     = make_df(200)
        result = engine.run(df, self._rsi_strategy())
        assert result.final_capital > 0

    def test_initial_capital_preserved_in_result(self):
        """BacktestResult içinde initial_capital korunmalı."""
        engine = self._engine()
        df     = make_df(200)
        result = engine.run(df, self._rsi_strategy())
        assert result.initial_capital == 10_000.0

    def test_trades_list_is_list(self):
        """Trades bir liste olmalı."""
        engine = self._engine()
        df     = make_df(200)
        result = engine.run(df, self._rsi_strategy())
        assert isinstance(result.trades, list)

    def test_all_trades_have_valid_direction(self):
        """Tüm işlemler geçerli direction değerine sahip olmalı."""
        engine = self._engine()
        df     = make_df(300)
        result = engine.run(df, self._rsi_strategy())
        for trade in result.trades:
            assert trade.direction in {"LONG", "SHORT"}

    def test_all_exit_reasons_valid(self):
        """Tüm exit_reason değerleri geçerli olmalı."""
        valid  = {"SIGNAL", "STOP_LOSS", "TAKE_PROFIT", "END_OF_DATA"}
        engine = self._engine()
        df     = make_df(300)
        result = engine.run(df, self._rsi_strategy())
        for trade in result.trades:
            assert trade.exit_reason in valid

    def test_insufficient_data_returns_empty_result(self):
        """Yetersiz veriyle boş sonuç döndürmeli."""
        engine = self._engine()
        df     = make_df(10)
        result = engine.run(df, self._rsi_strategy(), warmup_bars=50)
        assert result.final_capital == result.initial_capital
        assert len(result.trades) == 0

    def test_pa_range_strategy_runs(self):
        """PA Range stratejisi de backtest'te çalışabilmeli."""
        engine   = self._engine()
        df       = make_df(300)
        strategy = PARangeStrategy(use_regime_filter=False)
        result   = engine.run(df, strategy)
        assert isinstance(result, BacktestResult)

    def test_commission_reduces_profit(self):
        """Komisyon olmadan getiri daha yüksek olmalı."""
        df       = make_df(300, seed=10)
        strategy = RSIStrategy()

        engine_with    = BacktestEngine(initial_capital=10_000, commission=0.001)
        engine_without = BacktestEngine(initial_capital=10_000, commission=0.0)

        result_with    = engine_with.run(df, strategy)
        result_without = engine_without.run(df, strategy)

        # Komisyon olan final sermaye <= komisyonsuz final sermaye
        assert result_with.final_capital <= result_without.final_capital


# ─────────────────────────────────────────────────────────────────────────────
# Trade SINIFI TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestTrade:
    """Trade dataclass'ının özelliklerini test eder."""

    def test_is_winner_true_for_positive_pnl(self):
        assert make_trade(pnl=100, winner=True).is_winner is True

    def test_is_winner_false_for_negative_pnl(self):
        assert make_trade(pnl=-100, winner=False).is_winner is False

    def test_duration_hours_positive(self):
        t1 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc)
        trade = Trade(t1, t2, 50000, 51000, "LONG", 0.001, 100, 2.0, "SIGNAL")
        assert trade.duration_hours == pytest.approx(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceMetrics TESTLERİ
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceMetrics:
    """PerformanceMetrics hesaplamalarını test eder."""

    def _make_result(self, trades: list, initial: float = 10_000.0) -> BacktestResult:
        """Test için BacktestResult objesi üretir."""
        final = initial + sum(t.pnl for t in trades)
        equity = pd.Series(
            [initial + i * 10 for i in range(100)],
            index=pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        )
        return BacktestResult(
            strategy_name   = "Test",
            symbol          = "BTC/USDT",
            timeframe       = "1h",
            start_date      = "2024-01-01",
            end_date        = "2024-04-10",
            initial_capital = initial,
            final_capital   = final,
            trades          = trades,
            equity_curve    = equity,
        )

    def test_empty_trades_returns_zeros(self):
        """İşlem yoksa tüm metrikler 0 olmalı."""
        result  = self._make_result([])
        metrics = PerformanceMetrics.calculate(result)
        assert metrics["total_trades"] == 0
        assert metrics["win_rate_pct"] == 0.0

    def test_win_rate_100_percent(self):
        """Tüm işlemler kazanıyorsa win_rate %100 olmalı."""
        trades  = [make_trade(pnl=100, winner=True) for _ in range(10)]
        result  = self._make_result(trades)
        metrics = PerformanceMetrics.calculate(result)
        assert metrics["win_rate_pct"] == pytest.approx(100.0)

    def test_win_rate_0_percent(self):
        """Hiç kazanan yoksa win_rate %0 olmalı."""
        trades  = [make_trade(pnl=-50, winner=False) for _ in range(5)]
        result  = self._make_result(trades)
        metrics = PerformanceMetrics.calculate(result)
        assert metrics["win_rate_pct"] == pytest.approx(0.0)

    def test_win_rate_50_percent(self):
        """5 kazanan 5 kaybeden → %50 win rate."""
        trades  = [make_trade(100, winner=True)] * 5 + [make_trade(-50, winner=False)] * 5
        result  = self._make_result(trades)
        metrics = PerformanceMetrics.calculate(result)
        assert metrics["win_rate_pct"] == pytest.approx(50.0)

    def test_profit_factor_calculation(self):
        """Profit factor = toplam kazanç / toplam kayıp."""
        trades  = [make_trade(200, winner=True)] * 3 + [make_trade(-100, winner=False)] * 2
        result  = self._make_result(trades)
        metrics = PerformanceMetrics.calculate(result)
        # 600 kazanç / 200 kayıp = 3.0
        assert metrics["profit_factor"] == pytest.approx(3.0, abs=0.01)

    def test_total_trades_count(self):
        """Toplam işlem sayısı doğru olmalı."""
        trades  = [make_trade() for _ in range(7)]
        result  = self._make_result(trades)
        metrics = PerformanceMetrics.calculate(result)
        assert metrics["total_trades"] == 7

    def test_max_consecutive_losses(self):
        """Üst üste max kayıp serisi doğru hesaplanmalı."""
        pattern = [True, False, False, False, True, False, True]
        trades  = [make_trade(winner=w) for w in pattern]
        result  = self._make_result(trades)
        metrics = PerformanceMetrics.calculate(result)
        assert metrics["max_consec_losses"] == 3

    def test_compare_returns_dataframe(self):
        """compare() pandas DataFrame döndürmeli."""
        trades  = [make_trade() for _ in range(5)]
        result1 = self._make_result(trades)
        result2 = self._make_result(trades)
        result1.strategy_name = "RSI"
        result2.strategy_name = "PA_RANGE"
        df = PerformanceMetrics.compare([result1, result2])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_sharpe_positive_for_rising_equity(self):
        """Sürekli yükselen equity curve için Sharpe pozitif olmalı."""
        trades = [make_trade(100, winner=True) for _ in range(10)]
        final  = 10_000.0 + sum(t.pnl for t in trades)
        equity = pd.Series(
            [10_000 + i * 100 for i in range(100)],
            index=pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        )
        result = BacktestResult(
            strategy_name="Test", symbol="BTC/USDT", timeframe="1h",
            start_date="2024-01-01", end_date="2024-04-10",
            initial_capital=10_000.0, final_capital=final,
            trades=trades, equity_curve=equity,
        )
        metrics = PerformanceMetrics.calculate(result)
        assert metrics["sharpe_ratio"] > 0
