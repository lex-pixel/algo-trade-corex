"""
tests/test_phase17_21.py
==========================
Phase 17-21 testleri:
  - Phase 17: scripts/report.py (otomatik raporlama)
  - Phase 18: scripts/dashboard.py (UTC+3, acik pozisyon paneli)
  - Phase 19: scripts/rr_calc.py (R:R hesaplayici)
  - Phase 21: backtesting/walk_forward.py + engine SHORT destegi
"""

import json
import math
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 17: Otomatik Raporlama
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase17Report:
    """scripts/report.py testleri."""

    def _make_state(self, n_trades=5, n_eq=50):
        """Test icin sahte bot_state uretir."""
        import random
        random.seed(42)

        now = datetime.now(timezone.utc)
        trades = []
        for i in range(n_trades):
            pnl = random.uniform(-50, 100)
            trades.append({
                "realized_pnl": pnl,
                "direction"   : "LONG",
                "quantity"    : 0.01,
                "entry_price" : 70000.0,
                "exit_price"  : 70000.0 * (1 + pnl / 700),
                "opened_at"   : (now - timedelta(hours=10)).isoformat(),
                "closed_at"   : now.isoformat(),
                "exit_reason" : "TP",
            })

        equity = 10000.0
        eq_history = []
        for i in range(n_eq):
            equity += random.uniform(-20, 30)
            eq_history.append({
                "ts"    : (now - timedelta(hours=n_eq - i)).isoformat(),
                "equity": equity,
                "price" : 70000.0,
            })

        return {
            "initial_capital": 10000.0,
            "capital"        : equity,
            "equity_peak"    : equity + 200,
            "max_drawdown"   : 2.5,
            "iteration"      : 100,
            "paper"          : True,
            "trades"         : trades,
            "open_positions" : [],
            "equity_history" : eq_history,
            "saved_at"       : now.isoformat(),
        }

    def test_build_report_keys(self):
        """build_report() dogru anahtarlari dondurur."""
        from scripts.report import build_report
        state  = self._make_state()
        report = build_report(state)

        required = [
            "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
            "win_rate_pct", "profit_factor", "total_trades",
            "total_return_pct", "calmar_ratio", "expectancy_usd",
        ]
        for key in required:
            assert key in report, f"Eksik anahtar: {key}"

    def test_build_report_no_trades(self):
        """Islem yoksa bile hata vermez."""
        from scripts.report import build_report
        state  = self._make_state(n_trades=0)
        report = build_report(state)
        assert report["total_trades"] == 0
        assert report["win_rate_pct"] == 0.0

    def test_sharpe_positive_equity(self):
        """Surekli artan equity -> pozitif Sharpe."""
        from scripts.report import calc_sharpe
        eq = [10000 + i * 10 for i in range(50)]
        s  = calc_sharpe(eq)
        assert s > 0

    def test_sharpe_flat_equity(self):
        """Duz equity -> Sharpe 0."""
        from scripts.report import calc_sharpe
        eq = [10000.0] * 50
        s  = calc_sharpe(eq)
        assert s == 0.0

    def test_max_drawdown_negative(self):
        """Max drawdown negatif deger dondurur."""
        from scripts.report import calc_max_drawdown
        eq = [10000, 10500, 10200, 9800, 10100]
        dd = calc_max_drawdown(eq)
        assert dd < 0, "Max drawdown negatif olmali"

    def test_max_drawdown_no_loss(self):
        """Kayip yoksa drawdown 0."""
        from scripts.report import calc_max_drawdown
        eq = [10000, 10100, 10200, 10300]
        dd = calc_max_drawdown(eq)
        assert dd == 0.0

    def test_profit_factor_wins_only(self):
        """Sadece kazanli islemler -> inf PF."""
        from scripts.report import calc_profit_factor
        trades = [{"realized_pnl": 100}, {"realized_pnl": 50}]
        pf     = calc_profit_factor(trades)
        assert pf == float("inf")

    def test_profit_factor_mixed(self):
        """Karisik islemler -> 1.0 ustu."""
        from scripts.report import calc_profit_factor
        trades = [
            {"realized_pnl": 200},
            {"realized_pnl": -100},
        ]
        pf = calc_profit_factor(trades)
        assert pf == pytest.approx(2.0, rel=0.01)

    def test_expectancy_positive(self):
        """Kazanli sistemde beklenti pozitif."""
        from scripts.report import calc_expectancy
        trades = [
            {"realized_pnl": 100},
            {"realized_pnl": 100},
            {"realized_pnl": -30},
        ]
        exp = calc_expectancy(trades)
        assert exp > 0

    def test_format_txt_runs(self):
        """format_txt() string dondurur."""
        from scripts.report import build_report, format_txt
        state  = self._make_state()
        report = build_report(state)
        txt    = format_txt(report)
        assert isinstance(txt, str)
        assert "Sharpe" in txt
        assert "Win Rate" in txt

    def test_sortino_ratio(self):
        """Sortino hesaplamasi calisir."""
        from scripts.report import calc_sortino
        eq = [10000 + i * 5 + ((-1) ** i) * 10 for i in range(50)]
        s  = calc_sortino(eq)
        assert isinstance(s, float)

    def test_calmar_ratio(self):
        """Calmar orani hesaplamasi."""
        from scripts.report import calc_calmar
        calmar = calc_calmar(15.0, -5.0)
        assert calmar == pytest.approx(3.0, rel=0.01)

    def test_calmar_zero_dd(self):
        """DD sifir ise calmar 0 doner."""
        from scripts.report import calc_calmar
        assert calc_calmar(10.0, 0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 18: Dashboard UTC+3
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase18Dashboard:
    """scripts/dashboard.py UTC+3 ve yeni metrikler."""

    def test_to_tr_conversion(self):
        """UTC timestamp'i UTC+3'e cevirmeli."""
        from scripts.dashboard import _to_tr
        # 2026-03-13 12:00 UTC -> 2026-03-13 15:00 UTC+3
        ts  = "2026-03-13T12:00:00+00:00"
        out = _to_tr(ts)
        assert "15:00" in out

    def test_to_tr_naive_treated_as_utc(self):
        """Timezone bilgisi olmayan ts UTC kabul edilir."""
        from scripts.dashboard import _to_tr
        ts  = "2026-03-13T10:00:00"
        out = _to_tr(ts)
        assert "13:00" in out

    def test_to_tr_invalid_returns_fallback(self):
        """Gecersiz ts hata vermez."""
        from scripts.dashboard import _to_tr
        out = _to_tr("invalid")
        assert isinstance(out, str)

    def test_calc_sharpe_increasing(self):
        """Artan equity -> pozitif Sharpe."""
        from scripts.dashboard import _calc_sharpe
        eq = [10000 + i * 5 for i in range(30)]
        s  = _calc_sharpe(eq)
        assert s > 0

    def test_calc_sharpe_insufficient(self):
        """Az veri -> 0 doner."""
        from scripts.dashboard import _calc_sharpe
        assert _calc_sharpe([10000, 10010]) == 0.0

    def test_calc_sharpe_flat(self):
        """Duz equity -> 0."""
        from scripts.dashboard import _calc_sharpe
        eq = [10000.0] * 20
        assert _calc_sharpe(eq) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 19: R:R Modulu
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase19RR:
    """scripts/rr_calc.py testleri."""

    def test_long_valid_rr(self):
        """LONG gecerli R:R hesaplar."""
        from scripts.rr_calc import calc_rr
        r = calc_rr(entry=70000, sl=68000, tp=76000, direction="LONG")
        assert r["valid"] is True
        assert r["rr_ratio"] == pytest.approx(3.0, rel=0.01)
        assert r["rr_str"] == "1:3.00"

    def test_short_valid_rr(self):
        """SHORT gecerli R:R hesaplar."""
        from scripts.rr_calc import calc_rr
        r = calc_rr(entry=70000, sl=72000, tp=64000, direction="SHORT")
        assert r["valid"] is True
        assert r["rr_ratio"] == pytest.approx(3.0, rel=0.01)

    def test_invalid_sl_long(self):
        """LONG'da SL giris uzerinde -> hatali."""
        from scripts.rr_calc import calc_rr
        r = calc_rr(entry=70000, sl=72000, tp=76000, direction="LONG")
        assert r["valid"] is False

    def test_invalid_tp_long(self):
        """LONG'da TP giris altinda -> hatali."""
        from scripts.rr_calc import calc_rr
        r = calc_rr(entry=70000, sl=68000, tp=65000, direction="LONG")
        assert r["valid"] is False

    def test_risk_pct_correct(self):
        """Risk yuzdesi dogru hesaplanir."""
        from scripts.rr_calc import calc_rr
        r = calc_rr(entry=100, sl=98, tp=106, direction="LONG")
        assert r["risk_pct"] == pytest.approx(2.0, rel=0.01)
        assert r["reward_pct"] == pytest.approx(6.0, rel=0.01)

    def test_position_size_basic(self):
        """Lot hesaplamasi dogru calisir."""
        from scripts.rr_calc import calc_position_size
        pos = calc_position_size(
            capital=10000, risk_pct=2.0,
            entry=70000, sl=69300, direction="LONG"
        )
        assert pos["valid"] is True
        assert pos["actual_risk_usd"] == pytest.approx(200, rel=0.01)

    def test_position_size_zero_sl_distance(self):
        """SL ile giris ayni -> hatali."""
        from scripts.rr_calc import calc_position_size
        pos = calc_position_size(10000, 2.0, 70000, 70000)
        assert pos["valid"] is False

    def test_cascade_tp_single(self):
        """Tek TP ile kaskade calisir."""
        from scripts.rr_calc import calc_cascade_tp
        c = calc_cascade_tp(entry=70000, sl=68000, tp1=76000, qty=1.0)
        assert len(c) == 1
        assert c[0]["pnl_usd"] == pytest.approx(6000, rel=0.01)

    def test_cascade_tp_multiple(self):
        """Uc TP ile kaskade — agirliklar dogru bolunur."""
        from scripts.rr_calc import calc_cascade_tp
        c = calc_cascade_tp(
            entry=70000, sl=68000,
            tp1=74000, tp2=77000, tp3=82000,
            qty=1.0
        )
        assert len(c) == 3
        # Toplam agirlik %100
        total_w = sum(x["weight_pct"] for x in c)
        assert total_w == pytest.approx(100.0, rel=0.01)

    def test_cascade_pnl_positive_long(self):
        """LONG kaskade PnL pozitif olmali."""
        from scripts.rr_calc import calc_cascade_tp
        c = calc_cascade_tp(entry=70000, sl=68000, tp1=74000, tp2=78000, qty=0.1)
        for item in c:
            assert item["pnl_usd"] > 0

    def test_rr_ratio_below_one_valid(self):
        """R:R 1'in altinda olsa da gecerli (kullanici uyarilir)."""
        from scripts.rr_calc import calc_rr
        r = calc_rr(entry=70000, sl=69000, tp=70500, direction="LONG")
        assert r["valid"] is True
        assert r["rr_ratio"] < 1.0

    def test_print_rr_report_runs(self, capsys):
        """print_rr_report() hata vermez."""
        from scripts.rr_calc import calc_rr, print_rr_report
        rr = calc_rr(70000, 68000, 76000, "LONG")
        print_rr_report(rr)
        out = capsys.readouterr().out
        assert "R:R" in out or "Giris" in out


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 21: Backtesting Walk-Forward + SHORT
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Test icin sahte OHLCV DataFrame uretir."""
    rng    = np.random.default_rng(seed)
    closes = 70000 + np.cumsum(rng.normal(0, 100, n))
    highs  = closes + abs(rng.normal(0, 50, n))
    lows   = closes - abs(rng.normal(0, 50, n))
    opens  = closes + rng.normal(0, 30, n)
    vols   = abs(rng.normal(50, 20, n))

    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    }, index=idx)


class TestPhase21BacktestEngine:
    """backtesting/engine.py SHORT destegi ve yeni ozellikler."""

    def test_bnb_commission_lower(self):
        """BNB ile komisyon %25 daha dusuk."""
        from backtesting.engine import BacktestEngine
        e1 = BacktestEngine(commission=0.001, commission_bnb=False)
        e2 = BacktestEngine(commission=0.001, commission_bnb=True)
        assert e2.commission < e1.commission
        assert e2.commission == pytest.approx(0.00075, rel=0.01)

    def test_volume_slippage_high_volume(self):
        """Yuksek hacim -> dusuk slipaj katsayisi."""
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine(slippage_volume_adj=True)
        bar_high = pd.Series({"volume": 10000.0})
        bar_low  = pd.Series({"volume": 1.0})
        f_high   = engine._volume_slippage(bar_high)
        f_low    = engine._volume_slippage(bar_low)
        assert f_high < f_low

    def test_volume_slippage_disabled(self):
        """Hacim ayari kapali -> her zaman 1.0 doner."""
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine(slippage_volume_adj=False)
        bar    = pd.Series({"volume": 10000.0})
        assert engine._volume_slippage(bar) == 1.0

    def test_short_position_opens(self):
        """allow_short=True ile SAT sinyali SHORT acar."""
        from backtesting.engine import BacktestEngine
        from strategies.base_strategy import Signal

        engine = BacktestEngine(allow_short=True)
        pos = engine._open_position(
            Signal(action="SAT", confidence=0.8, stop_loss=72000, take_profit=65000),
            price=70000, time=datetime.now(), capital=10000, direction="SHORT"
        )
        assert pos["direction"] == "SHORT"
        # SHORT'da giris fiyati biraz asagidan
        assert pos["entry_price"] < 70000

    def test_long_position_opens(self):
        """LONG pozisyon acilir, giris biraz yuksekcten."""
        from backtesting.engine import BacktestEngine
        from strategies.base_strategy import Signal

        engine = BacktestEngine()
        pos = engine._open_position(
            Signal(action="AL", confidence=0.8, stop_loss=68000, take_profit=76000),
            price=70000, time=datetime.now(), capital=10000, direction="LONG"
        )
        assert pos["direction"] == "LONG"
        assert pos["entry_price"] > 70000

    def test_short_exit_sl_check(self):
        """SHORT stop-loss: high >= sl -> STOP_LOSS."""
        from backtesting.engine import BacktestEngine
        engine   = BacktestEngine()
        position = {"direction": "SHORT", "stop_loss": 72000, "take_profit": 65000}
        bar_hit  = pd.Series({"low": 68000, "high": 73000})
        bar_miss = pd.Series({"low": 68000, "high": 71000})
        assert engine._check_exit(bar_hit, position) == "STOP_LOSS"
        assert engine._check_exit(bar_miss, position) is None

    def test_short_exit_tp_check(self):
        """SHORT take-profit: low <= tp -> TAKE_PROFIT."""
        from backtesting.engine import BacktestEngine
        engine   = BacktestEngine()
        position = {"direction": "SHORT", "stop_loss": 72000, "take_profit": 65000}
        bar_hit  = pd.Series({"low": 64000, "high": 68000})
        bar_miss = pd.Series({"low": 66000, "high": 68000})
        assert engine._check_exit(bar_hit, position) == "TAKE_PROFIT"
        assert engine._check_exit(bar_miss, position) is None

    def test_long_exit_sl_check(self):
        """LONG stop-loss: low <= sl -> STOP_LOSS."""
        from backtesting.engine import BacktestEngine
        engine   = BacktestEngine()
        position = {"direction": "LONG", "stop_loss": 68000, "take_profit": 76000}
        bar_hit  = pd.Series({"low": 67500, "high": 71000})
        assert engine._check_exit(bar_hit, position) == "STOP_LOSS"

    def test_short_pnl_positive_on_decline(self):
        """SHORT pozisyonda fiyat dustugunde PnL pozitif olmali."""
        from backtesting.engine import BacktestEngine
        from strategies.base_strategy import Signal

        engine = BacktestEngine(allow_short=True, slippage=0, commission=0)
        pos    = engine._open_position(
            Signal(action="SAT", confidence=0.8, stop_loss=72000, take_profit=65000),
            price=70000, time=datetime.now(), capital=10000, direction="SHORT"
        )
        trade, new_cap = engine._close_position(pos, 67000, datetime.now(), 10000, "TP")
        assert trade.pnl > 0

    def test_long_pnl_positive_on_rise(self):
        """LONG pozisyonda fiyat yuksektiginde PnL pozitif."""
        from backtesting.engine import BacktestEngine
        from strategies.base_strategy import Signal

        engine = BacktestEngine(slippage=0, commission=0)
        pos    = engine._open_position(
            Signal(action="AL", confidence=0.8, stop_loss=68000, take_profit=76000),
            price=70000, time=datetime.now(), capital=10000, direction="LONG"
        )
        trade, new_cap = engine._close_position(pos, 74000, datetime.now(), 10000, "TP")
        assert trade.pnl > 0

    def test_engine_run_no_short(self):
        """allow_short=False ile klasik LONG-only backtest calisir."""
        from backtesting.engine import BacktestEngine
        from strategies.rsi_strategy import RSIStrategy

        df     = _make_ohlcv(200)
        engine = BacktestEngine(allow_short=False)
        result = engine.run(df, RSIStrategy(), warmup_bars=30)
        assert result is not None
        # Tum islemler LONG olmali
        for t in result.trades:
            assert t.direction == "LONG"


class TestPhase21WalkForward:
    """backtesting/walk_forward.py testleri."""

    def test_walk_forward_produces_periods(self):
        """Walk-forward en az 1 donem uretir."""
        from backtesting.walk_forward import WalkForwardValidator
        from strategies.rsi_strategy import RSIStrategy

        df        = _make_ohlcv(300)
        validator = WalkForwardValidator(method="rolling", train_bars=100, test_bars=50)
        result    = validator.run(df, RSIStrategy())
        assert result.total_periods >= 1

    def test_walk_forward_expanding(self):
        """Expanding window da calisir."""
        from backtesting.walk_forward import WalkForwardValidator
        from strategies.rsi_strategy import RSIStrategy

        df        = _make_ohlcv(300)
        validator = WalkForwardValidator(method="expanding", train_bars=100, test_bars=50)
        result    = validator.run(df, RSIStrategy())
        assert result.total_periods >= 1

    def test_walk_forward_insufficient_data(self):
        """Yetersiz veriyle bos sonuc doner."""
        from backtesting.walk_forward import WalkForwardValidator
        from strategies.rsi_strategy import RSIStrategy

        df        = _make_ohlcv(50)
        validator = WalkForwardValidator(train_bars=200, test_bars=100)
        result    = validator.run(df, RSIStrategy())
        assert result.total_periods == 0

    def test_walk_forward_period_structure(self):
        """Her donem tarih bilgisi icerir."""
        from backtesting.walk_forward import WalkForwardValidator
        from strategies.rsi_strategy import RSIStrategy

        df        = _make_ohlcv(300)
        validator = WalkForwardValidator(method="rolling", train_bars=100, test_bars=50)
        result    = validator.run(df, RSIStrategy())
        for p in result.periods:
            assert p.train_start != ""
            assert p.test_start  != ""
            assert p.period_num  >= 1

    def test_combined_return_calculation(self):
        """combined_return_pct() hesaplama calisir."""
        from backtesting.walk_forward import WalkForwardValidator
        from strategies.rsi_strategy import RSIStrategy

        df        = _make_ohlcv(300)
        validator = WalkForwardValidator(method="rolling", train_bars=100, test_bars=50)
        result    = validator.run(df, RSIStrategy())
        cr = result.combined_return_pct()
        assert isinstance(cr, float)

    def test_avg_sharpe_is_float(self):
        """avg_sharpe float dondurur."""
        from backtesting.walk_forward import WalkForwardValidator
        from strategies.rsi_strategy import RSIStrategy

        df        = _make_ohlcv(300)
        validator = WalkForwardValidator(method="rolling", train_bars=100, test_bars=50)
        result    = validator.run(df, RSIStrategy())
        assert isinstance(result.avg_sharpe, float)

    def test_print_summary_runs(self, capsys):
        """print_summary() hata vermez."""
        from backtesting.walk_forward import WalkForwardValidator
        from strategies.rsi_strategy import RSIStrategy

        df        = _make_ohlcv(300)
        validator = WalkForwardValidator(method="rolling", train_bars=100, test_bars=50)
        result    = validator.run(df, RSIStrategy())
        validator.print_summary(result)
        out = capsys.readouterr().out
        assert "WALK-FORWARD" in out

    def test_step_bars_custom(self):
        """Ozel step_bars ile adim buyuklugu ayarlanir."""
        from backtesting.walk_forward import WalkForwardValidator
        from strategies.rsi_strategy import RSIStrategy

        df        = _make_ohlcv(500)
        validator = WalkForwardValidator(
            method="rolling", train_bars=150, test_bars=50, step_bars=25
        )
        result = validator.run(df, RSIStrategy())
        # Daha kucuk step -> daha fazla donem
        assert result.total_periods >= 2
