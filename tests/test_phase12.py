"""
tests/test_phase12.py
======================
Phase 12 gelistirmelerini test eder:
    - 12A: Trailing Stop (breakeven, partial close)
    - 12B: PA Range (volume, fakeout, RSI divergence)
    - 12C: Short/Long ayni anda
    - 12D: ML external data

Calistirmak icin:
    powershell -Command "& 'C:\\Users\\rk209\\AppData\\Local\\Programs\\Python\\Python312\\python.exe' -m pytest tests/test_phase12.py -v"
"""

import sys
from pathlib import Path

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd


# ── Yardimci: Test DataFrame uret ────────────────────────────────────────────

def make_df(n=120, seed=42, trend="range"):
    """Test icin sahte OHLCV DataFrame uretir."""
    np.random.seed(seed)
    closes = []
    price  = 50000.0
    for i in range(n):
        if trend == "range":
            target = 50000 + 1000 * np.sin(i * 0.15)
            price  = price + (target - price) * 0.1 + np.random.uniform(-100, 100)
        elif trend == "up":
            price += np.random.uniform(0, 200)
        elif trend == "down":
            price -= np.random.uniform(0, 200)
        closes.append(max(price, 1.0))

    df = pd.DataFrame({
        "open"  : [c * 0.999 for c in closes],
        "high"  : [c * 1.003 for c in closes],
        "low"   : [c * 0.997 for c in closes],
        "close" : closes,
        "volume": [np.random.uniform(100, 500) for _ in closes],
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 12A — Trailing Stop
# ══════════════════════════════════════════════════════════════════════════════

class TestTrailingStop:
    """Trailing stop mekanizmasini test eder."""

    def setup_method(self):
        from trading.position_tracker import PositionTracker
        self.pt = PositionTracker(initial_capital=10_000.0, max_positions=2)

    def _open_long(self, entry=50000.0, qty=0.001):
        """Test icin LONG pozisyon ac."""
        return self.pt.open_position(
            symbol="BTC/USDT", direction="LONG",
            entry_price=entry, quantity=qty,
            stop_loss=entry * 0.98, take_profit=entry * 1.05,
            strategy="test",
        )

    def test_breakeven_triggered_at_1_5_pct(self):
        """Kar %1.5'e ulasinca BREAKEVEN action donmeli."""
        pos = self._open_long(entry=50000.0, qty=0.001)
        assert pos is not None

        # %1.5 kar: 50000 * 1.015 = 50750
        actions = self.pt.check_trailing_stops(
            current_price=50750.0,
            breakeven_pct=0.015,
            partial_close_pct=0.030,
            trail_sl_pct=0.015,
        )
        assert len(actions) == 1
        pid, action, data = actions[0]
        assert action == "BREAKEVEN"
        assert data["new_sl"] == 50000.0   # SL = entry

    def test_partial_close_at_3_pct(self):
        """Kar %3'e ulasinca PARTIAL_CLOSE action donmeli."""
        pos = self._open_long(entry=50000.0, qty=0.002)
        assert pos is not None

        # %3 kar: 50000 * 1.03 = 51500
        actions = self.pt.check_trailing_stops(
            current_price=51500.0,
            breakeven_pct=0.015,
            partial_close_pct=0.030,
            trail_sl_pct=0.015,
        )
        assert len(actions) == 1
        pid, action, data = actions[0]
        assert action == "PARTIAL_CLOSE"
        assert data["close_quantity"] == 0.001  # Yarisi (0.002 / 2)
        assert data["new_sl"] == round(50000.0 * 1.015, 2)  # %1.5 uzerinde

    def test_no_action_below_breakeven(self):
        """Kar %1.5 altinda ise hicbir action olmamali."""
        pos = self._open_long(entry=50000.0, qty=0.001)
        assert pos is not None

        # %1 kar — esik altinda
        actions = self.pt.check_trailing_stops(
            current_price=50500.0,   # %1 kar
            breakeven_pct=0.015,
            partial_close_pct=0.030,
        )
        assert len(actions) == 0

    def test_partial_close_not_repeated(self):
        """Partial close ikinci kez tetiklenmemeli."""
        pos = self._open_long(entry=50000.0, qty=0.002)
        assert pos is not None

        pid = pos.position_id

        # Ilk partial close
        actions = self.pt.check_trailing_stops(current_price=51500.0,
                                               breakeven_pct=0.015,
                                               partial_close_pct=0.030)
        assert len(actions) == 1

        # Partial close uygula
        self.pt.partial_close_position(
            pid, close_quantity=0.001, exit_price=51500.0
        )

        # Tekrar kontrol — artik PARTIAL_CLOSE gelmemeli
        actions2 = self.pt.check_trailing_stops(current_price=51600.0,
                                                breakeven_pct=0.015,
                                                partial_close_pct=0.030)
        partial_actions = [a for a in actions2 if a[1] == "PARTIAL_CLOSE"]
        assert len(partial_actions) == 0

    def test_partial_close_reduces_quantity(self):
        """Partial close sonrasi pozisyon miktari yariya dusumeli."""
        pos = self._open_long(entry=50000.0, qty=0.002)
        assert pos is not None

        trade = self.pt.partial_close_position(
            pos.position_id,
            close_quantity=0.001,
            exit_price=51000.0,
        )
        assert trade is not None
        assert trade.quantity == 0.001
        assert pos.quantity == pytest.approx(0.001, abs=1e-7)
        assert pos.partial_closed == True

    def test_partial_close_records_pnl(self):
        """Partial close PnL dogru hesaplanmali (komisyon dahil)."""
        pos = self._open_long(entry=50000.0, qty=0.002)
        assert pos is not None

        trade = self.pt.partial_close_position(
            pos.position_id,
            close_quantity=0.001,
            exit_price=51000.0,
        )
        assert trade is not None
        # gross = 0.001 * (51000 - 50000) = 1.0 USDT
        # fee   = 0.001 * 51000 * 0.001 = 0.051 USDT (komisyon %0.1)
        # net   = 1.0 - 0.051 ≈ 0.949 USDT
        assert trade.realized_pnl == pytest.approx(0.949, abs=0.01)

    def test_breakeven_disabled(self):
        """enabled=False iken hicbir action olmamali."""
        pos = self._open_long(entry=50000.0)
        assert pos is not None

        actions = self.pt.check_trailing_stops(
            current_price=51500.0,
            enabled=False,
        )
        assert len(actions) == 0

    def test_short_partial_close(self):
        """SHORT pozisyon icin partial close dogru calisnali."""
        pos = self.pt.open_position(
            symbol="BTC/USDT", direction="SHORT",
            entry_price=50000.0, quantity=0.002,
            strategy="test",
        )
        assert pos is not None

        # %3 kar SHORT: fiyat 50000 * 0.97 = 48500
        actions = self.pt.check_trailing_stops(
            current_price=48500.0,
            breakeven_pct=0.015,
            partial_close_pct=0.030,
            trail_sl_pct=0.015,
        )
        assert len(actions) == 1
        pid, action, data = actions[0]
        assert action == "PARTIAL_CLOSE"
        # SHORT breakeven SL: entry * (1 - trail_sl_pct)
        assert data["new_sl"] == round(50000.0 * (1 - 0.015), 2)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 12B — PA Range Gelistirmeleri
# ══════════════════════════════════════════════════════════════════════════════

class TestPARangeEnhancements:
    """PA Range volume, fakeout, RSI divergence testleri."""

    def setup_method(self):
        from strategies.pa_range_strategy import PARangeStrategy
        self.strategy_default = PARangeStrategy(
            symbol="BTC/USDT", timeframe="1h",
            lookback=50, rsi_oversold=40, rsi_overbought=60,
            use_regime_filter=False,
            volume_confirm_mult=1.5,
            fakeout_filter=True,
            rsi_divergence=True,
        )
        self.strategy_no_filters = PARangeStrategy(
            symbol="BTC/USDT", timeframe="1h",
            lookback=50, rsi_oversold=40, rsi_overbought=60,
            use_regime_filter=False,
            volume_confirm_mult=1.5,
            fakeout_filter=False,  # Fakeout filtresi kapali
            rsi_divergence=False,  # RSI div kapali
        )

    def test_strategy_params_stored(self):
        """Parametreler dogru saklanmali."""
        s = self.strategy_default
        assert s.volume_confirm_mult == 1.5
        assert s.fakeout_filter == True
        assert s.rsi_divergence == True

    def test_strategy_no_filter_params(self):
        """Filtre kapali parametreler dogru saklanmali."""
        s = self.strategy_no_filters
        assert s.fakeout_filter == False
        assert s.rsi_divergence == False

    def test_generate_signal_returns_signal(self):
        """generate_signal valid Signal donmeli."""
        from strategies.base_strategy import Signal
        df = make_df(n=120)
        sig = self.strategy_default.generate_signal(df)
        assert isinstance(sig, Signal)
        assert sig.action in ("AL", "SAT", "BEKLE")
        assert 0.0 <= sig.confidence <= 1.0

    def test_insufficient_data_returns_bekle(self):
        """Az veri ile BEKLE donmeli."""
        df = make_df(n=10)
        sig = self.strategy_default.generate_signal(df)
        assert sig.action == "BEKLE"

    def test_volume_confirm_mult_from_config(self):
        """volume_confirm_mult = 1.0 yapilirsa daha fazla sinyal uretmeli."""
        from strategies.pa_range_strategy import PARangeStrategy
        s_loose = PARangeStrategy(
            symbol="BTC/USDT", timeframe="1h",
            lookback=50, rsi_oversold=40, rsi_overbought=60,
            use_regime_filter=False,
            volume_confirm_mult=1.0,   # Her hacimde gecmeli
            fakeout_filter=False,
            rsi_divergence=False,
        )
        df = make_df(n=120)
        # Hata vermemeli
        sig = s_loose.generate_signal(df)
        assert sig.action in ("AL", "SAT", "BEKLE")

    def test_confidence_bounded(self):
        """Guven skoru her zaman 0-1 araliginda olmali."""
        df = make_df(n=200)
        for i in range(70, 200, 5):
            sig = self.strategy_default.generate_signal(df.iloc[:i+1])
            assert 0.0 <= sig.confidence <= 1.0, \
                f"Bar {i}: confidence={sig.confidence} aralik disi"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 12C — Short/Long Ayni Anda
# ══════════════════════════════════════════════════════════════════════════════

class TestShortLongSimultaneous:
    """LONG ve SHORT pozisyonlarin ayni anda acilabildigini test eder."""

    def setup_method(self):
        from trading.position_tracker import PositionTracker
        self.pt = PositionTracker(initial_capital=10_000.0, max_positions=2)

    def test_has_long_position_false_initially(self):
        """Baslangicta LONG pozisyon olmamali."""
        assert self.pt.has_long_position() == False

    def test_has_short_position_false_initially(self):
        """Baslangicta SHORT pozisyon olmamali."""
        assert self.pt.has_short_position() == False

    def test_open_long_detected(self):
        """LONG acinca has_long_position True olmali."""
        self.pt.open_position("BTC/USDT", "LONG", 50000.0, 0.001, strategy="t")
        assert self.pt.has_long_position() == True
        assert self.pt.has_short_position() == False

    def test_open_short_detected(self):
        """SHORT acinca has_short_position True olmali."""
        self.pt.open_position("BTC/USDT", "SHORT", 50000.0, 0.001, strategy="t")
        assert self.pt.has_short_position() == True
        assert self.pt.has_long_position() == False

    def test_long_and_short_simultaneously(self):
        """LONG ve SHORT ayni anda acilabilmeli (max_positions=2)."""
        long_pos  = self.pt.open_position("BTC/USDT", "LONG",  50000.0, 0.001)
        short_pos = self.pt.open_position("BTC/USDT", "SHORT", 50000.0, 0.001)
        assert long_pos  is not None
        assert short_pos is not None
        assert len(self.pt.open_positions()) == 2
        assert self.pt.has_long_position()  == True
        assert self.pt.has_short_position() == True

    def test_max_positions_enforced(self):
        """max_positions=2 iken 3. pozisyon acilmamali."""
        self.pt.open_position("BTC/USDT", "LONG",  50000.0, 0.001)
        self.pt.open_position("BTC/USDT", "SHORT", 50000.0, 0.001)
        third = self.pt.open_position("BTC/USDT", "LONG",  50000.0, 0.001)
        assert third is None

    def test_has_long_with_symbol_filter(self):
        """Symbol filtreli has_long_position dogru calisnali."""
        self.pt.open_position("BTC/USDT", "LONG", 50000.0, 0.001)
        assert self.pt.has_long_position("BTC/USDT")  == True
        assert self.pt.has_long_position("ETH/USDT")  == False

    def test_close_removes_direction(self):
        """Pozisyon kapaninca direction kontrolu guncellenmeli."""
        pos = self.pt.open_position("BTC/USDT", "LONG", 50000.0, 0.001)
        assert pos is not None
        self.pt.close_position(pos.position_id, 51000.0)
        assert self.pt.has_long_position() == False

    def test_long_short_independent_pnl(self):
        """LONG ve SHORT pozisyonlarin PnL bagimsiz hesaplanmali."""
        long_pos  = self.pt.open_position("BTC/USDT", "LONG",  50000.0, 0.001)
        short_pos = self.pt.open_position("BTC/USDT", "SHORT", 50000.0, 0.001)

        # Fiyat 51000'e cikti: LONG +1 USDT, SHORT -1 USDT
        self.pt.update(51000.0)

        long_pos.update_price(51000.0)
        short_pos.update_price(51000.0)

        assert long_pos.unrealized_pnl  > 0   # LONG kar
        assert short_pos.unrealized_pnl < 0   # SHORT zarar


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 12D — ML External Data
# ══════════════════════════════════════════════════════════════════════════════

class TestMLExternalData:
    """ML external data ve feature engineering testleri."""

    def test_external_data_fetcher_import(self):
        """ExternalDataFetcher import edilebilmeli."""
        from ml.external_data import ExternalDataFetcher, get_external_fetcher
        fetcher = ExternalDataFetcher()
        assert fetcher is not None

    def test_external_data_fetch_all_returns_dict(self):
        """fetch_all dict donmeli (timeout ile)."""
        from ml.external_data import ExternalDataFetcher
        # 1 saniye timeout ile test — API'ye baglanmak zorunda degil
        fetcher = ExternalDataFetcher(timeout=1)
        result = fetcher.fetch_all()
        assert isinstance(result, dict)
        assert "fear_greed" in result
        assert "btc_dominance" in result
        # Deger ya float ya da None olmali
        for k, v in result.items():
            assert v is None or isinstance(v, float)

    def test_external_data_cache(self):
        """Onbellek TTL icinde ikinci cagri API'ye gitmemeli."""
        from ml.external_data import ExternalDataFetcher
        fetcher = ExternalDataFetcher(cache_ttl=60, timeout=1)
        # Ilk cagri
        r1 = fetcher.fetch_all()
        # Onbellege manuel deger koy
        fetcher._set_cache("fear_greed", 42.0)
        fetcher._set_cache("btc_dominance", 55.0)
        # Ikinci cagri — onbellekten gelmeli
        assert fetcher.get_fear_greed() == 42.0
        assert fetcher.get_btc_dominance() == 55.0

    def test_feature_engineer_external_disabled(self):
        """use_external_data=False iken external feature olmamalı."""
        from ml.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(use_external_data=False)
        df = make_df(n=200)
        X, y, names = fe.build(df)
        assert "fear_greed_norm" not in names
        assert "btc_dominance_norm" not in names

    def test_feature_engineer_external_enabled(self):
        """use_external_data=True iken external feature olmali."""
        from ml.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(use_external_data=True)
        # External fetcher'i mock et (API'ye gitmeden)
        from ml.external_data import ExternalDataFetcher
        mock_fetcher = ExternalDataFetcher(timeout=1)
        mock_fetcher._set_cache("fear_greed", 50.0)
        mock_fetcher._set_cache("btc_dominance", 52.0)
        fe._ext_fetcher = mock_fetcher

        df = make_df(n=200)
        X, y, names = fe.build(df)
        assert "fear_greed_norm" in names
        assert "btc_dominance_norm" in names
        # Degerler dogru normalize edilmis mi?
        assert X["fear_greed_norm"].iloc[0]    == pytest.approx(0.50, abs=1e-5)
        assert X["btc_dominance_norm"].iloc[0] == pytest.approx(0.52, abs=1e-5)

    def test_feature_engineer_fg_rsi_diverge(self):
        """fg_rsi_diverge feature mevcut olmali."""
        from ml.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(use_external_data=True)
        from ml.external_data import ExternalDataFetcher
        mock_fetcher = ExternalDataFetcher(timeout=1)
        mock_fetcher._set_cache("fear_greed", 70.0)
        mock_fetcher._set_cache("btc_dominance", 50.0)
        fe._ext_fetcher = mock_fetcher

        df = make_df(n=200)
        X, y, names = fe.build(df)
        assert "fg_rsi_diverge" in names


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTLERI
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigUpdates:
    """Yeni config alanlari dogru yukleniyor mu?"""

    def setup_method(self):
        # Her test oncesi config onbellegi temizle
        from config.loader import reload_config
        self.cfg = reload_config()

    def test_trailing_stop_config_loaded(self):
        """trailing_stop config yuklenmeli."""
        ts = self.cfg.trailing_stop
        assert ts.enabled == True
        assert ts.breakeven_pct == pytest.approx(0.015)
        assert ts.partial_close_pct == pytest.approx(0.030)
        assert ts.trail_sl_pct == pytest.approx(0.015)

    def test_max_open_positions_2(self):
        """max_open_positions 2 olmali (LONG+SHORT)."""
        assert self.cfg.risk.max_open_positions == 2

    def test_pa_range_new_params(self):
        """PA Range yeni parametreler yuklenmeli."""
        pa = self.cfg.strategies.pa_range
        assert pa.volume_confirm_mult == pytest.approx(1.5)
        assert pa.fakeout_filter == True
        assert pa.rsi_divergence == True
