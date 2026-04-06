"""
trading/multi_coin_bot.py
==========================
AMACI:
    Birden fazla coin'i ayni anda paralel olarak trade eden bot.
    Her coin icin bagimsiz:
        - Veri cekimi (1h + 4h + 15m)
        - Sinyal uretimi (RSI + PA + ML)
        - Risk yonetimi (KillSwitch + ATR pozisyon boyutu)
        - State dosyasi (data/bot_state_{symbol}.json)

    Portfoy sermayesi: toplam sermaye coinler arasinda paylastirilir.
        Ornek: $10,000 toplam | BTC:%50 ($5000) ETH:%30 ($3000) SOL:%20 ($2000)

CALISTIRMA:
    # Varsayilan: BTC + ETH
    python -m trading.multi_coin_bot

    # Ozel coinler
    python -m trading.multi_coin_bot --coins BTC/USDT ETH/USDT SOL/USDT

    # Ozel sermaye paylasimi
    python -m trading.multi_coin_bot --coins BTC/USDT ETH/USDT --weights 0.6 0.4

    # Bir tur calistir, cik
    python -m trading.multi_coin_bot --once
"""

from __future__ import annotations
import asyncio
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import get_config
from data.fetcher import BinanceFetcher
from data.cleaner import OHLCVCleaner
from strategies.rsi_strategy import RSIStrategy
from strategies.pa_range_strategy import PARangeStrategy
from strategies.regime_detector import MarketRegimeDetector, Regime
from trading.order_manager import OrderManager
from trading.position_tracker import PositionTracker, ClosedTrade
from risk.risk_manager import RiskManager
from ml.predictor import MLPredictor
from utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Coin konfigurasyonu ────────────────────────────────────────────────────

@dataclass
class CoinConfig:
    """Tek bir coin icin konfigurasyonu tutar."""
    symbol:    str          # "BTC/USDT"
    timeframe: str = "1h"   # Ana timeframe
    alloc_pct: float = 1.0  # Portfoyden pay orani (0.0-1.0)
    model_path: str | None = None  # ML model yolu (None = XGBoost default)


# ── Coin Worker ────────────────────────────────────────────────────────────

class CoinWorker:
    """
    Tek bir coin icin bagimsiz trading worker.
    MultiCoinBot tarafindan olusturulur ve yonetilir.
    """

    def __init__(
        self,
        coin_cfg:  CoinConfig,
        capital:   float,
        paper:     bool = True,
        testnet:   bool = True,
    ):
        self.symbol    = coin_cfg.symbol
        self.timeframe = coin_cfg.timeframe
        self.capital   = capital
        self.paper     = paper

        # Symbol'u dosya adina uygun hale getir: BTC/USDT -> BTC_USDT
        safe_sym = self.symbol.replace("/", "_")

        # State dosyasi
        self.state_file = DATA_DIR / f"bot_state_{safe_sym}.json"

        # Veri cekiciler
        self.fetcher     = BinanceFetcher(testnet=testnet, symbol=self.symbol, timeframe=self.timeframe)
        self.fetcher_4h  = BinanceFetcher(testnet=testnet, symbol=self.symbol, timeframe="4h")
        self.fetcher_15m = BinanceFetcher(testnet=testnet, symbol=self.symbol, timeframe="15m")
        self.cleaner     = OHLCVCleaner()

        # Rejim dedektorleri
        self.detector    = MarketRegimeDetector()
        self.detector_4h = MarketRegimeDetector()

        # Emir + pozisyon yonetimi
        self.order_manager = OrderManager(
            symbol=self.symbol, paper=paper, commission=0.001, slippage=0.0005
        )
        self.position_tracker = PositionTracker(
            initial_capital=capital, max_positions=1, commission=0.001
        )
        self.risk_manager = RiskManager(
            initial_capital    = capital,
            max_risk_pct       = 0.02,
            max_open_positions = 1,
            min_confidence     = 0.30,
            sl_atr_mult        = 2.0,
            tp_atr_mult        = 3.0,
        )

        # ML model yukle
        self.ml_predictor: MLPredictor | None = None
        self._load_ml_model(coin_cfg.model_path)

        # Iteration ve equity gecmisi
        self._iteration     = 0
        self._equity_history: list[dict] = []
        self._errors        = 0

        # Onceki state yukle
        self.load_state()

        # KillSwitch gun-basi sifirla (yeniden baslatmada eski kayip bugun sayilmasin)
        actual_capital = self.position_tracker.capital
        self.risk_manager.kill_switch.update_day_start(actual_capital)

        logger.info(
            f"[{self.symbol}] CoinWorker hazir | "
            f"Sermaye: ${capital:,.2f} | "
            f"ML: {'YUKLU' if self.ml_predictor else 'YOK'}"
        )

    def _load_ml_model(self, model_path: str | None) -> None:
        """ML modelini yukler. Bulunamazsa None birakilir."""
        safe_sym = self.symbol.replace("/", "_").lower()
        # Oncelik sirasi: verilen yol -> sembol-ozel model -> varsayilan BTC modeli
        candidates = []
        if model_path:
            candidates.append(Path(model_path))

        # Ensemble model klasoru
        ens_dir = Path(__file__).parent.parent / "ml" / "models" / f"ensemble_{safe_sym}_1h"
        if ens_dir.exists():
            candidates.append(ens_dir)

        # XGBoost tek model
        xgb_path = Path(__file__).parent.parent / "ml" / "models" / f"xgb_{safe_sym}_1h.json"
        candidates.append(xgb_path)

        # BTC modeli fallback (BTC disindaki coinler icin)
        if "btc" not in safe_sym:
            candidates.append(Path(__file__).parent.parent / "ml" / "models" / "xgb_btc_1h.json")

        for p in candidates:
            if not p.exists():
                continue
            try:
                if p.is_dir():
                    self.ml_predictor = MLPredictor.from_ensemble(p, symbol=self.symbol)
                else:
                    self.ml_predictor = MLPredictor.from_file(p, symbol=self.symbol)
                logger.info(f"[{self.symbol}] ML model yuklendi: {p.name}")
                return
            except Exception as e:
                logger.warning(f"[{self.symbol}] ML model yuklenemedi ({p}): {e}")

        logger.info(f"[{self.symbol}] ML model yok, kuralsiz strateji ile calisiliyor.")

    # ── Ana Tick ──────────────────────────────────────────────────────────────

    async def tick(self) -> dict:
        """
        Tek bir dongu adimi. Coin durumunu dict olarak doner.
        MultiCoinBot tarafindan paralel cagrilir.
        """
        self._iteration += 1

        # 1. Veri cek (paralel)
        df, df_4h, df_15m = await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(None, self._fetch, self.fetcher,     200),
            asyncio.get_event_loop().run_in_executor(None, self._fetch, self.fetcher_4h,  100),
            asyncio.get_event_loop().run_in_executor(None, self._fetch, self.fetcher_15m, 100),
        )

        if df is None or df.empty:
            logger.warning(f"[{self.symbol}] Veri alinamadi, tick atlaniyor.")
            return self._status_dict(error="veri_yok")

        current_price = float(df["close"].iloc[-1])
        regime        = self.detector.detect(df)
        big_regime    = self.detector_4h.detect(df_4h) if df_4h is not None else None

        logger.info(
            f"[{self.symbol}] Tick #{self._iteration} | "
            f"Fiyat: ${current_price:,.2f} | "
            f"1h:{regime.value} 4h:{big_regime.value if big_regime else 'N/A'}"
        )

        # 2. SL/TP kontrol
        capital  = self.position_tracker.capital
        open_pnl = self.position_tracker.update(current_price)["total_unrealized"]
        locked   = sum(p.notional for p in self.position_tracker.open_positions())
        equity   = capital + locked + open_pnl

        exits = self.risk_manager.check_exit_conditions(
            position_tracker = self.position_tracker,
            current_price    = current_price,
            current_capital  = equity,
            open_pnl         = open_pnl,
        )
        for pid, reason in exits:
            self._close_position(pid, current_price, reason)

        # 3. Sinyal uret
        action, confidence, rsi_sig, pa_sig, ml_sig = self._get_signal(
            df, regime, big_regime, df_15m
        )

        logger.info(
            f"[{self.symbol}] Sinyal: {action} | Guven: {confidence:.2f} | "
            f"RSI:{rsi_sig.action} PA:{pa_sig.action} "
            f"ML:{'yok' if not ml_sig else ml_sig.action}"
        )

        # 4. ATR
        atr_val = self._calc_atr(df)

        # 5. Risk karar
        decision = self.risk_manager.evaluate_signal(
            action               = action,
            confidence           = confidence,
            current_capital      = equity,
            open_pnl             = open_pnl,
            price                = current_price,
            atr                  = atr_val,
            open_positions_count = len(self.position_tracker.open_positions()),
        )

        if decision.approved:
            direction  = "LONG" if action == "AL" else "SHORT"
            already_open = (
                direction == "LONG"  and self.position_tracker.has_long_position(self.symbol)
                or direction == "SHORT" and self.position_tracker.has_short_position(self.symbol)
            )
            if not already_open:
                order = self.order_manager.place_market_order(
                    side          = "buy" if direction == "LONG" else "sell",
                    quantity      = decision.quantity,
                    current_price = current_price,
                )
                self.position_tracker.open_position(
                    symbol      = self.symbol,
                    direction   = direction,
                    entry_price = order.filled_price or current_price,
                    quantity    = decision.quantity,
                    stop_loss   = decision.stop_loss,
                    take_profit = decision.take_profit,
                    strategy    = f"RSI+PA+ML ({action})",
                    order_id    = order.order_id,
                    entry_fee   = order.fee,
                )
                self.risk_manager.record_trade_executed()
                logger.info(
                    f"[{self.symbol}] {direction} pozisyon acildi | "
                    f"Giris: ${order.filled_price or current_price:,.2f} | "
                    f"Qty: {decision.quantity:.6f}"
                )

        # 6. Equity kaydet
        summary = self.position_tracker.update(current_price)
        self._equity_history.append({
            "ts"    : datetime.now(timezone.utc).isoformat(),
            "equity": round(summary["equity"], 2),
            "price" : round(current_price, 2),
        })
        if len(self._equity_history) > 1000:
            self._equity_history = self._equity_history[-1000:]

        # 7. State kaydet
        self.save_state()

        return self._status_dict(
            price    = current_price,
            equity   = summary["equity"],
            open_pos = summary["open_positions"],
            signal   = action,
            conf     = confidence,
        )

    # ── Sinyal ────────────────────────────────────────────────────────────────

    def _get_signal(self, df, regime, big_regime, df_15m):
        """RSI + PA + ML sinyal birlestirme (main_loop.get_combined_signal ile ayni mantik)."""
        cfg = get_config()
        rsi_cfg = cfg.strategies.rsi
        pa_cfg  = cfg.strategies.pa_range

        rsi_strategy = RSIStrategy(
            symbol=self.symbol, timeframe=self.timeframe,
            rsi_period=rsi_cfg.rsi_period, oversold=rsi_cfg.oversold,
            overbought=rsi_cfg.overbought, stop_pct=rsi_cfg.stop_pct,
            tp_pct=rsi_cfg.tp_pct,
        )
        pa_strategy = PARangeStrategy(
            symbol=self.symbol, timeframe=self.timeframe,
            lookback=pa_cfg.lookback, rsi_period=pa_cfg.rsi_period,
            rsi_oversold=pa_cfg.rsi_oversold, rsi_overbought=pa_cfg.rsi_overbought,
            proximity_pct=pa_cfg.proximity_pct, stop_pct=pa_cfg.stop_pct,
            tp_pct=pa_cfg.tp_pct, use_regime_filter=pa_cfg.use_regime_filter,
            volume_confirm_mult=pa_cfg.volume_confirm_mult,
            fakeout_filter=pa_cfg.fakeout_filter,
            rsi_divergence=pa_cfg.rsi_divergence,
        )

        rsi_sig = rsi_strategy.generate_signal(df)
        pa_sig  = pa_strategy.generate_signal(df)
        ml_sig  = None
        if self.ml_predictor:
            try:
                ml_sig = self.ml_predictor.predict(df)
            except Exception as e:
                logger.warning(f"[{self.symbol}] ML tahmin hatasi: {e}")

        signals    = [rsi_sig, pa_sig] + ([ml_sig] if ml_sig else [])
        vote_counts = {"AL": 0, "SAT": 0, "BEKLE": 0}
        vote_conf   = {"AL": 0.0, "SAT": 0.0, "BEKLE": 0.0}
        for s in signals:
            vote_counts[s.action] += 1
            vote_conf[s.action]   += s.confidence

        al_c  = vote_counts["AL"]
        sat_c = vote_counts["SAT"]

        if al_c > sat_c and al_c > 0:
            action     = "AL"
            confidence = min(1.0, vote_conf["AL"] / al_c + 0.05 * (al_c - 1))
        elif sat_c > al_c and sat_c > 0:
            action     = "SAT"
            confidence = min(1.0, vote_conf["SAT"] / sat_c + 0.05 * (sat_c - 1))
        else:
            action, confidence = "BEKLE", 0.0

        # MTF filtresi
        mtf_cfg = cfg.mtf
        penalty = 1.0 - (mtf_cfg.penalty_pct if mtf_cfg else 0.30)
        if big_regime and action != "BEKLE":
            if action == "AL" and big_regime == Regime.TREND_DOWN:
                confidence = round(confidence * penalty, 3)
            elif action == "SAT" and big_regime == Regime.TREND_UP:
                confidence = round(confidence * penalty, 3)

        # 15m giris filtresi
        ob_15m = getattr(mtf_cfg, "entry_15m_overbought", 78) if mtf_cfg else 78
        os_15m = getattr(mtf_cfg, "entry_15m_oversold",   22) if mtf_cfg else 22
        if df_15m is not None and action != "BEKLE" and len(df_15m) >= 15:
            try:
                import pandas_ta as ta
                rsi_15m     = ta.rsi(df_15m["close"], length=14)
                rsi_15m_val = float(rsi_15m.iloc[-1]) if rsi_15m is not None else None
                if rsi_15m_val is not None:
                    if (action == "AL" and rsi_15m_val > ob_15m) or \
                       (action == "SAT" and rsi_15m_val < os_15m):
                        action, confidence = "BEKLE", 0.0
            except Exception:
                pass

        return action, round(confidence, 3), rsi_sig, pa_sig, ml_sig

    # ── Yardimci ──────────────────────────────────────────────────────────────

    def _fetch(self, fetcher, limit: int):
        """Senkron veri cekimi (executor icinde calisir)."""
        try:
            raw = fetcher.fetch_ohlcv(limit=limit)
            return self.cleaner.clean(raw) if not raw.empty else None
        except Exception as e:
            logger.warning(f"[{self.symbol}] Veri hatasi: {e}")
            return None

    def _close_position(self, pid: str, price: float, reason: str) -> None:
        pos = self.position_tracker.get_position(pid)
        if not pos:
            return
        order = self.order_manager.place_market_order(
            side          = "sell" if pos.direction == "LONG" else "buy",
            quantity      = pos.quantity,
            current_price = price,
        )
        self.position_tracker.close_position(
            position_id = pid,
            exit_price  = order.filled_price or price,
            exit_reason = reason,
            exit_fee    = order.fee,
        )
        logger.info(
            f"[{self.symbol}] Pozisyon kapatildi | "
            f"Neden: {reason} | Fiyat: ${price:,.2f}"
        )

    @staticmethod
    def _calc_atr(df, period: int = 14):
        try:
            import pandas_ta as ta
            atr = ta.atr(df["high"], df["low"], df["close"], length=period)
            v   = atr.iloc[-1]
            import pandas as pd
            return float(v) if not pd.isna(v) else None
        except Exception:
            return None

    def _status_dict(self, **kwargs) -> dict:
        """Tick durum ozeti dict olarak doner."""
        return {"symbol": self.symbol, "iteration": self._iteration, **kwargs}

    # ── State Kayit / Yukle ───────────────────────────────────────────────────

    def save_state(self) -> None:
        pt     = self.position_tracker
        trades = []
        for t in pt._history:
            trades.append({
                "position_id" : t.position_id,
                "symbol"      : t.symbol,
                "direction"   : t.direction,
                "entry_price" : t.entry_price,
                "exit_price"  : t.exit_price,
                "quantity"    : t.quantity,
                "realized_pnl": t.realized_pnl,
                "realized_pct": t.realized_pct,
                "exit_reason" : t.exit_reason,
                "opened_at"   : t.opened_at.isoformat(),
                "closed_at"   : t.closed_at.isoformat(),
                "strategy"    : t.strategy,
                "total_fee"   : t.total_fee,
            })

        open_pos = []
        for pos in pt.open_positions():
            open_pos.append({
                "position_id" : pos.position_id,
                "symbol"      : pos.symbol,
                "direction"   : pos.direction,
                "entry_price" : pos.entry_price,
                "quantity"    : pos.quantity,
                "stop_loss"   : pos.stop_loss,
                "take_profit" : pos.take_profit,
                "strategy"    : pos.strategy,
                "opened_at"   : pos.opened_at.isoformat(),
                "entry_fee"   : pos.entry_fee,
            })

        state = {
            "symbol"         : self.symbol,
            "saved_at"       : datetime.now(timezone.utc).isoformat(),
            "capital"        : round(pt.capital, 4),
            "initial_capital": round(pt.initial_capital, 4),
            "equity_peak"    : round(pt._equity_peak, 4),
            "max_drawdown"   : round(pt._max_drawdown, 6),
            "iteration"      : self._iteration,
            "paper"          : self.paper,
            "trades"         : trades,
            "open_positions" : open_pos,
            "equity_history" : self._equity_history,
        }

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self) -> bool:
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)

            pt = self.position_tracker
            pt.capital         = state["capital"]
            pt.initial_capital = state["initial_capital"]
            pt._equity_peak    = state.get("equity_peak", state["capital"])
            pt._max_drawdown   = state.get("max_drawdown", 0.0)
            self._iteration    = state.get("iteration", 0)
            self._equity_history = state.get("equity_history", [])

            for t in state.get("trades", []):
                pt._history.append(ClosedTrade(
                    position_id = t["position_id"],
                    symbol      = t["symbol"],
                    direction   = t["direction"],
                    entry_price = t["entry_price"],
                    exit_price  = t["exit_price"],
                    quantity    = t["quantity"],
                    realized_pnl= t["realized_pnl"],
                    realized_pct= t["realized_pct"],
                    exit_reason = t["exit_reason"],
                    opened_at   = datetime.fromisoformat(t["opened_at"]),
                    closed_at   = datetime.fromisoformat(t["closed_at"]),
                    strategy    = t.get("strategy", "unknown"),
                    total_fee   = t.get("total_fee", 0.0),
                ))

            logger.info(
                f"[{self.symbol}] State yuklendi: "
                f"sermaye=${pt.capital:,.2f} iter={self._iteration}"
            )
            return True
        except Exception as e:
            logger.warning(f"[{self.symbol}] State yuklenemedi: {e}")
            return False


# ── MultiCoinBot ───────────────────────────────────────────────────────────

class MultiCoinBot:
    """
    Birden fazla coin'i paralel olarak yoneten ana bot.

    Her coin icin bir CoinWorker olusturulur.
    asyncio.gather ile hepsi ayni anda tick atar.

    Parametreler:
        coins    : [CoinConfig, ...]
        capital  : Toplam portfoy sermayesi (USDT)
        paper    : True = paper trading
        interval : Kac saniyede bir guncelleme
    """

    def __init__(
        self,
        coins:    list[CoinConfig],
        capital:  float = 10_000.0,
        paper:    bool  = True,
        interval: int   = 60,
    ):
        self.interval = interval
        self._running = False

        # Portfoy dagitimi
        total_w = sum(c.alloc_pct for c in coins)
        self.workers: list[CoinWorker] = []
        for cfg in coins:
            alloc = round(capital * (cfg.alloc_pct / total_w), 2)
            w = CoinWorker(
                coin_cfg = cfg,
                capital  = alloc,
                paper    = paper,
                testnet  = True,
            )
            self.workers.append(w)

        symbols_info = " | ".join(
            f"{w.symbol} ${w.capital:,.0f}"
            for w in self.workers
        )
        logger.info(
            f"MultiCoinBot baslatildi | {len(self.workers)} coin | "
            f"Toplam: ${capital:,.2f} | {symbols_info}"
        )

    async def run(self, once: bool = False) -> None:
        """Ana dongu — tum worker'lari paralel calistirir."""
        self._running = True
        logger.info(f"MultiCoinBot dongusu basliyor. Aralik: {self.interval}sn")

        while self._running:
            try:
                # Tum coinleri ayni anda tick at
                results = await asyncio.gather(
                    *[w.tick() for w in self.workers],
                    return_exceptions=True,
                )

                # Ozet
                self._print_portfolio_summary(results)

            except Exception as e:
                logger.error(f"MultiCoinBot dongu hatasi: {e}")

            if once:
                break

            await asyncio.sleep(self.interval)

        logger.info("MultiCoinBot dongusu bitti.")

    def _print_portfolio_summary(self, results: list) -> None:
        """Tum coinlerin anlık durumunu terminale yazdirir."""
        total_equity = 0.0
        lines = []
        for res in results:
            if isinstance(res, Exception):
                lines.append(f"  HATA: {res}")
                continue
            sym    = res.get("symbol", "?")
            eq     = res.get("equity", 0.0)
            price  = res.get("price", 0.0)
            signal = res.get("signal", "-")
            conf   = res.get("conf", 0.0)
            op     = res.get("open_pos", 0)
            err    = res.get("error")
            if err:
                lines.append(f"  {sym:<12} HATA: {err}")
            else:
                lines.append(
                    f"  {sym:<12} ${eq:>9,.2f} | "
                    f"Fiyat:${price:>9,.2f} | "
                    f"Sinyal:{signal:<5} ({conf:.2f}) | "
                    f"Pozisyon:{op}"
                )
                total_equity += eq

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        logger.info(f"=== PORTFOY OZETI [{ts}] ===")
        for line in lines:
            logger.info(line)
        logger.info(f"  TOPLAM EQUITY  : ${total_equity:,.2f}")
        logger.info("=" * 50)

    def portfolio_state(self) -> dict:
        """Tum coin state'lerini birlesik dict olarak doner."""
        coins_data = []
        for w in self.workers:
            pt = w.position_tracker
            open_positions = pt.open_positions()
            open_pnl = (
                sum(p.unrealized_pnl for p in open_positions)
                if open_positions and hasattr(open_positions[0], "unrealized_pnl")
                else 0
            )
            coins_data.append({
                "symbol"     : w.symbol,
                "capital"    : round(pt.capital, 2),
                "equity"     : round(pt.capital, 2),
                "trades"     : len(pt._history),
                "open_pos"   : len(pt.open_positions()),
                "iteration"  : w._iteration,
            })
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "coins"        : coins_data,
            "total_equity" : sum(c["equity"] for c in coins_data),
        }


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Coklu coin trading botu")
    parser.add_argument(
        "--coins", nargs="+",
        default=["BTC/USDT", "ETH/USDT"],
        help="Trade edilecek coinler (varsayilan: BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float,
        help="Her coin icin portfoy orani (toplam 1.0 olmali). "
             "Verilmezse esit dagitilir.",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Toplam portfoy sermayesi USDT (varsayilan: 10000)",
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Guncelleme araligi saniye (varsayilan: 60)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Bir tur calistir ve cik",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live mod (varsayilan: paper)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Agirliklar
    weights = args.weights
    if weights is None:
        n = len(args.coins)
        weights = [1.0 / n] * n
    elif len(weights) != len(args.coins):
        print(f"HATA: --weights sayisi ({len(weights)}) --coins sayisiyla ({len(args.coins)}) eslesmiyor")
        sys.exit(1)

    # CoinConfig listesi
    coins = [
        CoinConfig(symbol=sym, alloc_pct=w)
        for sym, w in zip(args.coins, weights)
    ]

    print("=" * 60)
    print("  MULTI-COIN BOT BASLATILIYOR")
    print(f"  Mod     : {'PAPER' if not args.live else 'LIVE (DIKKAT!)'}")
    print(f"  Coinler : {', '.join(c.symbol for c in coins)}")
    print(f"  Sermaye : ${args.capital:,.2f}")
    print(f"  Aralik  : {args.interval}sn")
    print("=" * 60)

    bot = MultiCoinBot(
        coins    = coins,
        capital  = args.capital,
        paper    = not args.live,
        interval = args.interval,
    )

    try:
        asyncio.run(bot.run(once=args.once))
    except KeyboardInterrupt:
        print("\nBot durduruldu (Ctrl+C)")


if __name__ == "__main__":
    main()
