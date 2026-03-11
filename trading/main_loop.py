"""
trading/main_loop.py
======================
AMACI:
    Asenkron (asyncio) ana trading dongusu.
    Her N saniyede bir:
      1. Veri cek (Binance Testnet REST API)
      2. Stratejilerden sinyal al
      3. Emir yonet (acik pozisyonlar icin SL/TP kontrol)
      4. Yeni pozisyon ac (sinyal + risk kurallarina gore)
      5. Telegram bildirimi gonder

MODLAR:
    paper=True  : Sahte emir, gercek fiyat (guvenli test)
    paper=False : Gercek emir (Binance Testnet veya Live)

CALISTIRMAK ICIN:
    python -m trading.main_loop             # 60sn paper trading
    python -m trading.main_loop --interval 30
    python -m trading.main_loop --once      # Bir sinyal uret, cik

DURDURMA:
    Ctrl+C — acik pozisyonlar korunur (kill-switch icin bkz. risk/kill_switch.py)
"""

from __future__ import annotations
import asyncio
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Proje kokunu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import get_config
from data.fetcher import BinanceFetcher
from data.cleaner import OHLCVCleaner
from strategies.rsi_strategy import RSIStrategy
from strategies.pa_range_strategy import PARangeStrategy
from strategies.regime_detector import MarketRegimeDetector, Regime
from trading.order_manager import OrderManager
from trading.position_tracker import PositionTracker
from risk.risk_manager import RiskManager
from ml.predictor import MLPredictor
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Pozisyon boyutu hesaplayici ───────────────────────────────────────────────

def calc_position_size(
    capital: float,
    price: float,
    atr: float | None,
    risk_pct: float = 0.02,
    stop_pct: float = 0.02,
) -> float:
    """
    ATR veya yuzde tabanli dinamik pozisyon boyutu.
    Sermayenin risk_pct'ini riske atmak icin gereken miktar.

    Formul: risk_amount / (stop_distance * price)
    """
    risk_amount = capital * risk_pct

    if atr and atr > 0:
        # ATR tabanli: 2x ATR stop mesafesi
        stop_distance = 2.0 * atr / price
    else:
        stop_distance = stop_pct

    if stop_distance <= 0:
        return 0.0

    qty = risk_amount / (stop_distance * price)

    # Minimum ve maksimum koruma
    min_qty = 0.0001   # BTC minimum lot
    max_qty = capital * 0.30 / price   # Sermayenin maks %30'u

    return round(max(min_qty, min(qty, max_qty)), 6)


# ── Sinyal Birlestirme ────────────────────────────────────────────────────────

def get_combined_signal(df, cfg, regime, ml_predictor=None, big_regime=None, df_15m=None):
    """
    RSI + PA Range + XGBoost sinyallerini cogunluk oyuyla birlestir.

    Kural:
        - 3 sinyal: RSI, PA, ML (ML yoksa 2 sinyal)
        - Cogunluk: en az 2 sinyal ayni yonde ise o yon secilir
        - Guven: oy verenlerin ortalama guveni + her ek oy icin +0.05 bonus
        - Catisma (hepsi farkli): BEKLE

    big_regime: 4h timeframe rejimi (MTF filtresi)
        - 4h TREND_DOWN iken AL sinyali gelirse -> BEKLE (kontra-trend engeli)
        - 4h TREND_UP   iken SAT sinyali gelirse -> BEKLE (kontra-trend engeli)

    df_15m: 15m giriş zamanlaması (MTF entry filtresi)
        - AL sinyali: 15m RSI > 65 ise BEKLE (15m aşırı alım, geç kalındı)
        - SAT sinyali: 15m RSI < 35 ise BEKLE (15m aşırı satım, geç kalındı)
    """
    rsi_cfg = cfg.strategies.rsi
    pa_cfg  = cfg.strategies.pa_range

    rsi_strategy = RSIStrategy(
        symbol=cfg.general.symbol, timeframe=cfg.general.timeframe,
        rsi_period=rsi_cfg.rsi_period, oversold=rsi_cfg.oversold,
        overbought=rsi_cfg.overbought, stop_pct=rsi_cfg.stop_pct,
        tp_pct=rsi_cfg.tp_pct,
    )
    pa_strategy = PARangeStrategy(
        symbol=cfg.general.symbol, timeframe=cfg.general.timeframe,
        lookback=pa_cfg.lookback, rsi_period=pa_cfg.rsi_period,
        rsi_oversold=pa_cfg.rsi_oversold, rsi_overbought=pa_cfg.rsi_overbought,
        proximity_pct=pa_cfg.proximity_pct, stop_pct=pa_cfg.stop_pct,
        tp_pct=pa_cfg.tp_pct, use_regime_filter=pa_cfg.use_regime_filter,
    )

    rsi_signal = rsi_strategy.generate_signal(df)
    pa_signal  = pa_strategy.generate_signal(df)

    # ML sinyal — model yuklu degilse None
    ml_signal = None
    if ml_predictor is not None:
        try:
            ml_signal = ml_predictor.predict(df)
        except Exception as e:
            logger.warning(f"ML tahmin hatasi: {e}")

    # Tum sinyalleri topla
    signals = [rsi_signal, pa_signal]
    if ml_signal is not None:
        signals.append(ml_signal)

    # Her yone verilen oy sayisi ve toplam guven
    vote_counts = {"AL": 0, "SAT": 0, "BEKLE": 0}
    vote_conf   = {"AL": 0.0, "SAT": 0.0, "BEKLE": 0.0}
    for sig in signals:
        vote_counts[sig.action] += 1
        vote_conf[sig.action]   += sig.confidence

    # Oylama: BEKLE oylarini yoksay, sadece AL vs SAT karsilastir.
    # AL kazanirsa → AL, SAT kazanirsa → SAT, esitse → BEKLE
    # Bu sayede tek bir AL/SAT sinyali bile yeterliyse islem acar.
    al_count  = vote_counts["AL"]
    sat_count = vote_counts["SAT"]

    if al_count > sat_count and al_count > 0:
        avg_conf   = vote_conf["AL"] / al_count
        bonus      = 0.05 * (al_count - 1)
        confidence = min(1.0, avg_conf + bonus)
        action     = "AL"
    elif sat_count > al_count and sat_count > 0:
        avg_conf   = vote_conf["SAT"] / sat_count
        bonus      = 0.05 * (sat_count - 1)
        confidence = min(1.0, avg_conf + bonus)
        action     = "SAT"
    elif al_count > 0 and sat_count > 0:
        # AL ve SAT catisiyor — guvensiz
        action     = "BEKLE"
        confidence = 0.0
    else:
        # Hepsi BEKLE
        action     = "BEKLE"
        confidence = 0.0

    ml_info = f"ML:{ml_signal.action}({ml_signal.confidence:.2f})" if ml_signal else "ML:yok"
    logger.debug(
        f"Sinyal oylari | RSI:{rsi_signal.action}({rsi_signal.confidence:.2f}) "
        f"PA:{pa_signal.action}({pa_signal.confidence:.2f}) {ml_info} "
        f"-> {action}({confidence:.2f})"
    )

    # ── MTF Filtresi: 4h kontra-trend ise guveni %30 dusur (bloke etme) ─────────
    # Eski davranis: tamamen bloke ederdi → hic islem yok
    # Yeni davranis: guveni azalt, RiskManager min_confidence ile filtrele
    if big_regime is not None and action != "BEKLE":
        if action == "AL" and big_regime == Regime.TREND_DOWN:
            confidence = round(confidence * 0.70, 3)
            logger.info(
                f"MTF filtre: 4h={big_regime.value} | AL kontra-trend, "
                f"guven -%30 → {confidence:.2f}"
            )
        elif action == "SAT" and big_regime == Regime.TREND_UP:
            confidence = round(confidence * 0.70, 3)
            logger.info(
                f"MTF filtre: 4h={big_regime.value} | SAT kontra-trend, "
                f"guven -%30 → {confidence:.2f}"
            )
        else:
            logger.info(
                f"MTF filtre: 4h={big_regime.value} | {action} trend yonunde, tam guven"
            )

    # ── 15m Giriş Zamanlaması: Aşırı Zonda Girme ────────────────────────────
    # AL sinyali var ama 15m'de fiyat zaten cok yukseldi → girme (gec kaldin)
    # SAT sinyali var ama 15m'de fiyat zaten cok dustu → girme (gec kaldin)
    if df_15m is not None and action != "BEKLE" and len(df_15m) >= 15:
        try:
            import pandas_ta as ta
            rsi_15m = ta.rsi(df_15m["close"], length=14)
            rsi_15m_val = float(rsi_15m.iloc[-1]) if rsi_15m is not None else None

            if rsi_15m_val is not None:
                if action == "AL" and rsi_15m_val > 78:
                    logger.info(
                        f"15m giris filtresi: RSI={rsi_15m_val:.1f} > 78 | "
                        f"AL sinyali asiri alim, BEKLE"
                    )
                    action     = "BEKLE"
                    confidence = 0.0
                elif action == "SAT" and rsi_15m_val < 22:
                    logger.info(
                        f"15m giris filtresi: RSI={rsi_15m_val:.1f} < 22 | "
                        f"SAT sinyali asiri satim, BEKLE"
                    )
                    action     = "BEKLE"
                    confidence = 0.0
                else:
                    logger.info(
                        f"15m giris filtresi: RSI={rsi_15m_val:.1f} | "
                        f"{action} icin uygun giris zamani"
                    )
        except Exception as e:
            logger.warning(f"15m RSI hesaplanamadi: {e}")

    return action, round(confidence, 3), rsi_signal, pa_signal, ml_signal


# ── TradingBot ────────────────────────────────────────────────────────────────

class TradingBot:
    """
    Paper / Live trading botu.

    Ozellikler:
        - Asyncio tabanli ana dongu
        - OrderManager + PositionTracker entegrasyonu
        - Strateji sinyallerini birlestirme
        - SL/TP otomatik kontrol
        - Hata yonetimi + otomatik yeniden baglanti

    Parametreler:
        paper    : True = paper trading (guvenli)
        interval : Kac saniyede bir guncelleme yapilsin
        capital  : Baslangic sermayesi (USDT)
    """

    def __init__(
        self,
        paper: bool    = True,
        interval: int  = 60,
        capital: float = 10_000.0,
    ):
        self.paper    = paper
        self.interval = interval

        self.cfg      = get_config()
        self.fetcher  = BinanceFetcher(
            testnet   = self.cfg.general.testnet,
            symbol    = self.cfg.general.symbol,
            timeframe = self.cfg.general.timeframe,
        )
        # MTF: 4h buyuk trend icin ayri fetcher
        self.fetcher_4h = BinanceFetcher(
            testnet   = self.cfg.general.testnet,
            symbol    = self.cfg.general.symbol,
            timeframe = "4h",
        )
        # MTF: 15m giris zamanlama icin ayri fetcher
        self.fetcher_15m = BinanceFetcher(
            testnet   = self.cfg.general.testnet,
            symbol    = self.cfg.general.symbol,
            timeframe = "15m",
        )
        self.cleaner    = OHLCVCleaner()
        self.detector   = MarketRegimeDetector()
        self.detector_4h= MarketRegimeDetector()   # 4h rejim dedektoru

        self.order_manager    = OrderManager(
            symbol    = self.cfg.general.symbol,
            paper     = paper,
            commission= 0.001,
            slippage  = 0.0005,
        )
        self.position_tracker = PositionTracker(
            initial_capital = capital,
            max_positions   = 1,   # Bir seferde maks 1 BTC/USDT pozisyonu
            commission      = 0.001,
        )
        # RiskManager: pozisyon boyutu + kill switch + min guven esigi
        self.risk_manager = RiskManager(
            initial_capital    = capital,
            max_risk_pct       = 0.02,     # Tek islemde maks %2 risk
            max_open_positions = 1,
            min_confidence     = 0.30,
            sl_atr_mult        = 2.0,
            tp_atr_mult        = 3.0,
        )

        # ML model — dosya varsa yukle, yoksa ML olmadan calis
        _model_path = Path(__file__).parent.parent / "ml" / "models" / "xgb_btc_1h.json"
        if _model_path.exists():
            try:
                self.ml_predictor = MLPredictor.from_file(
                    _model_path,
                    symbol    = self.cfg.general.symbol,
                    timeframe = self.cfg.general.timeframe,
                )
                logger.info(f"ML model yuklendi: {_model_path.name}")
            except Exception as e:
                logger.warning(f"ML model yuklenemedi ({e}), ML olmadan devam ediliyor.")
                self.ml_predictor = None
        else:
            logger.info("ML model dosyasi bulunamadi, ML olmadan calisiliyor.")
            self.ml_predictor = None

        self._running        = False
        self._iteration      = 0
        self._errors         = 0
        self._equity_history = []   # [{ts, equity, price}] — dashboard icin
        self._last_retrain   = 0    # Son retrain'in iteration numarasi

        # Onceki oturum varsa yukle
        self.load_state()

        mode = "PAPER" if paper else "LIVE (TESTNET)"
        logger.info(
            f"TradingBot baslatildi | Mod: {mode} | "
            f"{self.cfg.general.symbol} {self.cfg.general.timeframe} | "
            f"Sermaye: ${capital:,.2f}"
        )

    # ── Ana Dongu ─────────────────────────────────────────────────────────────

    async def run(self, once: bool = False) -> None:
        """Ana asyncio dongusu."""
        self._running = True
        logger.info(f"Bot dongusu basliyor. Aralik: {self.interval}sn")

        while self._running:
            try:
                await self._tick()
                self._errors = 0
                self.risk_manager.clear_errors()   # Basarili tick: hata sayacini sifirla
            except Exception as e:
                self._errors += 1
                self.risk_manager.record_error()   # Kill switch hata sayacini artir
                logger.error(f"Tick hatasi #{self._errors}: {e}")
                if self._errors >= 5:
                    logger.critical("Ust uste 5 hata! Bot durduruluyor.")
                    self._running = False
                    break

            if once:
                break

            # Sonraki tick'i bekle
            await asyncio.sleep(self.interval)

        logger.info("Bot dongusu bitti.")
        self.position_tracker.print_summary()

    async def _tick(self) -> None:
        """
        Tek bir dongu adimi:
        1. Veri cek
        2. Acik pozisyonlar icin SL/TP kontrol
        3. Yeni sinyal uret
        4. Pozisyon ac/kapat
        """
        self._iteration += 1
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        logger.info(f"--- Tick #{self._iteration} | {now} ---")

        # 1. Veri cek (1h ve 4h paralel)
        df = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_data
        )
        if df is None or df.empty:
            logger.warning("Veri alinamadi, tick atlaniyor.")
            return

        # MTF: 4h trend + 15m giris verisi (hata olursa None, bot yine de calisir)
        df_4h, df_15m = await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(None, self._fetch_data_4h),
            asyncio.get_event_loop().run_in_executor(None, self._fetch_data_15m),
        )
        big_regime = None
        if df_4h is not None and not df_4h.empty:
            big_regime = self.detector_4h.detect(df_4h)

        current_price = float(df["close"].iloc[-1])
        regime        = self.detector.detect(df)

        logger.info(
            f"Fiyat: ${current_price:,.2f} | "
            f"1h Rejim: {regime.value} | "
            f"4h Rejim: {big_regime.value if big_regime else 'N/A'} | "
            f"15m: {'OK' if df_15m is not None else 'N/A'} | "
            f"Veri: {len(df)} bar"
        )

        # 2. Acik pozisyonlarda SL/TP + kill switch kontrol
        capital  = self.position_tracker.capital
        open_pnl = self.position_tracker.update(current_price)["total_unrealized"]
        # Gercek equity: nakit + kilitli notional + unrealized P&L
        # (Sadece nakit gonderilirse KillSwitch pozisyon acarken yanlis alarm verir)
        locked_notional = sum(
            p.notional for p in self.position_tracker.open_positions()
        )
        true_equity = capital + locked_notional + open_pnl
        exits = self.risk_manager.check_exit_conditions(
            position_tracker = self.position_tracker,
            current_price    = current_price,
            current_capital  = true_equity,
            open_pnl         = open_pnl,
        )
        for position_id, reason in exits:
            self._close_position(position_id, current_price, reason)

        # 3. Sinyal uret (RSI + PA + ML + 4h trend filtresi + 15m giris zamanlama)
        action, confidence, rsi_sig, pa_sig, ml_sig = get_combined_signal(
            df, self.cfg, regime, self.ml_predictor,
            big_regime=big_regime, df_15m=df_15m
        )

        ml_part = (
            f"ML: {ml_sig.action} ({ml_sig.confidence:.2f})"
            if ml_sig else "ML: yok"
        )
        logger.info(
            f"Sinyal: {action} | Guven: {confidence:.2f} | "
            f"RSI: {rsi_sig.action} ({rsi_sig.confidence:.2f}) | "
            f"PA: {pa_sig.action} ({pa_sig.confidence:.2f}) | "
            f"{ml_part}"
        )

        # ATR hesapla (pozisyon boyutu + SL/TP icin)
        try:
            import pandas_ta as ta
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
            atr_val = float(atr_series.iloc[-1]) if atr_series is not None else None
        except Exception:
            atr_val = None

        # 4. RiskManager ile sinyal degerlendir
        open_positions_count = len(self.position_tracker.open_positions())
        decision = self.risk_manager.evaluate_signal(
            action               = action,
            confidence           = confidence,
            current_capital      = true_equity,
            open_pnl             = open_pnl,
            price                = current_price,
            atr                  = atr_val,
            open_positions_count = open_positions_count,
        )

        logger.info(f"Risk karari: {decision}")

        if decision.approved:
            direction = "LONG" if action == "AL" else "SHORT"

            # Emir gonder (RiskManager'dan gelen miktar)
            order = self.order_manager.place_market_order(
                side          = "buy" if direction == "LONG" else "sell",
                quantity      = decision.quantity,
                current_price = current_price,
            )

            # Pozisyon ac (RiskManager'dan gelen SL/TP)
            self.position_tracker.open_position(
                symbol      = self.cfg.general.symbol,
                direction   = direction,
                entry_price = order.filled_price or current_price,
                quantity    = decision.quantity,
                stop_loss   = decision.stop_loss,
                take_profit = decision.take_profit,
                strategy    = f"RSI+PA+ML ({action})" if self.ml_predictor else f"RSI+PA ({action})",
                order_id    = order.order_id,
                entry_fee   = order.fee,
            )

            # Kill switch'e islem kaydet
            self.risk_manager.record_trade_executed()

        # Guncel durumu logla
        summary = self.position_tracker.update(current_price)
        logger.info(
            f"Equity: ${summary['equity']:,.2f} | "
            f"Unrealized PnL: ${summary['total_unrealized']:,.2f} | "
            f"Acik pos: {summary['open_positions']}"
        )

        # Equity gecmisine ekle (dashboard icin — son 1000 nokta tutulur)
        self._equity_history.append({
            "ts"    : datetime.now(timezone.utc).isoformat(),
            "equity": round(summary["equity"], 2),
            "price" : round(current_price, 2),
        })
        if len(self._equity_history) > 1000:
            self._equity_history = self._equity_history[-1000:]

        # Auto-retrain: her 720 tick'te bir (1h bot = 30 gun) ML modeli yenile
        self._maybe_retrain()

        # State'i diske kaydet (PC kapansa bile devam eder)
        self.save_state()

    def _fetch_data(self):
        """Senkron 1h veri cekimi (executor icinde calisir)."""
        try:
            df_raw = self.fetcher.fetch_ohlcv(limit=200)
            if df_raw.empty:
                return None
            return self.cleaner.clean(df_raw)
        except Exception as e:
            logger.warning(f"1h veri alinamadi: {e}")
            return None

    def _fetch_data_4h(self):
        """MTF: 4h trend verisi cekimi. Hata olursa None doner (bot durdurmaz)."""
        try:
            df_raw = self.fetcher_4h.fetch_ohlcv(limit=100)
            if df_raw.empty:
                return None
            return self.cleaner.clean(df_raw)
        except Exception as e:
            logger.warning(f"4h MTF verisi alinamadi: {e}")
            return None

    def _fetch_data_15m(self):
        """MTF: 15m giris zamanlama verisi cekimi. Hata olursa None doner (bot durdurmaz)."""
        try:
            df_raw = self.fetcher_15m.fetch_ohlcv(limit=100)
            if df_raw.empty:
                return None
            return self.cleaner.clean(df_raw)
        except Exception as e:
            logger.warning(f"15m MTF verisi alinamadi: {e}")
            return None

    def _close_position(
        self, position_id: str, price: float, reason: str
    ) -> None:
        """Pozisyon kapatir ve kapanma emri gonderir."""
        pos = self.position_tracker.get_position(position_id)
        if not pos:
            return

        # Kapanma emri
        close_side = "sell" if pos.direction == "LONG" else "buy"
        order = self.order_manager.place_market_order(
            side          = close_side,
            quantity      = pos.quantity,
            current_price = price,
        )

        # Pozisyon kapat
        self.position_tracker.close_position(
            position_id = position_id,
            exit_price  = order.filled_price or price,
            exit_reason = reason,
            exit_fee    = order.fee,
        )

    # ── State Kayit / Yukle ───────────────────────────────────────────────────

    _STATE_FILE = Path(__file__).parent.parent / "data" / "bot_state.json"

    def save_state(self) -> None:
        """
        Bot durumunu JSON dosyasina kaydeder.
        Her tick sonunda otomatik cagrilir — PC kapansa bile devam eder.

        Kaydedilenler:
            - Guncel sermaye
            - Tum kapanmis islemler (P&L gecmisi)
            - Equity peak / max drawdown
            - Toplam tick sayisi
        """
        pt = self.position_tracker

        # Kapanmis islemleri serialize et
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

        # Acik pozisyonlari serialize et (basit bilgi)
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
            "saved_at"       : datetime.now(timezone.utc).isoformat(),
            "capital"        : round(pt.capital, 4),
            "initial_capital": round(pt.initial_capital, 4),
            "equity_peak"    : round(pt._equity_peak, 4),
            "max_drawdown"   : round(pt._max_drawdown, 6),
            "iteration"      : self._iteration,
            "paper"          : self.paper,
            "trades"         : trades,
            "open_positions" : open_pos,
            "equity_history" : self._equity_history,   # dashboard icin
        }

        self._STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self._STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_state(self) -> bool:
        """
        Kaydedilmis durumu yukler.
        __init__ icinde cagrilir — eski oturum varsa kaldigi yerden devam eder.

        Returns:
            True = state yuklendi, False = ilk calistirma
        """
        if not self._STATE_FILE.exists():
            return False

        try:
            with open(self._STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)

            pt = self.position_tracker

            # Sermaye geri yukle
            pt.capital        = state["capital"]
            pt.initial_capital= state["initial_capital"]
            pt._equity_peak   = state.get("equity_peak", state["capital"])
            pt._max_drawdown  = state.get("max_drawdown", 0.0)
            self._iteration   = state.get("iteration", 0)

            # Kapanmis islem gecmisini geri yukle
            from trading.position_tracker import ClosedTrade
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

            # Equity gecmisini geri yukle
            self._equity_history = state.get("equity_history", [])

            # KillSwitch gun-baslangic sermayesini guncelle
            # (yoksa yuklenen sermaye ile initial_capital arasindaki fark
            #  yanlis gunluk zarar olarak algilanir)
            self.risk_manager.update_day_start(pt.capital)

            saved_at = state.get("saved_at", "?")
            n_trades = len(state.get("trades", []))
            logger.info(
                f"Onceki oturum yuklendi | "
                f"Kaydedilme: {saved_at[:16]} | "
                f"Sermaye: ${pt.capital:,.2f} | "
                f"{n_trades} kapanmis islem"
            )
            return True

        except Exception as e:
            logger.warning(f"State yuklenemedi ({e}), sifirdan baslanıyor.")
            return False

    # ── Auto-Retrain ──────────────────────────────────────────────────────────

    _RETRAIN_EVERY = 720   # 1h bot icin 720 tick = 30 gun

    def _maybe_retrain(self) -> None:
        """
        Her 720 tick'te bir (yaklasik 30 gun) ML modelini arka planda yeniden egitir.
        Bot calismasini engellemez — ayri thread icinde calisir.
        """
        ticks_since = self._iteration - self._last_retrain
        if ticks_since < self._RETRAIN_EVERY:
            return

        self._last_retrain = self._iteration
        logger.info(f"Auto-retrain tetiklendi (Tick #{self._iteration}, ~30 gun gecti)")

        import threading
        def _retrain_worker():
            try:
                from ml.auto_retrain import retrain
                success = retrain(days=365, quiet=True)
                if success:
                    # Yeni modeli yukle
                    _model_path = Path(__file__).parent.parent / "ml" / "models" / "xgb_btc_1h.json"
                    if _model_path.exists() and self.ml_predictor is not None:
                        from ml.predictor import MLPredictor
                        self.ml_predictor = MLPredictor.from_file(
                            _model_path,
                            symbol=self.cfg.general.symbol,
                            timeframe=self.cfg.general.timeframe,
                        )
                        logger.info("Auto-retrain tamamlandi, yeni model yuklendi.")
                    else:
                        logger.info("Auto-retrain tamamlandi.")
                else:
                    logger.warning("Auto-retrain basarisiz, eski model kullaniliyor.")
            except Exception as e:
                logger.error(f"Auto-retrain hatasi: {e}")

        t = threading.Thread(target=_retrain_worker, daemon=True, name="auto-retrain")
        t.start()

    def stop(self) -> None:
        """Donguyu durdurur."""
        self._running = False
        logger.info("Bot durdurma sinyali alindi.")

    def get_status(self) -> dict:
        """Bot durumunu dict olarak dondurur."""
        risk = self.risk_manager.status()
        return {
            "running"          : self._running,
            "iteration"        : self._iteration,
            "errors"           : self._errors,
            "paper"            : self.paper,
            "open_positions"   : len(self.position_tracker.open_positions()),
            "capital"          : round(self.position_tracker.capital, 2),
            "kill_switch_level": risk["kill_switch_level"],
            "kill_switch_active": risk["kill_switch_active"],
            "day_trades"       : risk["day_trades"],
        }


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Algo Trade Codex - Trading Bot")
    parser.add_argument("--interval", type=int, default=60,
                        help="Guncelleme suresi (saniye)")
    parser.add_argument("--once",     action="store_true",
                        help="Bir kez calistir, cik")
    parser.add_argument("--capital",  type=float, default=10_000.0,
                        help="Baslangic sermayesi (USDT)")
    parser.add_argument("--live",     action="store_true",
                        help="DIKKAT: gercek emir moduну etkinlestirir")
    args = parser.parse_args()

    if args.live:
        print("\n[UYARI] LIVE MOD AKTIF — Gercek emirler gonderilecek!")
        print("Emin misiniz? (evet yazin, baska bir sey yazi iptal eder)")
        confirm = input("> ").strip()
        if confirm.lower() != "evet":
            print("Iptal edildi.")
            return

    bot = TradingBot(
        paper    = not args.live,
        interval = args.interval,
        capital  = args.capital,
    )

    try:
        await bot.run(once=args.once)
    except KeyboardInterrupt:
        bot.stop()
        print("\nBot durduruldu.")


if __name__ == "__main__":
    asyncio.run(main())
