"""
backtesting/engine.py
======================
AMACI:
    Herhangi bir stratejiyi geçmiş veriye karşı koşturur.
    Her mumda strateji çalıştırılır, sinyal gelirse işlem açılır/kapatılır.
    Gerçekçi maliyet modeli: komisyon + slipaj.

NE YAPAR:
    1. OHLCV verisini alır
    2. Her mumda stratejiyi çalıştırır (sinyale bakar)
    3. AL sinyali → Long pozisyon açar
    4. SAT sinyali veya stop-loss/take-profit → pozisyonu kapatır
    5. Her işlemi trade_log'a kaydeder
    6. Equity curve (sermaye eğrisi) oluşturur

BACKTEST vs GERÇEK HAYAT FARKI:
    - Lookahead bias: Bu engine mumun kapanışında sinyal üretir (gerçekçi)
    - Slipaj: Emir fiyatı = kapanış + küçük kayma (gerçekçi)
    - Komisyon: Binance Spot %0.1 her taraf = round-trip %0.2

KULLANIM:
    from backtesting.engine import BacktestEngine
    from strategies.rsi_strategy import RSIStrategy

    engine = BacktestEngine(initial_capital=10_000, commission=0.001, slippage=0.0005)
    result = engine.run(df, RSIStrategy())
    print(result.summary())
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy, Signal
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VERİ SINIFLARI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """
    Tamamlanmış bir işlemi temsil eder.
    Hem açılış hem kapanış bilgisini içerir.
    """
    entry_time:  datetime
    exit_time:   datetime
    entry_price: float
    exit_price:  float
    direction:   str        # 'LONG' | 'SHORT' (şimdilik sadece LONG)
    size:        float      # Pozisyon büyüklüğü (BTC miktarı)
    pnl:         float      # Net kar/zarar (USD, komisyon düşüldükten sonra)
    pnl_pct:     float      # Yüzde getiri
    exit_reason: str        # 'SIGNAL' | 'STOP_LOSS' | 'TAKE_PROFIT' | 'END_OF_DATA'

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def duration_hours(self) -> float:
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 3600


@dataclass
class BacktestResult:
    """
    Backtest sonuçlarını tutan sınıf.
    Engine.run() bu objeyi döndürür.
    """
    strategy_name:  str
    symbol:         str
    timeframe:      str
    start_date:     str
    end_date:       str
    initial_capital: float
    final_capital:  float

    trades:         list[Trade]           = field(default_factory=list)
    equity_curve:   pd.Series            = field(default_factory=pd.Series)

    def summary(self) -> dict:
        """Temel metrikleri dict olarak döndürür."""
        from backtesting.metrics import PerformanceMetrics
        return PerformanceMetrics.calculate(self)

    def print_summary(self) -> None:
        """Terminale güzel formatlanmış özet yazdırır."""
        s = self.summary()
        print("\n" + "=" * 55)
        print(f"  BACKTEST SONUCU — {self.strategy_name}")
        print("=" * 55)
        print(f"  Sembol       : {self.symbol} {self.timeframe}")
        print(f"  Tarih        : {self.start_date} -> {self.end_date}")
        print(f"  Baslangic    : ${self.initial_capital:,.2f}")
        print(f"  Bitis        : ${self.final_capital:,.2f}")
        print("-" * 55)
        print(f"  Toplam Getiri: %{s.get('total_return_pct', 0):.2f}")
        print(f"  Sharpe       : {s.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown : %{s.get('max_drawdown_pct', 0):.2f}")
        print(f"  Islem Sayisi : {s.get('total_trades', 0)}")
        print(f"  Kazanma Orani: %{s.get('win_rate_pct', 0):.1f}")
        print(f"  Profit Factor: {s.get('profit_factor', 0):.2f}")
        print(f"  Ort. Islem   : %{s.get('avg_trade_pct', 0):.3f}")
        print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Olay tabanlı (event-driven) basit backtest motoru.

    Her mumun kapanışında strateji çalıştırılır.
    Bu "bar-close" yaklaşımı gerçekçidir:
        - O mumu henüz görmemişiz gibi davranıyoruz
        - Sinyal o mumun kapanışında oluşuyor
        - Emir bir sonraki mumun açılışında gerçekleşiyor (slipaj eklenmiş)

    Parametreler:
        initial_capital : Başlangıç sermayesi (USD)
        commission      : İşlem başına komisyon oranı (0.001 = %0.1)
        slippage        : Slipaj oranı (0.0005 = %0.05) — emir kaymasi
        max_risk_per_trade: Tek işlemde riske atılacak max sermaye oranı
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        commission: float = 0.001,       # %0.1 — Binance Spot
        slippage: float = 0.0005,        # %0.05
        max_risk_per_trade: float = 0.02, # %2 risk per trade
    ):
        self.initial_capital    = initial_capital
        self.commission         = commission
        self.slippage           = slippage
        self.max_risk_per_trade = max_risk_per_trade

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        warmup_bars: int = 30,
    ) -> BacktestResult:
        """
        Backtest'i çalıştırır.

        Args:
            df          : OHLCV DataFrame (tüm geçmiş)
            strategy    : Çalıştırılacak strateji
            warmup_bars : İlk bu kadar mum indikatör ısınması için atlanır

        Returns:
            BacktestResult: Tüm işlemler + equity curve
        """
        if len(df) < warmup_bars + 2:
            logger.warning(f"Backtest icin yetersiz veri: {len(df)} bar")
            return self._empty_result(strategy, df)

        logger.info(
            f"Backtest basliyor | {strategy.name} | {strategy.symbol} | "
            f"{len(df)} bar | Sermaye: ${self.initial_capital:,.0f}"
        )

        capital      = self.initial_capital
        position     = None      # Açık pozisyon varsa dict, yoksa None
        trades: list[Trade] = []
        equity_values = []
        equity_times  = []

        for i in range(warmup_bars, len(df)):
            df_slice    = df.iloc[:i + 1]        # Strateji bu kadar veri görüyor
            current_bar = df.iloc[i]
            current_price = float(current_bar["close"])
            current_time  = df.index[i]

            # Equity curve: açık pozisyon varsa mark-to-market değeri
            if position:
                unrealized_pnl = (current_price - position["entry_price"]) * position["size"]
                equity_values.append(capital + unrealized_pnl)
            else:
                equity_values.append(capital)
            equity_times.append(current_time)

            # Açık pozisyon varsa stop-loss / take-profit kontrolü
            if position:
                exit_reason = self._check_exit(current_bar, position)
                if exit_reason:
                    trade, capital = self._close_position(
                        position, current_price, current_time, capital, exit_reason
                    )
                    trades.append(trade)
                    position = None
                    continue

            # Stratejiyi çalıştır
            try:
                signal = strategy.generate_signal(df_slice)
            except Exception as e:
                logger.warning(f"Strateji hatasi bar {i}: {e}")
                continue

            # Sinyal işleme
            if signal.action == "AL" and position is None:
                # Pozisyon aç
                position = self._open_position(
                    signal, current_price, current_time, capital
                )

            elif signal.action == "SAT" and position is not None:
                # Sinyal kapama
                trade, capital = self._close_position(
                    position, current_price, current_time, capital, "SIGNAL"
                )
                trades.append(trade)
                position = None

        # Veri bitti, açık pozisyon varsa kapat
        if position:
            last_price = float(df.iloc[-1]["close"])
            last_time  = df.index[-1]
            trade, capital = self._close_position(
                position, last_price, last_time, capital, "END_OF_DATA"
            )
            trades.append(trade)

        equity_curve = pd.Series(equity_values, index=equity_times)

        logger.info(
            f"Backtest tamamlandi | {len(trades)} islem | "
            f"Baslangic: ${self.initial_capital:,.2f} -> Bitis: ${capital:,.2f} | "
            f"Getiri: %{(capital/self.initial_capital - 1)*100:.2f}"
        )

        return BacktestResult(
            strategy_name   = strategy.name,
            symbol          = strategy.symbol,
            timeframe       = strategy.timeframe,
            start_date      = str(df.index[warmup_bars].date()),
            end_date        = str(df.index[-1].date()),
            initial_capital = self.initial_capital,
            final_capital   = capital,
            trades          = trades,
            equity_curve    = equity_curve,
        )

    # ── Pozisyon Yönetimi ─────────────────────────────────────────────────────

    def _open_position(
        self,
        signal: Signal,
        price: float,
        time: datetime,
        capital: float,
    ) -> dict:
        """
        Yeni pozisyon açar.
        Pozisyon büyüklüğü: max_risk_per_trade / stop_loss_distance ile hesaplanır.
        """
        # Slipaj: alış emri biraz daha pahalıya gerçekleşir
        entry_price = price * (1 + self.slippage)

        # Pozisyon boyutlandırma
        # Eğer stop_loss varsa: riske edilen sermaye / stop mesafesi
        if signal.stop_loss and signal.stop_loss < entry_price:
            risk_amount    = capital * self.max_risk_per_trade
            stop_distance  = entry_price - signal.stop_loss
            size           = risk_amount / stop_distance
            # Maksimum: tüm sermayenin %100'ü ile alınabilecek kadar
            max_size = (capital / entry_price) * 0.95
            size     = min(size, max_size)
        else:
            # Stop yoksa sabit %20 sermaye kullan
            size = (capital * 0.20) / entry_price

        size = max(size, 0.0001)   # Minimum pozisyon

        # Açılış komisyonu
        commission_cost = entry_price * size * self.commission

        return {
            "entry_price"  : entry_price,
            "entry_time"   : time,
            "size"         : size,
            "stop_loss"    : signal.stop_loss,
            "take_profit"  : signal.take_profit,
            "commission_in": commission_cost,
        }

    def _close_position(
        self,
        position: dict,
        price: float,
        time: datetime,
        capital: float,
        reason: str,
    ) -> tuple[Trade, float]:
        """
        Pozisyonu kapatır ve Trade objesi + yeni sermayeyi döndürür.
        """
        # Slipaj: satış emri biraz daha ucuza gerçekleşir
        exit_price = price * (1 - self.slippage)

        size         = position["size"]
        entry_price  = position["entry_price"]
        commission_out = exit_price * size * self.commission

        # Brüt P&L
        gross_pnl = (exit_price - entry_price) * size

        # Net P&L (iki taraf komisyon düşüldükten sonra)
        net_pnl = gross_pnl - position["commission_in"] - commission_out

        # Yüzde getiri (başlangıç değerine göre)
        invested = entry_price * size
        pnl_pct  = (net_pnl / invested) * 100 if invested > 0 else 0.0

        new_capital = capital + net_pnl

        trade = Trade(
            entry_time  = position["entry_time"],
            exit_time   = time,
            entry_price = entry_price,
            exit_price  = exit_price,
            direction   = "LONG",
            size        = size,
            pnl         = round(net_pnl, 4),
            pnl_pct     = round(pnl_pct, 4),
            exit_reason = reason,
        )

        return trade, new_capital

    def _check_exit(self, bar: pd.Series, position: dict) -> Optional[str]:
        """
        Mevcut mumda stop-loss veya take-profit tetiklendi mi kontrol eder.
        Low fiyatı stop-loss altına düştüyse → STOP_LOSS
        High fiyatı take-profit üstüne çıktıysa → TAKE_PROFIT
        """
        low  = float(bar["low"])
        high = float(bar["high"])

        if position.get("stop_loss") and low <= position["stop_loss"]:
            return "STOP_LOSS"

        if position.get("take_profit") and high >= position["take_profit"]:
            return "TAKE_PROFIT"

        return None

    def _empty_result(self, strategy: BaseStrategy, df: pd.DataFrame) -> BacktestResult:
        return BacktestResult(
            strategy_name    = strategy.name,
            symbol           = strategy.symbol,
            timeframe        = strategy.timeframe,
            start_date       = str(df.index[0].date()) if len(df) else "N/A",
            end_date         = str(df.index[-1].date()) if len(df) else "N/A",
            initial_capital  = self.initial_capital,
            final_capital    = self.initial_capital,
        )
