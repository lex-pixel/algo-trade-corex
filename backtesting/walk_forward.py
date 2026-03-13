"""
backtesting/walk_forward.py
============================
AMACI:
    Walk-forward validation (yuruyen pencere dogrulama).
    Geleneksel backtest'in "overfitting" sorununu cozer:
    - Veriyi egitim + test bolumlerine ayirir
    - Her pencerede strateji egitim veriyle "goruluyor", test veriyle olculuyor
    - Sonuclar gercek performansi daha iyi yansitir

YONTEMLER:
    1. Expanding Window (Buyuyen Pencere) — klasik walk-forward
       Egitim: [0 -> t], Test: [t -> t+step]
       Her adimda egitim verisi buyur, test penceresi sabit

    2. Rolling Window (Kayan Pencere) — sabit boyutlu
       Egitim: [t -> t+train_size], Test: [t+train_size -> t+train_size+test_size]
       Her adimda hem egitim hem test kayar

KULLANIM:
    from backtesting.walk_forward import WalkForwardValidator
    from strategies.pa_range_strategy import PARangeStrategy

    validator = WalkForwardValidator(
        method="rolling",
        train_bars=500,
        test_bars=100,
        step_bars=50,
    )
    results = validator.run(df, PARangeStrategy())
    validator.print_summary(results)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from strategies.base_strategy import BaseStrategy
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardPeriod:
    """Tek bir walk-forward donemini temsil eder."""
    period_num    : int
    train_start   : str
    train_end     : str
    test_start    : str
    test_end      : str
    train_bars    : int
    test_bars     : int
    result        : Optional[BacktestResult] = None

    @property
    def metrics(self) -> dict:
        if self.result:
            return self.result.summary()
        return {}

    @property
    def return_pct(self) -> float:
        m = self.metrics
        return m.get("total_return_pct", 0.0)

    @property
    def sharpe(self) -> float:
        m = self.metrics
        return m.get("sharpe_ratio", 0.0)


@dataclass
class WalkForwardResult:
    """Tum walk-forward donemi sonuclari."""
    method      : str
    strategy    : str
    symbol      : str
    periods     : list[WalkForwardPeriod] = field(default_factory=list)

    @property
    def all_metrics(self) -> list[dict]:
        return [p.metrics for p in self.periods if p.result]

    @property
    def avg_return_pct(self) -> float:
        rets = [p.return_pct for p in self.periods if p.result]
        return sum(rets) / len(rets) if rets else 0.0

    @property
    def avg_sharpe(self) -> float:
        sharpes = [p.sharpe for p in self.periods if p.result]
        return sum(sharpes) / len(sharpes) if sharpes else 0.0

    @property
    def win_periods(self) -> int:
        """Pozitif getiri saglayan donem sayisi."""
        return sum(1 for p in self.periods if p.return_pct > 0)

    @property
    def total_periods(self) -> int:
        return len([p for p in self.periods if p.result])

    @property
    def period_win_rate(self) -> float:
        if self.total_periods == 0:
            return 0.0
        return self.win_periods / self.total_periods * 100

    def combined_return_pct(self) -> float:
        """
        Tum donemler boyunca bilesik getiri.
        Her donemde baslangic sermayesi bir oncekinin bitis sermayesidir.
        """
        compound = 1.0
        for p in self.periods:
            if p.result:
                r = p.return_pct / 100
                compound *= (1 + r)
        return (compound - 1) * 100


class WalkForwardValidator:
    """
    Walk-forward backtest motoru.

    Args:
        method      : 'expanding' veya 'rolling'
        train_bars  : Egitim periyodu uzunlugu (mum sayisi)
        test_bars   : Test periyodu uzunlugu (mum sayisi)
        step_bars   : Her adimda ilerleme miktari (None = test_bars)
        commission  : Komisyon orani (varsayilan %0.1)
        slippage    : Slipaj orani (varsayilan %0.05)
        initial_capital: Baslangic sermayesi
    """

    def __init__(
        self,
        method: str = "rolling",
        train_bars: int = 500,
        test_bars: int = 100,
        step_bars: int | None = None,
        commission: float = 0.001,
        slippage: float = 0.0005,
        initial_capital: float = 10_000.0,
    ):
        self.method          = method
        self.train_bars      = train_bars
        self.test_bars       = test_bars
        self.step_bars       = step_bars or test_bars
        self.commission      = commission
        self.slippage        = slippage
        self.initial_capital = initial_capital

        self._engine = BacktestEngine(
            initial_capital  = initial_capital,
            commission       = commission,
            slippage         = slippage,
        )

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
    ) -> WalkForwardResult:
        """
        Walk-forward validasyonu calistirir.

        Args:
            df       : Tum OHLCV verisi
            strategy : Test edilecek strateji

        Returns:
            WalkForwardResult: Tum donem sonuclari
        """
        n = len(df)
        if n < self.train_bars + self.test_bars:
            logger.warning(
                f"Walk-forward icin yetersiz veri: {n} bar, "
                f"min {self.train_bars + self.test_bars} gerekli"
            )
            return WalkForwardResult(method=self.method, strategy=strategy.name, symbol=strategy.symbol)

        result = WalkForwardResult(
            method   = self.method,
            strategy = strategy.name,
            symbol   = strategy.symbol,
        )

        period_num = 0
        start_idx  = 0

        while True:
            period_num += 1

            if self.method == "expanding":
                # Buyuyen pencere: egitim hep bastan baslar
                train_start = 0
                train_end   = self.train_bars + (period_num - 1) * self.step_bars
                test_start  = train_end
                test_end    = test_start + self.test_bars

            else:  # rolling
                # Kayan pencere: sabit egitim boyutu
                train_start = start_idx
                train_end   = train_start + self.train_bars
                test_start  = train_end
                test_end    = test_start + self.test_bars

            if test_end > n:
                break

            train_df = df.iloc[train_start:train_end]
            test_df  = df.iloc[test_start:test_end]

            logger.info(
                f"WF Donem {period_num}: egitim [{train_start}:{train_end}] "
                f"test [{test_start}:{test_end}]"
            )

            # Tam veri ile test calis (egitim warmup + test)
            # Strateji egitim veriyi "gormus" sayilir, test uzerinde olculur
            combined_df = df.iloc[train_start:test_end]

            try:
                backtest_result = self._engine.run(
                    combined_df,
                    strategy,
                    warmup_bars=len(train_df),
                )
            except Exception as e:
                logger.warning(f"WF Donem {period_num} hatasi: {e}")
                backtest_result = None

            ts_fmt = lambda idx: str(df.index[idx].date()) if idx < n else "N/A"

            period = WalkForwardPeriod(
                period_num  = period_num,
                train_start = ts_fmt(train_start),
                train_end   = ts_fmt(train_end - 1),
                test_start  = ts_fmt(test_start),
                test_end    = ts_fmt(test_end - 1),
                train_bars  = len(train_df),
                test_bars   = len(test_df),
                result      = backtest_result,
            )
            result.periods.append(period)

            # Sonraki adim
            start_idx += self.step_bars
            if start_idx + self.train_bars + self.test_bars > n:
                break

        logger.info(
            f"Walk-forward tamamlandi: {period_num} donem | "
            f"Ort. Getiri: %{result.avg_return_pct:.2f} | "
            f"Donem Win Rate: %{result.period_win_rate:.0f}"
        )

        return result

    def print_summary(self, result: WalkForwardResult) -> None:
        """Terminal ozeti yazdirir."""
        print("\n" + "=" * 75)
        print(f"  WALK-FORWARD SONUCU — {result.strategy} ({result.method.upper()})")
        print("=" * 75)
        print(
            f"  {'Donem':>5}  {'Egitim':>22}  {'Test':>22}  "
            f"{'Getiri':>8}  {'Sharpe':>7}  {'Islem':>5}  {'WR':>6}"
        )
        print(f"  {'-' * 73}")

        for p in result.periods:
            m    = p.metrics
            ret  = m.get("total_return_pct", 0)
            sh   = m.get("sharpe_ratio", 0)
            tr   = m.get("total_trades", 0)
            wr   = m.get("win_rate_pct", 0)
            sign = "+" if ret >= 0 else ""
            print(
                f"  {p.period_num:>5}  "
                f"{p.train_start}..{p.train_end}  "
                f"{p.test_start}..{p.test_end}  "
                f"{sign}{ret:>6.2f}%  "
                f"{sh:>7.3f}  "
                f"{tr:>5}  "
                f"{wr:>5.1f}%"
            )

        print(f"  {'-' * 73}")
        print(f"  {'ORTALAMA':>5}  {'':>22}  {'':>22}  "
              f"{result.avg_return_pct:>+7.2f}%  "
              f"{result.avg_sharpe:>7.3f}")
        print()
        print(f"  Toplam Donem  : {result.total_periods}")
        print(f"  Kazanan Donem : {result.win_periods}  (%{result.period_win_rate:.0f})")
        print(f"  Bilesik Getiri: %{result.combined_return_pct():+.2f}")
        print("=" * 75 + "\n")
