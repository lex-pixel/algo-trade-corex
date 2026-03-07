"""
backtesting/metrics.py
=======================
AMACI:
    Backtest sonuçlarından performans metriklerini hesaplar.

METRİKLER VE ANLAMLARI:
    total_return_pct  : Toplam yüzde getiri. Örn: %15.3
    sharpe_ratio      : Risk-ayarlı getiri. >1 iyi, >2 çok iyi, <0 kötü.
                        Formül: (Ortalama Günlük Getiri / Std) * sqrt(365)
    sortino_ratio     : Sharpe gibi ama sadece aşağı volatilite cezalandırılır.
                        Daha adil bir ölçüt.
    max_drawdown_pct  : Zirve'den en derin düşüş. Örn: -%12.4
                        Bu ne kadar düşük olursa o kadar iyi.
    win_rate_pct      : Kazanan işlem yüzdesi. Örn: %55.0
    profit_factor     : Toplam kazanç / Toplam kayıp. >1.5 iyi, >2.0 çok iyi.
    avg_trade_pct     : Ortalama işlem getirisi. Pozitif olmalı.
    total_trades      : Toplam işlem sayısı.
    max_consecutive_losses: Üst üste max kayıp sayısı (psikoloji için önemli).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    BacktestResult üzerinden tüm metrikleri hesaplar.
    Tüm metodlar static — instance oluşturmaya gerek yok.
    """

    ANNUAL_FACTOR = 365      # Kripto 7/24 çalıştığı için 365 gün

    @classmethod
    def calculate(cls, result) -> dict:
        """
        Tüm metrikleri tek seferde hesaplar ve dict döndürür.

        Args:
            result: BacktestResult objesi

        Returns:
            dict: Tüm metrikler
        """
        trades       = result.trades
        equity_curve = result.equity_curve
        initial      = result.initial_capital
        final        = result.final_capital

        # Temel getiri
        total_return     = final - initial
        total_return_pct = (total_return / initial) * 100 if initial > 0 else 0.0

        # İşlem sayısı ve kazanma oranı
        total_trades = len(trades)
        if total_trades == 0:
            logger.warning("Hic islem yok, metrikler hesaplanamadi")
            return cls._empty_metrics(initial, final, total_return_pct)

        winners = [t for t in trades if t.is_winner]
        losers  = [t for t in trades if not t.is_winner]

        win_rate_pct    = (len(winners) / total_trades) * 100

        # Profit Factor: Toplam kazanç / Toplam kayıp
        total_profit = sum(t.pnl for t in winners) if winners else 0.0
        total_loss   = abs(sum(t.pnl for t in losers)) if losers else 0.001  # sıfıra bölmeyi önle
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Ortalama işlem
        avg_trade_pnl = np.mean([t.pnl for t in trades])
        avg_trade_pct = np.mean([t.pnl_pct for t in trades])

        # Max Drawdown (equity curve üzerinden)
        max_drawdown_pct = cls._calc_max_drawdown(equity_curve)

        # Sharpe & Sortino (günlük getiriler üzerinden)
        sharpe_ratio, sortino_ratio = cls._calc_risk_ratios(equity_curve)

        # Üst üste maksimum kayıp serisi
        max_consec_losses = cls._calc_max_consecutive_losses(trades)

        # Ortalama işlem süresi
        avg_duration_h = np.mean([t.duration_hours for t in trades]) if trades else 0.0

        metrics = {
            # Getiri
            "total_return_usd"  : round(total_return, 2),
            "total_return_pct"  : round(total_return_pct, 3),
            "initial_capital"   : initial,
            "final_capital"     : round(final, 2),

            # Risk metrikleri
            "sharpe_ratio"      : round(sharpe_ratio, 4),
            "sortino_ratio"     : round(sortino_ratio, 4),
            "max_drawdown_pct"  : round(max_drawdown_pct, 3),

            # İşlem istatistikleri
            "total_trades"      : total_trades,
            "winning_trades"    : len(winners),
            "losing_trades"     : len(losers),
            "win_rate_pct"      : round(win_rate_pct, 2),
            "profit_factor"     : round(profit_factor, 4),
            "avg_trade_usd"     : round(avg_trade_pnl, 4),
            "avg_trade_pct"     : round(avg_trade_pct, 4),
            "avg_duration_hours": round(avg_duration_h, 1),
            "max_consec_losses" : max_consec_losses,

            # En iyi / en kötü işlem
            "best_trade_pct"    : round(max(t.pnl_pct for t in trades), 3),
            "worst_trade_pct"   : round(min(t.pnl_pct for t in trades), 3),
        }

        return metrics

    # ── İç Hesaplama Metodları ────────────────────────────────────────────────

    @classmethod
    def _calc_max_drawdown(cls, equity_curve: pd.Series) -> float:
        """
        Equity curve'den maksimum drawdown'ı hesaplar.

        Drawdown = (Zirve - Dip) / Zirve * 100
        En büyük bu değeri döndürür.
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return 0.0

        rolling_max = equity_curve.cummax()
        drawdown    = (equity_curve - rolling_max) / rolling_max * 100
        return float(drawdown.min())   # Negatif değer — en derin düşüş

    @classmethod
    def _calc_risk_ratios(cls, equity_curve: pd.Series) -> tuple[float, float]:
        """
        Sharpe ve Sortino oranlarını hesaplar.

        Günlük getiri serisi kullanılır, yıllıklandırılır.
        """
        if equity_curve.empty or len(equity_curve) < 10:
            return 0.0, 0.0

        # Günlük getiriler (pct_change)
        daily_returns = equity_curve.pct_change(fill_method=None).dropna()

        if len(daily_returns) < 5 or daily_returns.std() == 0:
            return 0.0, 0.0

        mean_return  = daily_returns.mean()
        std_return   = daily_returns.std()

        # Sharpe: Tüm volatilite
        sharpe = (mean_return / std_return) * np.sqrt(cls.ANNUAL_FACTOR)

        # Sortino: Sadece negatif volatilite
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (mean_return / downside.std()) * np.sqrt(cls.ANNUAL_FACTOR)
        else:
            sortino = float("inf") if mean_return > 0 else 0.0

        return float(sharpe), float(sortino)

    @staticmethod
    def _calc_max_consecutive_losses(trades) -> int:
        """Üst üste en fazla kaç işlem kaybedildi?"""
        if not trades:
            return 0
        max_seq = current_seq = 0
        for t in trades:
            if not t.is_winner:
                current_seq += 1
                max_seq = max(max_seq, current_seq)
            else:
                current_seq = 0
        return max_seq

    @staticmethod
    def _empty_metrics(initial: float, final: float, total_return_pct: float) -> dict:
        return {
            "total_return_usd"  : round(final - initial, 2),
            "total_return_pct"  : round(total_return_pct, 3),
            "initial_capital"   : initial,
            "final_capital"     : round(final, 2),
            "sharpe_ratio"      : 0.0,
            "sortino_ratio"     : 0.0,
            "max_drawdown_pct"  : 0.0,
            "total_trades"      : 0,
            "winning_trades"    : 0,
            "losing_trades"     : 0,
            "win_rate_pct"      : 0.0,
            "profit_factor"     : 0.0,
            "avg_trade_usd"     : 0.0,
            "avg_trade_pct"     : 0.0,
            "avg_duration_hours": 0.0,
            "max_consec_losses" : 0,
            "best_trade_pct"    : 0.0,
            "worst_trade_pct"   : 0.0,
        }

    @staticmethod
    def compare(results: list) -> pd.DataFrame:
        """
        Birden fazla backtest sonucunu yan yana karşılaştırır.

        Args:
            results: BacktestResult listesi

        Returns:
            pd.DataFrame: Her strateji bir satır
        """
        rows = []
        for r in results:
            m = r.summary()
            rows.append({
                "Strateji"       : r.strategy_name,
                "Getiri %"       : m["total_return_pct"],
                "Sharpe"         : m["sharpe_ratio"],
                "Max DD %"       : m["max_drawdown_pct"],
                "Islem"          : m["total_trades"],
                "Win Rate %"     : m["win_rate_pct"],
                "Profit Factor"  : m["profit_factor"],
                "Ort. Islem %"   : m["avg_trade_pct"],
            })
        return pd.DataFrame(rows).set_index("Strateji")
