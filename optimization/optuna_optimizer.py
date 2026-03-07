"""
optimization/optuna_optimizer.py
==================================
AMACI:
    Strateji parametrelerini otomatik optimize eder.
    Bayesian optimizasyon (Optuna) kullanır — rastgele değil, akıllıca arar.

NASIL ÇALIŞIR:
    1. Parametre aralıkları tanımla (örn: RSI oversold 20-40 arası)
    2. Optuna her denemede farklı bir kombinasyon seçer
    3. Seçilen parametrelerle backtest çalışır
    4. Sharpe oranı hesaplanır (bu bizim "skor")
    5. Optuna bir sonraki denemeyi önceki sonuçlara göre seçer
    6. N deneme sonunda en iyi parametre seti bulunur

RASTGELE ARAMA vs BAYESIAN:
    Rastgele: 1000 kombinasyonu şans eseri dener
    Bayesian: "RSI 28'de iyi çalıştı, 25-30 arasını daha fazla dene" der
    Bayesian ~10x daha az denemeyle iyi parametre bulur.

ÇALIŞTIRMAK İÇİN:
    python -m optimization.optuna_optimizer               # RSI optimize et
    python -m optimization.optuna_optimizer --strategy pa # PA Range optimize et
    python -m optimization.optuna_optimizer --trials 200  # 200 deneme
"""

import argparse
import warnings
warnings.filterwarnings("ignore")  # Optuna trial gürültüsünü sustur

import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from data.fetcher import BinanceFetcher
from data.cleaner import OHLCVCleaner
from strategies.rsi_strategy import RSIStrategy
from strategies.pa_range_strategy import PARangeStrategy
from backtesting.engine import BacktestEngine
from utils.logger import get_logger

logger = get_logger(__name__)


def load_data(symbol: str = "BTC/USDT", timeframe: str = "1h") -> pd.DataFrame:
    """
    Önce kaydedilmiş Parquet'ten yükler, yoksa Binance'ten çeker.
    Optimizasyon sırasında aynı veri kullanılır (adil karşılaştırma).
    """
    from pathlib import Path
    parquet_path = Path("data/raw/backtest_btc_usdt_1h.parquet")

    if parquet_path.exists():
        logger.info("Kaydedilmis veri yukleniyor...")
        return BinanceFetcher.load_parquet(parquet_path)

    logger.info("Binance'ten veri cekiliyor...")
    from config.loader import get_config
    cfg     = get_config()
    fetcher = BinanceFetcher(testnet=cfg.general.testnet, symbol=symbol, timeframe=timeframe)
    df_raw  = fetcher.fetch_since(since_days=90)
    cleaner = OHLCVCleaner()
    df      = cleaner.clean(df_raw)
    fetcher.save_parquet(df, filename="backtest_btc_usdt_1h.parquet")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RSI STRATEJİSİ OPTİMİZASYONU
# ─────────────────────────────────────────────────────────────────────────────

def optimize_rsi(df: pd.DataFrame, n_trials: int = 100) -> dict:
    """
    RSI stratejisinin parametrelerini optimize eder.

    Aranacak parametreler:
        rsi_period  : 10-25
        oversold    : 20-40
        overbought  : 60-80
        stop_pct    : 0.01-0.03
        tp_pct      : 0.02-0.06

    Hedef: Sharpe oranını maksimize et
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna yuklu degil. Kurmak icin: pip install optuna")
        return {}

    engine = BacktestEngine(initial_capital=10_000, commission=0.001, slippage=0.0005)

    def objective(trial: optuna.Trial) -> float:
        # Parametre aralıklarını tanımla
        rsi_period  = trial.suggest_int("rsi_period",  10, 25)
        oversold    = trial.suggest_int("oversold",    20, 40)
        overbought  = trial.suggest_int("overbought",  60, 80)
        stop_pct    = trial.suggest_float("stop_pct",  0.008, 0.030)
        tp_pct      = trial.suggest_float("tp_pct",    0.015, 0.060)

        # oversold < overbought zorunlu
        if oversold >= overbought:
            return -999.0

        strategy = RSIStrategy(
            rsi_period=rsi_period, oversold=oversold,
            overbought=overbought, stop_pct=stop_pct, tp_pct=tp_pct,
        )

        result  = engine.run(df, strategy, warmup_bars=50)
        metrics = result.summary()

        sharpe = metrics.get("sharpe_ratio", -999)
        trades = metrics.get("total_trades", 0)

        # Az işlem cezalandır (overfitting önlemi)
        if trades < 3:
            return -999.0

        return sharpe

    study = optuna.create_study(direction="maximize", study_name="rsi_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["sharpe"] = round(study.best_value, 4)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# PA RANGE STRATEJİSİ OPTİMİZASYONU
# ─────────────────────────────────────────────────────────────────────────────

def optimize_pa_range(df: pd.DataFrame, n_trials: int = 100) -> dict:
    """
    PA Range stratejisinin parametrelerini optimize eder.

    Aranacak parametreler:
        lookback      : 30-100
        rsi_oversold  : 30-50
        rsi_overbought: 50-70
        proximity_pct : 0.01-0.05
        stop_pct      : 0.01-0.03
        tp_pct        : 0.02-0.06
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna yuklu degil. Kurmak icin: pip install optuna")
        return {}

    engine = BacktestEngine(initial_capital=10_000, commission=0.001, slippage=0.0005)

    def objective(trial: optuna.Trial) -> float:
        lookback       = trial.suggest_int("lookback",       30, 100)
        rsi_oversold   = trial.suggest_int("rsi_oversold",   25, 50)
        rsi_overbought = trial.suggest_int("rsi_overbought", 50, 75)
        proximity_pct  = trial.suggest_float("proximity_pct", 0.01, 0.05)
        stop_pct       = trial.suggest_float("stop_pct",      0.008, 0.030)
        tp_pct         = trial.suggest_float("tp_pct",        0.015, 0.060)

        if rsi_oversold >= rsi_overbought:
            return -999.0

        strategy = PARangeStrategy(
            lookback=lookback, rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought, proximity_pct=proximity_pct,
            stop_pct=stop_pct, tp_pct=tp_pct,
            use_regime_filter=True,
        )

        result  = engine.run(df, strategy, warmup_bars=max(lookback, 50))
        metrics = result.summary()

        sharpe = metrics.get("sharpe_ratio", -999)
        trades = metrics.get("total_trades", 0)

        if trades < 2:
            return -999.0

        return sharpe

    study = optuna.create_study(direction="maximize", study_name="pa_range_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["sharpe"] = round(study.best_value, 4)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# SONUÇLARI YAZDIRMA VE YAML'A KAYDETME
# ─────────────────────────────────────────────────────────────────────────────

def print_results(strategy_name: str, best: dict, original: dict) -> None:
    """Optimizasyon sonuçlarını orijinal değerlerle karşılaştırır."""
    print(f"\n{'=' * 55}")
    print(f"  OPTIMIZASYON SONUCU — {strategy_name}")
    print(f"{'=' * 55}")
    print(f"  {'Parametre':<20} {'Mevcut':>10} {'Optimum':>10}")
    print(f"  {'-'*40}")
    for key, opt_val in best.items():
        if key == "sharpe":
            continue
        orig_val = original.get(key, "N/A")
        changed  = "*" if str(orig_val) != str(opt_val) else " "
        print(f"  {key:<20} {str(orig_val):>10} {str(opt_val):>10} {changed}")
    print(f"  {'-'*40}")
    print(f"  {'Sharpe (optimum)':<20} {'':>10} {best.get('sharpe', 0):>10.4f}")
    print(f"{'=' * 55}")
    print(f"\n  * isaretli parametreler degisti")


def save_to_yaml(strategy_name: str, best: dict) -> None:
    """
    En iyi parametreleri settings.yaml'a öneri olarak yazar.
    Doğrudan değiştirmez — kullanıcı onaylamalı.
    """
    from pathlib import Path
    output_path = Path("config/optimized_params.yaml")

    import yaml
    data = {strategy_name: {k: v for k, v in best.items() if k != "sharpe"}}

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  Optimum parametreler kaydedildi: {output_path}")
    print("  Bu parametreleri settings.yaml'a manuel olarak kopyalayabilirsin.")


# ─────────────────────────────────────────────────────────────────────────────
# ANA SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def run(strategy: str = "rsi", n_trials: int = 100) -> None:
    print("=" * 55)
    print("  ALGO TRADE CODEX — Parametre Optimizasyonu")
    print("=" * 55)

    if not OPTUNA_AVAILABLE:
        print("\nOptuna yuklu degil!")
        print("Kurmak icin: pip install optuna")
        return

    # Veriyi yükle
    print("\nVeri yukleniyor...")
    df = load_data()
    print(f"Yuklendi: {len(df)} mum | {df.index[0].date()} -> {df.index[-1].date()}")

    if strategy == "rsi":
        # Mevcut değerler (settings.yaml'daki)
        original = {
            "rsi_period": 14, "oversold": 30,
            "overbought": 70, "stop_pct": 0.015, "tp_pct": 0.030,
        }
        print(f"\nRSI stratejisi optimize ediliyor ({n_trials} deneme)...")
        best = optimize_rsi(df, n_trials=n_trials)
        if best:
            print_results("RSIStrategy", best, original)
            save_to_yaml("rsi", best)

    elif strategy == "pa":
        original = {
            "lookback": 50, "rsi_oversold": 40, "rsi_overbought": 60,
            "proximity_pct": 0.02, "stop_pct": 0.015, "tp_pct": 0.030,
        }
        print(f"\nPA Range stratejisi optimize ediliyor ({n_trials} deneme)...")
        best = optimize_pa_range(df, n_trials=n_trials)
        if best:
            print_results("PARangeStrategy", best, original)
            save_to_yaml("pa_range", best)

    elif strategy == "all":
        print("\nTum stratejiler optimize ediliyor...")

        original_rsi = {"rsi_period": 14, "oversold": 30, "overbought": 70,
                        "stop_pct": 0.015, "tp_pct": 0.030}
        best_rsi = optimize_rsi(df, n_trials=n_trials)
        if best_rsi:
            print_results("RSIStrategy", best_rsi, original_rsi)
            save_to_yaml("rsi", best_rsi)

        original_pa = {"lookback": 50, "rsi_oversold": 40, "rsi_overbought": 60,
                       "proximity_pct": 0.02, "stop_pct": 0.015, "tp_pct": 0.030}
        best_pa = optimize_pa_range(df, n_trials=n_trials)
        if best_pa:
            print_results("PARangeStrategy", best_pa, original_pa)
            save_to_yaml("pa_range", best_pa)

    print("\nOptimizasyon tamamlandi!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strateji Parametre Optimizasyonu")
    parser.add_argument("--strategy", choices=["rsi", "pa", "all"], default="rsi")
    parser.add_argument("--trials",   type=int, default=100)
    args = parser.parse_args()
    run(strategy=args.strategy, n_trials=args.trials)
