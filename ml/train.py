"""
ml/train.py
============
AMACI:
    XGBoost modelini gercek BTC/USDT verisiyle eger ve kaydeder.
    Canli botta kullanilacak model bu script ile uretilir.

AKIS:
    1. Mevcut Parquet cache varsa yukle
    2. Cache yoksa / eski ise Binance'ten 90 gunluk veri cek ve kaydet
    3. FeatureEngineer ile 52 ozellik uret
    4. TimeSeriesSplit cross-validation (5 fold) — lookahead bias yok
    5. Final modeli tum veriyle egit
    6. ml/models/xgb_btc_1h.json olarak kaydet
    7. CV raporu + feature importance yazdir

CALISTIRMAK ICIN:
    python -m ml.train

    # Mevcut cache'i zorunlu olarak yenile:
    python -m ml.train --refresh

    # Farkli timeframe veya horizon:
    python -m ml.train --days 180 --horizon 5 --threshold 0.4

CIKTI:
    ml/models/xgb_btc_1h.json  (model)
    ml/models/xgb_btc_1h_meta.json  (meta bilgi — otomatik)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# Proje kokunu Python yoluna ekle (dogrudan calistirildiginda import hatasi olmasin)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fetcher import BinanceFetcher
from data.cleaner import OHLCVCleaner
from ml.predictor import MLPredictor
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

DEFAULT_SYMBOL    = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_DAYS      = 90       # Kac gunluk veri cekilesin
DEFAULT_HORIZON   = 3        # Kac bar ileri tahmin
DEFAULT_THRESHOLD = 0.3      # Yuzde degisim esigi: AL/SAT etiketi icin

MODEL_DIR  = PROJECT_ROOT / "ml" / "models"
MODEL_FILE = MODEL_DIR / "xgb_btc_1h.json"

CACHE_DIR  = PROJECT_ROOT / "data" / "raw"
CACHE_FILE = CACHE_DIR / "train_btc_usdt_1h.parquet"


# ── Veri Yukleme ──────────────────────────────────────────────────────────────

def _load_or_fetch(days: int, refresh: bool) -> pd.DataFrame:
    """
    Oncelikle Parquet cache'e bakar.
    Cache yoksa veya --refresh verilmisse Binance'ten ceker ve cache'ler.

    Returns:
        OHLCV DataFrame (timestamp index, open/high/low/close/volume)
    """
    if not refresh and CACHE_FILE.exists():
        logger.info(f"Cache bulundu, yukleniyor: {CACHE_FILE}")
        df = BinanceFetcher.load_parquet(CACHE_FILE)

        # Cache yeterince uzun mu? (istenen gunun %80'i)
        min_rows = int(days * 24 * 0.80)
        if len(df) >= min_rows:
            logger.info(f"Cache gecerli | {len(df)} mum | {df.index[0]} -> {df.index[-1]}")
            return df
        else:
            logger.info(f"Cache kisa ({len(df)} < {min_rows}), yeniden cekiliyor...")

    # Binance'ten cek
    logger.info(f"Binance Testnet'ten {days} gunluk veri cekiliyor...")
    fetcher = BinanceFetcher(testnet=True, symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME)

    try:
        df = fetcher.fetch_since(since_days=days, batch_size=500)
    except Exception as e:
        logger.error(f"Veri cekme hatasi: {e}")

        # Hata durumunda mevcut cache'i kullan (eski de olsa)
        if CACHE_FILE.exists():
            logger.warning("HATA: Eski cache kullaniliyor!")
            return BinanceFetcher.load_parquet(CACHE_FILE)
        raise RuntimeError("Veri cekelemedi ve cache de yok!") from e

    if df.empty:
        raise RuntimeError("Binance bos veri dondu!")

    # Cache'e kaydet
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE, engine="pyarrow")
    logger.info(f"Cache guncellendi: {CACHE_FILE} ({len(df)} mum)")

    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisini temizler: NaN kaldir, tip kontrol, OHLC mantik kontrolu."""
    try:
        cleaner = OHLCVCleaner()
        df_clean = cleaner.clean(df)
        logger.info(f"Temizleme sonrasi: {len(df_clean)} mum (oncesi: {len(df)})")
        return df_clean
    except Exception as e:
        logger.warning(f"DataCleaner hatasi ({e}), ham veri kullaniliyor")
        # Manuel temizlik fallback
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = df[df["close"] > 0]
        return df


# ── Egitim ────────────────────────────────────────────────────────────────────

def train(
    days: int      = DEFAULT_DAYS,
    horizon: int   = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
    refresh: bool  = False,
) -> MLPredictor:
    """
    Tam egitim pipeline'i calistirir.

    Args:
        days     : Kac gunluk gecmis veri kullanilsin
        horizon  : Kac bar ileri tahmin (hedef label)
        threshold: AL/SAT esigi (yuzde degisim)
        refresh  : True = cache'i yenile

    Returns:
        Egitilmis MLPredictor nesnesi
    """
    print("\n" + "=" * 65)
    print("  ALGO TRADE CODEX — XGBoost Egitimi")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

    # 1. Veri yukle
    print(f"\n[1/4] Veri yukleniyor ({days} gun)...")
    df_raw = _load_or_fetch(days, refresh)
    print(f"      {len(df_raw)} mum | {df_raw.index[0].date()} -> {df_raw.index[-1].date()}")

    # 2. Temizle
    print("\n[2/4] Veri temizleniyor...")
    df = _clean(df_raw)
    print(f"      {len(df)} temiz mum kaldi")

    if len(df) < 200:
        raise RuntimeError(
            f"Yeterli veri yok: {len(df)} < 200 satir. "
            f"'--days' parametresini artirin veya '--refresh' ile yeniden cekin."
        )

    # 3. Model olustur + egit
    print(f"\n[3/4] Model egitiliyor...")
    print(f"      Horizon: {horizon} bar | Threshold: {threshold}% | CV: 5-fold")

    predictor = MLPredictor(
        symbol        = DEFAULT_SYMBOL,
        timeframe     = DEFAULT_TIMEFRAME,
        horizon_bars  = horizon,
        threshold_pct = threshold,
        n_cv_splits   = 5,
    )

    cv_results = predictor.train(df)

    # 4. Kaydet
    print(f"\n[4/4] Model kaydediliyor...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictor.save(MODEL_FILE)
    print(f"      Kaydedildi: {MODEL_FILE}")

    # ── Raporlar ──────────────────────────────────────────────────────────────

    predictor.cv_report()
    predictor.feature_importance_report(top_n=15)

    # Ozet
    avg_acc  = cv_results.get("avg_accuracy", 0)
    avg_al   = cv_results.get("avg_al_f1", 0)
    avg_sat  = cv_results.get("avg_sat_f1", 0)

    print("\n" + "=" * 65)
    print("  EGITIM TAMAMLANDI")
    print("=" * 65)
    print(f"  Veri       : {len(df)} mum ({days} gun)")
    print(f"  Ort. Acc   : {avg_acc:.3f}")
    print(f"  Ort. AL F1 : {avg_al:.3f}")
    print(f"  Ort. SAT F1: {avg_sat:.3f}")
    print(f"  Model      : {MODEL_FILE}")
    print("=" * 65)

    # Uyarilar
    if avg_acc < 0.45:
        print("\n[UYARI] Accuracy cok dusuk (< 0.45). Veriye ve parametrelere bakin.")
    if avg_al < 0.30:
        print("[UYARI] AL F1 dusuk — model AL sinyalini iyi tanimiyor.")
    if avg_sat < 0.30:
        print("[UYARI] SAT F1 dusuk — model SAT sinyalini iyi tanimiyor.")

    print()
    return predictor


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost modelini BTC/USDT verisiyle egitir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help="Kac gunluk gecmis veri kullanilsin"
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON,
        help="Kac bar ileri tahmin (hedef label)"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="AL/SAT etiketi icin min yuzde degisim"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Cache'i yok say, Binance'ten yeniden cek"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        train(
            days      = args.days,
            horizon   = args.horizon,
            threshold = args.threshold,
            refresh   = args.refresh,
        )
    except Exception as e:
        logger.error(f"Egitim basarisiz: {e}")
        print(f"\n[HATA] {e}")
        sys.exit(1)
