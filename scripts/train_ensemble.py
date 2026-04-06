"""
scripts/train_ensemble.py
==========================
AMACI:
    Binance Testnet'ten BTC/USDT 1h veri ceker,
    XGBoost + LightGBM + RandomForest ensemble modelini egitir,
    ml/models/ensemble_btc_1h/ klasorune kaydeder.

KULLANIM:
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --days 365
    python scripts/train_ensemble.py --days 180 --no-consensus
"""

import argparse
import sys
from pathlib import Path

# Proje kokunu ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.cleaner import DataCleaner
from ml.feature_engineering import FeatureEngineer
from ml.ensemble_model import EnsembleModel
from utils.logger import get_logger

logger = get_logger(__name__)

SAVE_DIR = Path("ml/models/ensemble_btc_1h")


def main():
    parser = argparse.ArgumentParser(description="Ensemble model egitici")
    parser.add_argument("--days",        type=int,  default=365,   help="Kac gunluk veri (varsayilan: 365)")
    parser.add_argument("--no-consensus",action="store_true",       help="Konsensus modunu kapat")
    parser.add_argument("--weights",     type=float, nargs=3,
                        default=[0.40, 0.35, 0.25],
                        metavar=("XGB_W", "LGB_W", "RF_W"),
                        help="Model agirliklari (toplam 1.0 olmali)")
    parser.add_argument("--cv-splits",   type=int,  default=5,     help="CV fold sayisi")
    args = parser.parse_args()

    # Agirlik toplami kontrol
    w_sum = sum(args.weights)
    if abs(w_sum - 1.0) > 0.01:
        print(f"HATA: Agirliklar toplami {w_sum:.2f} olmaliydi, 1.0 olmalı")
        sys.exit(1)

    print("=" * 60)
    print("  ENSEMBLE MODEL EGITIMI")
    print(f"  XGBoost:{args.weights[0]} + LightGBM:{args.weights[1]} + RF:{args.weights[2]}")
    print(f"  Veri: {args.days} gun | CV: {args.cv_splits} fold")
    print(f"  Konsensus: {'KAPALI' if args.no_consensus else 'ACIK (2/3 esik)'}")
    print("=" * 60)

    # 1. Veri cek
    print("\n[1/4] Veri cekiliyor...")
    limit = args.days * 24 + 50   # 1h mumlar
    fetcher = DataFetcher()
    df_raw = fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=limit)
    if df_raw is None or df_raw.empty:
        print("HATA: Veri cekelemedi!")
        sys.exit(1)

    cleaner = DataCleaner()
    df = cleaner.clean(df_raw)
    print(f"    {len(df)} satir veri hazir ({df.index[0]} -> {df.index[-1]})")

    # 2. Feature engineering
    print("\n[2/4] Feature engineering...")
    fe = FeatureEngineer(horizon_bars=3, threshold_pct=0.3)
    X, y, feature_names = fe.build(df)
    print(f"    {X.shape[0]} satir, {X.shape[1]} ozellik")
    print(f"    AL:{(y==2).sum()} BEKLE:{(y==1).sum()} SAT:{(y==0).sum()}")

    # 3. Egit
    print("\n[3/4] Ensemble egitiliyor...")
    model = EnsembleModel(
        weights        = args.weights,
        n_splits       = args.cv_splits,
        consensus_only = not args.no_consensus,
    )

    cv_results = model.cross_validate(X, y)
    model.train(X, y)
    model.cv_report()

    # 4. Kaydet
    print(f"\n[4/4] Kaydediliyor: {SAVE_DIR}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVE_DIR)

    print("\n" + "=" * 60)
    print("  EGITIM TAMAMLANDI")
    print(f"  CV Ort. Accuracy : {cv_results['avg_accuracy']:.3f}")
    print(f"  Train Accuracy   : {model.training_meta.get('train_accuracy', 0):.3f}")
    print(f"  Model Klasoru    : {SAVE_DIR.resolve()}")
    print("=" * 60)

    # Ornek tahmin goster
    print("\n--- ORNEK OY DAGILIMI (Son Bar) ---")
    model.model_votes_report(X)


if __name__ == "__main__":
    main()
