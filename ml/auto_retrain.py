"""
ml/auto_retrain.py
===================
AMACI:
    ML modelini guncel veriyle yeniden egitir.
    Walk-forward validation: eskiye git-egit, ileri dogrula.

    Bot icinden otomatik tetiklenir (her ~30 gunde bir)
    veya elle calistirilabilir:
        python -m ml.auto_retrain
        python -m ml.auto_retrain --days 365

ATOMIK GUNCELLEME:
    Yeni model once xgb_btc_1h_new.json'a yazilir.
    Basarili olursa xgb_btc_1h.json ile yer degistirilir.
    Boylece bot hic bozuk model gormez.

WALK-FORWARD:
    Egitim: son days-30 gun
    Dogrulama: son 30 gun
    Dogrulama accuracy < 0.35 ise model guncellenmez (guvenlik)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH     = Path(__file__).parent / "models" / "xgb_btc_1h.json"
MODEL_PATH_NEW = Path(__file__).parent / "models" / "xgb_btc_1h_new.json"
MIN_ACCURACY   = 0.35   # Bu esik altinda model degistirilmez


def retrain(days: int = 365, quiet: bool = False) -> bool:
    """
    Modeli yeniden egitir.

    Args:
        days : Kac gunluk veri kullanilsin
        quiet: True ise cikti minimize edilir (bot icinden cagirilirken)

    Returns:
        True = basarili guncelleme, False = basarisiz / degismedi
    """
    if not quiet:
        print("=" * 55)
        print("  AUTO-RETRAIN — Walk-Forward ML Egitimi")
        print("=" * 55)

    try:
        # ── 1. Veri cek ──────────────────────────────────────────
        import ccxt
        import pandas as pd
        from data.fetcher import BinanceFetcher
        from data.cleaner import OHLCVCleaner

        logger.info(f"Egitim verisi cekiliyor ({days} gun)...")

        # Public API — API key gerekmez
        exchange = ccxt.binance({"apiKey": "", "secret": ""})
        since_ms = exchange.milliseconds() - days * 24 * 3600 * 1000

        all_ohlcv = []
        limit = 1000
        current = since_ms
        while True:
            batch = exchange.fetch_ohlcv("BTC/USDT", "1h", since=current, limit=limit)
            if not batch:
                break
            all_ohlcv.extend(batch)
            if len(batch) < limit:
                break
            current = batch[-1][0] + 3600 * 1000

        if len(all_ohlcv) < 200:
            logger.error(f"Yetersiz veri: {len(all_ohlcv)} mum")
            return False

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        cleaner = OHLCVCleaner()
        df = cleaner.clean(df)

        logger.info(f"Veri hazir: {len(df)} mum")

        # ── 2. Walk-forward bolum ─────────────────────────────────
        # Egitim: ilk %92, Dogrulama: son %8 (yaklasik 30 gun)
        split_idx = int(len(df) * 0.92)
        df_train = df.iloc[:split_idx]
        df_val   = df.iloc[split_idx:]

        if not quiet:
            print(f"\nEgitim: {len(df_train)} mum | Dogrulama: {len(df_val)} mum")

        # ── 3. Egit ───────────────────────────────────────────────
        from ml.predictor import MLPredictor

        predictor = MLPredictor(symbol="BTC/USDT", timeframe="1h")

        logger.info("Model egitiliyor...")
        cv_results = predictor.train(df_train)
        cv_acc = cv_results.get("avg_accuracy", 0.0)

        logger.info(f"Egitim tamamlandi | CV Accuracy: {cv_acc:.3f}")

        # ── 4. Walk-forward dogrulama ─────────────────────────────
        correct = 0
        total   = 0
        for i in range(50, len(df_val)):
            window = pd.concat([df_train.tail(150), df_val.iloc[:i]])
            try:
                sig = predictor.predict(window)
                # Gercek hareket: bir sonraki mumun kapanisi yukari mi asagi mi?
                if i + 1 < len(df_val):
                    real_up = df_val["close"].iloc[i + 1] > df_val["close"].iloc[i]
                    pred_up = sig.action == "AL"
                    if sig.action != "BEKLE":
                        correct += int(real_up == pred_up)
                        total   += 1
            except Exception:
                continue

        val_acc = correct / total if total > 0 else 0.0
        logger.info(f"Walk-forward dogrulama | Accuracy: {val_acc:.3f} ({correct}/{total})")

        if not quiet:
            print(f"Walk-forward accuracy: {val_acc:.3f}")

        # ── 5. Guvenlik kontrolu ──────────────────────────────────
        if val_acc < MIN_ACCURACY and total >= 10:
            logger.warning(
                f"Walk-forward accuracy {val_acc:.3f} < {MIN_ACCURACY} esigi | "
                f"Model guncellenmedi (eski model daha iyi)"
            )
            if not quiet:
                print(f"Model guncellenmedi: accuracy {val_acc:.3f} < {MIN_ACCURACY}")
            return False

        # ── 6. Atomik kayit ───────────────────────────────────────
        MODEL_PATH_NEW.parent.mkdir(parents=True, exist_ok=True)
        predictor.save(MODEL_PATH_NEW)

        # Atomik yer degistir
        import shutil
        shutil.move(str(MODEL_PATH_NEW), str(MODEL_PATH))

        logger.info(f"Model guncellendi: {MODEL_PATH}")
        if not quiet:
            print(f"\nModel basariyla guncellendi!")
            print(f"  CV Accuracy:          {cv_acc:.3f}")
            print(f"  Walk-forward Acc:     {val_acc:.3f}")
            print(f"  Kayit:                {MODEL_PATH}")

        return True

    except Exception as e:
        logger.error(f"Retrain hatasi: {e}")
        if not quiet:
            print(f"HATA: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Auto-Retrain")
    parser.add_argument("--days", type=int, default=365, help="Egitim icin kac gun veri")
    args = parser.parse_args()
    success = retrain(days=args.days, quiet=False)
    sys.exit(0 if success else 1)
