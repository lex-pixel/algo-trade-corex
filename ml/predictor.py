"""
ml/predictor.py
================
AMACI:
    Egitilmis XGBoost modelini canli veri uzerinde kullanir.
    Strateji motoruyla ayni Signal formatini dondurur —
    RSIStrategy ve PARangeStrategy ile entegre edilebilir.

    Akis:
        1. Gelen OHLCV DataFrame -> FeatureEngineer.build_live() ile ozellikler
        2. XGBoostModel.predict_single() ile tahmin
        3. Signal nesnesi dondurulur (strategies/base_strategy.py ile uyumlu)

KULLANIM (canli bot icinde):
    from ml.predictor import MLPredictor

    predictor = MLPredictor.from_file("ml/models/xgb_btc_1h.json")
    signal = predictor.predict(df)  # Signal nesnesi doner
    print(signal.action)            # "AL", "SAT" veya "BEKLE"
    print(signal.confidence)        # 0.0 - 1.0

EGITIM VE KAYIT:
    from ml.predictor import MLPredictor

    predictor = MLPredictor()
    predictor.train(df, horizon_bars=3, threshold_pct=0.3)
    predictor.save("ml/models/xgb_btc_1h.json")
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from ml.feature_engineering import FeatureEngineer, LABEL_BUY, LABEL_SELL, LABEL_HOLD
from ml.xgboost_model import XGBoostModel
from strategies.base_strategy import Signal
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum guvenlilir tahmin icin olasilik esigi
MIN_CONFIDENCE = 0.40


class MLPredictor:
    """
    Egitilmis XGBoost modelini canli veri uzerinde calistiran yuksek seviye sinif.

    Ozellikleri:
        - Egitim: train(df) - feature engineering + cv + model egitimi
        - Tahmin: predict(df) -> Signal (strategies ile entegre)
        - Kaydet: save(path)
        - Yukle: MLPredictor.from_file(path)
    """

    def __init__(
        self,
        symbol: str    = "BTC/USDT",
        timeframe: str = "1h",
        horizon_bars: int    = 3,
        threshold_pct: float = 0.3,
        min_confidence: float = MIN_CONFIDENCE,
        model_params: dict | None = None,
        n_cv_splits: int = 5,
    ):
        self.symbol         = symbol
        self.timeframe      = timeframe
        self.horizon_bars   = horizon_bars
        self.threshold_pct  = threshold_pct
        self.min_confidence = min_confidence
        self.n_cv_splits    = n_cv_splits

        self._fe    = FeatureEngineer(
            horizon_bars=horizon_bars,
            threshold_pct=threshold_pct,
        )
        self._model = XGBoostModel(
            params=model_params,
            n_splits=n_cv_splits,
        )
        self._cv_results: dict = {}

    # ── Egitim ───────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict:
        """
        Verilen OHLCV verisiyle modeli egitir.
        Cross-validation yapilir, sonra final model tum veriyle egitilir.

        Returns:
            cv_results: fold bazli metrikler
        """
        logger.info(
            f"MLPredictor egitimi basliyor | "
            f"symbol: {self.symbol} | "
            f"horizon: {self.horizon_bars} bar | "
            f"threshold: {self.threshold_pct}%"
        )

        # Feature matrix uret
        X, y, feature_names = self._fe.build(df)

        # Cross-validation
        logger.info("TimeSeriesSplit cross-validation yapiliyor...")
        self._cv_results = self._model.cross_validate(X, y)

        # Final model egit
        logger.info("Final model egitiliyor (tum veri)...")
        self._model.train(X, y)

        logger.info(
            f"Egitim tamamlandi | "
            f"CV avg accuracy: {self._cv_results.get('avg_accuracy', 0):.3f} | "
            f"CV avg AL F1: {self._cv_results.get('avg_al_f1', 0):.3f}"
        )

        return self._cv_results

    # ── Canli Tahmin ──────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Signal:
        """
        Son barin sinyalini tahmin eder.
        Yeterli guvenliligi yoksa BEKLE sinyali doner.

        Args:
            df: OHLCV DataFrame (en az 100 satir)

        Returns:
            Signal: strategies/base_strategy.py ile uyumlu sinyal nesnesi
        """
        if not self._model.is_trained():
            logger.warning("Model egitilmemis, BEKLE donuluyor")
            return Signal(
                action="BEKLE", confidence=0.0,
                strategy="XGBoostModel",
                reason="model_not_trained",
            )

        try:
            # Feature olustur (label olmadan — canli mod)
            X_live = self._build_live_features(df)
            if X_live is None or X_live.empty:
                raise ValueError("Feature olusturulamadi")

            label, confidence, probs = self._model.predict_single(X_live)

            # Guvenliligi dusuk ise BEKLE
            if confidence < self.min_confidence:
                action = "BEKLE"
                logger.debug(
                    f"Dusuk guven ({confidence:.2f} < {self.min_confidence}) -> BEKLE"
                )
            else:
                action = {LABEL_BUY: "AL", LABEL_SELL: "SAT", LABEL_HOLD: "BEKLE"}[label]

            current_price = float(df["close"].iloc[-1])
            atr_val       = self._calc_atr(df)

            # Stop-loss ve take-profit (ATR tabanli)
            stop_loss   = None
            take_profit = None
            if action == "AL" and atr_val:
                stop_loss   = current_price - 2.0 * atr_val
                take_profit = current_price + 3.0 * atr_val
            elif action == "SAT" and atr_val:
                stop_loss   = current_price + 2.0 * atr_val
                take_profit = current_price - 3.0 * atr_val

            probs_info = (
                f"AL:{probs[LABEL_BUY]:.2f} "
                f"BEKLE:{probs[LABEL_HOLD]:.2f} "
                f"SAT:{probs[LABEL_SELL]:.2f}"
            )
            signal = Signal(
                action      = action,
                confidence  = round(confidence, 4),
                stop_loss   = stop_loss,
                take_profit = take_profit,
                strategy    = "XGBoostModel",
                reason      = (
                    f"ML tahmin | {probs_info} | "
                    f"horizon={self.horizon_bars}bar"
                ),
            )

            logger.info(
                f"ML Tahmin | {action} | "
                f"Guven: {confidence:.3f} | "
                f"AL:{probs[LABEL_BUY]:.2f} "
                f"BEKLE:{probs[LABEL_HOLD]:.2f} "
                f"SAT:{probs[LABEL_SELL]:.2f}"
            )

            return signal

        except Exception as e:
            logger.error(f"Tahmin hatasi: {e}")
            return Signal(
                action="BEKLE", confidence=0.0,
                strategy="XGBoostModel",
                reason=f"hata: {e}",
            )

    # ── Canli Feature Olusturma (Label Olmadan) ───────────────────────────────

    def _build_live_features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Canli modda feature olusturur.
        Label (hedef) hesaplamasi yapilmaz — sadece son satir ozelliklerini doner.
        NaN satirlarini atar, model feature siralamasina gore hizalar.
        """
        try:
            # FeatureEngineer'i live mod icin kullan:
            # horizon_bars=0 ile hedef shiftleme olmadan tum satirlari dondur
            fe_live = FeatureEngineer(
                horizon_bars=1,
                threshold_pct=self.threshold_pct,
                min_rows=self.feature_rows_needed(),
            )
            # Not: build() son horizon_bars satiri atar (gelecek bilgisi yok)
            # Live modda bizi ilgilendiren son satir zaten son barin ozellikleridir
            X, _, _ = fe_live.build(df)

            if X.empty:
                return None

            # Model'in feature listesine gore hizala
            expected_features = self._model.feature_names
            if not expected_features:
                return X

            # Sadece egitimde kullanilan feature'lari sec
            missing = [f for f in expected_features if f not in X.columns]
            if missing:
                logger.warning(f"Eksik ozellikler: {missing}")
                return None

            return X[expected_features]

        except Exception as e:
            logger.warning(f"Live feature olusturulamadi: {e}")
            return None

    def feature_rows_needed(self) -> int:
        """Anlamli ozellik icin minimum satir sayisi."""
        return 60   # 50 + biraz tampon

    # ── Yardimci ─────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float | None:
        """ATR hesaplar (stop-loss icin)."""
        try:
            import pandas_ta as ta
            atr = ta.atr(df["high"], df["low"], df["close"], length=period)
            val = atr.iloc[-1]
            return float(val) if not pd.isna(val) else None
        except Exception:
            return None

    # ── Kaydet / Yukle ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Modeli kaydeder."""
        self._model.save(path)
        logger.info(f"MLPredictor modeli kaydedildi: {path}")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> "MLPredictor":
        """Kaydedilmis modeli yukler."""
        instance = cls(symbol=symbol, timeframe=timeframe)
        instance._model = XGBoostModel.load(path)
        logger.info(f"MLPredictor yuklendi: {path}")
        return instance

    # ── Ozellik Onem Raporu ───────────────────────────────────────────────────

    def feature_importance_report(self, top_n: int = 15) -> None:
        """Feature importance tablosunu terminale yazdirir."""
        if not self._model.is_trained():
            print("Model henuz egitilmemis.")
            return

        df_imp = self._model.get_feature_importance(top_n=top_n)
        print("\n" + "=" * 50)
        print(f"  EN ONEMLI {top_n} OZELLIK (XGBoost)")
        print("=" * 50)
        for _, row in df_imp.iterrows():
            bar = "#" * int(row["importance"] * 50)
            print(f"  {row['feature']:<30} {row['importance']:.4f}  {bar}")
        print("=" * 50)

    def cv_report(self) -> None:
        """Cross-validation sonuclarini terminale yazdirir."""
        if not self._cv_results:
            print("CV henuz yapilmamis.")
            return

        print("\n" + "=" * 60)
        print("  CROSS-VALIDATION SONUCLARI (TimeSeriesSplit)")
        print("=" * 60)
        print(f"  Fold sayisi : {self._cv_results.get('n_folds')}")
        print(f"  Ort. Accuracy: {self._cv_results.get('avg_accuracy', 0):.3f}")
        print(f"  Ort. AL F1  : {self._cv_results.get('avg_al_f1', 0):.3f}")
        print(f"  Ort. SAT F1 : {self._cv_results.get('avg_sat_f1', 0):.3f}")
        print("-" * 60)
        for r in self._cv_results.get("fold_results", []):
            print(
                f"  Fold {r['fold']} | "
                f"Train: {r['train_size']} | "
                f"Test: {r['test_size']} | "
                f"Acc: {r['accuracy']:.3f} | "
                f"AL_F1: {r['al_f1']:.3f} | "
                f"SAT_F1: {r['sat_f1']:.3f}"
            )
        print("=" * 60)
