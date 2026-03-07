"""
ml/xgboost_model.py
=====================
AMACI:
    XGBoost tabanli AL/SAT/BEKLE siniflandirma modeli.
    TimeSeriesSplit ile lookahead bias onlenmis cross-validation.
    SHAP degerleri ile ozellik onem analizi.

MODEL DETAYLARI:
    - XGBoostClassifier (3 sinif: 0=SAT, 1=BEKLE, 2=AL)
    - TimeSeriesSplit: zaman sirali cross-validation (gelecek veri sizintisi yok)
    - Sinif agirlandirmasi: veri dengesizligi icin otomatik compute_sample_weight
    - SHAP: hangi ozellik ne kadar etkiliyor?

KAYIT / YUKLE:
    model.save("ml/models/xgb_model.json")
    model = XGBoostModel.load("ml/models/xgb_model.json")

KULLANIM:
    from ml.feature_engineering import FeatureEngineer
    from ml.xgboost_model import XGBoostModel

    fe    = FeatureEngineer()
    X, y, names = fe.build(df)

    model = XGBoostModel()
    cv_results = model.cross_validate(X, y)
    model.train(X, y)
    model.save("ml/models/xgb_model.json")
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from utils.logger import get_logger

logger = get_logger(__name__)

# Sinif etiketleri
LABEL_SELL = 0
LABEL_HOLD = 1
LABEL_BUY  = 2


class XGBoostModel:
    """
    XGBoost tabanli 3-sinif zaman serisi siniflandirici.

    Parametreler (varsayilanlar Optuna ile optimize edilebilir):
        n_estimators    : agac sayisi
        max_depth       : agac derinligi
        learning_rate   : ogrenme hizi (eta)
        subsample       : her agac icin veri alt-ornekleme orani
        colsample_bytree: her agac icin ozellik alt-ornekleme orani
        min_child_weight: yaprak dugumu min agirlik
        gamma           : bolunme icin min kayip azalmasi
        n_splits        : TimeSeriesSplit bolme sayisi
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators"     : 300,
        "max_depth"        : 4,
        "learning_rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "min_child_weight" : 5,
        "gamma"            : 0.1,
        "reg_alpha"        : 0.1,    # L1 regularizasyon
        "reg_lambda"       : 1.0,    # L2 regularizasyon
        "objective"        : "multi:softprob",
        "num_class"        : 3,
        "eval_metric"      : "mlogloss",
        "use_label_encoder": False,
        "random_state"     : 42,
        "verbosity"        : 0,
    }

    def __init__(self, params: dict | None = None, n_splits: int = 5):
        self.params   = {**self.DEFAULT_PARAMS, **(params or {})}
        self.n_splits = n_splits
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []
        self.training_meta: dict = {}

    # ── Egitim ───────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Tum veri seti uzerinde final modeli egitir.
        Cross-validate()'dan sonra cagrilmali.
        """
        logger.info(f"XGBoost egitimi basliyor: {len(X)} satir, {X.shape[1]} ozellik")

        self.feature_names = list(X.columns)
        sample_weights = compute_sample_weight("balanced", y)

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X, y,
            sample_weight=sample_weights,
            verbose=False,
        )

        # Egitim metrikleri kaydet
        train_pred = self.model.predict(X)
        train_acc  = accuracy_score(y, train_pred)

        self.training_meta = {
            "n_samples"    : int(len(X)),
            "n_features"   : int(X.shape[1]),
            "train_accuracy": round(float(train_acc), 4),
            "class_dist"   : {
                "SAT"  : int((y == LABEL_SELL).sum()),
                "BEKLE": int((y == LABEL_HOLD).sum()),
                "AL"   : int((y == LABEL_BUY).sum()),
            }
        }

        logger.info(
            f"Egitim tamamlandi | "
            f"Train accuracy: {train_acc:.3f} | "
            f"AL:{self.training_meta['class_dist']['AL']} "
            f"SAT:{self.training_meta['class_dist']['SAT']} "
            f"BEKLE:{self.training_meta['class_dist']['BEKLE']}"
        )

    # ── Cross-Validation ──────────────────────────────────────────────────────

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, float]:
        """
        TimeSeriesSplit ile walk-forward cross-validation.
        Lookahead bias yoktur: her fold'da egitim seti test setinden oncedir.

        Returns:
            dict: ortalama accuracy, her fold sonuclari, rapor
        """
        logger.info(
            f"TimeSeriesSplit CV basliyor: {self.n_splits} fold, "
            f"{len(X)} satir, {X.shape[1]} ozellik"
        )

        tscv      = TimeSeriesSplit(n_splits=self.n_splits)
        fold_results = []

        X_arr = X.values
        y_arr = y.values

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            # Minimum egitim verisi kontrol
            if len(X_train) < 50:
                logger.warning(f"Fold {fold+1}: Yetersiz egitim verisi, atlaniyor")
                continue

            # Sinif agirlandirmasi
            weights = compute_sample_weight("balanced", y_train)

            fold_model = xgb.XGBClassifier(**self.params)
            fold_model.fit(X_train, y_train, sample_weight=weights, verbose=False)

            preds = fold_model.predict(X_test)
            acc   = accuracy_score(y_test, preds)

            # Sinif bazli metrikler
            report = classification_report(
                y_test, preds,
                labels=[LABEL_SELL, LABEL_HOLD, LABEL_BUY],
                target_names=["SAT", "BEKLE", "AL"],
                output_dict=True,
                zero_division=0,
            )

            fold_results.append({
                "fold"         : fold + 1,
                "train_size"   : len(X_train),
                "test_size"    : len(X_test),
                "accuracy"     : round(float(acc), 4),
                "al_f1"        : round(float(report["AL"]["f1-score"]), 4),
                "sat_f1"       : round(float(report["SAT"]["f1-score"]), 4),
                "bekle_f1"     : round(float(report["BEKLE"]["f1-score"]), 4),
            })

            logger.info(
                f"Fold {fold+1}/{self.n_splits} | "
                f"Accuracy: {acc:.3f} | "
                f"AL_F1: {report['AL']['f1-score']:.3f} | "
                f"SAT_F1: {report['SAT']['f1-score']:.3f}"
            )

        if not fold_results:
            logger.warning("Hic fold tamamlanamadi!")
            return {}

        # Ortalama metrikler
        avg_acc    = float(np.mean([r["accuracy"] for r in fold_results]))
        avg_al_f1  = float(np.mean([r["al_f1"]   for r in fold_results]))
        avg_sat_f1 = float(np.mean([r["sat_f1"]  for r in fold_results]))

        summary = {
            "n_folds"     : len(fold_results),
            "avg_accuracy": round(avg_acc, 4),
            "avg_al_f1"   : round(avg_al_f1, 4),
            "avg_sat_f1"  : round(avg_sat_f1, 4),
            "fold_results": fold_results,
        }

        logger.info(
            f"CV tamamlandi | "
            f"Ort. Accuracy: {avg_acc:.3f} | "
            f"Ort. AL F1: {avg_al_f1:.3f} | "
            f"Ort. SAT F1: {avg_sat_f1:.3f}"
        )

        return summary

    # ── Tahmin ───────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ham sinif tahminleri dondurur (0, 1, 2)."""
        self._check_trained()
        return self.model.predict(X[self.feature_names].values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Her sinif icin olasilik matrisi dondurur.
        Shape: (n_samples, 3) — [SAT_prob, BEKLE_prob, AL_prob]
        """
        self._check_trained()
        return self.model.predict_proba(X[self.feature_names].values)

    def predict_single(self, X: pd.DataFrame) -> tuple[int, float, np.ndarray]:
        """
        Son satir icin tahmin yapar.

        Returns:
            label     : 0=SAT, 1=BEKLE, 2=AL
            confidence: en yuksek sinifin olasiligi
            probs     : [SAT_prob, BEKLE_prob, AL_prob]
        """
        self._check_trained()
        probs      = self.predict_proba(X.tail(1))[0]
        label      = int(np.argmax(probs))
        confidence = float(probs[label])
        return label, confidence, probs

    # ── SHAP Analizi ─────────────────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        XGBoost dahili onem skorlarini dondurur.
        (SHAP yerine built-in importance — shap paketi opsiyonel)
        """
        self._check_trained()
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature"   : self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n)
        return df

    def get_shap_values(self, X: pd.DataFrame):
        """
        SHAP degerlerini dondurur (shap paketi yuklu olmalidir).
        Yuklu degilse None doner, hata vermez.
        """
        try:
            import shap
            self._check_trained()
            explainer   = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X[self.feature_names])
            return shap_values
        except ImportError:
            logger.warning("shap paketi yuklu degil. pip install shap")
            return None
        except Exception as e:
            logger.warning(f"SHAP hesaplanamadi: {e}")
            return None

    # ── Kaydet / Yukle ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Modeli ve meta-verisini kaydeder.
        2 dosya: model.json (XGBoost) + model_meta.json (feature names + params)
        """
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # XGBoost modelini kaydet
        self.model.save_model(str(path))

        # Meta verisini ayri dosyaya kaydet
        meta_path = path.parent / (path.stem + "_meta.json")
        meta = {
            "feature_names" : self.feature_names,
            "params"        : {k: v for k, v in self.params.items()
                               if isinstance(v, (int, float, str, bool))},
            "n_splits"      : self.n_splits,
            "training_meta" : self.training_meta,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Model kaydedildi: {path} + {meta_path}")

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        """Kaydedilmis modeli ve meta-verisini yukler."""
        path = Path(path)
        meta_path = path.parent / (path.stem + "_meta.json")

        if not path.exists():
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta dosyasi bulunamadi: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(params=meta.get("params"), n_splits=meta.get("n_splits", 5))
        instance.feature_names  = meta["feature_names"]
        instance.training_meta  = meta.get("training_meta", {})

        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(path))

        logger.info(
            f"Model yuklendi: {path} | "
            f"{len(instance.feature_names)} ozellik"
        )
        return instance

    # ── Yardimci ─────────────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model henuz egitilmedi. Once model.train(X, y) cagirin."
            )

    def is_trained(self) -> bool:
        return self.model is not None

    def summary(self) -> dict:
        """Model ozeti dondurur."""
        return {
            "trained"       : self.is_trained(),
            "n_features"    : len(self.feature_names),
            "feature_names" : self.feature_names[:5],  # ilk 5 ozellik
            "params"        : {
                k: v for k, v in self.params.items()
                if k in ["n_estimators", "max_depth", "learning_rate"]
            },
            "training_meta" : self.training_meta,
        }
