"""
ml/ensemble_model.py
=====================
AMACI:
    XGBoost + LightGBM + RandomForest uclu ensemble modeli.
    Her model ayri tahmin yapar, agirlikli oy birlestirmesiyle final karar verilir.

    Oylama Sistemi:
        - Her model: AL=2, BEKLE=1, SAT=0 tahmin eder
        - Soft voting: olasilik vektorleri agirlikli ortalama
        - Varsayilan agirliklar: XGB=0.4, LGB=0.35, RF=0.25
        - Konsensus modu: en az 2/3 model ayni yonde olmali

KULLANIM:
    from ml.ensemble_model import EnsembleModel

    model = EnsembleModel()
    cv_results = model.cross_validate(X, y)
    model.train(X, y)
    model.save("ml/models/ensemble_btc_1h")

    label, conf, probs = model.predict_single(X_live)

KAYIT / YUKLE:
    model.save("ml/models/ensemble_btc_1h")  # klasor olusturur
    model = EnsembleModel.load("ml/models/ensemble_btc_1h")
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb

from utils.logger import get_logger

logger = get_logger(__name__)

# Sinif etiketleri (xgboost_model.py ile ayni)
LABEL_SELL = 0
LABEL_HOLD = 1
LABEL_BUY  = 2


class EnsembleModel:
    """
    XGBoost + LightGBM + RandomForest soft-voting ensemble siniflandirici.

    Parametreler:
        weights       : [xgb_w, lgb_w, rf_w] toplam 1.0 olmali
        n_splits      : TimeSeriesSplit CV fold sayisi
        min_confidence: bu esik alti -> BEKLE donulur
        consensus_only: True ise sadece 2/3 model ayni yonde oy verirse sinyal
    """

    # XGBoost varsayilan parametreleri
    XGB_PARAMS: dict[str, Any] = {
        "n_estimators"     : 300,
        "max_depth"        : 4,
        "learning_rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "min_child_weight" : 5,
        "gamma"            : 0.1,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 1.0,
        "objective"        : "multi:softprob",
        "num_class"        : 3,
        "eval_metric"      : "mlogloss",
        "use_label_encoder": False,
        "random_state"     : 42,
        "verbosity"        : 0,
    }

    # LightGBM varsayilan parametreleri
    LGB_PARAMS: dict[str, Any] = {
        "n_estimators"    : 300,
        "max_depth"       : 5,
        "learning_rate"   : 0.05,
        "num_leaves"      : 31,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha"       : 0.1,
        "reg_lambda"      : 1.0,
        "objective"       : "multiclass",
        "num_class"       : 3,
        "metric"          : "multi_logloss",
        "random_state"    : 42,
        "verbosity"       : -1,
        "force_col_wise"  : True,
    }

    # RandomForest varsayilan parametreleri
    RF_PARAMS: dict[str, Any] = {
        "n_estimators" : 200,
        "max_depth"    : 6,
        "min_samples_leaf": 10,
        "max_features" : "sqrt",
        "class_weight" : "balanced",
        "random_state" : 42,
        "n_jobs"       : -1,
    }

    def __init__(
        self,
        weights: list[float] | None = None,
        n_splits: int = 5,
        min_confidence: float = 0.40,
        consensus_only: bool = True,
    ):
        # Agirliklar: [xgb, lgb, rf]
        self.weights        = weights or [0.40, 0.35, 0.25]
        self.n_splits       = n_splits
        self.min_confidence = min_confidence
        self.consensus_only = consensus_only

        # Modeller
        self._xgb: xgb.XGBClassifier | None = None
        self._lgb: lgb.LGBMClassifier | None = None
        self._rf:  RandomForestClassifier | None = None

        self.feature_names: list[str] = []
        self.training_meta: dict = {}
        self._cv_results: dict = {}

    # ── Egitim ───────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Tum veri seti uzerinde uc modeli de egitir."""
        logger.info(
            f"Ensemble egitimi basliyor: {len(X)} satir, "
            f"{X.shape[1]} ozellik | XGB+LGB+RF"
        )
        self.feature_names = list(X.columns)
        weights_bal = compute_sample_weight("balanced", y)

        # XGBoost
        self._xgb = xgb.XGBClassifier(**self.XGB_PARAMS)
        self._xgb.fit(X, y, sample_weight=weights_bal, verbose=False)
        logger.info("XGBoost egitildi")

        # LightGBM
        self._lgb = lgb.LGBMClassifier(**self.LGB_PARAMS)
        self._lgb.fit(
            X, y,
            sample_weight=weights_bal,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        logger.info("LightGBM egitildi")

        # RandomForest (class_weight=balanced parametresi var, ayrica weight vermiyoruz)
        self._rf = RandomForestClassifier(**self.RF_PARAMS)
        self._rf.fit(X, y)
        logger.info("RandomForest egitildi")

        # Egitim acc
        preds = self._predict_proba_ensemble(X)
        train_acc = accuracy_score(y, np.argmax(preds, axis=1))

        self.training_meta = {
            "n_samples"    : int(len(X)),
            "n_features"   : int(X.shape[1]),
            "train_accuracy": round(float(train_acc), 4),
            "weights"      : self.weights,
            "class_dist"   : {
                "SAT"  : int((y == LABEL_SELL).sum()),
                "BEKLE": int((y == LABEL_HOLD).sum()),
                "AL"   : int((y == LABEL_BUY).sum()),
            },
        }

        logger.info(
            f"Ensemble egitim tamamlandi | "
            f"Train acc: {train_acc:.3f} | "
            f"AL:{self.training_meta['class_dist']['AL']} "
            f"BEKLE:{self.training_meta['class_dist']['BEKLE']} "
            f"SAT:{self.training_meta['class_dist']['SAT']}"
        )

    # ── Cross-Validation ──────────────────────────────────────────────────────

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        TimeSeriesSplit CV — her fold'da uc model ayri ayri egitilir,
        ensemble olasiliklari ile fold metrikleri hesaplanir.
        """
        logger.info(
            f"Ensemble CV basliyor: {self.n_splits} fold, "
            f"{len(X)} satir"
        )

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_results = []
        all_acc = []

        X_arr = X.values
        y_arr = y.values

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_arr)):
            X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
            y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]

            if len(X_tr) < 50:
                logger.warning(f"Fold {fold+1}: az veri, atlandi")
                continue

            w = compute_sample_weight("balanced", y_tr)

            # XGB fold
            m_xgb = xgb.XGBClassifier(**self.XGB_PARAMS)
            m_xgb.fit(X_tr, y_tr, sample_weight=w, verbose=False)

            # LGB fold
            m_lgb = lgb.LGBMClassifier(**self.LGB_PARAMS)
            m_lgb.fit(
                X_tr, y_tr, sample_weight=w,
                callbacks=[lgb.log_evaluation(period=-1)],
            )

            # RF fold
            m_rf = RandomForestClassifier(**self.RF_PARAMS)
            m_rf.fit(X_tr, y_tr)

            # Ensemble proba
            p = (
                self.weights[0] * m_xgb.predict_proba(X_te) +
                self.weights[1] * m_lgb.predict_proba(X_te) +
                self.weights[2] * m_rf.predict_proba(X_te)
            )
            preds = np.argmax(p, axis=1)
            acc   = accuracy_score(y_te, preds)
            all_acc.append(acc)

            # Sinif bazli ACC
            al_mask  = y_te == LABEL_BUY
            sat_mask = y_te == LABEL_SELL
            al_acc  = accuracy_score(y_te[al_mask],  preds[al_mask])  if al_mask.any()  else 0.0
            sat_acc = accuracy_score(y_te[sat_mask], preds[sat_mask]) if sat_mask.any() else 0.0

            fold_results.append({
                "fold"      : fold + 1,
                "train_size": int(len(X_tr)),
                "test_size" : int(len(X_te)),
                "accuracy"  : round(float(acc), 4),
                "al_acc"    : round(float(al_acc), 4),
                "sat_acc"   : round(float(sat_acc), 4),
            })

            logger.info(
                f"Fold {fold+1} | "
                f"train:{len(X_tr)} test:{len(X_te)} | "
                f"Acc:{acc:.3f} AL_acc:{al_acc:.3f} SAT_acc:{sat_acc:.3f}"
            )

        avg_acc = float(np.mean(all_acc)) if all_acc else 0.0
        self._cv_results = {
            "n_folds"     : len(fold_results),
            "avg_accuracy": round(avg_acc, 4),
            "fold_results": fold_results,
        }

        logger.info(f"Ensemble CV tamamlandi | Ort. Acc: {avg_acc:.3f}")
        return self._cv_results

    # ── Tahmin ────────────────────────────────────────────────────────────────

    def predict_single(
        self, X: pd.DataFrame
    ) -> tuple[int, float, dict[int, float]]:
        """
        Son satir icin tahmin yapar.

        Returns:
            (label, confidence, probs_dict)
            label: LABEL_SELL=0, LABEL_HOLD=1, LABEL_BUY=2
            confidence: kazanan sinifin olasiligi
            probs_dict: {0: sat_p, 1: bekle_p, 2: al_p}
        """
        if not self.is_trained():
            return LABEL_HOLD, 0.0, {LABEL_SELL: 0.0, LABEL_HOLD: 1.0, LABEL_BUY: 0.0}

        X_last = X.iloc[[-1]]

        # Her modelin olasiligi
        p_xgb = self._xgb.predict_proba(X_last)[0]
        p_lgb = self._lgb.predict_proba(X_last)[0]
        p_rf  = self._rf.predict_proba(X_last)[0]

        # Agirlikli ortalama
        p_ens = (
            self.weights[0] * p_xgb +
            self.weights[1] * p_lgb +
            self.weights[2] * p_rf
        )

        label = int(np.argmax(p_ens))
        confidence = float(p_ens[label])

        # Konsensus kontrolu: 2/3 model ayni sinifi tahmin etmeli
        if self.consensus_only:
            votes = [
                int(np.argmax(p_xgb)),
                int(np.argmax(p_lgb)),
                int(np.argmax(p_rf)),
            ]
            vote_counts = {v: votes.count(v) for v in set(votes)}
            # Cogunluk yok mu -> BEKLE'ye don
            if vote_counts.get(label, 0) < 2:
                label = LABEL_HOLD
                confidence = float(p_ens[LABEL_HOLD])
                logger.debug(
                    f"Konsensus yok, BEKLE | "
                    f"XGB:{votes[0]} LGB:{votes[1]} RF:{votes[2]}"
                )

        probs = {
            LABEL_SELL: round(float(p_ens[LABEL_SELL]), 4),
            LABEL_HOLD: round(float(p_ens[LABEL_HOLD]), 4),
            LABEL_BUY : round(float(p_ens[LABEL_BUY]),  4),
        }

        logger.debug(
            f"Ensemble tahmin | label:{label} conf:{confidence:.3f} | "
            f"XGB:{int(np.argmax(p_xgb))} LGB:{int(np.argmax(p_lgb))} RF:{int(np.argmax(p_rf))}"
        )

        return label, confidence, probs

    def _predict_proba_ensemble(self, X) -> np.ndarray:
        """Tum satirlar icin ensemble olasiligi."""
        return (
            self.weights[0] * self._xgb.predict_proba(X) +
            self.weights[1] * self._lgb.predict_proba(X) +
            self.weights[2] * self._rf.predict_proba(X)
        )

    def is_trained(self) -> bool:
        return (
            self._xgb is not None and
            self._lgb is not None and
            self._rf  is not None
        )

    # ── Kaydet / Yukle ────────────────────────────────────────────────────────

    def save(self, dir_path: str | Path) -> None:
        """
        Modeli klasore kaydeder:
            dir_path/xgb.json
            dir_path/lgb.pkl
            dir_path/rf.pkl
            dir_path/meta.json
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        self._xgb.save_model(str(p / "xgb.json"))

        with open(p / "lgb.pkl", "wb") as f:
            pickle.dump(self._lgb, f)

        with open(p / "rf.pkl", "wb") as f:
            pickle.dump(self._rf, f)

        meta = {
            "feature_names" : self.feature_names,
            "weights"       : self.weights,
            "n_splits"      : self.n_splits,
            "min_confidence": self.min_confidence,
            "consensus_only": self.consensus_only,
            "training_meta" : self.training_meta,
            "cv_results"    : self._cv_results,
        }
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Ensemble model kaydedildi: {p}")

    @classmethod
    def load(cls, dir_path: str | Path) -> "EnsembleModel":
        """Kaydedilmis ensemble modelini yukler."""
        p = Path(dir_path)

        with open(p / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            weights        = meta["weights"],
            n_splits       = meta["n_splits"],
            min_confidence = meta["min_confidence"],
            consensus_only = meta["consensus_only"],
        )
        instance.feature_names  = meta["feature_names"]
        instance.training_meta  = meta["training_meta"]
        instance._cv_results    = meta.get("cv_results", {})

        instance._xgb = xgb.XGBClassifier()
        instance._xgb.load_model(str(p / "xgb.json"))

        with open(p / "lgb.pkl", "rb") as f:
            instance._lgb = pickle.load(f)

        with open(p / "rf.pkl", "rb") as f:
            instance._rf = pickle.load(f)

        logger.info(f"Ensemble model yuklendi: {p}")
        return instance

    # ── Raporlar ──────────────────────────────────────────────────────────────

    def cv_report(self) -> None:
        """CV sonuclarini terminale yazdirir."""
        if not self._cv_results:
            print("CV henuz yapilmamis.")
            return

        print("\n" + "=" * 65)
        print("  ENSEMBLE CV SONUCLARI (XGBoost + LightGBM + RandomForest)")
        print("=" * 65)
        print(f"  Fold sayisi  : {self._cv_results.get('n_folds')}")
        print(f"  Ort. Accuracy: {self._cv_results.get('avg_accuracy', 0):.3f}")
        print("-" * 65)
        for r in self._cv_results.get("fold_results", []):
            print(
                f"  Fold {r['fold']} | "
                f"Train:{r['train_size']:4d} Test:{r['test_size']:4d} | "
                f"Acc:{r['accuracy']:.3f} | "
                f"AL_acc:{r['al_acc']:.3f} | "
                f"SAT_acc:{r['sat_acc']:.3f}"
            )
        print("=" * 65)

    def model_votes_report(self, X: pd.DataFrame) -> None:
        """Son satir icin her modelin oyunu terminale yazdirir."""
        if not self.is_trained():
            print("Model egitilmemis.")
            return

        X_last = X.iloc[[-1]]
        label_map = {LABEL_SELL: "SAT", LABEL_HOLD: "BEKLE", LABEL_BUY: "AL"}

        p_xgb = self._xgb.predict_proba(X_last)[0]
        p_lgb = self._lgb.predict_proba(X_last)[0]
        p_rf  = self._rf.predict_proba(X_last)[0]
        p_ens = (
            self.weights[0] * p_xgb +
            self.weights[1] * p_lgb +
            self.weights[2] * p_rf
        )

        print("\n" + "=" * 55)
        print("  ENSEMBLE OY DAGILIMI (Son Bar)")
        print("=" * 55)
        header = f"  {'MODEL':<15} {'SAT':>8} {'BEKLE':>8} {'AL':>8}  {'KARAR':>6}"
        print(header)
        print("-" * 55)

        for name, p, w in [
            ("XGBoost", p_xgb, self.weights[0]),
            ("LightGBM", p_lgb, self.weights[1]),
            ("RandomForest", p_rf, self.weights[2]),
        ]:
            karar = label_map[int(np.argmax(p))]
            print(
                f"  {name:<15} "
                f"{p[LABEL_SELL]:>7.2%} "
                f"{p[LABEL_HOLD]:>8.2%} "
                f"{p[LABEL_BUY]:>7.2%}  "
                f"{karar:>6} (w={w})"
            )

        print("-" * 55)
        ens_karar = label_map[int(np.argmax(p_ens))]
        print(
            f"  {'ENSEMBLE':<15} "
            f"{p_ens[LABEL_SELL]:>7.2%} "
            f"{p_ens[LABEL_HOLD]:>8.2%} "
            f"{p_ens[LABEL_BUY]:>7.2%}  "
            f"{ens_karar:>6}"
        )
        print("=" * 55)
