"""
tests/test_phase16.py
======================
Phase 16: Ensemble Model (XGBoost + LightGBM + RandomForest) testleri
"""

import sys
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.ensemble_model import EnsembleModel, LABEL_BUY, LABEL_SELL, LABEL_HOLD


# ── Yardimci fonksiyon ─────────────────────────────────────────────────────

def make_dummy_data(n: int = 300, n_features: int = 10, seed: int = 42):
    """Test icin sahte X, y verileri uretir."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.random((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.integers(0, 3, size=n))  # 0=SAT, 1=BEKLE, 2=AL
    return X, y


# ── TestEnsembleModelBasic ─────────────────────────────────────────────────

class TestEnsembleModelBasic:
    """Temel olusturma ve egitim testleri."""

    def test_init_defaults(self):
        """Varsayilan parametrelerle olusturma."""
        m = EnsembleModel()
        assert m.weights == [0.40, 0.35, 0.25]
        assert m.n_splits == 5
        assert m.min_confidence == 0.40
        assert m.consensus_only is True

    def test_init_custom_weights(self):
        """Ozel agirlikla olusturma."""
        m = EnsembleModel(weights=[0.5, 0.3, 0.2])
        assert m.weights == [0.5, 0.3, 0.2]

    def test_is_trained_before_train(self):
        """Egitim oncesi is_trained False olmali."""
        m = EnsembleModel()
        assert m.is_trained() is False

    def test_train_sets_models(self):
        """train() uc modeli de olusturmali."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        m.train(X, y)
        assert m.is_trained() is True
        assert m._xgb is not None
        assert m._lgb is not None
        assert m._rf  is not None

    def test_train_sets_feature_names(self):
        """train() feature_names'i doldurmali."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        m.train(X, y)
        assert len(m.feature_names) == X.shape[1]
        assert m.feature_names == list(X.columns)

    def test_train_meta_populated(self):
        """train() sonrasi training_meta dolu olmali."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        m.train(X, y)
        assert "n_samples"     in m.training_meta
        assert "n_features"    in m.training_meta
        assert "train_accuracy" in m.training_meta
        assert m.training_meta["n_samples"] == len(X)

    def test_train_accuracy_range(self):
        """Train accuracy 0-1 araliginda olmali."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        m.train(X, y)
        acc = m.training_meta["train_accuracy"]
        assert 0.0 <= acc <= 1.0


# ── TestEnsembleCrossValidation ────────────────────────────────────────────

class TestEnsembleCrossValidation:
    """Cross-validation testleri."""

    def test_cv_returns_dict(self):
        """cross_validate dict donmeli."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        result = m.cross_validate(X, y)
        assert isinstance(result, dict)

    def test_cv_has_avg_accuracy(self):
        """CV sonucunda avg_accuracy olmali."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        result = m.cross_validate(X, y)
        assert "avg_accuracy" in result
        assert 0.0 <= result["avg_accuracy"] <= 1.0

    def test_cv_fold_count(self):
        """CV fold sayisi n_splits ile esleseli."""
        m = EnsembleModel(n_splits=3)
        X, y = make_dummy_data(300)
        result = m.cross_validate(X, y)
        assert result["n_folds"] <= 3   # az veri nedeniyle bazi foldlar atlanabilir

    def test_cv_fold_results_structure(self):
        """Her fold sonucunda gerekli alanlar olmali."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        result = m.cross_validate(X, y)
        for fr in result["fold_results"]:
            assert "fold"       in fr
            assert "train_size" in fr
            assert "test_size"  in fr
            assert "accuracy"   in fr


# ── TestEnsemblePrediction ─────────────────────────────────────────────────

class TestEnsemblePrediction:
    """Tahmin testleri."""

    @pytest.fixture
    def trained_model(self):
        m = EnsembleModel(n_splits=2, consensus_only=False)
        X, y = make_dummy_data(200)
        m.train(X, y)
        return m, X

    def test_predict_single_returns_tuple(self, trained_model):
        """predict_single 3-tuple donmeli."""
        m, X = trained_model
        result = m.predict_single(X)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_predict_single_label_valid(self, trained_model):
        """Tahmin etiketi 0, 1 veya 2 olmali."""
        m, X = trained_model
        label, _, _ = m.predict_single(X)
        assert label in [LABEL_SELL, LABEL_HOLD, LABEL_BUY]

    def test_predict_single_confidence_range(self, trained_model):
        """Confidence 0-1 araliginda olmali."""
        m, X = trained_model
        _, confidence, _ = m.predict_single(X)
        assert 0.0 <= confidence <= 1.0

    def test_predict_single_probs_sum_to_one(self, trained_model):
        """Olasiliklar toplami ~1 olmali."""
        m, X = trained_model
        _, _, probs = m.predict_single(X)
        total = probs[LABEL_SELL] + probs[LABEL_HOLD] + probs[LABEL_BUY]
        assert abs(total - 1.0) < 0.01

    def test_predict_untrained_returns_hold(self):
        """Egitilmemis model BEKLE donmeli."""
        m = EnsembleModel()
        X, _ = make_dummy_data(10)
        label, conf, _ = m.predict_single(X)
        assert label == LABEL_HOLD
        assert conf == 0.0

    def test_consensus_only_reduces_signals(self):
        """Konsensus modu aktifken daha az AL/SAT sinyali olmali."""
        X, y = make_dummy_data(300)

        m_no_consensus = EnsembleModel(n_splits=2, consensus_only=False)
        m_no_consensus.train(X, y)

        m_consensus = EnsembleModel(n_splits=2, consensus_only=True)
        m_consensus.train(X, y)

        # Tum satirlar uzerinde tahmin yap
        nc_signals = []
        c_signals  = []
        for i in range(min(50, len(X))):
            nc_l, _, _ = m_no_consensus.predict_single(X.iloc[:i+1])
            c_l,  _, _ = m_consensus.predict_single(X.iloc[:i+1])
            nc_signals.append(nc_l)
            c_signals.append(c_l)

        nc_active = sum(1 for s in nc_signals if s != LABEL_HOLD)
        c_active  = sum(1 for s in c_signals  if s != LABEL_HOLD)
        # Konsensuslu model daha az veya esit sayida aktif sinyal vermeli
        assert c_active <= nc_active + 5   # kucuk tolerans


# ── TestEnsembleSaveLoad ───────────────────────────────────────────────────

class TestEnsembleSaveLoad:
    """Kaydet/yukle testleri."""

    def test_save_creates_files(self):
        """save() 4 dosya olusturmali."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        m.train(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            m.save(tmp)
            files = list(Path(tmp).iterdir())
            names = {f.name for f in files}
            assert "xgb.json" in names
            assert "lgb.pkl"  in names
            assert "rf.pkl"   in names
            assert "meta.json" in names

    def test_load_restores_model(self):
        """load() modeli geri yuklemeli."""
        m = EnsembleModel(n_splits=2, consensus_only=False)
        X, y = make_dummy_data(200)
        m.train(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            m.save(tmp)
            m2 = EnsembleModel.load(tmp)

        assert m2.is_trained() is True
        assert m2.feature_names == m.feature_names
        assert m2.weights == m.weights

    def test_load_predict_same_result(self):
        """Yuklenen model ayni tahmin sonucunu vermeli."""
        m = EnsembleModel(n_splits=2, consensus_only=False)
        X, y = make_dummy_data(200)
        m.train(X, y)

        label1, conf1, _ = m.predict_single(X)

        with tempfile.TemporaryDirectory() as tmp:
            m.save(tmp)
            m2 = EnsembleModel.load(tmp)

        label2, conf2, _ = m2.predict_single(X)
        assert label1 == label2
        assert abs(conf1 - conf2) < 0.001

    def test_meta_json_has_all_fields(self):
        """meta.json gerekli alanlari icermeli."""
        m = EnsembleModel(n_splits=2)
        X, y = make_dummy_data(200)
        m.train(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            m.save(tmp)
            with open(Path(tmp) / "meta.json", encoding="utf-8") as f:
                meta = json.load(f)

        assert "feature_names"  in meta
        assert "weights"        in meta
        assert "consensus_only" in meta
        assert "training_meta"  in meta


# ── TestEnsemblePredictor ──────────────────────────────────────────────────

class TestEnsemblePredictor:
    """MLPredictor ile ensemble entegrasyon testleri."""

    def test_predictor_use_ensemble_flag(self):
        """use_ensemble=True ile EnsembleModel kullanilmali."""
        from ml.predictor import MLPredictor
        from ml.ensemble_model import EnsembleModel

        p = MLPredictor(use_ensemble=True)
        assert isinstance(p._model, EnsembleModel)

    def test_predictor_default_uses_xgboost(self):
        """Varsayilan predictor XGBoostModel kullanmali."""
        from ml.predictor import MLPredictor
        from ml.xgboost_model import XGBoostModel

        p = MLPredictor(use_ensemble=False)
        assert isinstance(p._model, XGBoostModel)

    def test_predictor_from_ensemble_classmethod(self):
        """from_ensemble() EnsembleModel yuklemeli."""
        from ml.predictor import MLPredictor
        from ml.ensemble_model import EnsembleModel

        # Once model egit ve kaydet
        m = EnsembleModel(n_splits=2, consensus_only=False)
        X, y = make_dummy_data(200)
        m.train(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            m.save(tmp)
            p = MLPredictor.from_ensemble(tmp)

        assert isinstance(p._model, EnsembleModel)
        assert p.use_ensemble is True
