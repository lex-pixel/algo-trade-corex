# ALGO TRADE CODEX — Yapılacaklar Listesi

Son güncelleme: 2026-03-07 (Oturum 6)
Durum: ✅ Tamamlandı | 🔄 Devam Ediyor | ⏳ Bekliyor | ❌ Engellendi

---

## PHASE 1 — Altyapı Kurulumu & Python Temelleri

### Tamamlananlar ✅
- [x] Proje klasör yapısı oluşturuldu (data, strategies, trading, ml, risk, monitoring, tests, config, utils, logs)
- [x] Her klasöre `__init__.py` eklendi
- [x] `.gitignore` oluşturuldu (API key, veri, model dosyaları korunuyor)
- [x] `.env.example` oluşturuldu (Binance/Bybit testnet şablonu)
- [x] `requirements.txt` oluşturuldu (tüm bağımlılıklar açıklamalı)
- [x] `strategies/base_strategy.py` — BaseStrategy soyut sınıfı + Signal dataclass
- [x] `strategies/hello_strategy.py` — İlk test stratejisi (momentum tabanlı)
- [x] Git başlatıldı, ilk commit atıldı
- [x] GitHub'a bağlandı ve push edildi → github.com/lex-pixel/algo-trade-corex
- [x] `SESSION_LOG.md` oluşturuldu (oturum takibi)

### Tamamlananlar (Oturum 2) ✅
- [x] `utils/logger.py` — loguru tabanlı merkezi log sistemi (terminal + 3 dosya)
- [x] `strategies/rsi_strategy.py` — RSI Mean Reversion stratejisi (pandas-ta)
- [x] `tests/test_strategies.py` — 24 unit test, 24/24 PASSED

### Tamamlananlar (Oturum 3) ✅
- [x] `config/settings.yaml` + `config/loader.py` — pydantic doğrulamalı YAML konfigürasyon
- [x] `main.py` — gerçek Testnet verisiyle çalışır hale getirildi

---

## PHASE 2 — Veri Katmanı ✅

- [x] CCXT kurulumu ve Binance Testnet API bağlantısı
- [x] `data/fetcher.py` → REST API ile OHLCV verisi çekme (200 gerçek BTC/USDT mumu test edildi)
- [x] `data/cleaner.py` → Veri temizleme pipeline (NaN, duplikat, anomali, OHLC mantık)
- [x] `tests/test_fetcher.py` → 21/21 PASSED
- [ ] `data/ws_listener.py` → WebSocket (Phase 6'ya ertelendi)
- [ ] TimescaleDB / Redis → (Phase 6'ya ertelendi)

---

## PHASE 3 — Strateji ve Sinyal Motoru ✅

- [x] `strategies/indicators.py` → RSI, MACD, BB, ATR, ADX merkezi hesaplama
- [x] `strategies/regime_detector.py` → ADX tabanlı RANGE/TREND/TRANSITION tespiti
- [x] `strategies/pa_range_strategy.py` → Destek/direnç + RSI + rejim filtresi
- [x] `strategies/rsi_strategy.py` → RSI Mean Reversion
- [x] Sinyal kombinasyonu → main.py'de iki strateji kombine ediliyor
- [x] `config/settings.yaml` + `config/loader.py` → PA Range parametreleri eklendi
- [x] `tests/test_phase3.py` → 26/26 PASSED (toplam 71/71)

---

## PHASE 4 — Backtesting ve Optimizasyon ✅

- [x] `backtesting/engine.py` → Bar-close olay tabanlı backtest motoru
- [x] Gerçekçi maliyet modeli → komisyon %0.1 + slipaj %0.05
- [x] `backtesting/metrics.py` → Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor
- [x] `backtesting/run_backtest.py` → RSI vs PA Range karşılaştırması, Parquet cache
- [x] `optimization/optuna_optimizer.py` → Bayesian parametre optimizasyonu (RSI + PA Range)
- [x] `dashboard.py` → Rich terminali ile canlı sinyal paneli
- [x] `tests/test_backtesting.py` → 22/22 PASSED (toplam 93/93)
- [ ] Monte Carlo simülasyonu → (Phase 8'e ertelendi)
- [ ] HTML rapor → (Phase 8'e ertelendi)

---

## PHASE 5 — ML / AI Modeli ✅

- [x] `ml/feature_engineering.py` → 52 teknik ozellik (lag, rolling, RSI/MACD/BB/ATR/ADX/hacim/momentum)
- [x] Lookahead bias onlemi → TimeSeriesSplit cross-validation (n_splits=5)
- [x] `ml/xgboost_model.py` → XGBoost siniflandirici (AL/SAT/BEKLE), SHAP destekli
- [x] `ml/predictor.py` → canli predict(), ATR tabanli SL/TP, save/load
- [x] `tests/test_phase5.py` → 35/35 PASSED (toplam 128/128)
- [ ] MLflow entegrasyonu → (Phase 8'e ertelendi)
- [ ] A/B test → ML sinyalli vs kural bazli → (Phase 8'e ertelendi)

---

## PHASE 6 — Canlı İşlem Altyapısı ✅

- [x] `trading/order_manager.py` → OrderManager: market/limit emir, iptal, simulate_fill, paper mod
- [x] `trading/position_tracker.py` → PositionTracker: LONG/SHORT pozisyon, SL/TP otomatik, P&L takibi
- [x] `trading/main_loop.py` → asyncio TradingBot: RSI+PA sinyal birlestirme, paper/live mod, hata yonetimi
- [x] Paper trading modu → paper=True ile sahte emir, gercek fiyat
- [x] `monitoring/telegram_notifier.py` → Telegram: AL/SAT/SL/TP/gunluk ozet bildirimleri, dry-run
- [x] `tests/test_phase6.py` → 54/54 PASSED (toplam 182/182)
- [ ] 2 hafta paper trading — performans kaydet (canli calistirma gerektirir)
- [ ] WebSocket canli veri → (Phase 8'e ertelendi)

---

## PHASE 7 — Risk Sistemi ✅

- [x] `risk/position_sizer.py` → FixedFraction + ATR + Kelly (Yari Kelly) + Conservative
- [x] `risk/kill_switch.py` → 3 seviyeli alarm: Sari %3 / Turuncu %5 / Kirmizi %15, drawdown, hata sayaci
- [x] `risk/risk_manager.py` → Merkezi risk: sinyal degerlendirme, SL/TP, audit log, KillSwitch entegre
- [x] `tests/test_phase7.py` → 52/52 PASSED (toplam 234/234)
- [ ] Portfolio korelasyon monitoru → (Phase 8'e ertelendi)
- [ ] API arizasi protokolu → (Phase 8'e ertelendi)
- [ ] Sunucu guvenligi (UFW, SSH) → (Deployment asamasinda)

---

## PHASE 8 — İzleme, Analiz ve Sürekli İyileştirme (Sürekli)

- [ ] Docker Compose → tüm servisleri ayağa kaldır (bot + DB + Redis + Grafana)
- [ ] Grafana + Prometheus → canlı P&L, sinyal akışı, API sağlık paneli
- [ ] Model drift monitörü → Evidently AI
- [ ] `scripts/weekly_review.py` → haftalık strateji review scripti
- [ ] Aylık performans raporu otomasyon scripti
- [ ] Yeni strateji A/B test pipeline
- [ ] `.github/workflows/main.yml` → CI/CD (GitHub Actions: test + Docker build + deploy)

---

## Notlar

- Borsa: Binance Testnet (sahte para) → live'a geçince `.env`'de `BINANCE_TESTNET=false`
- Python: 3.12.3 (rapor 3.11 diyor ama 3.12 uyumlu)
- Proje dizini: `C:\Users\rk209\Desktop\Cloud-Algo\`
- GitHub: github.com/lex-pixel/algo-trade-corex
