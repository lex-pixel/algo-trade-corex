# ALGO TRADE CODEX — Yapılacaklar Listesi

Son güncelleme: 2026-04-07 (Oturum 9)
Durum: ✅ Tamamlandı | 🔄 Devam Ediyor | ⏳ Bekliyor | ❌ Engellendi

---

## TAMAMLANAN FAZLAR

| Faz | Açıklama | Test | Durum |
|---|---|---|---|
| Phase 1 | Altyapı, klasör yapısı, git | 24/24 | ✅ |
| Phase 2 | Veri katmanı — CCXT, fetcher, cleaner | 21/21 | ✅ |
| Phase 3 | Strateji motoru — RSI, PA Range, regime detector | 26/26 | ✅ |
| Phase 4 | Backtesting, metrikler, Optuna optimizer | 22/22 | ✅ |
| Phase 5 | ML/AI — XGBoost 3-sınıf, feature engineering, SHAP | 35/35 | ✅ |
| Phase 6 | Canlı işlem — OrderManager, PositionTracker, MainLoop | 54/54 | ✅ |
| Phase 7 | Risk sistemi — PositionSizer, KillSwitch, RiskManager | 52/52 | ✅ |
| Phase 8 | MTF (4h trend + 15m entry zamanlama) | - | ✅ |
| Phase 9 | Görsel dashboard — Plotly 6 panel HTML | - | ✅ |
| Phase 10 | Walk-forward auto-retrain (720 tick / 30 gün) | - | ✅ |
| Phase 12 | Trailing stop — breakeven + partial close | - | ✅ |
| Phase 13-14 | SHORT desteği, paper mod iyileştirmeleri | - | ✅ |
| Phase 15 | Telegram bildirim entegrasyonu (main_loop.py) | 27/27 | ✅ |
| Phase 16 | Ensemble model — XGBoost + LightGBM + RandomForest | 24/24 | ✅ |
| Phase 17-21 | Otomatik rapor, dashboard, R:R, walk-forward, SHORT backtest | - | ✅ |
| Phase 22 | TradingView benzeri canlı mum grafiği (live_chart.py) | - | ✅ |
| Phase 20 | Multi-coin bot (MultiCoinBot, CoinWorker) | 20/20 | ✅ |
| Phase 23 | Kaldıraçlı trading (LeverageManager) | 39/39 | ✅ |
| Fix | KillSwitch yeniden başlatma hatası düzeltildi | - | ✅ |
| Fix | Test log izolasyonu (conftest.py) | - | ✅ |

**Toplam: 462/462 test PASSED**

---

## SIRADAKI GÖREVLER (Öncelik Sırasına Göre)

### 🔥 Yüksek Öncelik

- [ ] **Telegram token kurulumu** — `.env` dosyasına `TELEGRAM_TOKEN` ve `TELEGRAM_CHAT_ID` ekle
  - @BotFather ile bot oluştur → token al
  - @userinfobot ile chat_id bul
  - Şu an dry_run=True çalışıyor, gerçek bildirim gitmiyor

- [ ] **VPS/Server kurulumu** — 7/24 bot çalıştırmak için
  - Önerilen: DigitalOcean/Hetzner 2GB RAM Ubuntu 22.04 (~$6-12/ay)
  - Bot kodu git clone ile çekilir
  - systemd service ile otomatik başlatma
  - Telegram bildirimleri VPS'ten gelir → PC kapalıyken de bildirim alırsın

### 🟡 Orta Öncelik

- [ ] **Binance Futures/Margin API entegrasyonu** (Phase 23 tamamlayıcı)
  - Şu an kaldıraç hesapları sadece paper modda çalışıyor
  - Gerçek SHORT/futures emir için `ccxt.binanceusdm()` + futures testnet key gerekiyor
  - Phase 23'te altyapı hazır, sadece OrderManager güncellenmeli

- [ ] **Multi-coin bot aktif kullanımı** (Phase 20 tamamlayıcı)
  - BTC + ETH + SOL aynı anda çalıştırma testi
  - Her coin için ayrı Telegram bildirimi

- [ ] **CI/CD — GitHub Actions**
  - Push olunca otomatik `pytest tests/` çalışsın
  - `.github/workflows/test.yml` dosyası oluştur

### 🟢 Düşük Öncelik / Gelecek

- [ ] **Docker Compose** — bot + tüm bağımlılıkları container'a al
- [ ] **Grafana + Prometheus** — canlı P&L, API sağlık paneli
- [ ] **WebSocket canlı veri** — REST polling yerine gerçek zamanlı
- [ ] **Model drift monitörü** — ML modelinin bozulduğunu tespit et (Evidently AI)
- [ ] **Ensemble model gerçek eğitim** — `python scripts/train_ensemble.py --days 365`
  - Şu an XGBoost tek model aktif, ensemble sadece test ortamında
- [ ] **Haftalık otomatik strateji review** scripti
- [ ] **Live trading geçişi** — testnet → gerçek Binance (EN SON)
  - `.env`'de `BINANCE_TESTNET=false`
  - Minimum 2 ay başarılı paper/testnet sonucu şart

---

## Notlar

- Borsa: Binance Testnet (sahte para) — live'a geçince `.env`'de `BINANCE_TESTNET=false`
- Python: `C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe`
- Proje dizini: `C:\Users\rk209\Desktop\Cloud-Algo\`
- GitHub: github.com/lex-pixel/algo-trade-corex
- Bot state: `data/bot_state.json` — PC kapansa bile devam eder
- Açık pozisyon: SHORT @ $71,826 | SL: $70,749 | TP: $68,753 (2026-04-06'dan beri)
