# ALGO TRADE CODEX — Oturum Logu

Bu dosya her çalışma oturumunda güncellenir.
Limit bitip yeni oturum açıldığında buraya bakarak kaldığımız yerden devam ederiz.

---

## OTURUM 1 — 2026-03-06

### Tamamlananlar

**Klasör Yapısı Kuruldu:**
```
Cloud-Algo/
├── data/           -> Borsa API, veri çekme, TimescaleDB, Redis
├── strategies/     -> Strateji sınıfları (BaseStrategy, PA Range, RSI...)
├── trading/        -> Canlı işlem altyapısı, OrderManager
├── ml/             -> XGBoost, LSTM, feature engineering
├── risk/           -> RiskManager, KillSwitch (3 seviye)
├── monitoring/     -> Telegram, Grafana, Prometheus
├── tests/          -> pytest unit testler
├── config/         -> YAML konfigürasyon
├── utils/          -> Logger, yardımcı fonksiyonlar
└── logs/           -> Log dosyaları (git'e gitmez)
```

**Oluşturulan Dosyalar:**
| Dosya | Ne Yapar |
|---|---|
| `.gitignore` | API key, veri, model dosyalarını GitHub'dan korur |
| `.env.example` | Binance/Bybit testnet, DB, Telegram şablonu |
| `requirements.txt` | Tüm bağımlılıklar (ccxt, pandas, xgboost, vb.) |
| `strategies/base_strategy.py` | Soyut temel sınıf + Signal dataclass |
| `strategies/hello_strategy.py` | İlk test stratejisi (momentum tabanlı) |

**Git:** ilk commit 699a520

---

## OTURUM 4 — 2026-03-07

### PHASE 5: ML / AI Modeli

| Dosya | Aciklama |
|---|---|
| `ml/feature_engineering.py` | 52 ozellik: RSI/MACD/BB/ATR/ADX + hacim + momentum + lag + rolling + mum sekli |
| `ml/xgboost_model.py` | XGBoost siniflandirici: AL/SAT/BEKLE, TimeSeriesSplit CV, SHAP destekli, save/load |
| `ml/predictor.py` | Canli tahmin arayuzu: predict() -> Signal, ATR tabanli SL/TP, from_file() |
| `tests/test_phase5.py` | 35 test |

**Test:** 35/35 PASSED | Toplam: 128/128

---

## OTURUM 5 — 2026-03-07

### PHASE 6: Canli Islem Altyapisi

| Dosya | Aciklama |
|---|---|
| `trading/order_manager.py` | Market/limit emir, slipaj, komisyon, paper/live mod |
| `trading/position_tracker.py` | LONG/SHORT pozisyon, unrealized P&L, SL/TP otomatik |
| `trading/main_loop.py` | asyncio TradingBot: RSI+PA sinyal birlestirme |
| `monitoring/telegram_notifier.py` | Telegram Bot API bildirimleri |
| `tests/test_phase6.py` | 54 test |

**Test:** 54/54 PASSED | Toplam: 182/182

---

## OTURUM 6 — 2026-03-07

### PHASE 7: Risk Sistemi

| Dosya | Aciklama |
|---|---|
| `risk/position_sizer.py` | FixedFraction + ATR + Kelly (Yari) + Conservative |
| `risk/kill_switch.py` | 3 seviye: Sari/Turuncu/Kirmizi, drawdown, hata sayaci |
| `risk/risk_manager.py` | Merkezi risk kontrolu, audit log |
| `tests/test_phase7.py` | 52 test |

**Test:** 52/52 PASSED | Toplam: 234/234

---

## OTURUM 7 — 2026-03-08

### PHASE 8: MTF (Multi-Timeframe) Entegrasyonu

- 4h trend filtresi: TREND_DOWN + AL -> BEKLE (kontra-trend engeli)
- 15m giris zamanlama: RSI>65 AL engel, RSI<35 SAT engel
- `config/loader.py`: optimized_params.yaml otomatik yukleme
- ML egitim scripti: `python -m ml.train` (365 gun, CV acc: 0.421)
- Optuna optimizer: RSI + PA Range parametreleri optimize edildi
- `config/optimized_params.yaml` olusturuldu

**Test:** 234/234 PASSED

---

## OTURUM 8 — 2026-03-09

### PHASE 9: Gorsel Dashboard

| Dosya | Aciklama |
|---|---|
| `scripts/dashboard.py` | Plotly 6 panel HTML: equity curve, drawdown, PnL, win/loss |
| `scripts/summary.py` | Terminal ozet |
| `scripts/watch.py` | Canli log izleme |

**Kullanim:** `py -3.12 -m scripts.dashboard --open`

**Test:** 257/257 PASSED

---

## OTURUM 9 — 2026-03-09

### PHASE 10: Walk-Forward Auto-Retrain

| Dosya | Aciklama |
|---|---|
| `ml/auto_retrain.py` | 720 tick (~30 gun) tetikleyici, walk-forward validation, MIN_ACC=0.35 |

- Atomik model guncelleme: once _new.json'a yaz, basarili ise tasinir
- Bot icinden otomatik tetiklenir, elle de calistirilabilir

**Test:** 257/257 PASSED

---

## OTURUM 10 — 2026-03-10 / 11

### Bot Canlı Testleri + Bug Düzeltmeleri

**Sorunlar ve cozumler:**

| Sorun | Cozum | Dosya |
|---|---|---|
| 407 tick, 0 islem | Voting sistemi degistirildi: BEKLE oylar yok sayilir, AL vs SAT yaris | `main_loop.py` |
| 4h hard block | Soft penalty (%30 confidence dusurme) | `main_loop.py` |
| min_confidence cok yuksek | 0.55 -> 0.30 | `main_loop.py` |
| KillSwitch yanlis TURUNCU | Pozisyon notional nakit'ten dusuluyor, KillSwitch %10 zarar goruyor | `position_tracker.py`, `main_loop.py`, `kill_switch.py` |
| KillSwitch state yuklenince TURUNCU | load_state() sonrasi update_day_start() cagrisi eklendi | `main_loop.py` |
| Dashboard Max Drawdown %1019 | max_drawdown zaten yuzde, *100 kaldirildi | `dashboard.py` |
| auto_retrain dict->float hatasi | cv_results.get("avg_accuracy") ile duzeltildi | `auto_retrain.py` |

**Bot ilk islemleri:**
- T1: SHORT BTC @ $70,734 -> KillSwitch TURUNCU (bug) -> -$3.42 + $2 fee
- T2: SHORT BTC @ $70,052 -> Hala acik (TP: $60,668 | SL: $76,367)

**Python kurulum notu:**
- Sistem Python 3.12 bozuldu (python.exe 0 byte)
- Python 3.14 yuklendi ama pandas-ta uyumsuz (numba <3.14)
- Cozum: `py -3.12` launcher ile Python 3.12 kullan
- Bot calistirma: `py -3.12 -m trading.main_loop`

**Test:** 257/257 PASSED

---

## PHASE 12 — Tamamlandi (2026-03-12)

**Test Sonucu:** 289/289 PASSED (32 yeni test eklendi)

---

### PHASE 12A: Trailing Stop — TAMAMLANDI
Dosyalar: trading/position_tracker.py, trading/main_loop.py, config/settings.yaml, config/loader.py

Yapilan degisiklikler:
- Position dataclass: breakeven_triggered, partial_closed, partial_quantity alanlari eklendi
- partial_close_position(): pozisyonun istenen miktarini kapatir, kalan devam eder
- check_trailing_stops(): tum acik pozisyonlar icin trailing stop kontrol eder
  -> %1.5 karda: BREAKEVEN action (SL = entry)
  -> %3 karda: PARTIAL_CLOSE action (yarisi kapatilir, SL = entry + %1.5)
- main_loop._tick(): trailing stop check entegre edildi (2b adimi)
- settings.yaml: trailing_stop bolumu eklendi (enabled, breakeven_pct, partial_close_pct, trail_sl_pct)
- config/loader.py: TrailingStopConfig sinifi eklendi

### PHASE 12B: PA Range Gelistirmesi — TAMAMLANDI
Dosyalar: strategies/pa_range_strategy.py, config/settings.yaml, config/loader.py

Yapilan degisiklikler:
- Volume Confirmation: hacim 1.5x 20-bar ortalama uzerinde olmali (volume_confirm_mult)
  -> Hacim dusuksa sinyal uretilmez (fakeout riski)
  -> Hacim OK ise confidence +0.08 bonus
- Fakeout Filtresi: kapanisin destek/direnc bolgesinde olmasi kontrol edilir
  -> Sadece golge dokunen mumlar engellenir
  -> fakeout_filter=True ile settings.yaml'dan kontrol edilebilir
- RSI Divergence: fiyat dip/zirve yaparken RSI yapmiyorsa +0.10 confidence bonus
  -> Bullish: fiyat dip, RSI yukseliyor -> guclu AL sinyali
  -> Bearish: fiyat zirve, RSI dusuyor -> guclu SAT sinyali
  -> rsi_divergence=True ile ayarlanabilir

### PHASE 12C: Short/Long Ayni Anda — TAMAMLANDI (paper mod)
Dosyalar: trading/position_tracker.py, trading/main_loop.py, config/settings.yaml

Yapilan degisiklikler:
- has_long_position(symbol): acik LONG pozisyon var mi?
- has_short_position(symbol): acik SHORT pozisyon var mi?
- max_open_positions: 1 -> 2 (LONG + SHORT ayni anda)
- main_loop: ayni yonde pozisyon varsa yeni pozisyon acilmaz
  -> AL sinyali + zaten LONG var -> atla
  -> SAT sinyali + zaten SHORT var -> atla
  -> AL sinyali + SHORT var, LONG yok -> yeni LONG ac (her iki yon ayni anda calisiyor)
NOT: Gercek SHORT icin Binance Futures/Margin gerekir (paper modda simule edilir)

### PHASE 12D: ML External Data — TAMAMLANDI
Dosyalar: ml/external_data.py (yeni), ml/feature_engineering.py

Yapilan degisiklikler:
- ml/external_data.py: ExternalDataFetcher sinifi
  -> get_fear_greed(): alternative.me API, 0-100 (API anahtari gerektirmez)
  -> get_btc_dominance(): CoinGecko /global, 0-100 yuzde
  -> 1 saatlik onbellekleme (gereksiz API cagrisi yok)
  -> Hata durumunda None doner, bot calismasini engellemez
  -> get_external_fetcher(): singleton pattern
- feature_engineering.py:
  -> use_external_data parametresi eklendi (varsayilan: True)
  -> _add_external_features(): fear_greed_norm, btc_dominance_norm eklendi (0-1)
  -> fg_rsi_diverge: Fear & Greed ile RSI arasindaki fark (uyumsuzluk sinyali)

## Gelecek Gelistirmeler
- Binance Futures/Margin: gercek SHORT desteği
- ETH/USDT coklu parite feature (feature_engineering'e eklenecek)

## TAMAMLANAN GELISTIRMELER — 2026-03-13

### Phase 13 — PA Gelistirmeleri (PA-1'den PA-10'a)
Dosya: `strategies/pa_range_strategy.py`
| # | Özellik | Tip |
|---|---------|-----|
| PA-1 | EQ Seviyesi + S/R Flip | Confidence ±0.10 |
| PA-2 | Deviasyon Tespiti | Confidence +0.15 |
| PA-3 | Order Block | Confidence +0.12 |
| PA-4 | Market Yapısı (CHoCH/BOS) | Filtre |
| PA-5 | OTE Fibonacci 0.618-0.786 | Confidence +0.15 |
| PA-6 | Imbalance/GAP | TP hedefi değiştirir |
| PA-7 | Likidite Tabanlı TP | TP hedefi değiştirir |
| PA-8 | Key Levels (Daily/Weekly) | Confidence +0.10 |
| PA-9 | Power of 3 (AMD) | Filtre + +0.08 |
| PA-10 | SFP (Swing Failure Pattern) | Confidence +0.13 |

### Phase 14 — Veri Katmanı
- **VWAP**: `pa_range_strategy.py` — fiyat VWAP altı AL +0.10, üstü SAT +0.10
- **Funding Rate**: `external_data.py` + `feature_engineering.py` — `funding_rate_norm` ML özelliği
- **Open Interest**: `external_data.py` + `feature_engineering.py` — `oi_norm` ML özelliği
- fetch_all() artık 4 kaynak döndürüyor: F&G + BTC Dom + Funding Rate + OI
- ML modeli 53 özellik (bir sonraki auto-retrainde 53-feature model eğitilecek)

---

## TAMAMLANAN GELISTIRMELER — 2026-03-13 (2. oturum)

**Test Sonucu:** 339/339 PASSED (+50 yeni test)  |  commit: 1641fdc

### Phase 17 — Otomatik Raporlama — TAMAMLANDI
Dosya: `scripts/report.py`
- Sharpe, Sortino, Calmar, Profit Factor, Expectancy, Win Rate hesaplar
- A/B/C/D not sistemi (4 metrik kombinasyonu)
- TXT + JSON otomatik rapor -> `reports/report_TARIH.txt`
- `python scripts/report.py` ile calistirilir

### Phase 18 — Dashboard Yenileme — TAMAMLANDI
Dosya: `scripts/dashboard.py`
- UTC+3 timestamp (Turkiye saati) tum grafik ve tablolarda
- Sharpe + Profit Factor ozet tabloya eklendi
- 4. panel: acik pozisyonlar tablosu (yesil=LONG, kirmizi=SHORT)
- 60 saniyede bir auto-refresh (HTML meta tag)
- Dashboard yuksekligi 1000 -> 1250

### Phase 19 — R:R Modulu — TAMAMLANDI
Dosya: `scripts/rr_calc.py`
- LONG/SHORT R:R hesaplama, lot/miktar hesabi
- Kaskade TP destegi (TP1/TP2/TP3 farkli agirliklar: %50/%30/%20)
- Interaktif mod + CLI mod (`--entry --sl --tp --capital --risk`)
- Plotly HTML gorsel (risk/reward bolgeler, kaskade PnL bar)
- `python scripts/rr_calc.py --entry 70000 --sl 68000 --tp 76000 --capital 10000`

### Phase 21 — Backtesting Gelistirme — TAMAMLANDI
Dosyalar: `backtesting/engine.py`, `backtesting/walk_forward.py` (yeni)
- **Walk-Forward**: rolling + expanding window, bilesik getiri, donem ozeti
- **Engine: SHORT destegi** — SAT sinyaline SHORT pozisyon (allow_short=True)
- **Engine: BNB komisyon** — commission_bnb=True ile %0.075 (standart %0.1'in %25 alti)
- **Engine: Hacim bazli slipaj** — yuksek hacimde daha az kayma (slippage_volume_adj=True)
- SHORT PnL: (entry - exit) * size formulu

### Phase 22 — TradingView Benzeri Canli Grafik — TAMAMLANDI
Dosyalar: `scripts/live_chart.py` (yeni)
- **Mum Grafigi**: Binance Testnet OHLCV, Plotly dark theme (#131722)
- **Islem Isaretleri**: Giris ucgen (AL/SAT), cikis X ile isaret, PnL yazisi
- **Acik Pozisyon Cizgileri**: SL (kirmizi), TP (yesil), Giris (sari) yatay cizgi
- **Risk/Reward Shading**: SL bolgesi kirmizi, TP bolgesi yesil transparan
- **RSI Alt Paneli**: 30/50/70 seviye cizgileri, asiri al/sat etiketleri
- **Hacim Paneli**: Yukselis/dusus renkli bar
- **30sn Auto-Refresh**: meta http-equiv refresh
- **UTC+3 Timestamp + Ozet Kutu**: Kapital, PnL, WR, iterasyon
- 13 yeni test -> Toplam 352/352 PASSED
- Commit: 6c34687
- Kullanim: `python scripts/live_chart.py --open`

---

## PLANLANAN GELISTIRMELER — 2026-03-12

### 1. Telegram/Discord Bildirim Sistemi
- Pozisyon acildi/kapandi bildirimi
- KillSwitch tetiklendi alarmi
- Gunluk PnL ozeti (otomatik)
- TP/SL calistigi anlarda anlık mesaj
- PC basinda olmadan bot takibi saglar

### 2. Coklu Coin Destegi (ETH, SOL vb.)
- Simdi sadece BTC/USDT, paralel olarak diger coinler de izlenecek
- Her coin icin ayri sinyal + pozisyon yonetimi
- Portfoy cesitlendirmesi
- max_positions ve risk yonetimi yeniden duzenlenmeli

### 3. Sinyal Kalitesi Iyilestirme
- Ensemble model: XGBoost + LightGBM + RandomForest oylama
- Dinamik guven esigi (simdi sabit 0.3)
- Volatilite tabanli rejim tespiti (sadece trend degil)

### 4. Raporlama / Performans Analizi
- Gunluk/haftalik otomatik rapor (PDF veya metin)
- Sharpe Ratio, Win Rate, Max Drawdown ozeti
- Islem gecmisi analizi (saat dilimi bazli performans)

### 5. Backtesting Gelistirme
- Walk-forward backtest (simdi sadece basit backtest)
- Komisyon + slippage daha gercekci modelleme
- Farkli piyasa kosullarinda test (bull/bear/sideways)

---

## PLANLANAN GELISTIRME — PA (PRICE ACTION) IYILESTIRMELERI — 2026-03-12

### Kaynak
DD Finance Public Video Notları (PDF) incelendi — 2026-03-12

### Durum — 2026-03-13 TAMAMLANDI
`strategies/pa_range_strategy.py` — PA-1'den PA-10'a kadar tüm özellikler eklendi.

| # | Özellik | Commit |
|---|---------|--------|
| PA-1 | EQ Seviyesi + S/R Flip | 4f8220b |
| PA-2 | Deviasyon Tespiti | 4f8220b |
| PA-3 | Order Block | 4f8220b |
| PA-4 | Market Yapısı (CHoCH/BOS) | 9405435 |
| PA-5 | OTE Fibonacci 0.618-0.786 | 9c3dfe3 |
| PA-6 | Imbalance/GAP → TP hedefi | e9855a8 |
| PA-7 | Likidite Tabanlı TP | 410103d |
| PA-8 | Key Levels (Daily/Weekly Open) | — |
| PA-9 | Power of 3 (AMD) | — |
| PA-10 | SFP (Swing Failure Pattern) | — |

### DD Finance Notlarindan Cikan PA Gelistirme Listesi

#### ONCELIK 1 — En Etkili, Hemen Uygulanabilir

**1. EQ (Equilibrium) Seviyesi + S/R Flip Tespiti**
- Kaynak: "Price Action Range" + "S/R Flip" bolumu
- Range ortasi = EQ = (RL + RH) / 2
- 0.5-1.0 arasi: fiyatin pahali bolgesi → short tercih
- 0.0-0.5 arasi: fiyatin ucuz bolgesi → long tercih
- EQ noktasi S/R Flip gorevi gorur: eskiden destek olan bolgeden gecilince direnc, direncten gecilince destek
- Yapilacak: `generate_signal()` icerisine `price_position` hesabina gore confidence ayarlamasi
  - price_position > 0.5 → SAT konfirmasyon bonusu +0.10
  - price_position < 0.5 → AL konfirmasyon bonusu +0.10

**2. Deviasyon Tespiti (Range Disina Sapma + Geri Donus)**
- Kaynak: "Price Action Range" + "Power of 3 Box"
- Fiyat RL altina veya RH ustune cikarsa = deviasyon (sahte kirilim)
- Deviasyon sonrasi range icine geri girme = GUCLU AL/SAT sinyali
- Kural: Fiyat range disina cikti AMA kapanisin range icinde olmasi = deviasyon onaylandi
- Yapilacak: `_detect_deviation()` metodu
  - Son mum high > RH ama close <= RH → bearish deviasyon → SAT
  - Son mum low < RL ama close >= RL → bullish deviasyon → AL
  - Confidence +0.15 (deviasyon sonrasi re-test en guclu sinyal)

**3. Order Block (OB) Tespiti**
- Kaynak: "Price Action: OrderBlock"
- Tanim: 2 farkli renkli ardisik mum + fitil kurali
  - Bullish OB: kirmizi mum ardından buyuk yesil mum (onceki dususu kapatan)
  - Bearish OB: yesil mum ardından buyuk kirmizi mum
- Kural: Bullish OB fitilinin gunu, ayni yondeki mumun altinda olmamali
- RL bolgesinde bullish OB = AL konfirmasyon → confidence +0.12
- RH bolgesinde bearish OB = SAT konfirmasyon → confidence +0.12
- NOT: OB tek basina kullanilmaz, diger yapilarla birlikte
- Yapilacak: `_find_order_block()` metodu — son N mumdaki OB tespiti

**4. Market Yapisi (Swing High/Low + CHoCH/BOS)**
- Kaynak: "Market Yapisi" bolumu
- Swing High: her iki tarafinda daha dusuk mumlar olan tepe
- Swing Low: her iki tarafinda daha yuksek mumlar olan dip
- CHoCH (Change of Character): son tepe kirilindan once son dibin kaybedilmesi → trend degisimi
- BOS (Break of Structure): onceki swing High/Low kiriliyor → trend teyidi
- Kullanim: Market yapisi bullish → sadece AL tercih et, bearish → sadece SAT tercih et
- Oncelik sirasi: Breaker/MSB > BOS > CHoCH
- Yapilacak: `_detect_market_structure()` metodu → bullish/bearish/neutral dondur

#### ONCELIK 2 — Orta Zorlukta

**5. OTE (Optimal Trade Entry) — Fibonacci 0.618-0.705-0.786**
- Kaynak: "Fibonacci Kullanimi / Optimal Trade Entry"
- Swing Low ile Swing High arasina cekilen Fibonacci
- 0.618 - 0.705 - 0.786 araligi = optimal long bolgesi (AL icin)
- Short icin: Swing High - Swing Low arasi, ayni seviyeler
- 0.705 en cok kullanilan optimal giris noktasi
- Yapilacak: `_calc_ote_zone()` metodu → OTE bolgesi hesapla, fiyat bu bolgede mi kontrol et
- Confidence boost: fiyat OTE bolgesiyle RL/RH cakisiyorsa +0.15

**6. Imbalance / GAP Tespiti — TP Hedefi**
- Kaynak: "Price Action: Imbalance"
- Tanim: Hizli hareketle olusan bosluk — onceki mumun fityilinin gitmedi diger mumun kapanisina kadar
- Kural: Eger bir fitil onceki boslugu dolduruyorsa Imbalance sayilmaz
- Kullanim: TP hedefi olarak kullan → en yakin Imbalance bolgesi
- CME GAP: Hafta sonu bosluklar — dolurulma olasiligi yuksek
- Yapilacak: `_find_imbalance()` metodu → en yakin doldurulmamis bosluk → TP hedefi olarak don
- Mevcut TP formulu `price + range*0.5` yerine Imbalance hedefi kullan

**7. Likidite Tabanli TP Hedefi**
- Kaynak: "Price Action: Likidite"
- Likidite = swing noktalarin ustü/alti (stop avciligi bolgesi)
- Equal: Ayni seviyeye cok kez temas = likidite birkiyor → kirilim buyuk olur
- TP hedefi = bir onceki Swing High (bullish) veya Swing Low (bearish) likiditesi
- Kural: Likidite hedef noktasidir, GIRIS noktasi degil
- Yapilacak: TP hesabini `_find_liquidity_target()` ile guncelle

#### ONCELIK 3 — Karmasik / Uzun Vadeli

**8. Key Levels (Kurumsal Seviyeler)**
- Kaynak: "Anahtar Zaman Seviyeleri"
- Daily Open, Weekly Open, Monthly Open, Yearly Open, Monday High/Low
- Bu seviyelere yakin sinyal daha guclu
- Yapilacak: `data/fetcher.py` ile periyodik Open fiyatlarini cek, KEY_LEVELS listesi olustur
- Confidence boost: fiyat key level yakinindaysa +0.10

**9. Power of 3 (AMD) Yapisi**
- Kaynak: "Power of 3 Box"
- Accumulation (birikim, range) → Manipulation (deviasyon, stop avciligi) → Distribution (gercek hareket)
- Deviasyon = Power of 3 manipulasyon fazinda range disina sapma
- Geri donus + range icinde range EQ'yu gecme = distribution baslangici
- Yapilacak: Deviasyon + EQ kirimi kombosu → super sinyal (confidence 0.85+)

**10. Swing Failure Pattern (SFP)**
- Kaynak: "Swing Failure Pattern"
- Bearish SFP: Son tepenin ustune fitil atar, asagi kapanir → SAT
- Bullish SFP: Son dibin altina fitil atar, yukari kapanir → AL
- Kural: O bolgede kapanish olmamali (sadece fitil)
- Stop: Likiditey alan fityilin ustunde/altinda
- Yapilacak: `_detect_sfp()` metodu → son 5 mumdaki SFP tespiti

---

## PLANLANAN GELISTIRME — R:R (RISK:REWARD) HESAPLAMA SISTEMI — 2026-03-12

### Kaynak
RR hesaplama.png goruntusu incelendi — Bilesik buyume formulune dayali R:R sistemi

### Temel Matematik
```
Aylik buyume formulu   : r = hedef_pct / 100
Yillik bilesik getiri  : (1 + r)^12 - 1
Ornek: Aylik %10       : (1 + 0.10)^12 - 1 = 2.1384 = Yillik %213.84

Gerekli R:R formulu:
  Beklenen deger = (win_rate * avg_win) - (loss_rate * avg_loss) >= hedef_per_trade
  min_rr = (hedef_per_trade / risk_per_trade) / win_rate
```

### Yeni Modul: `risk/rr_calculator.py`

#### Sorumluluklar
1. Hedef getiriye gore gerekli minimum R:R hesapla
2. Her sinyale R:R skoru hesapla (SL/TP mesafesine gore)
3. Minimum R:R saglanmiyorsa islemi reddet (RiskManager entegrasyonu)
4. Kaldiraca gore lot boyutu onerisi
5. Bilesik buyume projeksiyonu (aylik/yillik tablo)

#### Ozellestirilebilir Parametreler
| Parametre | Aciklama | Varsayilan |
|---|---|---|
| `monthly_target_pct` | Hedef aylik buyume | 0.10 (%10) |
| `risk_per_trade_pct` | Her islemde riske edilecek sermaye | 0.02 (%2) |
| `min_rr_ratio` | Kabul edilecek minimum R:R | 1.5 |
| `win_rate_estimate` | Tahmini kazanma orani | 0.55 |
| `leverage` | Kaldirac katsayisi (Futures) | 1.0 (spot) |
| `max_lot_pct` | Maksimum lot (sermayenin yuzdesi) | 0.10 (%10) |

#### Temel Metotlar
```python
calc_required_rr(monthly_target, win_rate, risk_pct, trades_per_month)
  → Gerekli minimum R:R dondurur

calc_trade_rr(entry, stop_loss, take_profit)
  → Bu islemin R:R oranini dondurur

calc_lot_size(capital, risk_pct, entry, stop_loss, leverage)
  → Kaldiraca gore optimal lot buyuklugu dondurur

is_rr_acceptable(entry, stop_loss, take_profit) -> bool
  → R:R minimum esgigi sagliyor mu?

compound_projection(capital, monthly_pct, months)
  → Bilesik buyume tablosu (pandas DataFrame)

adjust_tp_for_rr(entry, stop_loss, min_rr) -> float
  → Minimum R:R'yi saglayacak TP fiyatini hesapla
```

#### RiskManager Entegrasyonu
- `evaluate_signal()` icerisine R:R kontrolu eklenir
- SL ve TP hesaplandiktan sonra R:R kontrolu yapilir:
  ```
  trade_rr = (take_profit - entry) / (entry - stop_loss)
  if trade_rr < min_rr_ratio:
      return reject("R:R yetersiz: {trade_rr:.2f} < {min_rr_ratio}")
  ```

#### Kaldiraca Gore Lot Hesabi
```
Spot (1x):
  lot = (capital * risk_pct) / (entry - stop_loss)

Futures (Nx):
  margin = capital * risk_pct  (riske edilen tutar)
  lot    = (margin * leverage) / (entry - stop_loss)
  max_lot = (capital * max_lot_pct * leverage) / entry
  lot    = min(lot, max_lot)
```

#### Bilesik Buyume Projeksiyonu (Ornek)
| Ay | %5/ay | %10/ay | %15/ay |
|---|---|---|---|
| 1 | $10,500 | $11,000 | $11,500 |
| 3 | $11,576 | $13,310 | $15,209 |
| 6 | $13,401 | $17,716 | $23,130 |
| 12 | $17,959 | $31,384 | $53,503 |

### Uygulama Sirasi
1. [ ] `risk/rr_calculator.py` — yeni modul olustur
2. [ ] `risk/risk_manager.py` — `evaluate_signal()` icerisine R:R kontrolu ekle
3. [ ] `risk/position_sizer.py` — kaldiraca gore lot hesabi guncelle
4. [ ] `trading/main_loop.py` — rr_calculator ile entegrasyon
5. [ ] Backtest: R:R filtresi acik/kapali karsilastirma

### Onemli Notlar
- R:R tek basina yeterli degil — market yapisi + sinyal kalitesi de onemli
- Minimum R:R 1.5:1 tavsiye (kazanc kayiptan 1.5 kat buyuk olmali)
- Kaldirach kullanim durumunda R:R hesabi margin bazinda yapilmali
- Bilesik buyume gercekci degil — her ay sabit getiri mumkun degil
  Gercekci hedef: Aylik %5-8 (yillik %80-150)

---

## PLANLANAN GELISTIRME — VWAP + FUNDING RATE + OPEN INTEREST — 2026-03-12

### Amac
Mevcut sistemi bosaltmadan (RSI/MACD/BB/ATR/ADX/Volume zaten var) 3 yuksek degerli
veri kaynagi eklenerek sinyal kalitesi arttirilacak.

### 1. VWAP (Volume Weighted Average Price)
- Tanim: Gunluk islem hacmiyle agirliklandirilmis ortalama fiyat
- Formul: VWAP = kumulatif(fiyat * hacim) / kumulatif(hacim)
- Kurumsal oyuncular destek/direnc olarak kullanir — en guclu intraday seviyesi
- Kullanim kurallari:
  - Fiyat > VWAP → bullish bias → AL sinyali guclu
  - Fiyat < VWAP → bearish bias → SAT sinyali guclu
  - Fiyat VWAP'a re-test → potansiyel giris noktasi
  - Confidence boost: sinyal yonu VWAP biasina uyuyorsa +0.08
- API/Hesaplama: Dis bagimlililik yok, OHLCV verisinden hesaplanir
- Eklenecek dosya: `strategies/indicators.py` icerisine `calc_vwap()` metodu
- Feature olarak ML modeline de eklenebilir: `vwap_distance` (fiyatin VWAP'tan uzakligi)

### 2. Funding Rate (Binance Futures API)
- Tanim: Futures piyasasinda long/short pozisyon dengesi gostergesi
- Her 8 saatte bir guncellenir (00:00, 08:00, 16:00 UTC)
- Yorum:
  - Yuksek pozitif (+0.1%+) → herkes long → asirimdi long → dusus riski yuksek
    - AL sinyali varsa confidence -0.10 (kontra sinyal)
    - SAT sinyali varsa confidence +0.10 (teyit)
  - Cok negatif (-0.05%-) → herkes short → asirimdi short → yukselis riski
    - SAT sinyali varsa confidence -0.10
    - AL sinyali varsa confidence +0.10
  - Noralde (+-0.01%) → tarafsiz, etki yok
- API: Binance Futures `/fapi/v1/fundingRate` endpoint, CCXT ile cekilir
- Onbellekleme: 8 saatlik (gereksiz API cagrisi engellenir)
- Eklenecek dosya: `ml/external_data.py` icerisine `get_funding_rate()` metodu
- Kaldirach sistemiyle entegrasyon: yuksek funding rate varken kaldirach azalt

### 3. Open Interest (OI)
- Tanim: Futures piyasasindaki toplam acik sozlesme sayisi/degeri
- Yorum (fiyat + OI kombinasyonu):
  | Fiyat | OI    | Anlam                              | Aksiyon         |
  |-------|-------|------------------------------------|-----------------|
  | Yukari | Yukari | Guclu trend, para giriyor          | Trendi teyit et |
  | Yukari | Asagi  | Zayif hareket, kapanis var         | Dikkatli ol     |
  | Asagi  | Yukari | Guclu dusus, yeni short aciliyor   | Trendi teyit et |
  | Asagi  | Asagi  | Panik kapanislar, dip yakin olabil | Dikkatli AL     |
- OI degisimi (delta) daha anlamli: ani OI artisi = buyuk pozisyon acildi
- API: Binance Futures `/fapi/v1/openInterest`, CCXT ile cekilir
- Onbellekleme: 1 saatlik
- Eklenecek dosya: `ml/external_data.py` icerisine `get_open_interest()` metodu
- Feature olarak ML modeline: `oi_change_pct` (1 saatlik OI degisim yuzdesi)

### Uygulama Sirasi
1. [ ] VWAP — `strategies/indicators.py` + `ml/feature_engineering.py`
2. [ ] Funding Rate — `ml/external_data.py` + `strategies/pa_range_strategy.py`
3. [ ] Open Interest — `ml/external_data.py` + `ml/feature_engineering.py`
4. [ ] Backtest: Bu 3 ozellik acik/kapali karsilastirma

### Onemli Notlar
- Funding Rate ve OI sadece Futures piyasasinda var — spot botumuz okuma amacli kullanir
- VWAP gunluk sifirlanan bir deger — 1h botumuz icin gunluk VWAP kullanilmali
- Bu 3 ozellik kaldirach sistemiyle birlikte gelistirilecek (funding rate ortak)
- Bunun otesinde (on-chain, order book, liquidation map) karmasi artar, fayda azalir

### Tespit Edilen Sorunlar ve Cozumler

#### 1. Basit Min/Max Yerine Swing High/Low Tespiti
- **Sorun:** `window.min()` / `window.max()` tek bir spike ile bozuluyor
- **Cozum:** Gercek destek = en az 2-3 kez test edilmis fiyat bolgesi
- **Yapilacak:** `_find_swing_levels()` metodu — her iki tarafinda daha dusuk/yuksek mum olan noktalari bul

#### 2. ATR Tabanli Proximity (Sabit %2 yerine)
- **Sorun:** Sabit `proximity_pct=0.02` volatiliteye gore cok genis veya dar kalab ilir
  - Range dar ise %2 = cok kucuk (hic sinyal yok)
  - Range genis ise %2 = cok buyuk (zayif bolgede de giris)
- **Cozum:** `proximity = 1.5 * ATR` — piyasa volatilitesine otomatik adapte olur
- **Yapilacak:** `PARangeStrategy.__init__` parametresi: `use_atr_proximity=True`

#### 3. Multi-Timeframe (MTF) Destek/Direnc
- **Sorun:** Sadece 1h seviyelerine bakiliyor; 4h seviyeleri cok daha guclu
- **Cozum:** 4h destek/direnci 1h sinyaliyle cakisiyorsa `confidence += 0.15`
- **Yapilacak:** `generate_signal()` icine 4h OHLCV alip ayri seviye hesabi

#### 4. Dinamik/Adaptif Lookback
- **Sorun:** Sabit `lookback=50` — trend piyasada 50 mumun min/max'i yanlis seviye verir
- **Cozum:** ADX'e gore lookback ayarla:
  - ADX < 20 (range):    lookback = 30
  - ADX 20-30 (karisik): lookback = 50
  - ADX > 30 (trend):    PA stratejisi devre disi (sadece trend takibi)
- **Yapilacak:** `_adaptive_lookback(adx)` yardimci metodu

#### 5. Birden Fazla Destek/Direnc Seviyesi (Cluster Analysis)
- **Sorun:** Tek bir destek/direnc seviyesi — piyasada genellikle birden fazla zone var
- **Cozum:** Fiyat yugunluk analizi: son 100 mumdaki kapanislar histogram ile kume analizi
  - En yogun bolgeler = gercek destek/direnc
- **Yapilacak:** `_find_price_clusters()` — numpy histogram ile top-3 seviye

### Uygulama Sirasi
1. [ ] ATR tabanli proximity (en kolay, en etkili)
2. [ ] Swing High/Low tespiti (min/max yerine)
3. [ ] Dinamik lookback (ADX entegrasyonu)
4. [ ] MTF destek/direnc (4h verisi gerekli)
5. [ ] Cluster analysis (en karmasik, en guclu)

### NOT: PA Stratejisi Detayli Inceleme
- Bir sonraki oturumda PA (Price Action Range) stratejisi detayli konusulacak
- Mevcut parametreler, iyilestirme alanlari, alternatif yaklasimlar tartisılacak

---

## PLANLANAN GELISTIRME — KALDIRAC (LEVERAGE) SISTEMI — 2026-03-12

### Hedef
Simdi spot trading yapiyoruz (Binance Testnet Spot). Gercek kaldiraci desteklemek icin
Binance Futures API entegrasyonu yapilacak.

### Mimari Degisiklikler

#### 1. Binance Futures API Baglantisi
- Yeni endpoint: `https://fapi.binance.com` (USD-M Futures)
- Testnet: `https://testnet.binancefuture.com`
- CCXT: `ccxt.binanceusdm()` veya `ccxt.binance({'options': {'defaultType': 'future'}})`
- `.env` dosyasina FUTURES_API_KEY / FUTURES_API_SECRET eklenmeli
- Mevcut spot key'den AYRI — Futures icin ayri API key gerekebilir

#### 2. OrderManager Guncellemesi (`trading/order_manager.py`)
- `open_long()` / `open_short()` metotlari Futures emri gonderecek
- Leverage ayari: `set_leverage(symbol, leverage)` cagrilmali (once emr acilmadan)
- Margin modu: `ISOLATED` (tavsiye) — her pozisyon kendi margini ile calisir
  - ISOLATED: Sadece o pozisyona ayirilan margin kaybolur (tum hesabi yakmaz)
  - CROSS: Tum hesap margini kullanir (tehlikeli)
- Position mode: ONE-WAY veya HEDGE — HEDGE mode ile LONG + SHORT ayni anda
- Emr tipleri: MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET

#### 3. PositionTracker Guncellemesi (`trading/position_tracker.py`)
- Futures PnL hesabi farkli: `pnl = (exit - entry) * qty * leverage` (LONG icin)
- Unrealized PnL: Binance API'dan cekerek gercek zamanli guncelleme
- Liquidation price takibi:
  - LONG liquidation: `entry_price * (1 - 1/leverage + maintenance_margin_rate)`
  - SHORT liquidation: `entry_price * (1 + 1/leverage - maintenance_margin_rate)`
- Funding rate: her 8 saatte bir pozisyon maliyetine eklenmeli

#### 4. RiskManager Guncellemesi (`risk/risk_manager.py`)
- `leverage` parametresi eklenmeli (1x-10x arasi tavsiye)
- Pozisyon boyutu hesabi degisir:
  - Spot: `qty = risk_budget / (price * stop_pct)`
  - Futures: `qty = (risk_budget * leverage) / (price * stop_pct)`
  - Ancak max_capital_pct kontrolu MARGIN bazinda yapilmali
- Liquidation risk kontrolu: SL, liquidation price'dan once olmali

#### 5. KillSwitch Guncellemesi (`risk/kill_switch.py`)
- Funding rate birikimi de gunluk zarar hesabina dahil edilmeli
- Liquidation yaklasiminda erken uyari (mesela %50 margin kullaniminda SARI alarm)

#### 6. Yeni: LeverageManager Modulu (`risk/leverage_manager.py`)
```
Sorumluluklar:
- Volatiliteye gore dinamik kaldirac onerisi
  ADX < 20 (range): max 5x
  ADX 20-30:        max 3x
  ADX > 30 (trend): max 2x (trend takibi)
- ATR tabanlı risk hesabi:
  safe_leverage = max_risk_per_trade / (atr_pct * stop_mult)
- Margin kullanim takibi (toplam hesap margini)
- Liquidation buffer: SL, liquidation'dan en az 2*ATR uzakta olmali
```

### Kaldırac Seviyeleri (Tavsiye)
| Piyasa Durumu | Max Kaldirac | Aciklama |
|---|---|---|
| Range (ADX<20) | 5x | Dar aralık, tahmin edilebilir |
| Karisik (ADX 20-30) | 3x | Orta risk |
| Trend (ADX>30) | 2x | Trend tersine donebilir |
| Yuksek volatilite | 1x (spot) | ATR anormal yuksekse kaldiracsiz |

### Uygulama Sirasi (Yapilacaklar)
1. [ ] Binance Futures Testnet hesabi ac + API key al
2. [ ] `trading/order_manager.py` — Futures emri desteği
3. [ ] `risk/leverage_manager.py` — yeni modul
4. [ ] `risk/position_sizer.py` — kaldiracli boyut hesabi
5. [ ] `trading/position_tracker.py` — Futures PnL + liquidation takibi
6. [ ] `risk/kill_switch.py` — funding rate + liquidation uyarisi
7. [ ] `trading/main_loop.py` — leverage parametresi geçişi
8. [ ] Testnet'te paper mode ile test (gercek para yok)
9. [ ] Backtest: kaldiraçli strateji performansi karsilastirma
10. [ ] Canli (gercek Futures) — EN SON

### Onemli Uyarilar
- Kaldiraç karı da zarar da buyutür: 5x kaldiraç = 20% fiyat düşüşü = %100 kayıp (likidite)
- Önce 2x ile baslayip test et
- Funding rate maliyeti (uzun sureli pozisyonlarda birikmektedir)
- Binance Futures ve Spot API key'leri farklı olabilir


---

## OTURUM 8 - 2026-04-06

### Phase 16 - Ensemble Model (XGBoost + LightGBM + RandomForest)

**Tamamlandi** | Commit: cb7d563

**Olusturulan Dosyalar:**
| Dosya | Aciklama |
|---|---|
| ml/ensemble_model.py | EnsembleModel sinifi - soft-voting, consensus modu, save/load |
| scripts/train_ensemble.py | Gercek veri ile ensemble egitim scripti |
| tests/test_phase16.py | 24 test - temel, CV, tahmin, kaydet/yukle, predictor entegrasyon |

**Teknik Detaylar:**
- Agirliklar: XGBoost=0.40, LightGBM=0.35, RandomForest=0.25
- Soft-voting: olasilik vektorleri agirlikli ortalama
- Consensus modu: 2/3 model ayni yonde olmali (varsayilan: ACIK)
- cv_report() ve model_votes_report() terminal raporlari
- MLPredictor.from_ensemble(dir) ile yukle
- lightgbm==4.6.0 kuruldu

**Test Sonucu:** 376/376 PASSED (24 yeni, 352 mevcut)

**Ensemble Egitimi icin:**
    python scripts/train_ensemble.py --days 365
    python scripts/train_ensemble.py --days 180 --no-consensus

**Sira:**
- [ ] Phase 15: Telegram bildirim sistemi
- [ ] Phase 20: Coklu coin (ETH, SOL)
- [ ] Phase 23: Kaldirackli trading (Binance Futures)

---

## OTURUM 8 (devam) - 2026-04-06

### Phase 20 - Coklu Coin (MultiCoinBot)

**Tamamlandi** | Commit: 0d8e31f

**Olusturulan Dosyalar:**
| Dosya | Aciklama |
|---|---|
| trading/multi_coin_bot.py | CoinConfig, CoinWorker, MultiCoinBot siniflar |
| tests/test_phase20.py | 20 test - config, init, state, portfoy, sinyal |

**Teknik Detaylar:**
- asyncio.gather ile tum coinler paralel tick atar
- Her coin icin bagimsiz: state dosyasi, risk manager, position tracker, ML model
- Portfoy sermaye dagitimi: ornek BTC:%50 ETH:%30 SOL:%20
- State dosyasi: data/bot_state_{SYMBOL}.json
- ML model oncelik sirasi: ensemble dir -> xgb_symbol -> xgb_btc (fallback)

**Calistirma:**
    python -m trading.multi_coin_bot                          # BTC+ETH
    python -m trading.multi_coin_bot --coins BTC/USDT ETH/USDT SOL/USDT
    python -m trading.multi_coin_bot --weights 0.5 0.3 0.2 --capital 10000
    python -m trading.multi_coin_bot --once   # bir tur calistir

**Test Sonucu:** 396/396 PASSED (20 yeni, 376 mevcut)

---

## OTURUM 8 (devam-2) - 2026-04-06

### Phase 23 - Kaldiracli Trading (LeverageManager)

**Tamamlandi** | Commit: 4698ef8

**Olusturulan/Guncellenen Dosyalar:**
| Dosya | Aciklama |
|---|---|
| risk/leverage_manager.py (YENI) | ADX bazli dinamik kaldirac, likidite buffer, funding rate |
| risk/position_sizer.py (GUNCELLENDI) | leveraged() metodu eklendi |
| trading/position_tracker.py (GUNCELLENDI) | leverage, liquidation_price, margin, is_near_liquidation |
| tests/test_phase23.py | 39 test |

**Teknik Detaylar:**
- ADX < 20 -> max 5x | ADX 20-30 -> max 3x | ADX > 30 -> max 2x | ATR>%3 -> 1x
- liquidation_price: LONG=entry*(1-1/lev+mmr), SHORT=entry*(1+1/lev-mmr)
- is_near_liquidation: liq'e %5 yaklasildiginda True (uyari log)
- funding_cost(): 8 saatlik periyot bazli maliyet hesabi
- margin_usage(): acik pozisyonlarin marjin kullanim orani
- PositionSizer.leveraged(): marjin*leverage/price

**NOT: Binance Futures API entegrasyonu (gercek emir) henuz yapilmadi**
- Mevcut: paper modda leverage hesaplamalari calisir
- Sonraki adim: Futures Testnet API key + ccxt binanceusdm()

**Test Sonucu:** 435/435 PASSED (39 yeni, 396 mevcut)


---

## OTURUM 9 - 2026-04-07

### Phase 15 - Telegram Bildirim Entegrasyonu

**Tamamlandi**

**Olusturulan/Guncellenen Dosyalar:**
| Dosya | Aciklama |
|---|---|
| trading/main_loop.py (GUNCELLENDI) | TelegramNotifier tam entegrasyon |
| tests/test_phase15.py (YENI) | 27 test |

**Entegrasyon Noktalari (main_loop.py):**
- __init__: TelegramNotifier(symbol) olusturuldu
- Bot baslatildiginda: send_bot_started(capital, paper)
- Sinyal olusup pozisyon acildiginda: send_signal() + send_position_opened()
- Pozisyon kapatildiginda: send_stop_loss() / send_take_profit() / send_position_closed()
- Ust uste 5 hata: send_error() + bot durduruluyor
- Her 24 tick (24 saat): send_daily_summary()
- Bot durduğunda: send_bot_stopped(total_pnl, win_rate)

**Notifier Davranisi:**
- .env'de TELEGRAM_TOKEN + TELEGRAM_CHAT_ID varsa gercek mesaj gonderir
- Yoksa otomatik dry_run=True -> sessizce loglar, bot etkilenmez

**KURULUM:**
    # .env dosyasina ekle:
    TELEGRAM_TOKEN=1234567890:ABC...
    TELEGRAM_CHAT_ID=-100123456789
    # @BotFather'dan bot olustur, @userinfobot ile chat_id bul

**Test Sonucu:** 462/462 PASSED (27 yeni, 435 mevcut)
