# Algo Trade Bot — Hizli Referans

## Bot Baslatma

```bash
# Ana bot (paper trading, saatte bir tick)
C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe -m trading.main_loop --interval 3600

# Test icin hizli (her 60 saniyede bir tick)
C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe -m trading.main_loop --interval 60
```

## Canli Izleme (ayri terminal)

```bash
# Bot calisirken log'u canli izle
C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe scripts/watch.py
```

## Ozet Rapor (bot duruyorken de calisir)

```bash
# Sermaye, kazan/kayip, islem gecmisi
C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe scripts/summary.py
```

## ML Modeli Yeniden Egitme

```bash
# 365 gunluk gercek veri ile XGBoost egit (~2-3 dk)
C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe -m ml.train --days 365 --refresh
```

## Backtest Calistirma

```bash
C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe -m backtesting.run_backtest
```

## Testler

```bash
powershell -Command "& 'C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/ -v"
```

---

## Dosyalar

| Dosya | Ne Yapar |
|-------|----------|
| `trading/main_loop.py` | Ana bot mantigI |
| `data/bot_state.json` | Bot durumu (sermaye, islemler) |
| `ml/models/xgb_btc_1h.json` | Egitilmis ML modeli |
| `logs/` | Bot log dosyalari |
| `.env` | API key'ler (dokunma) |

---

## Sistem Mimarisi (Kisaca)

```
Binance Testnet
    |
    v
Veri Cekme (CCXT, 1h mumlar)
    |
    v
Temizleme + Indiktorler (RSI, ATR, PA Range)
    |
    v
Rejim Tespiti (Yukari/Asagi/Yatay trend)
    |
    v
3 Strateji Oylama
  - RSI Stratejisi
  - PA Range Stratejisi
  - XGBoost ML (3-sinif: AL/SAT/BEKLE)
  Minimum 2/3 oy = sinyal olusur
    |
    v
Risk Manager
  - KillSwitch (Yellow %3 / Orange %5 / Red %15)
  - ATR bazli pozisyon boyutlama (max %2 risk)
    |
    v
Pozisyon Ac / Kapat
    |
    v
State Kaydet (data/bot_state.json)
```

---

## Fazlar

- [x] Phase 1-4: Strateji, veri, backtest altyapisi
- [x] Phase 5: ML entegrasyonu (XGBoost voting)
- [ ] Phase 6: Paper trading gozlem (20 saat+)
- [ ] Phase 7: Hyperparameter optimizasyonu
- [ ] Phase 8: Multi-Timeframe analiz (4h trend + 15m giris)
- [ ] Phase 9: Gorsel dashboard
- [ ] Phase 10: Live trading gecis
