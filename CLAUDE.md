# Claude Code - Proje Baslatma Talimatlari

## Her Konusma Basinda ZORUNLU Adimlar

Yeni bir konusma basladiginda su adimlari OTOMATIK olarak yap, kullanicidan onay bekleme:

1. `SESSION_LOG.md` dosyasini oku — son tamamlanan faz ve siradaki gorev nedir
2. `TODO.md` dosyasi varsa oku — hangi gorev yarım kaldi
3. Kullaniciya kisa ozet ver: "Kaldigi yer: [faz/gorev], Devam ediyorum..."
4. Direkt o noktadan gelistirmeye devam et

## Kullanici "devam et" yazarsa

SESSION_LOG.md'deki son PHASE 12X gorevini bul ve kodlamaya direkt basla.
Onay sorma, izin isteme.

## Proje Kurallari (OZET)

- Python: `C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe`
- Test komutu: `powershell -Command "& 'C:\Users\rk209\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/ -v"`
- Her fazdaki testler gecmeli, sonra git commit + push
- Log mesajlarinda `->` kullan, `->` degil (Windows CP1254)
- `SESSION_LOG.md` ve `TODO.md` her faz sonunda guncelle
- HIC SORU SORMA, direkt yap
