"""
data/cleaner.py
================
AMACI:
    Ham OHLCV verisini temizler ve doğrular.
    Borsadan gelen veri her zaman mükemmel değildir:
        - Eksik satırlar (NaN değerler)
        - Duplikat timestamp'ler (aynı mum iki kez)
        - Fiyat anomalileri (0 fiyat, high < low, vb.)
        - Sütun adı yanlışlıkları

NEDEN GEREKLİ?
    Strateji ve ML modelleri temiz veri bekler.
    Kirli veri → yanlış RSI hesabı → yanlış sinyal → para kaybı.

KULLANIM:
    from data.cleaner import OHLCVCleaner

    cleaner = OHLCVCleaner()
    df_clean = cleaner.clean(df_raw)
    report   = cleaner.validate(df_clean)
"""

import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Bir mumun fiyatı bu katsayıdan fazla sapıyorsa anomali sayılır
PRICE_CHANGE_THRESHOLD = 0.20   # %20 — tek mumda %20 fiyat değişimi şüpheli


class OHLCVCleaner:
    """
    OHLCV DataFrame temizleyici.

    Yaptığı işlemler (sırasıyla):
        1. Sütun adlarını normalleştirir (küçük harf)
        2. Gerekli sütunların varlığını kontrol eder
        3. Sayısal olmayan değerleri NaN yapar
        4. NaN içeren satırları kaldırır
        5. Duplikat timestamp'leri kaldırır
        6. Kronolojik sıraya sokar
        7. Fiyat mantığını doğrular (high >= low, close > 0 vb.)
        8. Ani fiyat sıçramalarını işaretler (opsiyonel kaldırma)
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, remove_anomalies: bool = False):
        """
        Args:
            remove_anomalies: True ise %20'den fazla tek-mum değişimlerini kaldırır.
                              Genellikle False bırakılır — anomaliyi loga yazıp geçeriz.
        """
        self.remove_anomalies = remove_anomalies

    # ── Ana Temizleme Metodu ──────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'i temizler ve temiz kopyasını döndürür.

        Args:
            df: Ham OHLCV DataFrame

        Returns:
            pd.DataFrame: Temizlenmiş DataFrame

        Raises:
            ValueError: Gerekli sütunlar eksikse
        """
        if df.empty:
            logger.warning("Bos DataFrame temizlemeye verildi")
            return df.copy()

        df = df.copy()   # Orijinali değiştirme
        original_len = len(df)

        # 1. Sütun adlarını normalleştir
        df = self._normalize_columns(df)

        # 2. Gerekli sütunları kontrol et
        self._check_required_columns(df)

        # 3. Sayısal tiplere zorla
        df = self._coerce_numeric(df)

        # 4. NaN satırları kaldır
        before_nan = len(df)
        df.dropna(subset=list(self.REQUIRED_COLUMNS), inplace=True)
        nan_removed = before_nan - len(df)
        if nan_removed > 0:
            logger.warning(f"NaN satirlar kaldirildi: {nan_removed}")

        # 5. Duplikat timestamp'leri kaldır (index üzerinden)
        before_dup = len(df)
        df = df[~df.index.duplicated(keep="last")]
        dup_removed = before_dup - len(df)
        if dup_removed > 0:
            logger.warning(f"Duplikat timestamp'ler kaldirildi: {dup_removed}")

        # 6. Kronolojik sıraya koy
        df.sort_index(inplace=True)

        # 7. Fiyat mantığı kontrolü (OHLC ilişkisi)
        df = self._fix_ohlc_logic(df)

        # 8. Anomali tespiti
        anomalies = self._detect_anomalies(df)
        if anomalies.any():
            count = anomalies.sum()
            logger.warning(
                f"Fiyat anomalisi tespit edildi: {count} mum "
                f"(tek mumda >%{PRICE_CHANGE_THRESHOLD*100:.0f} degisim)"
            )
            if self.remove_anomalies:
                df = df[~anomalies]
                logger.warning(f"Anomali mumlar kaldirildi: {count}")

        final_len = len(df)
        removed   = original_len - final_len
        logger.info(
            f"Temizleme tamamlandi | "
            f"Baslangic: {original_len} | Son: {final_len} | "
            f"Kaldirilan: {removed}"
        )

        return df

    # ── Doğrulama Raporu ─────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Temizlenmiş verinin kalitesini raporlar.

        Returns:
            dict: {
                "rows"         : satır sayısı,
                "date_start"   : ilk timestamp,
                "date_end"     : son timestamp,
                "nan_count"    : NaN sayısı,
                "duplicate_ts" : duplikat timestamp sayısı,
                "zero_volume"  : sıfır hacimli mum sayısı,
                "anomalies"    : fiyat anomalisi sayısı,
                "is_clean"     : True/False
            }
        """
        if df.empty:
            return {"rows": 0, "is_clean": False, "error": "Bos DataFrame"}

        nan_count    = df[list(self.REQUIRED_COLUMNS)].isna().sum().sum()
        dup_ts       = df.index.duplicated().sum()
        zero_vol     = (df["volume"] <= 0).sum()
        anomalies    = self._detect_anomalies(df).sum()

        is_clean = bool(nan_count == 0 and dup_ts == 0 and zero_vol == 0 and anomalies == 0)

        report = {
            "rows"        : len(df),
            "date_start"  : str(df.index[0]),
            "date_end"    : str(df.index[-1]),
            "nan_count"   : int(nan_count),
            "duplicate_ts": int(dup_ts),
            "zero_volume" : int(zero_vol),
            "anomalies"   : int(anomalies),
            "is_clean"    : is_clean,
        }

        if is_clean:
            logger.info(f"Veri dogrulamasi GECTI | {len(df)} satir | {report['date_start']} -> {report['date_end']}")
        else:
            logger.warning(f"Veri dogrulamasi SORUNLU | {report}")

        return report

    # ── Yardımcı Metodlar ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Sütun adlarını küçük harfe çevirir: 'Close' → 'close'"""
        df.columns = [c.lower().strip() for c in df.columns]
        return df

    def _check_required_columns(self, df: pd.DataFrame) -> None:
        """Gerekli sütunların varlığını kontrol eder, eksikse ValueError fırlatır."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Eksik sutunlar: {missing}. Mevcut: {set(df.columns)}")

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Fiyat/hacim sütunlarını float'a çevirir, hataları NaN yapar."""
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _fix_ohlc_logic(df: pd.DataFrame) -> pd.DataFrame:
        """
        Mantıksız OHLC değerlerini düzeltir veya kaldırır.
            - close <= 0 → kaldır
            - high < low → kaldır (imkansız durum)
            - high < close veya high < open → high'ı max(open,close) yap
            - low > close veya low > open  → low'ı min(open,close) yap
        """
        before = len(df)

        # Sıfır veya negatif fiyat — tamamen kaldır
        df = df[df["close"] > 0]
        df = df[df["open"]  > 0]
        df = df[df["high"]  > 0]
        df = df[df["low"]   > 0]

        # High < Low imkansız — kaldır
        invalid_hl = df["high"] < df["low"]
        if invalid_hl.any():
            logger.warning(f"high < low olan {invalid_hl.sum()} mum kaldirildi")
            df = df[~invalid_hl]

        # High'ı düzelt: en az open, close ve mevcut high'ın max'ı olmalı
        df["high"] = df[["open", "high", "close"]].max(axis=1)

        # Low'ı düzelt: en fazla open, low ve close'un min'i olmalı
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        removed = before - len(df)
        if removed > 0:
            logger.warning(f"OHLC mantik hatasi nedeniyle {removed} mum kaldirildi")

        return df

    @staticmethod
    def _detect_anomalies(df: pd.DataFrame) -> pd.Series:
        """
        Tek bir mumda close fiyatının önceki close'a göre
        PRICE_CHANGE_THRESHOLD'dan fazla değiştiği satırları işaretler.

        Returns:
            pd.Series[bool]: True = anomali
        """
        pct_change = df["close"].pct_change(fill_method=None).abs()
        return pct_change > PRICE_CHANGE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# TEST BLOGU
# python -m data.cleaner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ALGO TRADE CODEX — OHLCVCleaner Test")
    print("=" * 60)

    import numpy as np

    np.random.seed(42)
    closes = [50000.0]
    for _ in range(98):
        closes.append(closes[-1] * (1 + np.random.uniform(-0.005, 0.005)))

    df_raw = pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.002 for c in closes],
        "low":    [c * 0.998 for c in closes],
        "close":  closes,
        "volume": [np.random.uniform(100, 500) for _ in range(99)],
    }, index=pd.date_range("2024-01-01", periods=99, freq="1h", tz="UTC"))

    # Bilinçli hatalar ekle
    df_raw.iloc[10, df_raw.columns.get_loc("close")] = float("nan")   # NaN
    df_raw.iloc[20, df_raw.columns.get_loc("volume")] = 0              # Sıfır hacim
    df_raw = pd.concat([df_raw, df_raw.iloc[[30]]])                    # Duplikat

    print(f"\nHam veri: {len(df_raw)} satir (NaN, duplikat, sifir hacim eklendi)\n")

    cleaner  = OHLCVCleaner()
    df_clean = cleaner.clean(df_raw)
    report   = cleaner.validate(df_clean)

    print(f"\nTemiz veri: {len(df_clean)} satir")
    print(f"\nDogrulama raporu:")
    for k, v in report.items():
        print(f"  {k:15s}: {v}")

    print("\nBASARI: OHLCVCleaner calisiyor!")
