"""
risk/leverage_manager.py
=========================
AMACI:
    Volatilite ve piyasa rejimi bazli dinamik kaldirac yonetimi.

    Kaldiraci ne zaman artirip ne zaman azaltacagimizi belirler:
        - ADX < 20  (range piyasa)  : max 5x  — tahmin edilebilir, sakin
        - ADX 20-30 (karisik)       : max 3x
        - ADX > 30  (trend piyasa)  : max 2x  — trend tersine donebilir
        - ATR anormal yuksekse      : 1x (spot, kaldiracsiz)

    Guvenlik kontrolleri:
        - SL, likidite fiyatindan en az 2*ATR uzakta olmali
        - Funding rate birikimi marjin maliyetini arttirir
        - Max margin kullanim orani: %80 (boyle olursa kill-switch sari alarm)

KULLANIM:
    from risk.leverage_manager import LeverageManager

    lm = LeverageManager()
    leverage = lm.suggest_leverage(adx=25.0, atr_pct=0.015)

    ok, msg = lm.check_liquidation_buffer(
        direction="LONG", entry=70000, sl=68000,
        leverage=5, atr=800
    )

    cost = lm.funding_cost(notional=5000, leverage=3, hours=8)
"""

from __future__ import annotations
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)

# Varsayilan parametreler
DEFAULT_MAX_LEVERAGE    = 5       # Hic bir kosulda bu degeri asmayiz
MAINTENANCE_MARGIN_RATE = 0.004   # Binance BTC: %0.4 (yaklaşık)
FUNDING_RATE_8H         = 0.0001  # Varsayilan 8 saatlik funding rate: %0.01


@dataclass
class LeverageDecision:
    """suggest_leverage() cikti nesnesi."""
    leverage:     int    # Onerilen kaldirac (1-10x)
    max_leverage: int    # Bu rejimde izin verilen maksimum
    reason:       str    # Neden bu karar alindi
    adx:          float  # Giren ADX degeri
    atr_pct:      float  # Giren ATR yuzde degeri
    regime:       str    # "RANGE" | "MIXED" | "TREND" | "HIGH_VOL"


class LeverageManager:
    """
    Dinamik kaldirac yoneticisi.

    Parametreler:
        max_leverage        : Sistemin mutlak maksimum kaldirac limiti
        atr_high_threshold  : Bu ATR%'nin uzerinde kaldirac dusurulur (spot'a)
        min_buffer_atr_mult : Likidite fiyatindan SL mesafesi (ATR carpani)
        funding_rate_8h     : 8 saatlik funding rate (varsayilan %0.01)
    """

    def __init__(
        self,
        max_leverage:        int   = DEFAULT_MAX_LEVERAGE,
        atr_high_threshold:  float = 0.03,    # ATR/fiyat > %3 -> kaldiracsiz
        min_buffer_atr_mult: float = 2.0,     # SL, liq'den en az 2x ATR uzakta
        funding_rate_8h:     float = FUNDING_RATE_8H,
    ):
        self.max_leverage        = max_leverage
        self.atr_high_threshold  = atr_high_threshold
        self.min_buffer_atr_mult = min_buffer_atr_mult
        self.funding_rate_8h     = funding_rate_8h

    # ── Kaldirac Onerisi ──────────────────────────────────────────────────────

    def suggest_leverage(
        self,
        adx:     float,
        atr_pct: float,   # ATR / fiyat (ornegin 0.012 = %1.2)
    ) -> LeverageDecision:
        """
        ADX ve ATR yuzdesine gore uygun kaldiraci oneriri.

        Args:
            adx     : ADX gostergesi degeri (0-100)
            atr_pct : ATR / current_price (ornegin 800/70000 = 0.0114)

        Returns:
            LeverageDecision: onerilen kaldirac ve gerekcesi
        """
        # Yuksek volatilite kontrolu — once kontrol et
        if atr_pct > self.atr_high_threshold:
            leverage = 1
            regime   = "HIGH_VOL"
            reason   = (
                f"ATR/fiyat={atr_pct:.2%} > esik={self.atr_high_threshold:.2%} | "
                f"Cok yuksek volatilite, kaldirac kullanma"
            )
            return LeverageDecision(
                leverage=leverage, max_leverage=1,
                reason=reason, adx=adx, atr_pct=atr_pct, regime=regime,
            )

        # ADX bazli maksimum kaldirac
        if adx < 20:
            max_lev = min(5, self.max_leverage)
            regime  = "RANGE"
        elif adx < 30:
            max_lev = min(3, self.max_leverage)
            regime  = "MIXED"
        else:
            max_lev = min(2, self.max_leverage)
            regime  = "TREND"

        # ATR'ye gore ek guvenlik: yuksek volatilite -> kaldirac azalt
        atr_factor = 1.0
        if atr_pct > 0.02:      # ATR/fiyat > %2
            atr_factor = 0.5
        elif atr_pct > 0.015:   # ATR/fiyat > %1.5
            atr_factor = 0.75

        leverage = max(1, round(max_lev * atr_factor))

        reason = (
            f"ADX={adx:.1f} ({regime}) | "
            f"ATR/fiyat={atr_pct:.2%} | "
            f"max_lev={max_lev}x | "
            f"atr_factor={atr_factor:.2f} | "
            f"onerilen={leverage}x"
        )

        logger.debug(f"Kaldirac onerisi: {reason}")

        return LeverageDecision(
            leverage=leverage, max_leverage=max_lev,
            reason=reason, adx=adx, atr_pct=atr_pct, regime=regime,
        )

    # ── Likidite Buffer Kontrolu ───────────────────────────────────────────────

    def check_liquidation_buffer(
        self,
        direction: str,    # "LONG" | "SHORT"
        entry:     float,  # Giris fiyati
        sl:        float,  # Stop-loss fiyati
        leverage:  int,    # Kullanilan kaldirac
        atr:       float,  # ATR degeri (USDT)
        maintenance_margin_rate: float = MAINTENANCE_MARGIN_RATE,
    ) -> tuple[bool, str]:
        """
        SL'in likidite fiyatindan yeterince uzak olup olmadigini kontrol eder.

        Likidite fiyati formulleri:
            LONG:  liq = entry * (1 - 1/leverage + mmr)
            SHORT: liq = entry * (1 + 1/leverage - mmr)

        Kural: |SL - liq| >= min_buffer_atr_mult * ATR

        Returns:
            (ok: bool, mesaj: str)
        """
        if leverage <= 1:
            return True, "Kaldirac yok (spot), likidite riski yok"

        # Likidite fiyatini hesapla
        if direction == "LONG":
            liq_price = entry * (1 - 1.0 / leverage + maintenance_margin_rate)
            sl_to_liq = sl - liq_price   # Pozitif olmali: SL liq'den yukarda olmali
        else:  # SHORT
            liq_price = entry * (1 + 1.0 / leverage - maintenance_margin_rate)
            sl_to_liq = liq_price - sl   # Pozitif olmali: SL liq'den asagida olmali

        buffer_needed = self.min_buffer_atr_mult * atr
        ok = sl_to_liq >= buffer_needed

        msg = (
            f"{direction} | Giris:{entry:,.2f} | SL:{sl:,.2f} | "
            f"Liq:{liq_price:,.2f} | "
            f"SL-Liq mesafe:{sl_to_liq:,.2f} | "
            f"Gereken buffer:{buffer_needed:,.2f} ({self.min_buffer_atr_mult}x ATR) | "
            f"{'OK' if ok else 'YETERSIZ - SL liq cok yakin!'}"
        )

        if not ok:
            logger.warning(f"Likidite buffer yetersiz: {msg}")
        else:
            logger.debug(f"Likidite buffer OK: {msg}")

        return ok, msg

    # ── Kaldiracli Pozisyon Boyutu ─────────────────────────────────────────────

    def leveraged_position_size(
        self,
        capital:      float,   # Toplam sermaye (USDT)
        margin_pct:   float,   # Bu islem icin kullanilacak marjin orani (0.0-1.0)
        leverage:     int,     # Kullanilan kaldirac
        price:        float,   # Giris fiyati
        max_margin_pct: float = 0.10,  # Toplam sermayenin max %10'u marjin
    ) -> float:
        """
        Kaldiracli pozisyon icin miktar hesaplar.

        Formul:
            marjin_miktari = capital * margin_pct
            notional       = marjin_miktari * leverage
            quantity       = notional / price

        Args:
            capital      : Toplam sermaye
            margin_pct   : Bu islem icin ayrilan marjin orani
            leverage     : Kaldirac katsayisi
            price        : Coin fiyati
            max_margin_pct: Maksimum izin verilen marjin orani

        Returns:
            Miktar (coin cinsiyle)
        """
        # Guvenli marjin orani
        safe_margin_pct = min(margin_pct, max_margin_pct)
        margin_amount   = capital * safe_margin_pct
        notional        = margin_amount * leverage
        quantity        = notional / price

        logger.debug(
            f"Kaldiracli boyut | "
            f"marjin:{margin_amount:.2f} USDT | "
            f"leverage:{leverage}x | "
            f"notional:{notional:.2f} | "
            f"qty:{quantity:.6f}"
        )

        return round(quantity, 6)

    # ── Funding Rate Maliyet Hesabi ───────────────────────────────────────────

    def funding_cost(
        self,
        notional:     float,  # Pozisyon USDT degeri (marjin * leverage)
        leverage:     int,
        hours:        float,  # Kac saattir acik pozisyon
        funding_rate: float | None = None,  # None = varsayilan kullan
    ) -> float:
        """
        Futures pozisyon icin birikmis funding rate maliyet hesaplar.
        Her 8 saatte bir odenir.

        Returns:
            Toplam funding maliyet (USDT)
        """
        rate     = funding_rate if funding_rate is not None else self.funding_rate_8h
        periods  = hours / 8.0  # 8 saatlik periyot sayisi
        cost     = notional * rate * periods

        logger.debug(
            f"Funding maliyeti | notional:{notional:.2f} | "
            f"leverage:{leverage}x | {hours:.1f}saat | "
            f"rate:{rate:.4%}/8h | maliyet:{cost:.4f} USDT"
        )

        return round(cost, 4)

    # ── Likidite Fiyati Hesabi ────────────────────────────────────────────────

    @staticmethod
    def liquidation_price(
        direction:  str,
        entry:      float,
        leverage:   int,
        maintenance_margin_rate: float = MAINTENANCE_MARGIN_RATE,
    ) -> float:
        """
        Likidite fiyatini hesaplar.

        LONG:  liq = entry * (1 - 1/leverage + mmr)
        SHORT: liq = entry * (1 + 1/leverage - mmr)
        """
        if leverage <= 1:
            return 0.0 if direction == "LONG" else float("inf")

        if direction == "LONG":
            return entry * (1.0 - 1.0 / leverage + maintenance_margin_rate)
        else:
            return entry * (1.0 + 1.0 / leverage - maintenance_margin_rate)

    # ── Marjin Kullanim Orani ─────────────────────────────────────────────────

    @staticmethod
    def margin_usage(
        open_notionals: list[float],
        leverage:       int,
        capital:        float,
    ) -> float:
        """
        Hesabin kac yuzdesi marjin olarak kullaniliyor?

        Returns:
            Oran (0.0 - 1.0): 0.8 = %80 kullanim
        """
        total_margin = sum(n / leverage for n in open_notionals)
        return total_margin / capital if capital > 0 else 0.0

    # ── Ozet Rapor ────────────────────────────────────────────────────────────

    def regime_table(self) -> None:
        """ADX-kaldirac tablosunu terminale yazdirir."""
        print("\n" + "=" * 55)
        print("  KALDIRAC REJiM TABLOSU")
        print("=" * 55)
        print(f"  {'Piyasa Durumu':<20} {'ADX':<12} {'Max Kaldirac':>12}")
        print("-" * 55)
        print(f"  {'Range':<20} {'< 20':<12} {'5x':>12}")
        print(f"  {'Karisik':<20} {'20-30':<12} {'3x':>12}")
        print(f"  {'Trend':<20} {'> 30':<12} {'2x':>12}")
        print(f"  {'Yuksek Volatilite':<20} {'ATR>%3':<12} {'1x (spot)':>12}")
        print("=" * 55)
        print(f"  Sistem max kaldirac limiti: {self.max_leverage}x")
        print(f"  Likidite buffer: {self.min_buffer_atr_mult}x ATR")
        print(f"  Funding rate (8h): {self.funding_rate_8h:.4%}")
        print("=" * 55)
