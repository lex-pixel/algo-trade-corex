"""
risk/position_sizer.py
=======================
AMACI:
    Her islem icin ne kadar BTC alacagimizi hesaplar.
    Iki yontem desteklenir — hangisi daha muhafazakar cikiyorsa o kullanilir.

YONTEMLER:
    1. FIXED FRACTION (Sabit Kesir)
       En basit yontem. Sermayenin belirli bir yuzdesi her islemde riske atilir.
       risk_amount = capital * risk_pct
       quantity    = risk_amount / (stop_distance * price)

    2. ATR TABANLI
       Stop mesafesi ATR'ye gore otomatik hesaplanir.
       Volatil piyasada kucuk pozisyon, sakin piyasada buyuk pozisyon.
       stop_distance = atr_multiplier * ATR / price

    3. KELLY KRITERi (Tam Kelly cok riskli — Yari Kelly kullanilir)
       Uzun vadede serveyi maksimize eden matematiksel formul.
       f* = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
       Pratikte yari Kelly (f*/2) kullanilir — daha muhafazakar.

GUVENLIK LIMITLERI:
    min_qty    : Minimum lot buyuklugu (0.0001 BTC)
    max_pct    : Sermayenin max yuzdesi (varsayilan %10)
    max_qty    : Tek islemde max BTC miktari (varsayilan 0.1 BTC)

KULLANIM:
    from risk.position_sizer import PositionSizer

    sizer = PositionSizer(max_risk_pct=0.02)

    qty = sizer.fixed_fraction(
        capital=10_000, price=95000, stop_pct=0.02
    )

    qty = sizer.atr_based(
        capital=10_000, price=95000, atr=500.0
    )

    qty = sizer.kelly(
        capital=10_000, price=95000,
        win_rate=0.55, avg_win_pct=0.03, avg_loss_pct=0.02
    )
"""

from __future__ import annotations
import math
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────
MIN_QTY          = 0.0001    # BTC minimum lot
DEFAULT_MAX_PCT  = 0.10      # Sermayenin maks %10'u tek islemde
DEFAULT_MAX_QTY  = 0.10      # Max 0.1 BTC tek islemde (testnet uyumlu)
DEFAULT_RISK_PCT = 0.02      # Varsayilan risk: sermayenin %2'si


class PositionSizer:
    """
    Dinamik pozisyon boyutu hesaplayici.

    Parametreler:
        max_risk_pct  : Tek islemde maks risk yuzdesi (0.02 = %2)
        max_capital_pct: Sermayenin maks yuzdesi (0.10 = %10)
        max_qty       : Tek islemde maks BTC miktari
        kelly_fraction: Kelly kriterinin kac kati uygulansin (0.5 = Yari Kelly)
    """

    def __init__(
        self,
        max_risk_pct: float    = DEFAULT_RISK_PCT,
        max_capital_pct: float = DEFAULT_MAX_PCT,
        max_qty: float         = DEFAULT_MAX_QTY,
        kelly_fraction: float  = 0.5,
    ):
        self.max_risk_pct    = max_risk_pct
        self.max_capital_pct = max_capital_pct
        self.max_qty         = max_qty
        self.kelly_fraction  = kelly_fraction

        logger.info(
            f"PositionSizer baslatildi | "
            f"max_risk=%{max_risk_pct*100:.1f} | "
            f"max_capital=%{max_capital_pct*100:.0f} | "
            f"max_qty={max_qty} BTC"
        )

    # ── Yontem 1: Sabit Kesir ─────────────────────────────────────────────────

    def fixed_fraction(
        self,
        capital: float,
        price: float,
        stop_pct: float | None = None,
        stop_price: float | None = None,
    ) -> float:
        """
        Sabit kesir yontemi: Sermayenin belirli bir yuzdesi riske atilir.

        Parametreler:
            capital    : Mevcut sermaye (USDT)
            price      : Giris fiyati (USDT)
            stop_pct   : Stop mesafesi yuzde olarak (orn. 0.02 = %2)
            stop_price : Veya gercek stop fiyati (stop_pct yerine)

        Returns:
            float: BTC miktari (guvenlik limitleri uygulanmis)
        """
        if price <= 0 or capital <= 0:
            return 0.0

        risk_amount = capital * self.max_risk_pct

        # Stop mesafesini hesapla
        if stop_price is not None:
            stop_distance = abs(price - stop_price) / price
        elif stop_pct is not None:
            stop_distance = abs(stop_pct)
        else:
            stop_distance = self.max_risk_pct  # Fallback: risk_pct kadar stop

        if stop_distance <= 0:
            logger.warning("Stop mesafesi sifir, MIN_QTY donuluyor.")
            return MIN_QTY

        # Pozisyon buyuklugu = risk_miktari / (stop_mesafesi * fiyat)
        qty = risk_amount / (stop_distance * price)

        qty = self._apply_limits(qty, capital, price)
        logger.debug(
            f"FixedFraction | capital=${capital:,.0f} | price=${price:,.0f} | "
            f"stop={stop_distance*100:.1f}% | qty={qty:.6f} BTC"
        )
        return qty

    # ── Yontem 2: ATR Tabanli ─────────────────────────────────────────────────

    def atr_based(
        self,
        capital: float,
        price: float,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> float:
        """
        ATR tabanli pozisyon boyutu.
        Stop mesafesi = atr_multiplier * ATR

        Volatil piyasada daha kucuk pozisyon acar (ayni risk icin).

        Parametreler:
            capital        : Mevcut sermaye (USDT)
            price          : Giris fiyati (USDT)
            atr            : Average True Range degeri (USDT)
            atr_multiplier : Stop mesafesi = multiplier * ATR (varsayilan 2.0)

        Returns:
            float: BTC miktari
        """
        if atr <= 0 or price <= 0 or capital <= 0:
            return 0.0

        risk_amount    = capital * self.max_risk_pct
        stop_distance  = atr_multiplier * atr   # USDT cinsinden stop mesafesi

        if stop_distance <= 0:
            return 0.0

        qty = risk_amount / stop_distance
        qty = self._apply_limits(qty, capital, price)

        logger.debug(
            f"ATR-based | capital=${capital:,.0f} | price=${price:,.0f} | "
            f"ATR={atr:.1f} | stop=${stop_distance:.1f} | qty={qty:.6f} BTC"
        )
        return qty

    # ── Yontem 3: Kelly Kriteri ───────────────────────────────────────────────

    def kelly(
        self,
        capital: float,
        price: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> float:
        """
        Kelly kriteri ile optimal pozisyon boyutu.
        Yari Kelly kullanilir (kelly_fraction=0.5) — tam Kelly cok riskli.

        Formul:
            f* = (p * b - q) / b
            p  = kazanma olasiligi (win_rate)
            q  = kayip olasiligi (1 - win_rate)
            b  = ortalama kazanc / ortalama kayip (oran)

        Parametreler:
            capital     : Mevcut sermaye (USDT)
            price       : Giris fiyati (USDT)
            win_rate    : Kazanma olasiligi (0-1)
            avg_win_pct : Ortalama kazanc yuzdesi (0.03 = %3)
            avg_loss_pct: Ortalama kayip yuzdesi (0.02 = %2)

        Returns:
            float: BTC miktari
        """
        if price <= 0 or capital <= 0:
            return 0.0

        if avg_loss_pct <= 0:
            logger.warning("Kelly: avg_loss_pct sifir veya negatif, MIN_QTY donuluyor.")
            return MIN_QTY

        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p
        b = avg_win_pct / avg_loss_pct   # kazanc/kayip orani

        # Kelly formulu
        kelly_f = (p * b - q) / b if b > 0 else 0.0

        # Negatif Kelly = pozisyon acma
        if kelly_f <= 0:
            logger.info(f"Kelly negatif ({kelly_f:.3f}), islem onerilmiyor.")
            return 0.0

        # Yari Kelly uygula
        kelly_f *= self.kelly_fraction

        # Sermayenin Kelly fraksiyonu kadarini riske at
        risk_amount = capital * kelly_f
        qty         = risk_amount / price

        qty = self._apply_limits(qty, capital, price)

        logger.debug(
            f"Kelly | win_rate={p:.1%} | b={b:.2f} | "
            f"f*={kelly_f:.3f} | qty={qty:.6f} BTC"
        )
        return qty

    # ── Kombinasyon: En Muhafazakar ───────────────────────────────────────────

    def conservative(
        self,
        capital: float,
        price: float,
        atr: float | None       = None,
        stop_pct: float | None  = None,
        win_rate: float | None  = None,
        avg_win_pct: float      = 0.03,
        avg_loss_pct: float     = 0.02,
    ) -> float:
        """
        Kullanilabilir yontemler arasinda EN KUCUK miktari dondurur.
        Muhafazakar yaklasim: birden fazla hesap varsa en kucugunu sec.

        Returns:
            float: BTC miktari (en muhafazakar)
        """
        quantities = []

        # Fixed fraction (her zaman hesaplanir)
        ff_qty = self.fixed_fraction(
            capital, price, stop_pct=stop_pct or self.max_risk_pct
        )
        quantities.append(("FixedFraction", ff_qty))

        # ATR tabanli (ATR varsa)
        if atr and atr > 0:
            atr_qty = self.atr_based(capital, price, atr)
            quantities.append(("ATR", atr_qty))

        # Kelly (win_rate varsa)
        if win_rate is not None and win_rate > 0:
            kelly_qty = self.kelly(
                capital, price, win_rate, avg_win_pct, avg_loss_pct
            )
            if kelly_qty > 0:
                quantities.append(("Kelly", kelly_qty))

        # En kucugunu sec
        best_name, best_qty = min(quantities, key=lambda x: x[1])
        logger.debug(
            f"Conservative | secilen={best_name} | qty={best_qty:.6f} BTC | "
            f"hesaplar={[(n, round(q,6)) for n,q in quantities]}"
        )
        return best_qty

    # ── Guvenlik Limitleri ────────────────────────────────────────────────────

    def _apply_limits(
        self, qty: float, capital: float, price: float
    ) -> float:
        """
        Guvenlik limitlerini uygular:
        1. Negatif veya NaN -> MIN_QTY
        2. Max sermaye yuzdesi asimini engelle
        3. Max BTC miktarini asimini engelle
        4. Minimum lot buyuklugu

        Returns:
            float: Limitler uygulanmis miktar, 6 ondalik hassasiyet
        """
        if qty is None or not math.isfinite(qty) or qty <= 0:
            return MIN_QTY

        # Max sermaye limiti
        max_by_capital = (capital * self.max_capital_pct) / price
        qty = min(qty, max_by_capital)

        # Max BTC limiti
        qty = min(qty, self.max_qty)

        # Minimum lot
        qty = max(qty, MIN_QTY)

        return round(qty, 6)

    # ── Bilgi Fonksiyonlari ────────────────────────────────────────────────────

    def risk_summary(
        self,
        capital: float,
        price: float,
        qty: float,
        stop_pct: float = 0.02,
    ) -> dict:
        """
        Bir islemin risk ozeti.

        Returns:
            dict: notional, max_loss, risk_pct, risk_to_reward (1:?)
        """
        notional  = qty * price
        max_loss  = notional * stop_pct
        risk_pct  = max_loss / capital * 100 if capital > 0 else 0.0

        return {
            "quantity"      : qty,
            "price"         : price,
            "notional_usdt" : round(notional, 2),
            "max_loss_usdt" : round(max_loss, 2),
            "risk_pct"      : round(risk_pct, 3),
            "capital_used_pct": round(notional / capital * 100, 2) if capital > 0 else 0.0,
        }
