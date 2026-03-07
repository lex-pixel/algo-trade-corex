"""
trading/order_manager.py
==========================
AMACI:
    Binance Testnet uzerinde emir gondermek, takip etmek ve iptal etmek.
    Paper trading modunda gercek API cagrisi yapilmaz — sahte emir kaydedilir.

MODLAR:
    paper=True  : Sahte emir (API'ye dokunmaz, lokal kayit tutar)
    paper=False : Gercek emir (Binance Testnet veya Live API)

DESTEKLENEN EMIR TIPLERI:
    MARKET : Anlik piyasa fiyatindan al/sat
    LIMIT  : Belirli fiyattan al/sat (dolmayabilir)

KULLANIM:
    from trading.order_manager import OrderManager

    om = OrderManager(exchange=exchange, paper=True)

    # Emir gonder
    order = om.place_market_order("BTC/USDT", "buy", quantity=0.001)

    # Acik emirleri goruntule
    om.list_open_orders()

    # Emir iptal et
    om.cancel_order(order_id)

    # Ozet
    om.print_summary()
"""

from __future__ import annotations
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Sabitler ──────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"


class OrderStatus(str, Enum):
    OPEN      = "open"       # Beklemede
    FILLED    = "filled"     # Doldu
    CANCELLED = "cancelled"  # Iptal edildi
    REJECTED  = "rejected"   # Reddedildi


# ── Veri Sinifi ───────────────────────────────────────────────────────────────

@dataclass
class Order:
    """
    Bir emiri temsil eder.

    Alanlar:
        order_id    : Benzersiz kimlik (Binance ID veya paper UUID)
        symbol      : Islem cifti (orn. BTC/USDT)
        side        : buy veya sell
        type        : market veya limit
        quantity    : Miktar (BTC cinsinden)
        price       : Limit fiyat (market emirlerde None)
        filled_price: Doldurulan ortalama fiyat
        status      : Emirin guncel durumu
        created_at  : Emir olusturulma zamani
        filled_at   : Dolma zamani (yoksa None)
        paper       : True = sahte, False = gercek emir
    """
    order_id    : str
    symbol      : str
    side        : str           # "buy" | "sell"
    type        : str           # "market" | "limit"
    quantity    : float
    price       : Optional[float]        # limit icin
    filled_price: Optional[float] = None # doldurulan fiyat
    status      : str = OrderStatus.OPEN
    created_at  : datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at   : Optional[datetime] = None
    paper       : bool = True
    exchange_id : Optional[str] = None  # Binance'teki gercek ID
    fee         : float = 0.0           # Komisyon (USDT)
    pnl         : float = 0.0           # Bu emirden kar/zarar

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_open(self) -> bool:
        return self.status == OrderStatus.OPEN

    @property
    def notional(self) -> float:
        """Emirin toplam USDT degeri."""
        p = self.filled_price or self.price or 0.0
        return self.quantity * p

    def __str__(self) -> str:
        price_str = f"{self.filled_price:.2f}" if self.filled_price else "bekliyor"
        return (
            f"Order({self.order_id[:8]}... | "
            f"{self.side.upper()} {self.quantity:.6f} {self.symbol} | "
            f"fiyat={price_str} | {self.status}"
            f"{' [PAPER]' if self.paper else ''}"
        )


# ── OrderManager ──────────────────────────────────────────────────────────────

class OrderManager:
    """
    Emir yonetim sistemi.

    Paper modda hic API cagrisi yapilmaz:
    - Emir aninda 'filled' sayilir
    - Son fiyat + slipaj uygulanir
    - Komisyon dusulur

    Parametreler:
        exchange      : CCXT exchange nesnesi (paper=False gerektirir)
        symbol        : Varsayilan islem cifti
        paper         : True = paper trading (API'ye dokunma)
        commission    : Komisyon orani (0.001 = %0.1)
        slippage      : Slipaj orani (0.0005 = %0.05)
    """

    def __init__(
        self,
        exchange=None,
        symbol: str  = "BTC/USDT",
        paper: bool  = True,
        commission: float  = 0.001,
        slippage: float    = 0.0005,
    ):
        self.exchange   = exchange
        self.symbol     = symbol
        self.paper      = paper
        self.commission = commission
        self.slippage   = slippage

        # Emir kayitlari
        self._orders: dict[str, Order] = {}   # order_id -> Order

        # Istatistikler
        self._total_filled  = 0
        self._total_cancelled = 0

        mode = "PAPER" if paper else "LIVE"
        logger.info(f"OrderManager baslatildi | Mod: {mode} | {symbol}")

    # ── Emir Gonder ───────────────────────────────────────────────────────────

    def place_market_order(
        self,
        side: str,
        quantity: float,
        current_price: float,
        symbol: str | None = None,
    ) -> Order:
        """
        Market emri gonderir (anlik fiyattan).

        Parametreler:
            side          : "buy" veya "sell"
            quantity      : Miktar (BTC cinsinden)
            current_price : Su anki piyasa fiyati (slipaj icin)
            symbol        : Islem cifti (None ise varsayilan kullanilir)

        Returns:
            Order nesnesi
        """
        symbol = symbol or self.symbol
        side   = side.lower()

        if side not in ("buy", "sell"):
            raise ValueError(f"Gecersiz side: {side}. 'buy' veya 'sell' olmali.")
        if quantity <= 0:
            raise ValueError(f"Miktar sifirdan buyuk olmali: {quantity}")

        # Slipaj uygula (buy = biraz yukari, sell = biraz asagi)
        if side == "buy":
            fill_price = current_price * (1 + self.slippage)
        else:
            fill_price = current_price * (1 - self.slippage)

        commission_fee = quantity * fill_price * self.commission

        order = Order(
            order_id    = str(uuid.uuid4()),
            symbol      = symbol,
            side        = side,
            type        = OrderType.MARKET,
            quantity    = quantity,
            price       = None,
            filled_price= fill_price,
            status      = OrderStatus.FILLED,   # Market emir aninda dolar
            filled_at   = datetime.now(timezone.utc),
            paper       = self.paper,
            fee         = commission_fee,
        )

        if not self.paper:
            order = self._send_to_exchange(order)
        else:
            logger.info(
                f"[PAPER] Market emir doldu | "
                f"{side.upper()} {quantity:.6f} @ ${fill_price:,.2f} | "
                f"Komisyon: ${commission_fee:.4f}"
            )

        self._orders[order.order_id] = order
        self._total_filled += 1
        return order

    def place_limit_order(
        self,
        side: str,
        quantity: float,
        limit_price: float,
        symbol: str | None = None,
    ) -> Order:
        """
        Limit emri gonderir (belirli fiyattan).
        Paper modda aninda dolu sayilmaz — simule_fill() ile doldurun.

        Returns:
            Order nesnesi (status=OPEN)
        """
        symbol = symbol or self.symbol
        side   = side.lower()

        order = Order(
            order_id = str(uuid.uuid4()),
            symbol   = symbol,
            side     = side,
            type     = OrderType.LIMIT,
            quantity = quantity,
            price    = limit_price,
            status   = OrderStatus.OPEN,
            paper    = self.paper,
        )

        if not self.paper:
            order = self._send_to_exchange(order)
        else:
            logger.info(
                f"[PAPER] Limit emir acildi | "
                f"{side.upper()} {quantity:.6f} @ ${limit_price:,.2f}"
            )

        self._orders[order.order_id] = order
        return order

    # ── Emir Dolum Simulasyonu (Paper) ────────────────────────────────────────

    def simulate_fill(self, order_id: str, current_price: float) -> bool:
        """
        Paper modda limit emrin dolup dolmadigini kontrol eder.

        Buy limit: current_price <= limit_price ise dolu
        Sell limit: current_price >= limit_price ise dolu

        Returns:
            True = doldu, False = hala bekliyor
        """
        order = self._orders.get(order_id)
        if not order or not order.is_open:
            return False

        filled = False
        if order.side == "buy" and current_price <= order.price:
            filled = True
        elif order.side == "sell" and current_price >= order.price:
            filled = True

        if filled:
            order.status      = OrderStatus.FILLED
            order.filled_price = current_price
            order.filled_at   = datetime.now(timezone.utc)
            order.fee         = order.quantity * current_price * self.commission
            self._total_filled += 1
            logger.info(
                f"[PAPER] Limit emir doldu | "
                f"{order.side.upper()} @ ${current_price:,.2f}"
            )

        return filled

    # ── Emir Iptal ────────────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        """
        Emiri iptal eder.

        Returns:
            True = basarili, False = emir bulunamadi veya zaten kapali
        """
        order = self._orders.get(order_id)
        if not order:
            logger.warning(f"Iptal edilecek emir bulunamadi: {order_id[:8]}...")
            return False

        if not order.is_open:
            logger.warning(f"Emir zaten kapali: {order.status}")
            return False

        if not self.paper and self.exchange:
            try:
                self.exchange.cancel_order(order.exchange_id, order.symbol)
            except Exception as e:
                logger.error(f"API iptal hatasi: {e}")
                return False

        order.status = OrderStatus.CANCELLED
        self._total_cancelled += 1
        logger.info(f"Emir iptal edildi: {order_id[:8]}...")
        return True

    def cancel_all_open_orders(self, symbol: str | None = None) -> int:
        """Tum acik emirleri iptal eder. Iptal edilen sayi doner."""
        symbol  = symbol or self.symbol
        targets = [
            o for o in self._orders.values()
            if o.is_open and o.symbol == symbol
        ]
        count = 0
        for order in targets:
            if self.cancel_order(order.order_id):
                count += 1
        logger.info(f"{count} emir iptal edildi ({symbol})")
        return count

    # ── Sorgulama ─────────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Order | None:
        """Belirli bir emiri dondurur."""
        return self._orders.get(order_id)

    def list_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Acik emirlerin listesini dondurur."""
        symbol = symbol or self.symbol
        return [o for o in self._orders.values() if o.is_open and o.symbol == symbol]

    def list_filled_orders(self, symbol: str | None = None) -> list[Order]:
        """Dolan emirlerin listesini dondurur."""
        symbol = symbol or self.symbol
        return [o for o in self._orders.values() if o.is_filled and o.symbol == symbol]

    def all_orders(self) -> list[Order]:
        """Tum emirleri dondurur."""
        return list(self._orders.values())

    # ── Ozet ─────────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Emir ozetini terminale yazdirir."""
        open_   = len(self.list_open_orders())
        filled  = len(self.list_filled_orders())
        total_fee = sum(o.fee for o in self._orders.values() if o.is_filled)
        print(f"\n{'='*50}")
        print(f"  OrderManager Ozeti {'[PAPER]' if self.paper else '[LIVE]'}")
        print(f"{'='*50}")
        print(f"  Acik emirler   : {open_}")
        print(f"  Dolan emirler  : {filled}")
        print(f"  Toplam komisyon: ${total_fee:.4f}")
        print(f"{'='*50}\n")

    def summary_dict(self) -> dict:
        """Ozet bilgileri dict olarak dondurur."""
        return {
            "paper"            : self.paper,
            "open_orders"      : len(self.list_open_orders()),
            "filled_orders"    : len(self.list_filled_orders()),
            "total_filled"     : self._total_filled,
            "total_cancelled"  : self._total_cancelled,
            "total_fee_usdt"   : round(sum(
                o.fee for o in self._orders.values() if o.is_filled
            ), 4),
        }

    # ── Gercek API (paper=False) ──────────────────────────────────────────────

    def _send_to_exchange(self, order: Order) -> Order:
        """
        Gercek CCXT API'ye emir gonderir.
        Sadece paper=False modunda cagirilir.
        """
        if not self.exchange:
            raise RuntimeError("Exchange nesnesi tanimli degil (paper=False icin gerekli)")

        try:
            if order.type == OrderType.MARKET:
                response = self.exchange.create_order(
                    symbol=order.symbol,
                    type="market",
                    side=order.side,
                    amount=order.quantity,
                )
            else:
                response = self.exchange.create_order(
                    symbol=order.symbol,
                    type="limit",
                    side=order.side,
                    amount=order.quantity,
                    price=order.price,
                )

            # Binance response'unu parse et
            order.exchange_id  = str(response.get("id", ""))
            order.filled_price = response.get("average") or response.get("price")
            order.status       = response.get("status", OrderStatus.OPEN)
            order.fee          = float(
                response.get("fee", {}).get("cost", 0) or 0
            )

            logger.info(
                f"[LIVE] Emir gonderildi | "
                f"{order.side.upper()} {order.quantity:.6f} @ "
                f"${order.filled_price or order.price:,.2f} | "
                f"ID: {order.exchange_id}"
            )
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"API emir hatasi: {e}")

        return order
