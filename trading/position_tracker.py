"""
trading/position_tracker.py
=============================
AMACI:
    Acik pozisyonlari ve tum islem gecmisini takip eder.
    Her bar guncellenmesiyle guncel P&L, max drawdown hesaplar.
    Paper ve live modda ayni sekilde calisir.

KAVRAMLAR:
    Position : Acik bir islem (henuz kapatilmamis)
    Trade    : Tamamlanmis bir islem (giris + cikis)
    P&L      : Profit & Loss — kar veya zarar

KULLANIM:
    from trading.position_tracker import PositionTracker

    pt = PositionTracker(initial_capital=10_000.0)

    # Pozisyon ac
    pt.open_position(order)

    # Fiyat guncelle ve P&L hesapla
    summary = pt.update(current_price=95000.0)

    # Stop/TP kontrol et
    actions = pt.check_exit_conditions(current_price=95000.0)

    # Pozisyon kapat
    pt.close_position(position_id, order)

    # Ozet
    pt.print_summary()
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Veri Siniflari ────────────────────────────────────────────────────────────

@dataclass
class Position:
    """
    Acik bir pozisyonu temsil eder.

    Alanlar:
        position_id  : Benzersiz kimlik
        symbol       : Islem cifti (BTC/USDT)
        direction    : "LONG" (yukari) veya "SHORT" (asagi)
        entry_price  : Giris fiyati
        quantity     : Miktar (BTC)
        stop_loss    : Zarar kes fiyati (None = ayarlanmamis)
        take_profit  : Kar al fiyati (None = ayarlanmamis)
        entry_order_id: Giris emirinin order_id'si
        opened_at    : Acilis zamani
        strategy     : Sinyali ureten strateji adi
    """
    position_id   : str
    symbol        : str
    direction     : str          # "LONG" | "SHORT"
    entry_price   : float
    quantity      : float
    stop_loss     : Optional[float] = None
    take_profit   : Optional[float] = None
    entry_order_id: Optional[str]  = None
    opened_at     : datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy      : str = "unknown"
    entry_fee     : float = 0.0

    # Canli guncellenen alanlar
    current_price : float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pct: float = 0.0
    max_unrealized: float = 0.0   # En iyi noktadaki unrealized PnL
    min_unrealized: float = 0.0   # En kotu noktadaki unrealized PnL

    @property
    def notional(self) -> float:
        """Pozisyonun USDT degeri (giris fiyatina gore)."""
        return self.quantity * self.entry_price

    @property
    def duration_minutes(self) -> float:
        """Pozisyon kac dakikadir acik?"""
        now = datetime.now(timezone.utc)
        return (now - self.opened_at).total_seconds() / 60

    def update_price(self, price: float) -> None:
        """Fiyat guncellenir ve unrealized P&L hesaplanir."""
        self.current_price = price
        if self.direction == "LONG":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

        self.unrealized_pct = (self.unrealized_pnl / self.notional) * 100

        # En iyi / en kotu noktayi kaydet
        self.max_unrealized = max(self.max_unrealized, self.unrealized_pnl)
        self.min_unrealized = min(self.min_unrealized, self.unrealized_pnl)

    def should_stop_loss(self, price: float) -> bool:
        """Stop-loss tetiklendi mi?"""
        if self.stop_loss is None:
            return False
        if self.direction == "LONG":
            return price <= self.stop_loss
        else:
            return price >= self.stop_loss

    def should_take_profit(self, price: float) -> bool:
        """Take-profit tetiklendi mi?"""
        if self.take_profit is None:
            return False
        if self.direction == "LONG":
            return price >= self.take_profit
        else:
            return price <= self.take_profit

    def __str__(self) -> str:
        pnl_sign = "+" if self.unrealized_pnl >= 0 else ""
        return (
            f"Position({self.position_id[:8]}... | "
            f"{self.direction} {self.quantity:.6f} {self.symbol} | "
            f"Giris: ${self.entry_price:,.2f} | "
            f"Suanki: ${self.current_price:,.2f} | "
            f"PnL: {pnl_sign}{self.unrealized_pnl:.2f} USDT "
            f"({pnl_sign}{self.unrealized_pct:.2f}%))"
        )


@dataclass
class ClosedTrade:
    """
    Kapatilmis bir islemi temsil eder (giris + cikis tamamlandi).

    Alanlar:
        position_id  : Ilgili pozisyon kimlixi
        symbol       : Islem cifti
        direction    : LONG | SHORT
        entry_price  : Giris fiyati
        exit_price   : Cikis fiyati
        quantity     : Miktar
        realized_pnl : Gerceklesen kar/zarar (komisyon dusulmus)
        realized_pct : Yuzde kar/zarar
        exit_reason  : SIGNAL | STOP_LOSS | TAKE_PROFIT | MANUAL | TIMEOUT
        duration_min : Islem suresi (dakika)
    """
    position_id : str
    symbol      : str
    direction   : str
    entry_price : float
    exit_price  : float
    quantity    : float
    realized_pnl: float
    realized_pct: float
    exit_reason : str
    opened_at   : datetime
    closed_at   : datetime
    strategy    : str = "unknown"
    total_fee   : float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.realized_pnl > 0

    @property
    def duration_min(self) -> float:
        return (self.closed_at - self.opened_at).total_seconds() / 60


# ── PositionTracker ───────────────────────────────────────────────────────────

class PositionTracker:
    """
    Acik pozisyonlari ve tum islem gecmisini yoneten sinif.

    Ozellikler:
        - Birden fazla acik pozisyon desteklenir
        - Her bar: update() ile P&L guncellenir
        - Stop-loss / Take-profit otomatik kontrol
        - Cikis kaydedilir -> ClosedTrade gecmisi
        - Ozet istatistikler: win rate, toplam PnL, max drawdown

    Parametreler:
        initial_capital : Baslangic sermayesi (USDT)
        max_positions   : Maksimum ayni anda acik pozisyon sayisi
        commission      : Komisyon orani
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        max_positions: int     = 3,
        commission: float      = 0.001,
    ):
        self.initial_capital = initial_capital
        self.capital         = initial_capital
        self.max_positions   = max_positions
        self.commission      = commission

        self._positions: dict[str, Position] = {}   # acik pozisyonlar
        self._history: list[ClosedTrade]     = []   # kapatilmis islemler

        # Equity takibi
        self._equity_peak = initial_capital
        self._max_drawdown = 0.0

        logger.info(
            f"PositionTracker baslatildi | "
            f"Sermaye: ${initial_capital:,.2f} | "
            f"Maks pozisyon: {max_positions}"
        )

    # ── Pozisyon Ac ───────────────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        stop_loss: float | None    = None,
        take_profit: float | None  = None,
        strategy: str              = "unknown",
        order_id: str | None       = None,
        entry_fee: float           = 0.0,
    ) -> Position | None:
        """
        Yeni pozisyon acar.

        Kural kontrolleri:
            - max_positions asilmamali
            - Yeterli sermaye olmali

        Returns:
            Position nesnesi veya None (kural ihlali)
        """
        if len(self._positions) >= self.max_positions:
            logger.warning(
                f"Maks pozisyon sayisina ulasildi ({self.max_positions}), "
                f"yeni pozisyon acilmiyor."
            )
            return None

        notional = quantity * entry_price
        if notional > self.capital:
            logger.warning(
                f"Yetersiz sermaye: gerekli ${notional:,.2f}, "
                f"mevcut ${self.capital:,.2f}"
            )
            return None

        pos = Position(
            position_id    = str(uuid.uuid4()),
            symbol         = symbol,
            direction      = direction.upper(),
            entry_price    = entry_price,
            quantity       = quantity,
            stop_loss      = stop_loss,
            take_profit    = take_profit,
            entry_order_id = order_id,
            strategy       = strategy,
            entry_fee      = entry_fee,
            current_price  = entry_price,
        )

        # Sermayeyi bloke et
        self.capital -= notional + entry_fee
        self._positions[pos.position_id] = pos

        sl_str = f"${stop_loss:,.2f}" if stop_loss else "yok"
        tp_str = f"${take_profit:,.2f}" if take_profit else "yok"
        logger.info(
            f"Pozisyon acildi | {direction.upper()} {quantity:.6f} {symbol} "
            f"@ ${entry_price:,.2f} | SL: {sl_str} | TP: {tp_str} | "
            f"Strateji: {strategy}"
        )
        return pos

    # ── Fiyat Guncelle ────────────────────────────────────────────────────────

    def update(self, current_price: float, symbol: str | None = None) -> dict:
        """
        Tum acik pozisyonlari gunceller, equity ve drawdown hesaplar.

        Returns:
            dict: guncel P&L ozeti
        """
        symbol = symbol or "BTC/USDT"

        total_unrealized = 0.0
        for pos in self._positions.values():
            if pos.symbol == symbol:
                pos.update_price(current_price)
                total_unrealized += pos.unrealized_pnl

        # Guncel equity (kilitli notional dahil — KillSwitch yanlis alarm vermesin)
        locked = sum(p.notional for p in self._positions.values())
        equity = self.capital + locked + total_unrealized
        self._equity_peak = max(self._equity_peak, equity)

        # Max drawdown guncelle
        if self._equity_peak > 0:
            drawdown = (self._equity_peak - equity) / self._equity_peak * 100
            self._max_drawdown = max(self._max_drawdown, drawdown)

        return {
            "capital"          : round(self.capital, 2),
            "total_unrealized" : round(total_unrealized, 2),
            "equity"           : round(equity, 2),
            "max_drawdown_pct" : round(self._max_drawdown, 3),
            "open_positions"   : len(self._positions),
        }

    # ── Cikis Sartlarini Kontrol ──────────────────────────────────────────────

    def check_exit_conditions(
        self, current_price: float
    ) -> list[tuple[str, str]]:
        """
        Tum acik pozisyonlarda SL/TP tetiklenip tetiklenmedigini kontrol eder.

        Returns:
            list of (position_id, reason): cikmasi gereken pozisyonlar
        """
        exits = []
        for pid, pos in self._positions.items():
            pos.update_price(current_price)
            if pos.should_stop_loss(current_price):
                exits.append((pid, "STOP_LOSS"))
                logger.warning(
                    f"STOP-LOSS tetiklendi! {pos.symbol} @ ${current_price:,.2f} "
                    f"(SL: ${pos.stop_loss:,.2f})"
                )
            elif pos.should_take_profit(current_price):
                exits.append((pid, "TAKE_PROFIT"))
                logger.info(
                    f"TAKE-PROFIT! {pos.symbol} @ ${current_price:,.2f} "
                    f"(TP: ${pos.take_profit:,.2f})"
                )
        return exits

    # ── Pozisyon Kapat ────────────────────────────────────────────────────────

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str = "SIGNAL",
        exit_fee: float  = 0.0,
    ) -> ClosedTrade | None:
        """
        Pozisyonu kapatir ve ClosedTrade kaydeder.

        Returns:
            ClosedTrade nesnesi veya None (pozisyon bulunamazsa)
        """
        pos = self._positions.get(position_id)
        if not pos:
            logger.warning(f"Pozisyon bulunamadi: {position_id[:8]}...")
            return None

        # PnL hesapla
        if pos.direction == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        # Komisyon dusulmus PnL
        exit_fee   = exit_fee or (pos.quantity * exit_price * self.commission)
        net_pnl    = gross_pnl - pos.entry_fee - exit_fee
        net_pct    = (net_pnl / pos.notional) * 100

        trade = ClosedTrade(
            position_id  = position_id,
            symbol       = pos.symbol,
            direction    = pos.direction,
            entry_price  = pos.entry_price,
            exit_price   = exit_price,
            quantity     = pos.quantity,
            realized_pnl = round(net_pnl, 4),
            realized_pct = round(net_pct, 4),
            exit_reason  = exit_reason,
            opened_at    = pos.opened_at,
            closed_at    = datetime.now(timezone.utc),
            strategy     = pos.strategy,
            total_fee    = round(pos.entry_fee + exit_fee, 4),
        )

        # Sermayeyi geri al
        notional       = pos.quantity * exit_price
        self.capital  += notional - exit_fee + gross_pnl

        del self._positions[position_id]
        self._history.append(trade)

        sign = "+" if net_pnl >= 0 else ""
        logger.info(
            f"Pozisyon kapatildi | {pos.direction} {pos.symbol} | "
            f"Giris: ${pos.entry_price:,.2f} -> Cikis: ${exit_price:,.2f} | "
            f"PnL: {sign}{net_pnl:.2f} USDT ({sign}{net_pct:.2f}%) | "
            f"Sebep: {exit_reason}"
        )
        return trade

    def close_all_positions(
        self, current_price: float, reason: str = "MANUAL"
    ) -> list[ClosedTrade]:
        """Tum acik pozisyonlari kapatir (orn. emergency stop icin)."""
        trades = []
        for pid in list(self._positions.keys()):
            t = self.close_position(pid, current_price, exit_reason=reason)
            if t:
                trades.append(t)
        logger.warning(f"{len(trades)} pozisyon kapatildi. Sebep: {reason}")
        return trades

    # ── Sorgulama ─────────────────────────────────────────────────────────────

    def get_position(self, position_id: str) -> Position | None:
        return self._positions.get(position_id)

    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    def closed_trades(self) -> list[ClosedTrade]:
        return list(self._history)

    def has_open_position(self, symbol: str | None = None) -> bool:
        if symbol:
            return any(p.symbol == symbol for p in self._positions.values())
        return len(self._positions) > 0

    def can_open_position(self) -> bool:
        """Yeni pozisyon acilabilir mi?"""
        return len(self._positions) < self.max_positions

    # ── Istatistikler ─────────────────────────────────────────────────────────

    def performance_summary(self) -> dict:
        """Tamamlanan islemler uzerinden performans metrikleri."""
        trades = self._history
        if not trades:
            return {
                "total_trades"  : 0,
                "win_rate_pct"  : 0.0,
                "total_pnl"     : 0.0,
                "max_drawdown_pct": round(self._max_drawdown, 3),
                "capital"       : round(self.capital, 2),
                "return_pct"    : round(
                    (self.capital - self.initial_capital) / self.initial_capital * 100, 3
                ),
            }

        winners = [t for t in trades if t.is_winner]
        total_pnl = sum(t.realized_pnl for t in trades)

        return {
            "total_trades"    : len(trades),
            "winning_trades"  : len(winners),
            "win_rate_pct"    : round(len(winners) / len(trades) * 100, 2),
            "total_pnl"       : round(total_pnl, 2),
            "avg_pnl"         : round(total_pnl / len(trades), 4),
            "best_trade"      : round(max(t.realized_pnl for t in trades), 2),
            "worst_trade"     : round(min(t.realized_pnl for t in trades), 2),
            "avg_duration_min": round(
                sum(t.duration_min for t in trades) / len(trades), 1
            ),
            "total_fee"       : round(sum(t.total_fee for t in trades), 4),
            "max_drawdown_pct": round(self._max_drawdown, 3),
            "capital"         : round(self.capital, 2),
            "return_pct"      : round(
                (self.capital - self.initial_capital) / self.initial_capital * 100, 3
            ),
        }

    def print_summary(self) -> None:
        """Ozeti terminale yazdirir."""
        s = self.performance_summary()
        print(f"\n{'='*55}")
        print(f"  PositionTracker Ozeti")
        print(f"{'='*55}")
        print(f"  Baslangic Sermaye : ${self.initial_capital:,.2f}")
        print(f"  Guncel Sermaye    : ${s['capital']:,.2f}")
        print(f"  Toplam Getiri     : %{s['return_pct']:.2f}")
        print(f"  Toplam Islem      : {s['total_trades']}")
        if s["total_trades"] > 0:
            print(f"  Win Rate          : %{s['win_rate_pct']:.1f}")
            print(f"  Toplam PnL        : ${s['total_pnl']:,.2f}")
            print(f"  En Iyi Islem      : ${s['best_trade']:,.2f}")
            print(f"  En Kotu Islem     : ${s['worst_trade']:,.2f}")
            print(f"  Max Drawdown      : %{s['max_drawdown_pct']:.2f}")
        print(f"{'='*55}\n")
