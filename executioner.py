"""
executioner.py — order execution and portfolio state.

PATCHES APPLIED (v2):
  1. [KRITIS] _resolve_live_taker_fill timeout: 1.0s → 3.5s
     Mencegah false-negative fill detection yang menyebabkan satu sisi
     ter-cancel padahal order sudah tereksekusi di exchange.

  2. [KRITIS] Pisah SDK lock: _sdk_call_lock → _sdk_read_lock (Semaphore(3))
     + _sdk_write_lock (Lock). Mencegah bottleneck dimana polling fill
     memblokir semua order baru selama window kritis.

  3. [KRITIS] Insurance placement dipindah ke dalam _run_phase2_twap,
     setelah chunk pertama confirmed. Mencegah insurance ter-arm saat
     TWAP gagal, yang menyebabkan posisi terbuka satu arah yang rugi.
     Brain.py perlu disesuaikan: flag phase2_insurance_placed sekarang
     di-set dari dalam _run_phase2_twap via helper baru.

  4. [MINOR] Straddle: state.straddle_placed hanya True jika kedua leg
     berhasil. Jika salah satu gagal, posisi yang ada di-cancel.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Deque, Dict, List, Optional

import config
from config import POLYGON_RPCS

log = logging.getLogger("executioner")

_WEB3_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
_ZERO_BYTES32 = "0x" + "00" * 32
_CTF_REDEEM_ABI = [{
    "inputs": [
        {"internalType": "address", "name": "collateralToken", "type": "address"},
        {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
        {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
        {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"},
    ],
    "name": "redeemPositions",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function",
}]


class MarketExpiredError(Exception):
    """Raised when the CLOB rejects an order because the market no longer exists."""


class Side(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELED = "CANCELED"


LIMIT_FILL_PHASES = {
    "PHASE1_ENTRY",
    "PHASE1_HEDGE",
    "PHASE2_HEDGE",
    "PERFECT_HEDGE",
    "GOD_HEDGE",
    "PHASE2_INSURANCE",
    "PHASE3_SNIPER",
}


@dataclass
class Order:
    order_id: str
    condition_id: str
    market_label: str
    asset: str
    timeframe: str
    token_id: str
    side: Side
    price: float
    size: float
    phase: str
    status: OrderStatus = OrderStatus.OPEN
    fill_price: float = 0.0
    fill_size: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Fill:
    order_id: str
    condition_id: str
    market_label: str
    asset: str
    timeframe: str
    token_id: str
    side: str
    price: float
    size: float
    shares: float
    phase: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    token_id: str
    bids: List[BookLevel]
    asks: List[BookLevel]
    tick_size: float
    last_trade_price: float = 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def best_ask_size(self) -> float:
        return self.asks[0].size if self.asks else 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_bid_size(self) -> float:
        return self.bids[0].size if self.bids else 0.0

    @property
    def bid_volume(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def ask_volume(self) -> float:
        return sum(level.size for level in self.asks)


@dataclass
class Position:
    condition_id: str
    market_label: str
    asset: str
    timeframe: str
    token_id: str
    side: Side
    shares: float = 0.0
    cost_basis: float = 0.0
    avg_entry_price: float = 0.0


@dataclass(frozen=True)
class PositionSnapshot:
    condition_id: str
    market_label: str
    asset: str
    timeframe: str
    token_id: str
    side: str
    shares: float
    cost_basis: float
    avg_entry_price: float


@dataclass(frozen=True)
class ChunkExecution:
    timestamp: float
    action: str
    market_id: str
    market_label: str
    asset: str
    timeframe: str
    side: str
    phase: str
    size_usdc: float
    price: float
    shares: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    available_balance: float
    locked_margin: float
    realized_pnl: float
    total_equity: float
    current_balance: float
    active_positions: Dict[str, Dict[str, PositionSnapshot]]
    open_orders: List[Order]
    recent_chunks: List[ChunkExecution]


@dataclass(frozen=True)
class SettlementResult:
    condition_id: str
    market_label: str
    winning_side: Optional[str]
    payout: float
    realized_pnl: float
    released_orders: int
    settled: bool = True
    error: str = ""
    claim_tx_hash: str = ""


class OrderManagerBase:
    async def place_limit_order(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        price: float,
        size_usdc: float,
        phase: str,
    ) -> Optional[Order]:
        raise NotImplementedError

    async def place_dual_limit_orders(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        up_token_id: str,
        down_token_id: str,
        up_price: float,
        down_price: float,
        size_per_side: float,
        phase: str,
    ) -> List[Order]:
        raise NotImplementedError

    async def get_order_book(
        self,
        token_id: str,
        *,
        fallback_best_ask: float = 0.0,
        fallback_best_bid: float = 0.0,
    ) -> Optional[OrderBookSnapshot]:
        raise NotImplementedError

    async def execute_taker_buy(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        phase: str,
        aggressive_price: float,
        expected_fill_price: float,
        target_shares: float,
        max_size_usdc: float,
    ) -> Optional[Fill]:
        raise NotImplementedError

    async def execute_taker_sell(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        phase: str,
        target_shares: float,
        expected_fill_price: float,
    ) -> Optional[Fill]:
        raise NotImplementedError

    async def execute_sniper(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        size_usdc: float,
        current_best_ask: float,
    ) -> Optional[Fill]:
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    async def cancel_all_for_market(self, condition_id: str) -> int:
        raise NotImplementedError

    async def process_limit_crosses(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        best_ask_up: float,
        best_ask_down: float,
    ) -> List[Fill]:
        raise NotImplementedError

    async def settle_market(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        winning_side: Optional[Side],
    ) -> SettlementResult:
        raise NotImplementedError

    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        raise NotImplementedError

    async def clear_expired_market(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        winning_side: Optional[Side],
    ) -> SettlementResult:
        raise NotImplementedError

    async def sync_live_balance(self) -> None:
        pass


class _PortfolioMixin:
    def __init__(self, initial_balance: float) -> None:
        self._available_balance = initial_balance
        self._locked_margin = 0.0
        self._realized_pnl = 0.0
        self._orders: Dict[str, Order] = {}
        self._active_positions: Dict[str, Dict[Side, Position]] = {}
        self._recent_chunks: Deque[ChunkExecution] = deque(maxlen=100)
        self._lock = asyncio.Lock()

    @property
    def available_balance(self) -> float:
        return self._available_balance

    @property
    def locked_margin(self) -> float:
        return self._locked_margin

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def total_equity(self) -> float:
        return self._available_balance + self._locked_margin

    def get_open_orders(self, condition_id: str = "") -> List[Order]:
        return [
            replace(order)
            for order in self._orders.values()
            if order.status == OrderStatus.OPEN
            and (not condition_id or order.condition_id == condition_id)
        ]

    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        async with self._lock:
            return self._snapshot_locked()

    async def clear_expired_market(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        winning_side: Optional[Side],
    ) -> SettlementResult:
        """Force-clear in-memory positions and locked margin for an expired market.

        Called when on-chain settlement keeps failing so the dashboard no longer
        shows stale positions or inflated locked exposure.  The actual wallet
        balance is reconciled on the next sync_live_balance() call.
        """
        async with self._lock:
            return self._settle_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                winning_side=winning_side,
            )

    def _snapshot_locked(self) -> PortfolioSnapshot:
        positions: Dict[str, Dict[str, PositionSnapshot]] = {}
        for condition_id, side_map in self._active_positions.items():
            market_positions: Dict[str, PositionSnapshot] = {}
            for side, position in side_map.items():
                if position.shares <= 0 or position.cost_basis <= 0:
                    continue
                market_positions[side.value] = PositionSnapshot(
                    condition_id=condition_id,
                    market_label=position.market_label,
                    asset=position.asset,
                    timeframe=position.timeframe,
                    token_id=position.token_id,
                    side=side.value,
                    shares=position.shares,
                    cost_basis=position.cost_basis,
                    avg_entry_price=position.avg_entry_price,
                )
            if market_positions:
                positions[condition_id] = market_positions

        open_orders = [
            replace(order)
            for order in self._orders.values()
            if order.status == OrderStatus.OPEN
        ]

        return PortfolioSnapshot(
            available_balance=self._available_balance,
            locked_margin=self._locked_margin,
            realized_pnl=self._realized_pnl,
            total_equity=self.total_equity,
            current_balance=self.total_equity,
            active_positions=positions,
            open_orders=open_orders,
            recent_chunks=list(self._recent_chunks)[:10],
        )

    def _record_chunk_locked(
        self,
        *,
        action: str,
        market_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        side: str,
        phase: str,
        size_usdc: float,
        price: float,
        shares: float,
    ) -> None:
        self._recent_chunks.appendleft(
            ChunkExecution(
                timestamp=time.time(),
                action=action,
                market_id=market_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                side=side,
                phase=phase,
                size_usdc=size_usdc,
                price=price,
                shares=shares,
            )
        )

    def _reserve_locked(self, size_usdc: float) -> bool:
        if size_usdc <= 0:
            return False
        if size_usdc > self._available_balance + 1e-9:
            return False
        self._available_balance -= size_usdc
        self._locked_margin += size_usdc
        return True

    def _release_locked(self, size_usdc: float) -> None:
        self._available_balance += size_usdc
        self._locked_margin = max(0.0, self._locked_margin - size_usdc)

    def _ensure_position_locked(
        self,
        *,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
    ) -> Position:
        market_positions = self._active_positions.setdefault(condition_id, {})
        if side not in market_positions:
            market_positions[side] = Position(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=token_id,
                side=side,
            )
        return market_positions[side]

    def _existing_position_locked(
        self,
        *,
        condition_id: str,
        side: Side,
    ) -> Optional[Position]:
        return self._active_positions.get(condition_id, {}).get(side)

    def _apply_fill_locked(self, order: Order, fill_price: float, action: str) -> Fill:
        shares = order.size / fill_price if fill_price > 0 else 0.0
        return self._apply_custom_fill_locked(
            order,
            fill_price=fill_price,
            fill_size_usdc=order.size,
            shares=shares,
            action=action,
        )

    def _apply_custom_fill_locked(
        self,
        order: Order,
        *,
        fill_price: float,
        fill_size_usdc: float,
        shares: float,
        action: str,
    ) -> Fill:
        fill_size_usdc = max(0.0, min(fill_size_usdc, order.size))
        shares = max(0.0, shares)
        unused_reserve = max(0.0, order.size - fill_size_usdc)

        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_size = fill_size_usdc

        if unused_reserve > 0:
            self._release_locked(unused_reserve)

        position = self._ensure_position_locked(
            condition_id=order.condition_id,
            market_label=order.market_label,
            asset=order.asset,
            timeframe=order.timeframe,
            token_id=order.token_id,
            side=order.side,
        )
        position.cost_basis += fill_size_usdc
        position.shares += shares
        position.avg_entry_price = (
            position.cost_basis / position.shares if position.shares > 0 else 0.0
        )

        fill = Fill(
            order_id=order.order_id,
            condition_id=order.condition_id,
            market_label=order.market_label,
            asset=order.asset,
            timeframe=order.timeframe,
            token_id=order.token_id,
            side=order.side.value,
            price=fill_price,
            size=fill_size_usdc,
            shares=shares,
            phase=order.phase,
        )

        self._record_chunk_locked(
            action=action,
            market_id=order.condition_id,
            market_label=order.market_label,
            asset=order.asset,
            timeframe=order.timeframe,
            side=order.side.value,
            phase=order.phase,
            size_usdc=fill_size_usdc,
            price=fill_price,
            shares=shares,
        )
        return fill

    def _apply_sell_fill_locked(
        self,
        *,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        phase: str,
        fill_price: float,
        shares: float,
        action: str,
        order_id: Optional[str] = None,
    ) -> Optional[Fill]:
        position = self._existing_position_locked(condition_id=condition_id, side=side)
        if position is None or position.shares <= 0 or position.cost_basis <= 0:
            return None

        sold_shares = min(max(0.0, shares), position.shares)
        if sold_shares <= 0 or fill_price <= 0:
            return None

        prior_shares = position.shares
        prior_cost = position.cost_basis
        released_cost = prior_cost * (sold_shares / prior_shares)
        proceeds = sold_shares * fill_price
        realized = proceeds - released_cost

        position.shares = max(0.0, prior_shares - sold_shares)
        position.cost_basis = max(0.0, prior_cost - released_cost)
        position.avg_entry_price = (
            position.cost_basis / position.shares if position.shares > 0 else 0.0
        )

        self._locked_margin = max(0.0, self._locked_margin - released_cost)
        self._available_balance += proceeds
        self._realized_pnl += realized

        if position.shares <= 1e-9:
            side_map = self._active_positions.get(condition_id, {})
            side_map.pop(side, None)
            if not side_map:
                self._active_positions.pop(condition_id, None)

        fill = Fill(
            order_id=order_id or f"sell-{uuid.uuid4().hex[:12]}",
            condition_id=condition_id,
            market_label=market_label,
            asset=asset,
            timeframe=timeframe,
            token_id=token_id,
            side=side.value,
            price=fill_price,
            size=proceeds,
            shares=sold_shares,
            phase=phase,
        )

        self._record_chunk_locked(
            action=action,
            market_id=condition_id,
            market_label=market_label,
            asset=asset,
            timeframe=timeframe,
            side=side.value,
            phase=phase,
            size_usdc=proceeds,
            price=fill_price,
            shares=sold_shares,
        )
        return fill

    def _cancel_order_locked(self, order: Order) -> bool:
        if order.status != OrderStatus.OPEN:
            return False
        order.status = OrderStatus.CANCELED
        self._release_locked(order.size)
        self._record_chunk_locked(
            action="Cancel",
            market_id=order.condition_id,
            market_label=order.market_label,
            asset=order.asset,
            timeframe=order.timeframe,
            side=order.side.value,
            phase=order.phase,
            size_usdc=order.size,
            price=order.price,
            shares=0.0,
        )
        return True

    def _cancel_all_for_market_locked(self, condition_id: str) -> int:
        canceled = 0
        for order in self._orders.values():
            if order.condition_id != condition_id:
                continue
            if self._cancel_order_locked(order):
                canceled += 1
        return canceled

    def _create_order_locked(
        self,
        *,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        price: float,
        size_usdc: float,
        phase: str,
        order_id: Optional[str] = None,
    ) -> Order:
        order = Order(
            order_id=order_id or f"ord-{uuid.uuid4().hex[:12]}",
            condition_id=condition_id,
            market_label=market_label,
            asset=asset,
            timeframe=timeframe,
            token_id=token_id,
            side=side,
            price=price,
            size=size_usdc,
            phase=phase,
        )
        self._orders[order.order_id] = order
        return order

    def _track_order_placement_locked(self, order: Order) -> None:
        self._record_chunk_locked(
            action="Place",
            market_id=order.condition_id,
            market_label=order.market_label,
            asset=order.asset,
            timeframe=order.timeframe,
            side=order.side.value,
            phase=order.phase,
            size_usdc=order.size,
            price=order.price,
            shares=0.0,
        )

    def _settle_locked(
        self,
        *,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        winning_side: Optional[Side],
    ) -> SettlementResult:
        released_orders = self._cancel_all_for_market_locked(condition_id)
        payout = 0.0
        total_cost = 0.0
        winning_side_label = winning_side.value if winning_side else None

        side_map = self._active_positions.pop(condition_id, {})
        for side, position in side_map.items():
            if position.cost_basis <= 0:
                continue
            total_cost += position.cost_basis
            if winning_side is not None and side == winning_side:
                payout += position.shares
            self._locked_margin = max(0.0, self._locked_margin - position.cost_basis)

        if payout > 0:
            self._available_balance += payout

        realized = payout - total_cost
        self._realized_pnl += realized

        if total_cost > 0 or released_orders > 0:
            self._record_chunk_locked(
                action="Settle",
                market_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                side=winning_side_label or "-",
                phase="SETTLE",
                size_usdc=payout,
                price=1.0 if payout > 0 else 0.0,
                shares=payout,
            )

        return SettlementResult(
            condition_id=condition_id,
            market_label=market_label,
            winning_side=winning_side_label,
            payout=payout,
            realized_pnl=realized,
            released_orders=released_orders,
        )


class SimulatorOrderManager(_PortfolioMixin, OrderManagerBase):
    def __init__(self, initial_balance: float = config.INITIAL_BALANCE) -> None:
        super().__init__(initial_balance=initial_balance)
        # Real CLOB book fetching — makes simulation constraints identical to live.
        # Books are cached per token_id with a 1-second TTL to avoid API spam.
        self._clob_session: Optional[object] = None  # aiohttp.ClientSession, lazy-init
        self._clob_read_lock = asyncio.Semaphore(3)   # max 3 concurrent reads (same as live)
        self._book_cache: Dict[str, tuple[Optional[OrderBookSnapshot], float]] = {}

    async def place_limit_order(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        price: float,
        size_usdc: float,
        phase: str,
    ) -> Optional[Order]:
        if not token_id or size_usdc < config.PASSIVE_MIN_ORDER_USDC:
            return None

        async with self._lock:
            if not self._reserve_locked(size_usdc):
                return None

            order = self._create_order_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=token_id,
                side=side,
                price=price,
                size_usdc=size_usdc,
                phase=phase,
                order_id=f"sim-{uuid.uuid4().hex[:12]}",
            )
            self._track_order_placement_locked(order)
            return replace(order)

    async def place_dual_limit_orders(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        up_token_id: str,
        down_token_id: str,
        up_price: float,
        down_price: float,
        size_per_side: float,
        phase: str,
    ) -> List[Order]:
        if not up_token_id or not down_token_id:
            return []
        if size_per_side < config.PASSIVE_MIN_ORDER_USDC:
            return []

        total_size = size_per_side * 2
        async with self._lock:
            if not self._reserve_locked(total_size):
                return []

            up_order = self._create_order_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=up_token_id,
                side=Side.UP,
                price=up_price,
                size_usdc=size_per_side,
                phase=phase,
                order_id=f"sim-{uuid.uuid4().hex[:12]}",
            )
            down_order = self._create_order_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=down_token_id,
                side=Side.DOWN,
                price=down_price,
                size_usdc=size_per_side,
                phase=phase,
                order_id=f"sim-{uuid.uuid4().hex[:12]}",
            )
            self._track_order_placement_locked(up_order)
            self._track_order_placement_locked(down_order)
            return [replace(up_order), replace(down_order)]

    # ------------------------------------------------------------------
    # Real CLOB book fetching (public endpoint — no auth needed)
    # Gives simulation the same depth constraints as live mode.
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def _get_clob_session(self):
        try:
            import aiohttp
            if self._clob_session is None or self._clob_session.closed:
                timeout = aiohttp.ClientTimeout(connect=3.0, total=5.0)
                self._clob_session = aiohttp.ClientSession(timeout=timeout)
            return self._clob_session
        except ImportError:
            return None

    async def _fetch_clob_book(self, token_id: str) -> tuple[Optional[OrderBookSnapshot], bool]:
        """Fetch real depth from CLOB public API.

        Returns `(book, hard_missing)`. `hard_missing=True` means the CLOB
        explicitly reported that the token has no live orderbook, so SIM should
        not fabricate a synthetic book for that market.
        """
        session = await self._get_clob_session()
        if session is None:
            return None, False
        try:
            async with self._clob_read_lock:
                async with session.get(
                    f"{config.CLOB_HOST}/book",
                    params={"token_id": token_id},
                ) as resp:
                    if resp.status == 404:
                        log.debug("[SIM] No live orderbook for %s", token_id[:16])
                        return None, True
                    if resp.status != 200:
                        return None, False
                    data = await resp.json(content_type=None)
        except Exception as exc:
            log.debug("[SIM] CLOB book fetch failed %s: %s", token_id[:16], exc)
            return None, False

        raw_bids = data.get("bids") or []
        raw_asks = data.get("asks") or []
        bids = [
            BookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in raw_bids
            if "price" in b and "size" in b
        ]
        asks = [
            BookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in raw_asks
            if "price" in a and "size" in a
        ]
        if not bids and not asks:
            return None, True

        tick = self._safe_float(data.get("tick_size", 0.01), default=0.01)
        last_trade = self._safe_float(data.get("last_trade_price", 0.0), default=0.0)
        return OrderBookSnapshot(
            token_id=token_id,
            bids=bids,
            asks=asks,
            tick_size=tick,
            last_trade_price=last_trade,
        ), False

    async def get_order_book(
        self,
        token_id: str,
        *,
        fallback_best_ask: float = 0.0,
        fallback_best_bid: float = 0.0,
    ) -> Optional[OrderBookSnapshot]:
        if not token_id or fallback_best_ask <= 0:
            return None

        # Try real CLOB book first (1-second TTL cache to avoid API spam).
        now = time.time()
        cached = self._book_cache.get(token_id)
        if cached and now - cached[1] < 1.0:
            return cached[0]

        real_book, hard_missing = await self._fetch_clob_book(token_id)
        if real_book is not None:
            self._book_cache[token_id] = (real_book, now)
            return real_book
        if hard_missing or not config.SIM_FALLBACK_SYNTHETIC_ON_NETWORK_ERROR:
            self._book_cache[token_id] = (None, now)
            return None

        # Fallback — synthetic book when CLOB is unreachable.
        bid_size = max(config.SIMULATED_TOP_BOOK_SIZE, config.MIN_ORDER_SHARES * 10.0)
        ask_size = max(config.SIMULATED_TOP_BOOK_SIZE, config.MIN_ORDER_SHARES * 10.0)
        effective_bid = (
            fallback_best_bid
            if fallback_best_bid > 0
            else fallback_best_ask * 0.97
        )
        book = OrderBookSnapshot(
            token_id=token_id,
            bids=[BookLevel(price=max(0.0, effective_bid), size=bid_size)],
            asks=[BookLevel(price=fallback_best_ask, size=ask_size)],
            tick_size=config.SIMULATED_TICK_SIZE,
            last_trade_price=effective_bid,
        )
        self._book_cache[token_id] = (book, now)
        return book

    async def execute_taker_buy(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        phase: str,
        aggressive_price: float,
        expected_fill_price: float,
        target_shares: float,
        max_size_usdc: float,
    ) -> Optional[Fill]:
        if not token_id or target_shares < config.MIN_ORDER_SHARES:
            return None

        base_price = expected_fill_price if expected_fill_price > 0 else aggressive_price
        # Realistic buy slippage: 0.1% – 0.3% above expected price, capped at aggressive.
        slippage = random.uniform(1.001, 1.003)
        actual_fill_price = min(base_price * slippage, aggressive_price if aggressive_price > 0 else base_price * slippage)
        min_size_usdc = config.minimum_taker_order_usdc(actual_fill_price)
        if max_size_usdc + 1e-9 < min_size_usdc:
            return None
        actual_cost = target_shares * actual_fill_price
        if actual_cost + 1e-9 < min_size_usdc:
            return None

        reserve_size = max(actual_cost, max_size_usdc)
        async with self._lock:
            if not self._reserve_locked(reserve_size):
                return None

            order = self._create_order_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=token_id,
                side=side,
                price=aggressive_price,
                size_usdc=reserve_size,
                phase=phase,
                order_id=f"sim-taker-{uuid.uuid4().hex[:12]}",
            )
            self._track_order_placement_locked(order)
            fill = self._apply_custom_fill_locked(
                order,
                fill_price=actual_fill_price,
                fill_size_usdc=actual_cost,
                shares=target_shares,
                action="Taker Fill",
            )

        if fill is not None:
            log.info(
                "[EXEC][SIM] action=BUY | market=%s | side=%s | phase=%s"
                " | shares=%.4f | price=%.4f | cost=$%.2f",
                market_label, side.value, phase, fill.shares, fill.price, fill.size,
            )
        return fill

    async def execute_taker_sell(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        phase: str,
        target_shares: float,
        expected_fill_price: float,
    ) -> Optional[Fill]:
        if not token_id or target_shares <= 0 or expected_fill_price <= 0:
            return None

        # Realistic sell slippage: 0.1% – 0.3% below expected bid price.
        slippage = random.uniform(0.997, 0.999)
        actual_fill_price = expected_fill_price * slippage

        async with self._lock:
            fill = self._apply_sell_fill_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=token_id,
                side=side,
                phase=phase,
                fill_price=actual_fill_price,
                shares=target_shares,
                action="Taker Sell",
            )

        if fill is not None:
            log.info(
                "[EXEC][SIM] action=SELL | market=%s | side=%s | phase=%s"
                " | shares=%.4f | price=%.4f | cost=$%.2f",
                market_label, side.value, phase, fill.shares, fill.price, fill.size,
            )
        return fill

    async def execute_sniper(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        size_usdc: float,
        current_best_ask: float,
    ) -> Optional[Fill]:
        if current_best_ask <= 0:
            return None

        target_shares = size_usdc / current_best_ask
        return await self.execute_taker_buy(
            condition_id=condition_id,
            market_label=market_label,
            asset=asset,
            timeframe=timeframe,
            token_id=token_id,
            side=side,
            phase="PHASE2",
            aggressive_price=config.SNIPER_LIMIT_PRICE,
            expected_fill_price=current_best_ask,
            target_shares=target_shares,
            max_size_usdc=size_usdc,
        )

    async def cancel_order(self, order_id: str) -> bool:
        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return False
            return self._cancel_order_locked(order)

    async def cancel_all_for_market(self, condition_id: str) -> int:
        async with self._lock:
            return self._cancel_all_for_market_locked(condition_id)

    async def process_limit_crosses(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        best_ask_up: float,
        best_ask_down: float,
    ) -> List[Fill]:
        fills: List[Fill] = []

        async with self._lock:
            for order in self._orders.values():
                if order.condition_id != condition_id or order.status != OrderStatus.OPEN:
                    continue
                if order.phase not in LIMIT_FILL_PHASES:
                    continue

                best_ask = best_ask_up if order.side == Side.UP else best_ask_down
                if best_ask <= 0 or order.price < best_ask:
                    continue

                fills.append(self._apply_fill_locked(order, best_ask, "Limit Fill"))

        return fills

    async def settle_market(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        winning_side: Optional[Side],
    ) -> SettlementResult:
        async with self._lock:
            return self._settle_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                winning_side=winning_side,
            )


class LiveOrderManager(_PortfolioMixin, OrderManagerBase):
    def __init__(self) -> None:
        super().__init__(initial_balance=config.INITIAL_BALANCE)
        self._tick_size_cache: Dict[str, float] = {}
        self._order_book_cache: Dict[str, tuple[Optional[OrderBookSnapshot], float, str]] = {}

        # ----------------------------------------------------------------
        # FIX #2: Pisah SDK lock → read (Semaphore) + write (Lock)
        # Read calls (get_order, get_order_book) bisa concurrent max 3.
        # Write calls (post_order, create_and_post_order) tetap serial.
        # ----------------------------------------------------------------
        self._sdk_read_lock = asyncio.Semaphore(3)
        self._sdk_write_lock = asyncio.Lock()

        self._sdk_transport_mode = "unpatched"
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds, MarketOrderArgs, OrderArgs, OrderType
            from eth_account import Account as _Account

            self._patch_py_clob_http_transport()
            self._signer_address = _Account.from_key(config.PRIVATE_KEY).address
            self._signature_type = config.SIGNATURE_TYPE
            self._funder_address = self._resolve_funder_address(self._signer_address)

            self._client = ClobClient(
                host=config.CLOB_HOST,
                key=config.PRIVATE_KEY,
                chain_id=config.CHAIN_ID,
                signature_type=self._signature_type,
                funder=self._funder_address,
            )
            self._ensure_builder_funder()

            env_creds_complete = all((config.API_KEY, config.API_SECRET, config.API_PASSPHRASE))
            env_creds_partial = any((config.API_KEY, config.API_SECRET, config.API_PASSPHRASE))
            self._api_creds_source = "derived"
            try:
                creds = self._client.create_or_derive_api_creds()
                self._client.set_api_creds(creds)
            except Exception as exc:
                if env_creds_complete:
                    self._client.set_api_creds(
                        ApiCreds(
                            api_key=config.API_KEY,
                            api_secret=config.API_SECRET,
                            api_passphrase=config.API_PASSPHRASE,
                        )
                    )
                    self._api_creds_source = "env-fallback"
                    log.warning(
                        "[LIVE] API creds auto-derive failed (%s) — using env creds fallback.",
                        exc,
                    )
                else:
                    if env_creds_partial:
                        log.warning(
                            "[LIVE] Incomplete POLY_API_* credentials; API auto-derive also failed (%s).",
                            exc,
                        )
                    raise

            self._OrderArgs = OrderArgs
            self._MarketOrderArgs = MarketOrderArgs
            self._OrderType = OrderType
            self._taker_order_type = self._resolve_taker_order_type()
            self._sdk_ready = True
            log.info(
                "[LIVE] ClobClient ready — signature_type=%s | signer=%s | funder=%s | host=%s | chain=%s | taker=%s | api_creds=%s | transport=%s",
                self._signature_type_label(self._signature_type),
                self._signer_address,
                self._funder_address,
                config.CLOB_HOST.replace("https://", ""),
                config.CHAIN_ID,
                self._taker_order_type,
                self._api_creds_source,
                self._sdk_transport_mode,
            )
        except ImportError:
            self._sdk_ready = False
            log.error(
                "py_clob_client not installed. Run `pip install py_clob_client` or use DRY_RUN."
            )
        except Exception as exc:
            self._sdk_ready = False
            log.error("[LIVE] Failed to initialize CLOB client: %s", exc)
            raise RuntimeError(
                f"LIVE FATAL: API/client initialization failed — {exc}"
            ) from exc

    def _patch_py_clob_http_transport(self) -> None:
        try:
            import httpx
            from py_clob_client.http_helpers import helpers as http_helpers

            previous_client = getattr(http_helpers, "_http_client", None)
            timeout = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=10.0)
            limits = httpx.Limits(max_connections=20, max_keepalive_connections=5)
            http_helpers._http_client = httpx.Client(
                http2=False,
                timeout=timeout,
                limits=limits,
            )
            if previous_client is not None:
                try:
                    previous_client.close()
                except Exception:
                    pass
            self._sdk_transport_mode = "http1-serialized"
            log.info("[LIVE] Patched py_clob_client transport to HTTP/1.1 for CLOB stability.")
        except Exception as exc:
            self._sdk_transport_mode = f"patch-failed:{exc}"
            log.warning("[LIVE] Failed to patch py_clob_client transport: %s", exc)

    # ----------------------------------------------------------------
    # FIX #2: Dua SDK call helper — read (concurrent) vs write (serial)
    # ----------------------------------------------------------------
    async def _sdk_read_call(self, fn):
        """Concurrent-safe read — max 3 simultaneous (get_order, get_order_book, dll)."""
        async with self._sdk_read_lock:
            return await asyncio.get_running_loop().run_in_executor(None, fn)

    async def _sdk_write_call(self, fn):
        """Strictly serial write — satu per satu (post_order, create_and_post_order)."""
        async with self._sdk_write_lock:
            return await asyncio.get_running_loop().run_in_executor(None, fn)

    @staticmethod
    def _signature_type_label(signature_type: int) -> str:
        return {
            0: "EOA(0)",
            1: "POLYGON_PROXY(1)",
            2: "BROWSER_PROXY(2)",
        }.get(signature_type, f"UNKNOWN({signature_type})")

    @staticmethod
    def _is_valid_address(value: str) -> bool:
        return isinstance(value, str) and len(value) == 42 and value.startswith("0x")

    def _resolve_funder_address(self, signer_address: str) -> str:
        if self._signature_type in {1, 2}:
            if self._is_valid_address(config.FUNDER):
                return config.FUNDER
            raise RuntimeError(
                "LIVE FATAL: proxy wallet mode requires POLY_FUNDER to be set to the funded address."
            )
        if self._is_valid_address(config.FUNDER):
            return config.FUNDER
        if self._is_valid_address(config.RELAYER_API_KEY_ADDRESS):
            return config.RELAYER_API_KEY_ADDRESS
        log.warning(
            "[LIVE] POLY_FUNDER is unset; using signer address as funder (%s). Set POLY_FUNDER if your funded wallet is different.",
            signer_address,
        )
        return signer_address

    async def _fetch_balance_from_clob(self, loop: asyncio.AbstractEventLoop) -> Optional[float]:
        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp = await self._sdk_read_call(lambda: self._client.get_balance_allowance(params))
            raw = resp.get("balance") or resp.get("usdc") or resp.get("USDC")
            if raw is None:
                return None
            balance = round(float(raw) / 1_000_000, 2)
            log.info("[LIVE] CLOB balance_allowance balance: $%.2f", balance)
            return balance
        except Exception as exc:
            log.warning("[LIVE] CLOB balance_allowance failed (%s) — trying Web3 fallback.", exc)
            return None

    async def sync_live_balance(self) -> None:
        if not self._sdk_ready:
            raise RuntimeError("LIVE FATAL: py_clob_client SDK is not ready.")

        loop = asyncio.get_running_loop()
        balance: Optional[float] = await self._fetch_balance_from_clob(loop)

        if balance is None:
            balance = await self._run_balance_radar(loop)

        if balance is None:
            raise RuntimeError(
                "LIVE FATAL: Both CLOB API and Web3 radar failed to return a balance."
            )

        async with self._lock:
            old = self._available_balance
            self._available_balance = balance

        log.info("[LIVE] _available_balance: $%.2f → $%.2f", old, balance)

    async def _get_working_web3(self, loop: asyncio.AbstractEventLoop):
        try:
            from web3 import Web3
        except ImportError:
            log.error("[RADAR] web3 not installed.")
            return None

        for rpc_url in POLYGON_RPCS:
            try:
                candidate = Web3(
                    Web3.HTTPProvider(
                        rpc_url,
                        request_kwargs={"headers": _WEB3_HEADERS, "timeout": 10},
                    )
                )
                await loop.run_in_executor(None, lambda: candidate.eth.block_number)
                return candidate
            except Exception as exc:
                log.warning("[RADAR] RPC failed %s: %s", rpc_url, exc)
        return None

    async def _redeem_live_positions(self, condition_id: str) -> tuple[bool, str, str]:
        try:
            from eth_account import Account
            from py_clob_client.config import get_contract_config
            from web3 import Web3
        except ImportError as exc:
            return False, "", f"missing live settlement dependency: {exc}"

        loop = asyncio.get_running_loop()
        w3 = await self._get_working_web3(loop)
        if w3 is None:
            return False, "", "all Polygon RPC endpoints failed"

        try:
            contracts = get_contract_config(config.CHAIN_ID)
            signer = Account.from_key(config.PRIVATE_KEY)
            checksum_signer = Web3.to_checksum_address(signer.address)
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(contracts.conditional_tokens),
                abi=_CTF_REDEEM_ABI,
            )
            tx = contract.functions.redeemPositions(
                Web3.to_checksum_address(contracts.collateral),
                _ZERO_BYTES32,
                Web3.to_bytes(hexstr=condition_id),
                [1, 2],
            ).build_transaction({
                "from": checksum_signer,
                "nonce": w3.eth.get_transaction_count(checksum_signer, "pending"),
                "chainId": config.CHAIN_ID,
            })
            gas_estimate = await loop.run_in_executor(None, lambda: w3.eth.estimate_gas(tx))
            tx["gas"] = math.ceil(gas_estimate * 1.2)

            fee = w3.eth.gas_price
            tx["maxFeePerGas"] = fee * 2
            tx["maxPriorityFeePerGas"] = fee

            signed = signer.sign_transaction(tx)
            tx_hash = await loop.run_in_executor(None, lambda: w3.eth.send_raw_transaction(signed.raw_transaction))
            receipt = await loop.run_in_executor(None, lambda: w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120))
            tx_hash_hex = tx_hash.hex()
            if receipt.status != 1:
                return False, tx_hash_hex, "redeemPositions transaction reverted"
            return True, tx_hash_hex, ""
        except Exception as exc:
            detail = str(exc)
            if "result for condition not received yet" in detail.lower():
                return False, "", "result for condition not received yet"
            return False, "", detail

    async def _run_balance_radar(self, loop: asyncio.AbstractEventLoop) -> Optional[float]:
        try:
            from eth_account import Account
            from web3 import Web3
        except ImportError:
            log.error("[RADAR] web3 / eth_account not installed.")
            return None

        usdc_e = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        erc20_abi = [{
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }]

        eoa_address = Account.from_key(config.PRIVATE_KEY).address
        proxy_address = getattr(self, "_funder_address", "") or config.FUNDER or config.RELAYER_API_KEY_ADDRESS or eoa_address

        w3 = None
        for rpc_url in POLYGON_RPCS:
            try:
                candidate = Web3(
                    Web3.HTTPProvider(
                        rpc_url,
                        request_kwargs={"headers": _WEB3_HEADERS, "timeout": 10},
                    )
                )
                await loop.run_in_executor(None, lambda: candidate.eth.block_number)
                w3 = candidate
                break
            except Exception as exc:
                log.warning("[RADAR] RPC failed %s: %s", rpc_url, exc)

        if w3 is None:
            log.error("[RADAR] All RPCs failed.")
            return None

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(usdc_e),
            abi=erc20_abi,
        )

        def get_balance(addr: str) -> float:
            raw = contract.functions.balanceOf(Web3.to_checksum_address(addr)).call()
            return round(raw / 1_000_000, 2)

        try:
            eoa_usdc_e = await loop.run_in_executor(None, lambda: get_balance(eoa_address))
            proxy_usdc_e = await loop.run_in_executor(None, lambda: get_balance(proxy_address))
        except Exception as exc:
            log.error("[RADAR] Balance fetch failed: %s", exc)
            return None

        log.info(
            "WEALTH RADAR | EOA=%s usdc=$%.2f | Proxy=%s usdc=$%.2f",
            eoa_address,
            eoa_usdc_e,
            proxy_address,
            proxy_usdc_e,
        )

        if proxy_usdc_e == 0:
            if eoa_usdc_e > 0:
                log.warning(
                    "[RADAR] $%.2f ada di EOA, bukan di Proxy Polymarket.",
                    eoa_usdc_e,
                )
            else:
                log.warning("[RADAR] Tidak ada saldo USDC.e terdeteksi.")

        return proxy_usdc_e if proxy_usdc_e > 0 else None

    async def _get_tick_size(self, token_id: str) -> float:
        cached = self._tick_size_cache.get(token_id)
        if cached is not None:
            return cached

        try:
            resp = await self._sdk_read_call(lambda: self._client.get_tick_size(token_id))
            tick = float(resp) if resp else 0.01
            tick = tick if tick > 0 else 0.01
            self._tick_size_cache[token_id] = tick
            return tick
        except Exception as exc:
            log.debug(
                "[LIVE] get_tick_size failed for %s: %s — defaulting to 0.01",
                token_id[:16],
                exc,
            )
            return 0.01

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_taker_order_type(self):
        requested = config.TAKER_ORDER_TYPE
        if requested == "IOC":
            requested = "FAK"
        return getattr(self._OrderType, requested, self._OrderType.FAK)

    async def _round_price(self, token_id: str, price: float, *, aggressive: bool) -> float:
        tick = await self._get_tick_size(token_id)
        if aggressive:
            rounded_ticks = math.ceil(price / tick)
        else:
            rounded_ticks = round(price / tick)
        rounded = round(rounded_ticks * tick, 4)
        return max(tick, min(1.0 - tick, rounded))

    def _minimum_notional(self, price: float) -> float:
        return config.minimum_taker_order_usdc(price)

    def _quantize_usdc(self, value: float) -> float:
        if value <= 0:
            return 0.0
        return math.floor((value * 100.0) + 1e-9) / 100.0

    def _quantize_taker_shares(self, shares: float) -> float:
        if shares <= 0:
            return 0.0
        return math.floor((shares * 10_000.0) + 1e-9) / 10_000.0

    @staticmethod
    def _order_book_cache_ttl(cache_state: str) -> float:
        if cache_state == "missing":
            return config.LIVE_ORDERBOOK_MISSING_CACHE_TTL_SECONDS
        if cache_state == "error":
            return config.LIVE_ORDERBOOK_ERROR_CACHE_TTL_SECONDS
        return config.LIVE_ORDERBOOK_CACHE_TTL_SECONDS

    def _cache_order_book(
        self,
        token_id: str,
        book: Optional[OrderBookSnapshot],
        *,
        cache_state: str,
        cached_at: float,
    ) -> None:
        self._order_book_cache[token_id] = (book, cached_at, cache_state)

    def _cached_order_book(
        self,
        token_id: str,
        *,
        now: float,
    ) -> tuple[bool, Optional[OrderBookSnapshot]]:
        cached = self._order_book_cache.get(token_id)
        if cached is None:
            return False, None
        book, cached_at, cache_state = cached
        ttl = self._order_book_cache_ttl(cache_state)
        if now - cached_at <= ttl:
            return True, book
        return False, book

    def _stale_live_order_book(
        self,
        token_id: str,
        *,
        now: float,
    ) -> Optional[OrderBookSnapshot]:
        cached = self._order_book_cache.get(token_id)
        if cached is None:
            return None
        book, cached_at, cache_state = cached
        if book is None or cache_state != "ok":
            return None
        if now - cached_at <= config.LIVE_ORDERBOOK_STALE_GRACE_SECONDS:
            return book
        return None

    def _validate_buy_feasibility(self, spend: float, price_hint: float) -> Optional[str]:
        if price_hint <= 0:
            return "price hint is not available"
        min_spend = self._minimum_notional(price_hint)
        if spend + 1e-9 < min_spend:
            return f"size ${spend:.2f} is below taker minimum ${min_spend:.2f}"
        est_shares = spend / price_hint
        if est_shares + 1e-9 < config.MIN_ORDER_SHARES:
            return (
                f"estimated shares {est_shares:.4f} < minimum {config.MIN_ORDER_SHARES:.2f}; "
                f"needs >= ${min_spend:.2f} at ask {price_hint:.4f}"
            )
        return None

    def _validate_sell_feasibility(self, shares: float, price_hint: float) -> Optional[str]:
        if shares <= 0:
            return "share size is zero"
        if price_hint <= 0:
            return "price hint is not available"
        return None

    async def _submit_market_buy_order(
        self,
        *,
        token_id: str,
        amount_usdc: float,
        price_hint: float,
        fallback_price: float,
    ) -> tuple[dict, str]:
        self._ensure_builder_funder()
        spend = self._quantize_usdc(amount_usdc)
        price_value = round(price_hint if price_hint > 0 else fallback_price, 4)
        if spend <= 0:
            raise RuntimeError("rounded buy amount is zero after CLOB precision clamp")

        # Gunakan _sdk_write_call — order submission harus serial
        signed_order = await self._sdk_write_call(
            lambda: self._client.create_market_order(
                self._MarketOrderArgs(
                    token_id=token_id,
                    amount=spend,
                    side="BUY",
                    price=price_value,
                    fee_rate_bps=0,
                    order_type=self._taker_order_type,
                )
            )
        )
        response = await self._sdk_write_call(
            lambda: self._client.post_order(signed_order, self._taker_order_type)
        )
        order_id = ""
        if isinstance(response, dict):
            order_id = str(response.get("orderID") or response.get("orderId") or "")
        return response if isinstance(response, dict) else {}, order_id

    async def _submit_market_sell_order(
        self,
        *,
        token_id: str,
        shares: float,
        price_hint: float,
        fallback_price: float,
    ) -> tuple[dict, str]:
        self._ensure_builder_funder()
        sell_shares = self._quantize_taker_shares(shares)
        price_value = round(price_hint if price_hint > 0 else fallback_price, 4)
        if sell_shares <= 0:
            raise RuntimeError("rounded sell size is zero after CLOB precision clamp")

        # Gunakan _sdk_write_call — order submission harus serial
        signed_order = await self._sdk_write_call(
            lambda: self._client.create_market_order(
                self._MarketOrderArgs(
                    token_id=token_id,
                    amount=sell_shares,
                    side="SELL",
                    price=price_value,
                    fee_rate_bps=0,
                    order_type=self._taker_order_type,
                )
            )
        )
        response = await self._sdk_write_call(
            lambda: self._client.post_order(signed_order, self._taker_order_type)
        )
        order_id = ""
        if isinstance(response, dict):
            order_id = str(response.get("orderID") or response.get("orderId") or "")
        return response if isinstance(response, dict) else {}, order_id

    def _log_preflight(
        self,
        *,
        market_label: str,
        side: Side,
        token_id: str,
        requested_price: float,
        rounded_price: float,
        tick: float,
        size_usdc: float,
        size_shares: float,
        phase: str,
    ) -> None:
        pre_flight = (
            f"\n[ORDER PRE-FLIGHT] {'='*40}\n"
            f"  Market    : {market_label}\n"
            f"  Side      : {side.value}\n"
            f"  token_id  : {token_id}\n"
            f"  price     : {requested_price:.4f}  →  rounded={rounded_price:.4f}  (tick={tick})\n"
            f"  size_usdc : ${size_usdc:.4f}\n"
            f"  size_share: {size_shares:.4f}\n"
            f"  phase     : {phase}\n"
            f"{'='*50}"
        )
        log.debug(pre_flight)

    def _ensure_builder_funder(self) -> None:
        builder = getattr(self._client, "builder", None)
        if builder is None:
            return

        builder_funder = getattr(builder, "funder", "")
        if not self._is_valid_address(str(builder_funder)):
            log.warning(
                "[LIVE] builder.funder is invalid (%s) — overriding with %s.",
                builder_funder,
                self._funder_address,
            )
            builder.funder = self._funder_address
        elif str(builder_funder).lower() != self._funder_address.lower():
            log.warning(
                "[LIVE] builder.funder mismatch (%s != %s) — overriding.",
                builder_funder,
                self._funder_address,
            )
            builder.funder = self._funder_address

    async def place_limit_order(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        price: float,
        size_usdc: float,
        phase: str,
    ) -> Optional[Order]:
        if not self._sdk_ready or not token_id or size_usdc < config.PASSIVE_MIN_ORDER_USDC:
            return None

        tick = await self._get_tick_size(token_id)
        price_rounded = await self._round_price(token_id, price, aggressive=False)
        size_shares = round(size_usdc / price_rounded, 4) if price_rounded > 0 else 0.0
        if size_shares < config.MIN_ORDER_SHARES:
            log.warning(
                "[LIVE] Skip passive order %s %s — shares %.4f below minimum %.2f",
                market_label,
                side.value,
                size_shares,
                config.MIN_ORDER_SHARES,
            )
            return None

        self._log_preflight(
            market_label=market_label,
            side=side,
            token_id=token_id,
            requested_price=price,
            rounded_price=price_rounded,
            tick=tick,
            size_usdc=size_usdc,
            size_shares=size_shares,
            phase=phase,
        )
        self._ensure_builder_funder()

        try:
            # Limit order placement — write call
            response = await self._sdk_write_call(
                lambda: self._client.create_and_post_order(
                    self._OrderArgs(
                        token_id=token_id,
                        price=price_rounded,
                        size=size_shares,
                        side="BUY",
                        fee_rate_bps=0,
                    )
                )
            )
        except Exception as exc:
            exc_str = repr(exc).lower()
            if "does not exist" in exc_str or "orderbook" in exc_str:
                raise MarketExpiredError(
                    f"{market_label} {side.value} — orderbook does not exist"
                ) from exc
            log.error("[LIVE] Limit order failed — %s %s | %s", market_label, side.value, exc)
            return None

        order_id = (response or {}).get("orderID", "")
        if not order_id:
            log.error(
                "[LIVE] Rejected (no orderID) — %s %s $%.4f @ %.4f | response=%s",
                market_label,
                side.value,
                size_usdc,
                price_rounded,
                response,
            )
            return None

        async with self._lock:
            if not self._reserve_locked(size_usdc):
                return None
            order = self._create_order_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=token_id,
                side=side,
                price=price_rounded,
                size_usdc=size_usdc,
                phase=phase,
                order_id=order_id,
            )
            self._track_order_placement_locked(order)
            return replace(order)

    async def place_dual_limit_orders(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        up_token_id: str,
        down_token_id: str,
        up_price: float,
        down_price: float,
        size_per_side: float,
        phase: str,
    ) -> List[Order]:
        orders: List[Order] = []

        up_order = await self.place_limit_order(
            condition_id=condition_id,
            market_label=market_label,
            asset=asset,
            timeframe=timeframe,
            token_id=up_token_id,
            side=Side.UP,
            price=up_price,
            size_usdc=size_per_side,
            phase=phase,
        )
        if up_order:
            orders.append(up_order)

        down_order = await self.place_limit_order(
            condition_id=condition_id,
            market_label=market_label,
            asset=asset,
            timeframe=timeframe,
            token_id=down_token_id,
            side=Side.DOWN,
            price=down_price,
            size_usdc=size_per_side,
            phase=phase,
        )
        if down_order:
            orders.append(down_order)

        if len(orders) != 2:
            for order in orders:
                await self.cancel_order(order.order_id)
            return []

        return orders

    async def get_order_book(
        self,
        token_id: str,
        *,
        fallback_best_ask: float = 0.0,
        fallback_best_bid: float = 0.0,
    ) -> Optional[OrderBookSnapshot]:
        if not self._sdk_ready or not token_id:
            return None

        now = time.time()
        cache_hit, cached_book = self._cached_order_book(token_id, now=now)
        if cache_hit:
            return cached_book

        try:
            # Read call — bisa concurrent
            summary = await self._sdk_read_call(lambda: self._client.get_order_book(token_id))
        except Exception as exc:
            exc_str = repr(exc).lower()
            if (
                "404" in exc_str
                or "no orderbook exists" in exc_str
                or "does not exist" in exc_str
            ):
                self._cache_order_book(
                    token_id,
                    None,
                    cache_state="missing",
                    cached_at=now,
                )
                log.debug("Order book missing for %s: %s", token_id[:16], exc)
                return None

            stale_book = self._stale_live_order_book(token_id, now=now)
            if stale_book is not None:
                log.debug(
                    "Order book fetch failed for %s: %s — reusing cached book",
                    token_id[:16],
                    exc,
                )
                return stale_book

            self._cache_order_book(
                token_id,
                None,
                cache_state="error",
                cached_at=now,
            )
            log.debug("Order book fetch failed for %s: %s", token_id[:16], exc)
            return None

        bids = [
            BookLevel(price=float(level.price), size=float(level.size))
            for level in getattr(summary, "bids", []) or []
        ]
        asks = [
            BookLevel(price=float(level.price), size=float(level.size))
            for level in getattr(summary, "asks", []) or []
        ]
        tick_size = self._safe_float(getattr(summary, "tick_size", 0.01), default=0.01)
        last_trade = self._safe_float(
            getattr(summary, "last_trade_price", 0.0),
            default=fallback_best_bid or fallback_best_ask,
        )
        if not asks and not bids:
            self._cache_order_book(
                token_id,
                None,
                cache_state="missing",
                cached_at=now,
            )
            return None

        self._tick_size_cache[token_id] = tick_size
        book = OrderBookSnapshot(
            token_id=token_id,
            bids=bids,
            asks=asks,
            tick_size=tick_size,
            last_trade_price=last_trade,
        )
        self._cache_order_book(
            token_id,
            book,
            cache_state="ok",
            cached_at=now,
        )
        return book

    async def execute_taker_buy(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        phase: str,
        aggressive_price: float,
        expected_fill_price: float,
        target_shares: float,
        max_size_usdc: float,
    ) -> Optional[Fill]:
        if not self._sdk_ready or not token_id or target_shares < config.MIN_ORDER_SHARES:
            return None

        quantized_shares = self._quantize_taker_shares(target_shares)
        if quantized_shares < config.MIN_ORDER_SHARES:
            return None

        price_rounded = await self._round_price(token_id, aggressive_price, aggressive=True)
        tick = self._tick_size_cache.get(token_id, 0.01)
        reserve_size = self._quantize_usdc(
            max(
                max_size_usdc,
                quantized_shares * price_rounded,
                self._minimum_notional(price_rounded),
            )
        )
        reason = self._validate_buy_feasibility(reserve_size, price_rounded)
        if reason is not None:
            log.warning("[LIVE] Skip taker order %s %s — %s", market_label, side.value, reason)
            return None

        live_book = await self.get_order_book(token_id)
        if live_book is None or live_book.best_ask <= 0 or live_book.best_ask_size <= 0:
            log.warning("[LIVE] Skip taker order %s %s — live order book unavailable", market_label, side.value)
            return None

        self._log_preflight(
            market_label=market_label,
            side=side,
            token_id=token_id,
            requested_price=aggressive_price,
            rounded_price=price_rounded,
            tick=tick,
            size_usdc=reserve_size,
            size_shares=quantized_shares,
            phase=phase,
        )

        async with self._lock:
            if not self._reserve_locked(reserve_size):
                return None
            order = self._create_order_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=token_id,
                side=side,
                price=price_rounded,
                size_usdc=reserve_size,
                phase=phase,
            )
            self._track_order_placement_locked(order)

        try:
            response, posted_order_id = await self._submit_market_buy_order(
                token_id=token_id,
                amount_usdc=reserve_size,
                price_hint=price_rounded,
                fallback_price=expected_fill_price if expected_fill_price > 0 else price_rounded,
            )
        except Exception as exc:
            exc_str = repr(exc).lower()
            log.error("[LIVE] Taker order failed — %s %s | %s", market_label, side.value, exc)
            async with self._lock:
                live_order = self._orders.get(order.order_id)
                if live_order:
                    self._cancel_order_locked(live_order)
            if "does not exist" in exc_str or "orderbook" in exc_str:
                raise MarketExpiredError(
                    f"{market_label} {side.value} — orderbook does not exist"
                ) from exc
            return None

        if posted_order_id:
            async with self._lock:
                live_order = self._orders.get(order.order_id)
                if live_order:
                    del self._orders[live_order.order_id]
                    live_order.order_id = posted_order_id
                    self._orders[posted_order_id] = live_order

        fill = await self._resolve_live_taker_fill(
            order_id=posted_order_id or order.order_id,
            fallback_fill_price=expected_fill_price if expected_fill_price > 0 else price_rounded,
            fallback_shares=quantized_shares,
        )
        if fill is None:
            async with self._lock:
                live_order = self._orders.get(posted_order_id or order.order_id)
                if live_order and live_order.status == OrderStatus.OPEN:
                    self._cancel_order_locked(live_order)
            return None

        async with self._lock:
            live_order = self._orders.get(posted_order_id or order.order_id)
            if not live_order:
                return None
            result = self._apply_custom_fill_locked(
                live_order,
                fill_price=fill["price"],
                fill_size_usdc=fill["cost"],
                shares=fill["shares"],
                action="Taker Fill",
            )

        if result is not None:
            log.info(
                "[EXEC][LIVE] action=BUY | market=%s | side=%s | phase=%s"
                " | shares=%.4f | price=%.4f | cost=$%.2f",
                market_label, side.value, phase, result.shares, result.price, result.size,
            )
        return result

    async def execute_taker_sell(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        phase: str,
        target_shares: float,
        expected_fill_price: float,
    ) -> Optional[Fill]:
        if not self._sdk_ready or not token_id or target_shares <= 0:
            return None

        quantized_shares = self._quantize_taker_shares(target_shares)
        if quantized_shares <= 0:
            return None

        live_book = await self.get_order_book(token_id)
        if live_book is None or live_book.best_bid <= 0 or live_book.best_bid_size <= 0:
            log.warning("[LIVE] Skip taker sell %s %s — live order book unavailable", market_label, side.value)
            return None

        price_hint = live_book.best_bid
        reason = self._validate_sell_feasibility(quantized_shares, price_hint)
        if reason is not None:
            log.warning("[LIVE] Skip taker sell %s %s — %s", market_label, side.value, reason)
            return None

        try:
            _, posted_order_id = await self._submit_market_sell_order(
                token_id=token_id,
                shares=quantized_shares,
                price_hint=price_hint,
                fallback_price=expected_fill_price if expected_fill_price > 0 else price_hint,
            )
        except Exception as exc:
            exc_str = repr(exc).lower()
            log.error("[LIVE] Taker sell failed — %s %s | %s", market_label, side.value, exc)
            if "does not exist" in exc_str or "orderbook" in exc_str:
                raise MarketExpiredError(
                    f"{market_label} {side.value} — orderbook does not exist"
                ) from exc
            return None

        fill = await self._resolve_live_taker_fill(
            order_id=posted_order_id,
            fallback_fill_price=expected_fill_price if expected_fill_price > 0 else price_hint,
            fallback_shares=quantized_shares,
        )
        if fill is None:
            return None

        async with self._lock:
            result = self._apply_sell_fill_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                token_id=token_id,
                side=side,
                phase=phase,
                fill_price=fill["price"],
                shares=fill["shares"],
                action="Taker Sell",
                order_id=posted_order_id or None,
            )

        if result is not None:
            log.info(
                "[EXEC][LIVE] action=SELL | market=%s | side=%s | phase=%s"
                " | shares=%.4f | price=%.4f | cost=$%.2f",
                market_label, side.value, phase, result.shares, result.price, result.size,
            )
        return result

    async def _resolve_live_taker_fill(
        self,
        *,
        order_id: str,
        fallback_fill_price: float,
        fallback_shares: float,
        # ----------------------------------------------------------------
        # FIX #1: Naikkan timeout dari 1.0s → 3.5s
        # Polymarket CLOB butuh waktu lebih untuk mengkonfirmasi fill,
        # terutama saat network congestion. 1s terlalu agresif dan
        # menyebabkan false-negative → bot cancel order yang sebenarnya filled.
        # ----------------------------------------------------------------
        timeout: float = 3.5,
    ) -> Optional[dict]:
        if not order_id:
            return None

        deadline = time.time() + timeout
        last_payload: dict | None = None
        poll_interval = 0.15  # sedikit lebih lambat dari 0.1 untuk kurangi SDK load

        while time.time() < deadline:
            try:
                # Read call — bisa concurrent dengan operasi lain
                payload = await self._sdk_read_call(lambda: self._client.get_order(order_id))
            except Exception as exc:
                log.debug("Order fill lookup failed for %s: %s", order_id[:12], exc)
                await asyncio.sleep(poll_interval)
                continue

            if not isinstance(payload, dict):
                log.debug(
                    "Order fill lookup returned non-dict payload for %s: %r",
                    order_id[:12],
                    payload,
                )
                await asyncio.sleep(poll_interval)
                continue

            last_payload = payload
            matched_shares = self._safe_float(payload.get("size_matched"), default=0.0)
            if matched_shares > 0:
                fill_price = self._safe_float(payload.get("price"), default=fallback_fill_price)
                if fill_price <= 0:
                    fill_price = fallback_fill_price
                log.debug(
                    "[LIVE] Fill confirmed | order=%s | shares=%.4f | price=%.4f",
                    order_id[:12],
                    matched_shares,
                    fill_price,
                )
                return {
                    "shares": matched_shares,
                    "price": fill_price,
                    "cost": matched_shares * fill_price,
                }

            status = str(payload.get("status", "")).upper()
            if status in {"CANCELED", "CANCELLED", "REJECTED"}:
                log.warning(
                    "[LIVE] Order %s returned status=%s before fill detected.",
                    order_id[:12],
                    status,
                )
                break
            await asyncio.sleep(poll_interval)

        # Satu pengecekan terakhir setelah timeout
        if isinstance(last_payload, dict):
            matched_shares = self._safe_float(last_payload.get("size_matched"), default=0.0)
            if matched_shares > 0:
                fill_price = self._safe_float(last_payload.get("price"), default=fallback_fill_price)
                if fill_price <= 0:
                    fill_price = fallback_fill_price
                log.warning(
                    "[LIVE] Fill recovered at deadline | order=%s | shares=%.4f",
                    order_id[:12],
                    matched_shares,
                )
                return {
                    "shares": matched_shares,
                    "price": fill_price,
                    "cost": matched_shares * fill_price,
                }

        log.warning(
            "[LIVE] Fill not confirmed after %.1fs | order=%s | last_status=%s",
            timeout,
            order_id[:12],
            str(last_payload.get("status", "unknown")).upper() if last_payload else "no-response",
        )
        return None

    async def execute_sniper(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        side: Side,
        size_usdc: float,
        current_best_ask: float,
    ) -> Optional[Fill]:
        if current_best_ask <= 0:
            return None
        target_shares = size_usdc / current_best_ask
        return await self.execute_taker_buy(
            condition_id=condition_id,
            market_label=market_label,
            asset=asset,
            timeframe=timeframe,
            token_id=token_id,
            side=side,
            phase="PHASE2",
            aggressive_price=config.SNIPER_LIMIT_PRICE,
            expected_fill_price=current_best_ask,
            target_shares=target_shares,
            max_size_usdc=size_usdc,
        )

    async def cancel_order(self, order_id: str) -> bool:
        if not self._sdk_ready:
            return False

        try:
            await self._sdk_write_call(lambda: self._client.cancel(order_id))
        except Exception as exc:
            log.error("Cancel failed %s: %s", order_id, exc)
            return False

        async with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return False
            return self._cancel_order_locked(order)

    async def cancel_all_for_market(self, condition_id: str) -> int:
        order_ids = [order.order_id for order in self.get_open_orders(condition_id)]
        if not order_ids:
            return 0
        results = await asyncio.gather(*[self.cancel_order(order_id) for order_id in order_ids])
        return sum(1 for result in results if result)

    async def process_limit_crosses(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        best_ask_up: float,
        best_ask_down: float,
    ) -> List[Fill]:
        fills: List[Fill] = []
        open_orders = [
            order
            for order in self.get_open_orders(condition_id)
            if order.phase in LIMIT_FILL_PHASES
        ]
        if not open_orders:
            return fills

        for order in open_orders:
            try:
                # Read call — concurrent OK
                payload = await self._sdk_read_call(lambda oid=order.order_id: self._client.get_order(oid))
            except Exception as exc:
                log.debug("Live limit fill lookup failed for %s: %s", order.order_id[:12], exc)
                continue

            if not isinstance(payload, dict):
                continue

            status = str(payload.get("status", "")).upper()
            matched_shares = self._safe_float(payload.get("size_matched"), default=0.0)
            fill_price = self._safe_float(payload.get("price"), default=order.price)
            if fill_price <= 0:
                fill_price = order.price

            if matched_shares > 0:
                async with self._lock:
                    live_order = self._orders.get(order.order_id)
                    if live_order is None or live_order.status != OrderStatus.OPEN:
                        continue
                    fills.append(
                        self._apply_custom_fill_locked(
                            live_order,
                            fill_price=fill_price,
                            fill_size_usdc=matched_shares * fill_price,
                            shares=matched_shares,
                            action="Limit Fill",
                        )
                    )
                continue

            if status in {"CANCELED", "CANCELLED", "REJECTED"}:
                async with self._lock:
                    live_order = self._orders.get(order.order_id)
                    if live_order is not None and live_order.status == OrderStatus.OPEN:
                        self._cancel_order_locked(live_order)

        return fills

    async def settle_market(
        self,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        winning_side: Optional[Side],
    ) -> SettlementResult:
        should_claim = False
        async with self._lock:
            positions = self._active_positions.get(condition_id, {})
            should_claim = any(position.shares > 0 and position.cost_basis > 0 for position in positions.values())

        claim_tx_hash = ""
        if should_claim:
            claimed, claim_tx_hash, error = await self._redeem_live_positions(condition_id)
            if not claimed:
                log.warning(
                    "[LIVE] Settlement pending | %s | condition=%s | reason=%s",
                    market_label,
                    condition_id[:10],
                    error,
                )
                return SettlementResult(
                    condition_id=condition_id,
                    market_label=market_label,
                    winning_side=winning_side.value if winning_side else None,
                    payout=0.0,
                    realized_pnl=0.0,
                    released_orders=0,
                    settled=False,
                    error=error,
                    claim_tx_hash=claim_tx_hash,
                )

            log.info(
                "[LIVE] Claim confirmed | %s | condition=%s | tx=%s",
                market_label,
                condition_id[:10],
                claim_tx_hash,
            )

        async with self._lock:
            result = self._settle_locked(
                condition_id=condition_id,
                market_label=market_label,
                asset=asset,
                timeframe=timeframe,
                winning_side=winning_side,
            )

        if should_claim:
            try:
                await self.sync_live_balance()
            except Exception as exc:
                log.warning("[LIVE] Balance sync after claim failed: %s", exc)

        return SettlementResult(
            condition_id=result.condition_id,
            market_label=result.market_label,
            winning_side=result.winning_side,
            payout=result.payout,
            realized_pnl=result.realized_pnl,
            released_orders=result.released_orders,
            settled=True,
            claim_tx_hash=claim_tx_hash,
        )


def build_order_manager() -> SimulatorOrderManager | LiveOrderManager:
    if config.DRY_RUN:
        log.info("DRY_RUN mode — using SimulatorOrderManager.")
        return SimulatorOrderManager()
    log.info("LIVE mode — using LiveOrderManager.")
    return LiveOrderManager()
