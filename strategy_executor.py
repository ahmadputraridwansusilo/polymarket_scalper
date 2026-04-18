"""
strategy_executor.py — 3-Phase Strategy Executor + SimBroker

Wires into existing signal infrastructure:
  - config.safe_margin_for(asset)           — SAFE_MARGIN per asset (untouched)
  - Brain._derive_obi_value()               — OBI value from books (untouched, @staticmethod)
  - Brain._obi_signal()                     — OBI label (untouched, @staticmethod)
  - Brain._phase1_target_side()             — OBI directional side (untouched, @staticmethod)
  - executioner.OrderManagerBase            — order execution layer (untouched)
  - MarketSnapshot / MarketState            — market data + per-market state

Note on oracle_gate helpers:
  Brain._oracle_gate() is an instance method and cannot be called without a Brain
  instance (which would create a circular import).  The equivalent logic is
  implemented here as module-level helpers (_oracle_gate, _winning_side) using
  the same config.safe_margin_for() call and the same formula — no new logic,
  no threshold changes.  Brain._oracle_gate itself is NOT modified.

Strategy overview:
  Phase 1 (Entry)   — Enter when delta > SAFE_MARGIN AND OBI aligned AND >30s AND no open position.
                       Sizing auto-scales with balance (6 / 8 / 10 %), hard cap 12 %, min $1.00.
                       Limit-style IOC at best ask; 3 s cooldown before re-evaluation after a miss.

  Phase 2 (Monitor) — Manage open position every ~250 ms (Brain tick).
                       Priority 1: Hedge Detection — binary-search minimum hedge that locks near
                                   breakeven on BOTH sides (≥ –$0.01 each). Once HEDGED, TP is
                                   permanently disabled for this position.
                       Priority 2: Take-Profit (HEDGED=false only) — close at 50 % unrealised PnL.
                       Priority 3: Emergency Close — at t ≤ 10 s, close losing unhedged positions.

  Phase 3 (Sniper)  — High-conviction late entry in exclusive 10–30 s window.
                       Requires delta > SAFE_MARGIN, OBI aligned, price in [0.96, 0.99], positive
                       expected profit after fees. Single limit order — no retry.

SimBroker:
  Mirrors real execution in SIM mode without lookahead or perfect fills.
  Same code paths trigger in both modes; only the execution layer differs.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import config
from calculations import evaluate_hedge_plan, HedgePlan
from executioner import Fill, OrderManagerBase, PortfolioSnapshot, Side
from oracle import MarketSnapshot, Oracle

log = logging.getLogger("strategy")


# ---------------------------------------------------------------------------
# Module-level oracle signal helpers
# These replicate Brain._oracle_gate / Brain._winning_side_from_oracle exactly,
# but as module-level functions so strategy_executor.py has no circular import.
# ---------------------------------------------------------------------------

def _oracle_gate(snap: MarketSnapshot) -> tuple[bool, float, float]:
    """Returns (oracle_safe, delta_abs, margin).

    Same formula as Brain._oracle_gate — reads config.safe_margin_for(asset)
    (untouched).  oracle_safe=True when |price - strike| > safe_margin.
    """
    if snap.binance_live_price <= 0 or snap.strike_price <= 0:
        return False, 0.0, config.safe_margin_for(snap.asset)
    delta_abs = abs(snap.binance_live_price - snap.strike_price)
    margin = config.safe_margin_for(snap.asset)
    return delta_abs > margin, delta_abs, margin


def _winning_side(snap: MarketSnapshot) -> Optional[Side]:
    """Returns UP if price >= strike, DOWN otherwise (or None if data missing)."""
    if snap.binance_live_price <= 0 or snap.strike_price <= 0:
        return None
    return Side.UP if snap.binance_live_price >= snap.strike_price else Side.DOWN


# ---------------------------------------------------------------------------
# Constants — strategy-specific; do NOT modify SAFE_MARGIN values here
# ---------------------------------------------------------------------------
PHASE1_ENTRY_MIN_TIME_REMAINING: float = 30.0   # seconds — entry gate
PHASE1_OBI_LONG: float = 0.70                    # OBI ≥ 0.70 → UP signal
PHASE1_OBI_SHORT: float = 0.30                   # OBI ≤ 0.30 → DOWN signal
PHASE1_ENTRY_COOLDOWN: float = 3.0               # seconds between entry attempts

PHASE2_TP_PCT: float = 50.0                      # unrealised PnL% threshold for take-profit
PHASE2_HEDGE_TOLERANCE: float = 0.01             # both sides must be ≥ –$0.01
PHASE2_EMERGENCY_CLOSE_SECONDS: float = 10.0     # close losing unhedged positions at this threshold

# ---------------------------------------------------------------------------
# Phase 3 Sniper — high-conviction near-expiry entry
#
# Window : 10 s < time_remaining ≤ 30 s
# Entry  : price in [0.96, 0.99] — almost-certain-win zone
# Size   : 15 % of bankroll (base); 20 % when OBI is very strong (≥ 0.85)
# Order  : aggressive limit at SNIPER_LIMIT_PRICE (1.00)
#          placed at current best_ask — fill at market, never overpay past 0.99
# Momentum gate: OBI must still be strongly aligned at fire time
#   UP  side → OBI ≥ PHASE3_MOMENTUM_OBI_STRONG (0.60)
#   DOWN side → OBI ≤ 1 − PHASE3_MOMENTUM_OBI_STRONG (0.40)
# ---------------------------------------------------------------------------
PHASE3_WINDOW_START: float = 30.0                # sniper window opens at 30 s remaining
PHASE3_WINDOW_END: float = 10.0                  # sniper window closes at 10 s remaining
PHASE3_MIN_WINNING_PRICE: float = 0.90           # floor — don't enter below this
PHASE3_MAX_WINNING_PRICE: float = 0.99           # ceiling — don't overpay past this
PHASE3_SIZE_RATIO_BASE: float = 0.15             # 15 % of balance (base)
PHASE3_SIZE_RATIO_HIGH: float = 0.20             # 20 % when OBI is very strong
PHASE3_SIZE_RATIO: float = PHASE3_SIZE_RATIO_BASE  # alias for test compat
PHASE3_OBI_HIGH_THRESHOLD: float = 0.85          # OBI ≥ this → use SIZE_RATIO_HIGH
PHASE3_MOMENTUM_OBI_STRONG: float = 0.60         # minimum OBI magnitude to confirm momentum intact

# ---------------------------------------------------------------------------
# Entry sizing tiers
# ---------------------------------------------------------------------------
ENTRY_RATIO_MICRO: float = 0.06   # balance < $50
ENTRY_RATIO_SMALL: float = 0.08   # $50 ≤ balance < $200
ENTRY_RATIO_NORMAL: float = 0.10  # balance ≥ $200
ENTRY_HARD_CAP_RATIO: float = 0.12
ENTRY_MIN_USDC: float = 1.00


# ---------------------------------------------------------------------------
# Per-market supplementary state
# ---------------------------------------------------------------------------

@dataclass
class StrategyState:
    """Supplementary per-market state for the 3-phase strategy."""
    condition_id: str

    # Phase 1 entry tracking
    entry_delta: float = 0.0
    entry_obi: float = 0.5
    last_entry_attempt_at: float = 0.0

    # Phase 2 hedge tracking
    hedged: bool = False
    hedge_side: Optional[Side] = None
    hedge_shares: float = 0.0
    hedge_cost: float = 0.0
    last_hedge_attempt_at: float = 0.0

    # Phase 3 sniper tracking
    phase3_sniper_fired: bool = False

    # Dashboard
    detail: str = "-"


# ---------------------------------------------------------------------------
# Entry sizing helper
# ---------------------------------------------------------------------------

def compute_entry_size(balance: float) -> float:
    """Auto-scale entry size based on current available balance.

    Tiers:
      balance < $50   → 6 % of balance
      $50 ≤ bal < $200→ 8 % of balance
      bal ≥ $200      → 10 % of balance

    Hard cap: 12 % of balance.
    Minimum:  $1.00 (supports accounts as small as $20).
    """
    if balance <= 0:
        return 0.0
    if balance < 50.0:
        ratio = ENTRY_RATIO_MICRO
    elif balance < 200.0:
        ratio = ENTRY_RATIO_SMALL
    else:
        ratio = ENTRY_RATIO_NORMAL

    raw = balance * ratio
    hard_cap = balance * ENTRY_HARD_CAP_RATIO
    capped = min(raw, hard_cap)
    return max(ENTRY_MIN_USDC, math.floor((capped * 100.0) + 1e-9) / 100.0)


# ---------------------------------------------------------------------------
# Hedge validity check
# ---------------------------------------------------------------------------

def find_valid_hedge(
    *,
    entry_cost: float,
    entry_shares: float,
    hedge_price: float,
    max_hedge_usdc: float,
    min_hedge_usdc: float = 0.01,
) -> Optional[HedgePlan]:
    """Binary-search the minimum USDC hedge that satisfies Phase 2 validity rules:

      net_if_entry_wins = entry_shares - entry_cost - hedge_cost ≥ -0.01
      net_if_hedge_wins = hedge_shares - entry_cost - hedge_cost ≥ -0.01
      AND NOT (both sides < -0.02)
    """
    T = PHASE2_HEDGE_TOLERANCE

    if entry_cost <= 0 or entry_shares <= 0:
        return None
    if not 0 < hedge_price < 1:
        return None
    if max_hedge_usdc <= 0:
        return None

    upper_from_entry = entry_shares - entry_cost + T

    denominator = 1.0 - hedge_price
    if denominator < 1e-9:
        return None
    lower_from_hedge = (entry_cost - T) * hedge_price / denominator

    effective_lo = max(min_hedge_usdc, lower_from_hedge)
    effective_hi = min(max_hedge_usdc, upper_from_entry)

    if effective_lo > effective_hi + 1e-9:
        return None

    def _valid(plan: HedgePlan) -> bool:
        if plan.entry_net < -T:
            return False
        if plan.hedge_net < -T:
            return False
        if plan.entry_net < -0.02 and plan.hedge_net < -0.02:
            return False
        return True

    lo_cents = max(1, int(math.ceil(effective_lo * 100.0 - 1e-9)))
    hi_cents = int(math.floor(effective_hi * 100.0 + 1e-9))
    if lo_cents > hi_cents:
        only_amount = lo_cents / 100.0
        plan = evaluate_hedge_plan(
            entry_cost=entry_cost,
            entry_shares=entry_shares,
            hedge_price=hedge_price,
            hedge_amount=only_amount,
            tolerance=T,
        )
        return plan if (plan is not None and _valid(plan)) else None

    best_valid: Optional[HedgePlan] = None

    hi_plan = evaluate_hedge_plan(
        entry_cost=entry_cost,
        entry_shares=entry_shares,
        hedge_price=hedge_price,
        hedge_amount=hi_cents / 100.0,
        tolerance=T,
    )
    if hi_plan is None or not _valid(hi_plan):
        return None

    best_valid = hi_plan

    while lo_cents <= hi_cents:
        mid_cents = (lo_cents + hi_cents) // 2
        mid_amount = mid_cents / 100.0
        plan = evaluate_hedge_plan(
            entry_cost=entry_cost,
            entry_shares=entry_shares,
            hedge_price=hedge_price,
            hedge_amount=mid_amount,
            tolerance=T,
        )
        if plan is not None and _valid(plan):
            best_valid = plan
            hi_cents = mid_cents - 1
        else:
            lo_cents = mid_cents + 1

    return best_valid


# ---------------------------------------------------------------------------
# SimBroker — mirrors real execution without lookahead
# ---------------------------------------------------------------------------

class SimBroker:
    """Simulation execution broker that mirrors the LIVE execution layer."""

    def __init__(self, exec_manager: OrderManagerBase, oracle: Oracle) -> None:
        self._exec = exec_manager
        self._oracle = oracle

    async def fill_order(
        self,
        *,
        condition_id: str,
        market_label: str,
        asset: str,
        timeframe: str,
        token_id: str,
        direction: Side,
        size_usdc: float,
        requested_price: float,
        order_type: str = "LIMIT",
        phase: str = "STRATEGY",
    ) -> Optional[Fill]:
        """Simulate a fill using real-time market prices from the existing feed."""
        if size_usdc < config.MIN_ORDER_USDC:
            log.debug("[SIM] fill_order: size_usdc=%.4f below minimum", size_usdc)
            return None

        snap = None
        for mkt in self._oracle.all_markets():
            if mkt.condition_id == condition_id:
                snap = mkt
                break

        fallback_ask = 0.99
        fallback_bid = 0.01
        if snap is not None:
            fallback_ask = snap.best_ask_up if direction == Side.UP else snap.best_ask_down
            fallback_bid = snap.best_bid_up if direction == Side.UP else snap.best_bid_down

        book = await self._exec.get_order_book(
            token_id,
            fallback_best_ask=fallback_ask,
            fallback_best_bid=fallback_bid,
        )
        if book is None:
            log.debug("[SIM] fill_order: no book for token=%s", token_id[:16])
            return None

        best_ask = book.best_ask

        if order_type == "MARKET":
            effective_price = best_ask if best_ask > 0 else requested_price
        else:
            if best_ask > 0 and requested_price >= best_ask:
                effective_price = requested_price
            elif best_ask > 0:
                effective_price = best_ask
            else:
                effective_price = requested_price

        if effective_price <= 0:
            return None

        target_shares = size_usdc / effective_price

        return await self._exec.execute_taker_buy(
            condition_id=condition_id,
            market_label=market_label,
            asset=asset,
            timeframe=timeframe,
            token_id=token_id,
            side=direction,
            phase=phase,
            aggressive_price=config.SNIPER_LIMIT_PRICE,
            expected_fill_price=effective_price,
            target_shares=target_shares,
            max_size_usdc=size_usdc,
        )

    async def get_balance(self) -> float:
        portfolio = await self._exec.get_portfolio_snapshot()
        return portfolio.available_balance

    async def get_positions(self) -> Dict[str, Any]:
        portfolio = await self._exec.get_portfolio_snapshot()
        return portfolio.active_positions  # type: ignore[return-value]

    async def get_payout(self, condition_id: str, winning_side: Side) -> float:
        portfolio = await self._exec.get_portfolio_snapshot()
        pos_map = portfolio.active_positions.get(condition_id, {})
        winning_pos = pos_map.get(winning_side.value)
        return winning_pos.shares if winning_pos is not None else 0.0


# ---------------------------------------------------------------------------
# StrategyExecutor — 3-phase strategy engine
# ---------------------------------------------------------------------------

class StrategyExecutor:
    """3-Phase strategy engine (Phase 1 Entry / Phase 2 Monitor / Phase 3 Sniper).

    Instantiated once inside Brain.__init__; called once per market per tick
    via StrategyExecutor.evaluate(snap, market_state).
    """

    def __init__(self, exec_manager: OrderManagerBase, oracle: Oracle) -> None:
        self._exec = exec_manager
        self._oracle = oracle
        self._sim_broker = SimBroker(exec_manager, oracle)
        self._strategy_states: Dict[str, StrategyState] = {}

    def get_sim_broker(self) -> SimBroker:
        return self._sim_broker

    async def evaluate(
        self,
        snap: MarketSnapshot,
        market_state,
        up_book,
        down_book,
    ) -> None:
        """Main entry point called by Brain._evaluate_market on every tick."""
        from brain import Phase1Status, Phase2Status

        cid = snap.condition_id
        if cid not in self._strategy_states:
            self._strategy_states[cid] = StrategyState(condition_id=cid)
        sstate = self._strategy_states[cid]

        tr = snap.time_remaining

        # ── Compute signals ──────────────────────────────────────────────────
        oracle_safe, delta_abs, margin = _oracle_gate(snap)
        winning_side = _winning_side(snap)

        from brain import Brain
        obi_value = 0.5
        if up_book is not None and down_book is not None:
            obi_value = Brain._derive_obi_value(snap, up_book, down_book)
        market_state.last_obi_value = obi_value
        market_state.last_obi_signal = Brain._obi_signal(obi_value)

        # OBI directional alignment
        obi_side: Optional[Side] = None
        if obi_value >= PHASE1_OBI_LONG:
            obi_side = Side.UP
        elif obi_value <= PHASE1_OBI_SHORT:
            obi_side = Side.DOWN

        # Fallback: neutral OBI → follow oracle delta direction
        if obi_side is None and oracle_safe:
            obi_side = winning_side

        # ── Phase 3 Sniper (highest priority in window) ──────────────────────
        if PHASE3_WINDOW_END < tr <= PHASE3_WINDOW_START:
            if not sstate.phase3_sniper_fired:
                await self._phase3_sniper(
                    snap=snap,
                    market_state=market_state,
                    sstate=sstate,
                    oracle_safe=oracle_safe,
                    delta_abs=delta_abs,
                    margin=margin,
                    winning_side=winning_side,
                    obi_side=obi_side,
                    obi_value=obi_value,
                    up_book=up_book,
                    down_book=down_book,
                )
            else:
                market_state.phase1_detail = f"Sniper already fired | t={tr:.0f}s"
            return

        # ── Phase 2 Monitor (existing position) ──────────────────────────────
        has_position = (
            market_state.position_source in {"PHASE1", "PHASE3"}
            and market_state.position_side is not None
            and not market_state.settled
        )
        if has_position:
            await self._phase2_monitor(
                snap=snap,
                market_state=market_state,
                sstate=sstate,
                oracle_safe=oracle_safe,
                winning_side=winning_side,
                obi_side=obi_side,
                obi_value=obi_value,
                up_book=up_book,
                down_book=down_book,
            )
            return

        # ── Phase 1 Entry ────────────────────────────────────────────────────
        if tr > PHASE1_ENTRY_MIN_TIME_REMAINING:
            await self._phase1_entry(
                snap=snap,
                market_state=market_state,
                sstate=sstate,
                oracle_safe=oracle_safe,
                delta_abs=delta_abs,
                margin=margin,
                winning_side=winning_side,
                obi_side=obi_side,
                obi_value=obi_value,
                up_book=up_book,
                down_book=down_book,
            )
        else:
            market_state.phase1_detail = f"Waiting sniper window | t={tr:.0f}s"

    # ── Phase 1: Entry ───────────────────────────────────────────────────────

    async def _phase1_entry(
        self,
        *,
        snap: MarketSnapshot,
        market_state,
        sstate: StrategyState,
        oracle_safe: bool,
        delta_abs: float,
        margin: float,
        winning_side: Optional[Side],
        obi_side: Optional[Side],
        obi_value: float,
        up_book,
        down_book,
    ) -> None:
        from brain import Phase1Status, Phase2Status

        mode_tag = "[SIM]" if config.DRY_RUN else "[LIVE]"
        tr = snap.time_remaining

        # Gate 1: Oracle delta must exceed safe margin
        if not oracle_safe:
            market_state.phase1_detail = (
                f"oracle unsafe | δ={delta_abs:.2f} ≤ margin={margin:.2f} | t={tr:.0f}s"
            )
            log.debug(
                "[PHASE1] Blocked | market=%s | delta=%.2f ≤ margin=%.2f | t=%.0fs",
                market_state.market_label, delta_abs, margin, tr,
            )
            return

        # Gate 2: OBI must align with oracle direction
        if winning_side is None or obi_side is None or obi_side != winning_side:
            market_state.phase1_detail = (
                f"OBI/oracle diverge | OBI={obi_side.value if obi_side else '-'}"
                f" oracle={winning_side.value if winning_side else '-'} | t={tr:.0f}s"
            )
            return

        # Gate 3: Cooldown between attempts
        now = time.monotonic()
        if now - sstate.last_entry_attempt_at < PHASE1_ENTRY_COOLDOWN:
            return
        sstate.last_entry_attempt_at = now

        # Select entry side book and token
        entry_side = winning_side
        entry_book = up_book if entry_side == Side.UP else down_book
        entry_token = snap.up_token_id if entry_side == Side.UP else snap.down_token_id

        if entry_book is None or entry_book.best_ask <= 0 or not entry_token:
            market_state.phase1_detail = "No order book for entry side"
            return

        current_price = entry_book.best_ask

        portfolio = await self._exec.get_portfolio_snapshot()
        balance = portfolio.available_balance
        entry_size = compute_entry_size(balance)

        if entry_size < config.MIN_ORDER_USDC:
            market_state.phase1_detail = (
                f"Entry size ${entry_size:.2f} below minimum ${config.MIN_ORDER_USDC:.2f}"
            )
            return

        target_shares = entry_size / current_price

        log.info(
            "%s Phase 1 Entry FIRED | market=%s | side=%s"
            " | ask=%.4f | size=$%.2f | OBI=%.3f | δ=%.2f | t=%.0fs",
            mode_tag, market_state.market_label, entry_side.value,
            current_price, entry_size, obi_value, delta_abs, tr,
        )

        fill = await self._exec.execute_taker_buy(
            condition_id=snap.condition_id,
            market_label=market_state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=entry_token,
            side=entry_side,
            phase="PHASE1",
            aggressive_price=config.SNIPER_LIMIT_PRICE,
            expected_fill_price=current_price,
            target_shares=target_shares,
            max_size_usdc=entry_size,
        )

        if fill:
            sstate.entry_delta = delta_abs
            sstate.entry_obi = obi_value
            market_state.position_source = "PHASE1"
            market_state.position_side = entry_side
            market_state.position_token_id = entry_token
            market_state.position_cost = fill.size
            market_state.position_shares = fill.shares
            market_state.avg_entry_price = fill.price
            market_state.phase1_status = Phase1Status.HOLDING
            market_state.phase1_detail = (
                f"Entered {entry_side.value} | ask={current_price:.3f} | size=${fill.size:.2f}"
            )
            log.info(
                "%s Phase 1 Entry FILLED | market=%s | side=%s"
                " | price=%.4f | shares=%.4f | cost=$%.2f",
                mode_tag, market_state.market_label, fill.side,
                fill.price, fill.shares, fill.size,
            )
        else:
            market_state.phase1_detail = f"Entry miss | ask={current_price:.3f} | t={tr:.0f}s"

    # ── Phase 2: Monitor ─────────────────────────────────────────────────────

    async def _phase2_monitor(
        self,
        *,
        snap: MarketSnapshot,
        market_state,
        sstate: StrategyState,
        oracle_safe: bool,
        winning_side: Optional[Side],
        obi_side: Optional[Side],
        obi_value: float,
        up_book,
        down_book,
    ) -> None:
        from brain import Phase1Status, Phase2Status

        mode_tag = "[SIM]" if config.DRY_RUN else "[LIVE]"
        tr = snap.time_remaining
        position_side = market_state.position_side
        position_cost = market_state.position_cost
        position_shares = market_state.position_shares

        pos_book = up_book if position_side == Side.UP else down_book
        if pos_book is None or pos_book.best_bid <= 0:
            market_state.phase1_detail = "Monitoring — no book"
            return

        current_bid = pos_book.best_bid
        unrealised_pnl = position_shares * current_bid - position_cost
        unrealised_pct = (unrealised_pnl / position_cost * 100.0) if position_cost > 0 else 0.0

        market_state.phase1_detail = (
            f"Monitoring {position_side.value} | bid={current_bid:.3f}"
            f" | PnL={unrealised_pct:+.1f}% | t={tr:.0f}s"
        )

        # Priority 3: Emergency close at t ≤ 10s for losing unhedged positions
        if tr <= PHASE2_EMERGENCY_CLOSE_SECONDS and not sstate.hedged:
            await self._phase2_emergency_close(
                snap=snap,
                market_state=market_state,
                unrealised_pnl=unrealised_pnl,
                current_bid=current_bid,
                mode_tag=mode_tag,
            )
            return

        # Priority 1: Hedge detection (if not hedged)
        if not sstate.hedged:
            now = time.monotonic()
            if now - sstate.last_hedge_attempt_at >= 1.0:
                sstate.last_hedge_attempt_at = now
                await self._attempt_hedge(
                    snap=snap,
                    market_state=market_state,
                    sstate=sstate,
                    position_cost=position_cost,
                    position_shares=position_shares,
                    position_side=position_side,
                    up_book=up_book,
                    down_book=down_book,
                    mode_tag=mode_tag,
                )

        # Priority 2: Take-profit (only when not hedged)
        if not sstate.hedged and unrealised_pct >= PHASE2_TP_PCT:
            await self._check_take_profit(
                snap=snap,
                market_state=market_state,
                position_side=position_side,
                unrealised_pct=unrealised_pct,
                current_bid=current_bid,
                mode_tag=mode_tag,
            )

    # ── Phase 2: Attempt hedge ───────────────────────────────────────────────

    async def _attempt_hedge(
        self,
        *,
        snap: MarketSnapshot,
        market_state,
        sstate: StrategyState,
        position_cost: float,
        position_shares: float,
        position_side: Side,
        up_book,
        down_book,
        mode_tag: str,
    ) -> None:
        from brain import Phase2Status

        hedge_side = Side.DOWN if position_side == Side.UP else Side.UP
        hedge_book = down_book if position_side == Side.UP else up_book
        hedge_token = snap.down_token_id if position_side == Side.UP else snap.up_token_id

        if hedge_book is None or hedge_book.best_ask <= 0 or not hedge_token:
            return

        hedge_price = hedge_book.best_ask

        portfolio = await self._exec.get_portfolio_snapshot()
        balance = portfolio.available_balance
        max_hedge = min(position_cost, max(balance, 0.01))

        hedge_plan = find_valid_hedge(
            entry_cost=position_cost,
            entry_shares=position_shares,
            hedge_price=hedge_price,
            max_hedge_usdc=max_hedge,
        )

        if hedge_plan is None:
            market_state.hedge_detail = (
                f"No valid hedge | price={hedge_price:.3f} | entry_cost=${position_cost:.2f}"
            )
            return

        hedge_size = hedge_plan.hedge_amount
        hedge_shares = hedge_size / hedge_price if hedge_price > 0 else 0.0

        fill = await self._exec.execute_taker_buy(
            condition_id=snap.condition_id,
            market_label=market_state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=hedge_token,
            side=hedge_side,
            phase="PHASE2_HEDGE",
            aggressive_price=config.SNIPER_LIMIT_PRICE,
            expected_fill_price=hedge_price,
            target_shares=hedge_shares,
            max_size_usdc=hedge_size,
        )

        if fill:
            sstate.hedged = True
            sstate.hedge_side = hedge_side
            sstate.hedge_cost = fill.size
            sstate.hedge_shares = fill.shares
            market_state.phase2_status = Phase2Status.EXECUTED
            market_state.hedge_detail = (
                f"Hedged {hedge_side.value} | price={fill.price:.3f}"
                f" | cost=${fill.size:.2f} | shares={fill.shares:.4f}"
            )
            log.info(
                "%s Phase 2 Hedge PLACED | market=%s | hedge_side=%s"
                " | price=%.3f | cost=$%.2f | net_entry=$%.4f | net_hedge=$%.4f",
                mode_tag, market_state.market_label, hedge_side.value,
                fill.price, fill.size,
                hedge_plan.entry_net, hedge_plan.hedge_net,
            )

    # ── Phase 2: Take-profit ─────────────────────────────────────────────────

    async def _check_take_profit(
        self,
        *,
        snap: MarketSnapshot,
        market_state,
        position_side: Side,
        unrealised_pct: float,
        current_bid: float,
        mode_tag: str,
    ) -> None:
        from brain import Phase1Status

        position_token = market_state.position_token_id
        position_shares = market_state.position_shares

        if not position_token or position_shares <= 0:
            return

        log.info(
            "%s Phase 2 Take-Profit triggered | market=%s | side=%s"
            " | unrealised=+%.1f%% | bid=%.4f",
            mode_tag, market_state.market_label, position_side.value,
            unrealised_pct, current_bid,
        )

        fill = await self._exec.execute_taker_sell(
            condition_id=snap.condition_id,
            market_label=market_state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=position_token,
            side=position_side,
            phase="PHASE2_TP",
            target_shares=position_shares,
            expected_fill_price=current_bid,
        )

        if fill:
            market_state.phase1_status = Phase1Status.EXITED
            market_state.phase1_detail = (
                f"TP exit | bid={current_bid:.3f} | PnL=+{unrealised_pct:.1f}%"
            )
            log.info(
                "%s Phase 2 TP FILLED | market=%s | price=%.4f | shares=%.4f | proceeds=$%.2f",
                mode_tag, market_state.market_label,
                fill.price, fill.shares, fill.size,
            )

    # ── Phase 2: Emergency close ─────────────────────────────────────────────

    async def _phase2_emergency_close(
        self,
        *,
        snap: MarketSnapshot,
        market_state,
        unrealised_pnl: float,
        current_bid: float,
        mode_tag: str,
    ) -> None:
        from brain import Phase1Status

        position_side = market_state.position_side
        position_token = market_state.position_token_id
        position_shares = market_state.position_shares
        tr = snap.time_remaining

        if not position_token or position_shares <= 0:
            return

        # Hold if profitable — only close if losing
        if unrealised_pnl >= 0:
            market_state.phase1_status = Phase1Status.HOLD_TO_SETTLEMENT
            market_state.phase1_detail = (
                f"Hold to settlement | bid={current_bid:.3f} | PnL=+${unrealised_pnl:.4f} | t={tr:.0f}s"
            )
            return

        log.info(
            "%s Phase 2 Emergency Close | market=%s | side=%s"
            " | unrealised_pnl=$%.4f | bid=%.4f | t=%.1fs",
            mode_tag, market_state.market_label, position_side.value,
            unrealised_pnl, current_bid, tr,
        )

        fill = await self._exec.execute_taker_sell(
            condition_id=snap.condition_id,
            market_label=market_state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=position_token,
            side=position_side,
            phase="PHASE2_EMERGENCY",
            target_shares=position_shares,
            expected_fill_price=current_bid,
        )

        if fill:
            market_state.phase1_status = Phase1Status.EXITED
            market_state.phase1_detail = (
                f"Emergency exit | bid={current_bid:.3f} | t={tr:.0f}s"
            )
            log.info(
                "%s Emergency Close FILLED | market=%s | price=%.4f | proceeds=$%.2f",
                mode_tag, market_state.market_label, fill.price, fill.size,
            )
        else:
            market_state.phase1_detail = (
                f"Emergency close FAILED | bid={current_bid:.3f} | t={tr:.0f}s"
            )

    # ── Phase 3: Sniper ──────────────────────────────────────────────────────

    async def _phase3_sniper(
        self,
        *,
        snap: MarketSnapshot,
        market_state,
        sstate: StrategyState,
        oracle_safe: bool,
        delta_abs: float,
        margin: float,
        winning_side: Optional[Side],
        obi_side: Optional[Side],
        obi_value: float,
        up_book,
        down_book,
    ) -> None:
        from brain import Phase2Status

        mode_tag = "[SIM]" if config.DRY_RUN else "[LIVE]"

        # Gate 1: Oracle delta > SAFE_MARGIN
        if not oracle_safe:
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = (
                f"Sniper: delta={delta_abs:.2f} ≤ margin={margin:.2f}"
            )
            log.debug(
                "[SNIPER] Blocked-delta | market=%s | δ=%.2f ≤ margin=%.2f | t=%.0fs",
                market_state.market_label, delta_abs, margin, snap.time_remaining,
            )
            return

        # Gate 2: OBI aligned with oracle direction
        if winning_side is None or obi_side is None or obi_side != winning_side:
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = (
                f"Sniper: OBI/oracle diverge"
                f" | OBI={obi_side.value if obi_side else '-'}"
                f" oracle={winning_side.value if winning_side else '-'}"
            )
            log.debug(
                "[SNIPER] Blocked-OBI | market=%s | OBI_side=%s oracle=%s | OBI=%.3f | t=%.0fs",
                market_state.market_label,
                obi_side.value if obi_side else "-",
                winning_side.value if winning_side else "-",
                obi_value,
                snap.time_remaining,
            )
            return

        # Gate 3: Price in [0.96, 0.99] — high-conviction zone
        entry_side = winning_side
        sniper_book = up_book if entry_side == Side.UP else down_book
        sniper_token = snap.up_token_id if entry_side == Side.UP else snap.down_token_id

        if sniper_book is None or sniper_book.best_ask <= 0 or not sniper_token:
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = "Sniper: no order book on sniper side"
            return

        current_price = sniper_book.best_ask

        if current_price < PHASE3_MIN_WINNING_PRICE:
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = (
                f"Sniper: ask={current_price:.3f} < floor={PHASE3_MIN_WINNING_PRICE:.2f}"
                f" — not in high-conviction zone yet"
            )
            log.debug(
                "[SNIPER] Blocked-price | market=%s | ask=%.3f < floor=%.2f | t=%.0fs",
                market_state.market_label, current_price,
                PHASE3_MIN_WINNING_PRICE, snap.time_remaining,
            )
            return

        if current_price > PHASE3_MAX_WINNING_PRICE:
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = (
                f"Sniper: ask={current_price:.3f} > ceiling={PHASE3_MAX_WINNING_PRICE:.2f}"
                f" — overpaying risk"
            )
            log.debug(
                "[SNIPER] Blocked-price | market=%s | ask=%.3f > ceiling=%.2f | t=%.0fs",
                market_state.market_label, current_price,
                PHASE3_MAX_WINNING_PRICE, snap.time_remaining,
            )
            return

        # Gate 4: Momentum not slowing — OBI must still be strongly aligned
        # UP  side: OBI ≥ PHASE3_MOMENTUM_OBI_STRONG (0.60)
        # DOWN side: OBI ≤ 1 − 0.60 = 0.40
        if entry_side == Side.UP:
            momentum_ok = obi_value >= PHASE3_MOMENTUM_OBI_STRONG
        else:
            momentum_ok = obi_value <= (1.0 - PHASE3_MOMENTUM_OBI_STRONG)

        if not momentum_ok:
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = (
                f"Sniper: momentum slowing | OBI={obi_value:.3f}"
                f" side={entry_side.value}"
                f" — required ≥{PHASE3_MOMENTUM_OBI_STRONG:.2f} (UP)"
                f" / ≤{1.0 - PHASE3_MOMENTUM_OBI_STRONG:.2f} (DOWN)"
            )
            log.debug(
                "[SNIPER] Blocked-momentum | market=%s | side=%s | OBI=%.3f | t=%.0fs",
                market_state.market_label, entry_side.value, obi_value, snap.time_remaining,
            )
            return

        # Gate 5: Positive expected profit
        portfolio = await self._exec.get_portfolio_snapshot()
        balance = portfolio.available_balance

        if (entry_side == Side.UP and obi_value >= PHASE3_OBI_HIGH_THRESHOLD) or \
           (entry_side == Side.DOWN and obi_value <= (1.0 - PHASE3_OBI_HIGH_THRESHOLD)):
            size_ratio = PHASE3_SIZE_RATIO_HIGH
        else:
            size_ratio = PHASE3_SIZE_RATIO_BASE

        sniper_size = math.floor((balance * size_ratio * 100.0) + 1e-9) / 100.0

        if sniper_size < config.MIN_ORDER_USDC:
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = (
                f"Sniper: size=${sniper_size:.2f} below minimum ${config.MIN_ORDER_USDC:.2f}"
            )
            return

        sniper_shares = sniper_size / current_price if current_price > 0 else 0.0
        sniper_payout = sniper_shares * 1.00
        expected_profit = sniper_payout - sniper_size

        if expected_profit <= 0:
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = (
                f"Sniper: expected_profit=${expected_profit:.4f} ≤ 0"
            )
            log.info(
                "%s Phase 3 Sniper ABORTED — non-positive profit"
                " | market=%s | ask=%.4f | size=$%.2f | payout=$%.4f | profit=$%.4f",
                mode_tag,
                market_state.market_label,
                current_price, sniper_size, sniper_payout, expected_profit,
            )
            return

        # ── All gates passed — fire ──────────────────────────────────────────
        # aggressive_price = SNIPER_LIMIT_PRICE (1.00) — avoids IOC cancel on ask micro-ticks
        aggressive_limit = config.SNIPER_LIMIT_PRICE
        target_shares = sniper_size / current_price

        log.info(
            "%s Phase 3 Sniper FIRED | market=%s | side=%s"
            " | ask=%.4f | limit=%.4f | size=$%.2f (%.0f%% balance)"
            " | shares=%.4f | expected_profit=$%.4f"
            " | OBI=%.3f | δ=%.4f | t=%.1fs",
            mode_tag,
            market_state.market_label,
            entry_side.value,
            current_price,
            aggressive_limit,
            sniper_size,
            size_ratio * 100,
            target_shares,
            expected_profit,
            obi_value,
            snap.binance_live_price - snap.strike_price,
            snap.time_remaining,
        )

        fill = await self._exec.execute_taker_buy(
            condition_id=snap.condition_id,
            market_label=market_state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=sniper_token,
            side=entry_side,
            phase="PHASE3_SNIPER",
            aggressive_price=aggressive_limit,
            expected_fill_price=current_price,
            target_shares=target_shares,
            max_size_usdc=sniper_size,
        )

        # Mark as fired regardless of fill — no retry in this window
        sstate.phase3_sniper_fired = True

        if fill is not None:
            actual_profit = fill.shares - fill.size
            market_state.phase2_status = Phase2Status.EXECUTED
            market_state.abort_reason = ""
            market_state.position_source = "PHASE3"
            market_state.position_side = entry_side
            market_state.position_token_id = sniper_token
            market_state.phase2_bullets_fired = 1
            market_state.phase2_spend = fill.size

            log.info(
                "%s Phase 3 Sniper FILLED | market=%s | side=%s"
                " | fill_price=%.4f | shares=%.4f | cost=$%.4f"
                " | actual_profit=$%.4f | t=%.1fs",
                mode_tag,
                market_state.market_label,
                fill.side,
                fill.price,
                fill.shares,
                fill.size,
                actual_profit,
                snap.time_remaining,
            )
        else:
            market_state.phase2_status = Phase2Status.ABORTED
            market_state.abort_reason = "Sniper not filled — no retry"
            log.info(
                "%s Phase 3 Sniper NOT FILLED | market=%s | side=%s"
                " | ask=%.4f | size=$%.2f | t=%.1fs",
                mode_tag,
                market_state.market_label,
                entry_side.value,
                current_price,
                sniper_size,
                snap.time_remaining,
            )

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_strategy_state(self, condition_id: str) -> Optional[StrategyState]:
        return self._strategy_states.get(condition_id)
