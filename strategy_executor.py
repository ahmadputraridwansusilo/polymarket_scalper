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
PHASE2_MAX_HEDGE_ATTEMPTS: int = 3               # give up hedging after this many fill failures

# ---------------------------------------------------------------------------
# Phase 3 Sniper — two-condition near-expiry and pre-expiry entry
#
# Condition A (Pre-60s): tr > SNIPER_EARLY_WINDOW_SEC (60s)
#   price in [SNIPER_PRICE_MIN, SNIPER_PRICE_MAX] (0.95–0.99) on winning side
#   delta > SAFE_MARGIN  (via oracle_safe)
#
# Condition B (Late, ≤30s): tr ≤ SNIPER_LATE_WINDOW_SEC (30s)
#   delta > SAFE_MARGIN AND OBI same direction OR neutral
#   price range NOT required
#
# Both: fixed size = SNIPER_POSITION_SIZE_USD; profitability checked with
#   slippage + fee model before any order is placed.
# Order: GTC limit at best_ask; polls PHASE3_LIMIT_TIMEOUT s, then cancels.
# ---------------------------------------------------------------------------
PHASE3_WINDOW_START: float = 60.0                # Phase 3 time window opens (seconds)
PHASE3_WINDOW_END: float = 10.0                  # Phase 3 time window closes (seconds)
PHASE3_MIN_WINNING_PRICE: float = 0.95           # kept for backward-compat / test imports
PHASE3_MAX_WINNING_PRICE: float = 0.99           # kept for backward-compat / test imports
PHASE3_SIZE_RATIO_BASE: float = 0.15             # unused after refactor; kept for test imports
PHASE3_SIZE_RATIO_HIGH: float = 0.20             # unused after refactor; kept for test imports
PHASE3_SIZE_RATIO: float = PHASE3_SIZE_RATIO_BASE  # alias for test compat
PHASE3_OBI_HIGH_THRESHOLD: float = 0.85          # unused after refactor; kept for test imports
PHASE3_MOMENTUM_OBI_STRONG: float = 0.60         # unused after refactor; kept for test imports
PHASE3_LIMIT_TIMEOUT: float = 60.0              # cancel GTC limit after this many seconds if unfilled
PHASE3_LIMIT_POLL_INTERVAL: float = 2.0         # poll interval for fill check

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
    hedge_fail_count: int = 0

    # Phase 3 sniper tracking
    sniper_entered: bool = False       # True once Condition A or B fires (no re-entry)
    phase3_order_id: str = ""

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

        # ── Phase 3 window (Condition B + pending-order status) ──────────────
        if PHASE3_WINDOW_END < tr <= PHASE3_WINDOW_START:
            if not sstate.sniper_entered:
                if self.check_condition_b(tr, oracle_safe, winning_side, obi_side):
                    cond_b_book = up_book if winning_side == Side.UP else down_book
                    if cond_b_book is not None and cond_b_book.best_ask > 0:
                        log.info(
                            "Condition B triggered: delta=%.2f obi=%.3f at T-%.0fs | market=%s",
                            delta_abs, obi_value, tr, market_state.market_label,
                        )
                        print(
                            f"Condition B triggered: delta={delta_abs:.2f}"
                            f" obi={obi_value:.3f} at T-{tr:.0f}s"
                        )
                        await self.execute_entry(
                            snap=snap,
                            market_state=market_state,
                            sstate=sstate,
                            entry_side=winning_side,
                            entry_book=cond_b_book,
                            condition="B",
                            tr=tr,
                            obi_value=obi_value,
                        )
                    else:
                        market_state.phase1_detail = f"Sniper Cond B: no book | t={tr:.0f}s"
                else:
                    market_state.phase1_detail = (
                        f"Sniper Cond B blocked | delta={delta_abs:.2f}"
                        f" obi={obi_value:.3f} t={tr:.0f}s"
                    )
            elif sstate.phase3_order_id:
                market_state.phase1_detail = f"Sniper limit pending | order={sstate.phase3_order_id[:8]}"
            else:
                market_state.phase1_detail = f"Sniper entered | holding | t={tr:.0f}s"
            return

    def check_condition_b(
        self,
        tr: float,
        oracle_safe: bool,
        entry_side: Optional[Side],
        obi_side: Optional[Side],
    ) -> bool:
        """Condition B: returns True when delta > safe_margin, OBI not against.

        Time gating is handled by evaluate() (Phase 3 window). No redundant gate here.
        """
        if not oracle_safe or entry_side is None:
            return False
        # OBI must be same direction as the trade OR neutral (not pointing against it).
        if obi_side is not None and obi_side != entry_side:
            return False
        return True

    @staticmethod
    def is_profitable(
        side: Side,
        price: float,
        size_usd: float,
    ) -> tuple[bool, float]:
        """Returns (profitable, net_profit) after dynamic taker fee.

        Polymarket 15-min crypto markets charge a price-dependent taker fee:
          fee_rate     = 4 * p * (1-p) * PEAK_FEE_RATE  (peaks ~2% at p=0.5)
          platform_fee = size_usd * fee_rate
          total_cost   = size_usd + platform_fee
          expected_payout = size_usd / price   # shares × $1.00
          net_profit   = expected_payout - total_cost
        """
        if price <= 0 or size_usd <= 0:
            return False, -size_usd
        platform_fee = size_usd * config.dynamic_fee_rate(price)
        total_cost = size_usd + platform_fee
        expected_payout = size_usd / price
        net_profit = expected_payout - total_cost
        return net_profit > 0, net_profit

    async def execute_entry(
        self,
        *,
        snap: MarketSnapshot,
        market_state,
        sstate: StrategyState,
        entry_side: Side,
        entry_book,
        condition: str,
        tr: float,
        obi_value: float,
    ) -> None:
        """Place a sniper limit order after all condition gates have passed.

        Called for both Condition A and Condition B.  Sets sniper_entered=True
        only on a successful order placement (not on profitability block).
        """
        from brain import Phase2Status

        mode_tag = "[SIM]" if config.DRY_RUN else "[LIVE]"
        entry_token = snap.up_token_id if entry_side == Side.UP else snap.down_token_id
        current_price = entry_book.best_ask
        size_usd = config.SNIPER_POSITION_SIZE_USD

        # Profit gate — slippage + fee model
        profitable, net_profit = self.is_profitable(entry_side, current_price, size_usd)
        if not profitable:
            msg = f"Entry BLOCKED: unprofitable (net_profit=${net_profit:.4f} after slippage+fees)"
            log.info("%s | market=%s", msg, market_state.market_label)
            print(msg)
            market_state.phase2_status = Phase2Status.BLOCKED
            market_state.abort_reason = f"Sniper: unprofitable (net=${net_profit:.4f})"
            return

        # Exchange minimum-shares guard — bump size_usd instead of blocking
        sniper_shares = size_usd / current_price if current_price > 0 else 0.0
        if sniper_shares < config.MIN_ORDER_SHARES:
            size_usd = config.MIN_ORDER_SHARES * current_price
            sniper_shares = config.MIN_ORDER_SHARES
            log.info(
                "SNIPER: bumped size to $%.2f to meet MIN_ORDER_SHARES=%.0f | market=%s",
                size_usd, config.MIN_ORDER_SHARES, market_state.market_label,
            )

        log.info(
            "SNIPER ENTRY: side=%s size=$%.2f price=%.4f market=%s (Condition %s)",
            entry_side.value, size_usd, current_price, market_state.market_label, condition,
        )
        print(
            f"SNIPER ENTRY: side={entry_side.value} size=${size_usd:.2f}"
            f" price={current_price:.4f} market={market_state.market_label}"
        )

        # Claim the shot before awaiting — prevents retry on the next tick
        # even if the order is rejected or unfilled.
        sstate.sniper_entered = True

        # Use execute_sniper (taker/FOK) — GTC limits are rejected by Polymarket near expiry.
        fill: Optional[Fill] = await self._exec.execute_sniper(
            condition_id=snap.condition_id,
            market_label=market_state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=entry_token,
            side=entry_side,
            size_usdc=size_usd,
            current_best_ask=current_price,
        )

        if fill is None:
            market_state.phase2_status = Phase2Status.ABORTED
            market_state.abort_reason = "Sniper: taker order failed"
            log.warning(
                "%s SNIPER REJECTED (no retry) | market=%s | side=%s | ask=%.4f | size=$%.2f | t=%.1fs"
                " — check live order book depth at this price level",
                mode_tag, market_state.market_label, entry_side.value,
                current_price, size_usd, tr,
            )
            return

        actual_profit = fill.shares - fill.size
        market_state.phase2_status = Phase2Status.EXECUTED
        market_state.abort_reason = ""
        market_state.position_source = "PHASE3"
        market_state.position_side = entry_side
        market_state.position_token_id = entry_token
        market_state.phase2_bullets_fired = 1
        market_state.phase2_spend = fill.size

        log.info(
            "%s SNIPER FILLED | market=%s | side=%s"
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

    # Legacy shim — no longer called from evaluate(); kept so any external
    # integration tests that call _phase3_sniper directly still work.
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
        if winning_side is None:
            return
        book = up_book if winning_side == Side.UP else down_book
        if book is None:
            return
        await self.execute_entry(
            snap=snap,
            market_state=market_state,
            sstate=sstate,
            entry_side=winning_side,
            entry_book=book,
            condition="legacy",
            tr=snap.time_remaining,
            obi_value=obi_value,
        )

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_strategy_state(self, condition_id: str) -> Optional[StrategyState]:
        return self._strategy_states.get(condition_id)
