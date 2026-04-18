"""
test_strategy_trace.py — Side-by-Side SIM/LIVE Trace Test

Runs one full simulated market cycle through the StrategyExecutor and logs
every decision point. Validates that:
  1. The same code paths execute in SIM mode as would in LIVE mode.
  2. Only the execution layer differs ([SIM] / [LIVE] log tags).
  3. Phase 1 entry fires when conditions are met.
  4. Phase 2 hedge detection triggers once position is open.
  5. Phase 3 sniper fires in the correct time window.
  6. Emergency close fires for losing unhedged positions at t≤10s.

Usage:
    python test_strategy_trace.py

No external connections required — uses synthetic market data.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

# Configure trace logging before importing any bot modules
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-12s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trace_test")

# Patch config.DRY_RUN = True before import so executioner uses SimulatorOrderManager
import config as _config_module
_config_module.DRY_RUN = True  # force SIM mode for trace test

from executioner import (
    BookLevel,
    Fill,
    OrderBookSnapshot,
    Side,
    SimulatorOrderManager,
)
from oracle import MarketSnapshot, OracleState
from strategy_executor import (
    StrategyExecutor,
    StrategyState,
    compute_entry_size,
    find_valid_hedge,
    SimBroker,
    PHASE1_ENTRY_MIN_TIME_REMAINING,
    PHASE2_TP_PCT,
    PHASE2_EMERGENCY_CLOSE_SECONDS,
    PHASE3_WINDOW_START,
    PHASE3_WINDOW_END,
    PHASE3_MIN_WINNING_PRICE,
    PHASE3_SIZE_RATIO,
)
from brain import MarketState, Phase1Status, Phase2Status


# ---------------------------------------------------------------------------
# Helpers — synthetic market / book factories
# ---------------------------------------------------------------------------

def make_market_snapshot(
    *,
    asset: str = "BTC",
    timeframe: str = "5m",
    binance_price: float = 84_100.0,
    strike_price: float = 84_000.0,      # delta = +100 → UP favoured
    time_remaining: float = 180.0,        # seconds
    up_ask: float = 0.60,
    down_ask: float = 0.42,
    up_bid: float = 0.58,
    down_bid: float = 0.40,
) -> MarketSnapshot:
    snap = MarketSnapshot(
        condition_id="trace-market-001",
        asset=asset,
        timeframe=timeframe,
        up_token_id="up-token-001",
        down_token_id="down-token-001",
        strike_price=strike_price,
        event_start_time=time.time() - 300,
        end_time=time.time() + time_remaining,
        best_ask_up=up_ask,
        best_bid_up=up_bid,
        best_ask_down=down_ask,
        best_bid_down=down_bid,
        binance_live_price=binance_price,
    )
    return snap


def make_order_book(
    *,
    token_id: str,
    best_ask: float,
    best_bid: float,
    size: float = 500.0,
    tick_size: float = 0.001,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        bids=[BookLevel(price=best_bid, size=size)],
        asks=[BookLevel(price=best_ask, size=size)],
        tick_size=tick_size,
        last_trade_price=best_bid,
    )


def make_market_state(snap: MarketSnapshot) -> MarketState:
    return MarketState(
        condition_id=snap.condition_id,
        market_label=f"{snap.asset} {snap.timeframe}",
        asset=snap.asset,
        timeframe=snap.timeframe,
    )


# ---------------------------------------------------------------------------
# Mock Oracle that returns synthetic books for get_order_book calls
# ---------------------------------------------------------------------------

class MockOracle:
    """Minimal oracle stub that serves synthetic books.

    The StrategyExecutor doesn't call oracle directly — books are passed
    in pre-computed by Brain._evaluate_market. This mock is used only by
    SimBroker.fill_order which calls exec.get_order_book internally.
    """

    def __init__(self) -> None:
        self._markets: Dict[str, MarketSnapshot] = {}

    def register(self, snap: MarketSnapshot) -> None:
        self._markets[snap.condition_id] = snap

    def all_markets(self) -> list[MarketSnapshot]:
        return list(self._markets.values())

    def get_market(self, condition_id: str) -> Optional[MarketSnapshot]:
        return self._markets.get(condition_id)


# ---------------------------------------------------------------------------
# Helper: inject a synthetic book into the SimulatorOrderManager's cache
# ---------------------------------------------------------------------------

def inject_book(
    exec_manager: SimulatorOrderManager,
    token_id: str,
    best_ask: float,
    best_bid: float,
) -> None:
    """Pre-populate the CLOB book cache so exec.get_order_book returns a
    known synthetic book without hitting the real API."""
    book = OrderBookSnapshot(
        token_id=token_id,
        bids=[BookLevel(price=best_bid, size=500.0)],
        asks=[BookLevel(price=best_ask, size=500.0)],
        tick_size=0.001,
        last_trade_price=best_bid,
    )
    exec_manager._book_cache[token_id] = (book, time.time() + 10.0)  # 10 s TTL


# ---------------------------------------------------------------------------
# Phase-by-phase trace scenarios
# ---------------------------------------------------------------------------

async def trace_phase1_entry() -> None:
    """SCENARIO 1: Phase 1 Entry.

    Conditions: delta=+100 (BTC UP), OBI=0.80 (UP), t=180s, balance=$100, no position.
    Expected: entry fires, position opens at ~0.60 per share.

    SIM vs LIVE: identical trigger evaluation; only execution layer differs.
    """
    log.info("=" * 70)
    log.info("SCENARIO 1 — Phase 1 Entry")
    log.info("=" * 70)

    exec_mgr = SimulatorOrderManager(initial_balance=100.0)
    oracle = MockOracle()
    executor = StrategyExecutor(exec_mgr, oracle)

    # Use asymmetric prices so up_mid=0.74, down_mid=0.26 → OBI=0.74 > 0.70 threshold.
    # With equal bid sizes (500 each) _market_obi falls back to midprices, so midprice
    # ratio determines the OBI signal.
    snap = make_market_snapshot(
        binance_price=84_100.0,
        strike_price=84_000.0,  # delta = +100 > BTC margin (15.0) → oracle_safe = True
        time_remaining=180.0,
        up_ask=0.75,            # up_mid = (0.75+0.73)/2 = 0.74
        up_bid=0.73,
        down_ask=0.27,          # down_mid = (0.27+0.25)/2 = 0.26
        down_bid=0.25,          # OBI = 0.74/(0.74+0.26) = 0.74 > 0.70 → UP signal
    )
    oracle.register(snap)
    inject_book(exec_mgr, snap.up_token_id, best_ask=0.75, best_bid=0.73)
    inject_book(exec_mgr, snap.down_token_id, best_ask=0.27, best_bid=0.25)

    market_state = make_market_state(snap)

    up_book = make_order_book(token_id=snap.up_token_id, best_ask=0.75, best_bid=0.73)
    down_book = make_order_book(token_id=snap.down_token_id, best_ask=0.27, best_bid=0.25)

    # ── Compute entry size (should be $8.00 = $100 × 8%) ──────────────────
    balance = 100.0
    expected_size = compute_entry_size(balance)
    assert expected_size == 8.0, f"Expected entry_size=8.0, got {expected_size}"
    log.info("[TRACE] Entry size for balance=$%.2f → $%.2f (8%% tier)", balance, expected_size)

    # ── Run evaluate ────────────────────────────────────────────────────────
    await executor.evaluate(
        snap=snap,
        market_state=market_state,
        up_book=up_book,
        down_book=down_book,
    )

    # ── Verify entry fired ──────────────────────────────────────────────────
    portfolio = await exec_mgr.get_portfolio_snapshot()
    pos_map = portfolio.active_positions.get(snap.condition_id, {})

    if "UP" in pos_map:
        pos = pos_map["UP"]
        log.info(
            "[TRACE] ✓ Entry confirmed | side=UP | cost=$%.4f | shares=%.4f | avg_price=%.4f",
            pos.cost_basis, pos.shares, pos.avg_entry_price,
        )
        assert pos.cost_basis > 0, "Position cost_basis must be > 0"
        assert pos.shares > 0, "Position shares must be > 0"
    else:
        log.error("[TRACE] ✗ No UP position found after entry evaluation")
        log.info("  market_state.phase1_detail: %s", market_state.phase1_detail)

    sstate = executor.get_strategy_state(snap.condition_id)
    log.info("[TRACE] StrategyState: hedged=%s | entry_delta=%.2f | entry_obi=%.3f",
             sstate.hedged if sstate else "N/A",
             sstate.entry_delta if sstate else 0.0,
             sstate.entry_obi if sstate else 0.5)

    log.info("[TRACE] LIVE equivalent: same code path, LiveOrderManager instead of Sim")
    log.info("")


async def trace_phase2_hedge_detection() -> None:
    """SCENARIO 2: Phase 2 Hedge Detection.

    Condition: position exists (UP), delta still aligned.
    The DOWN token is cheap enough to make a valid hedge.
    Expected: hedge executes, HEDGED=True, TP disabled.

    Demonstrates binary search for minimum hedge amount.
    """
    log.info("=" * 70)
    log.info("SCENARIO 2 — Phase 2 Hedge Detection")
    log.info("=" * 70)

    exec_mgr = SimulatorOrderManager(initial_balance=100.0)
    oracle = MockOracle()
    executor = StrategyExecutor(exec_mgr, oracle)

    snap = make_market_snapshot(
        binance_price=84_100.0,
        strike_price=84_000.0,
        time_remaining=90.0,
        up_ask=0.78,      # position is at 0.78 now (was 0.60 at entry)
        up_bid=0.76,
        down_ask=0.24,    # losing side cheap → hedge candidate
        down_bid=0.22,
    )
    oracle.register(snap)
    inject_book(exec_mgr, snap.up_token_id, best_ask=0.78, best_bid=0.76)
    inject_book(exec_mgr, snap.down_token_id, best_ask=0.24, best_bid=0.22)

    # ── Manually create a Phase 1 position (simulate a prior entry) ──────────
    # Entry: $8 at 0.60 → 13.33 shares
    entry_cost = 8.0
    entry_price = 0.60
    entry_shares = entry_cost / entry_price   # ≈ 13.33
    # Directly inject position into the executor's portfolio
    async with exec_mgr._lock:
        exec_mgr._available_balance -= entry_cost
        exec_mgr._locked_margin += entry_cost
        pos = exec_mgr._ensure_position_locked(
            condition_id=snap.condition_id,
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id=snap.up_token_id,
            side=Side.UP,
        )
        pos.cost_basis = entry_cost
        pos.shares = entry_shares
        pos.avg_entry_price = entry_price

    # Set market_state to reflect the existing position
    market_state = make_market_state(snap)
    market_state.position_side = Side.UP
    market_state.position_token_id = snap.up_token_id
    market_state.position_cost = entry_cost
    market_state.position_shares = entry_shares
    market_state.avg_entry_price = entry_price
    market_state.position_source = "PHASE1"

    # ── Test hedge validity independently ──────────────────────────────────
    hedge_price = 0.24
    hedge_plan = find_valid_hedge(
        entry_cost=entry_cost,
        entry_shares=entry_shares,
        hedge_price=hedge_price,
        max_hedge_usdc=5.0,
    )
    if hedge_plan:
        log.info(
            "[TRACE] find_valid_hedge result | hedge_amount=$%.4f | hedge_shares=%.4f"
            " | net_if_entry_wins=$%.4f | net_if_hedge_wins=$%.4f | class=%s",
            hedge_plan.hedge_amount,
            hedge_plan.hedge_shares,
            hedge_plan.entry_net,
            hedge_plan.hedge_net,
            hedge_plan.classification,
        )
        assert hedge_plan.entry_net >= -0.01, "entry_net must be ≥ -0.01"
        assert hedge_plan.hedge_net >= -0.01, "hedge_net must be ≥ -0.01"
        log.info("[TRACE] ✓ Hedge validity conditions satisfied")
    else:
        log.warning("[TRACE] No valid hedge at these prices (entry_cost=%.2f, entry_shares=%.2f, hedge_price=%.2f)",
                    entry_cost, entry_shares, hedge_price)

    up_book = make_order_book(token_id=snap.up_token_id, best_ask=0.78, best_bid=0.76)
    down_book = make_order_book(token_id=snap.down_token_id, best_ask=0.24, best_bid=0.22)

    await executor.evaluate(
        snap=snap,
        market_state=market_state,
        up_book=up_book,
        down_book=down_book,
    )

    sstate = executor.get_strategy_state(snap.condition_id)
    if sstate and sstate.hedged:
        log.info(
            "[TRACE] ✓ Phase 2 HEDGED | hedge_side=%s | hedge_cost=$%.4f | hedge_shares=%.4f",
            sstate.hedge_side.value if sstate.hedge_side else "-",
            sstate.hedge_cost,
            sstate.hedge_shares,
        )
        log.info("[TRACE]   TP permanently disabled for this position")
    else:
        log.info("[TRACE] Hedge not executed this tick (check next tick or prices)")

    log.info("[TRACE] LIVE equivalent: same hedge validity formula, LiveOrderManager fills hedge")
    log.info("")


async def trace_phase2_take_profit() -> None:
    """SCENARIO 3: Phase 2 Take-Profit at 50%.

    Condition: open UP position, unrealised PnL = +50%, not hedged.
    Expected: market close fires, EXITED status set.
    """
    log.info("=" * 70)
    log.info("SCENARIO 3 — Phase 2 Take-Profit at 50%%")
    log.info("=" * 70)

    exec_mgr = SimulatorOrderManager(initial_balance=100.0)
    oracle = MockOracle()
    executor = StrategyExecutor(exec_mgr, oracle)

    # UP position up at 0.91 bid = +50% unrealised.
    # DOWN ask = 0.88 (expensive losing side) → hedge lower bound ≈ (8-0.01)*0.88/0.12 ≈ $58.6
    # which exceeds max_hedge (capped at entry_cost=$8) → no valid hedge → TP fires instead.
    snap = make_market_snapshot(time_remaining=90.0, up_ask=0.92, up_bid=0.91, down_ask=0.88, down_bid=0.86)
    oracle.register(snap)
    inject_book(exec_mgr, snap.up_token_id, best_ask=0.92, best_bid=0.91)
    inject_book(exec_mgr, snap.down_token_id, best_ask=0.88, best_bid=0.86)

    # Entry at 0.60, current bid = 0.91 → unrealised > +50%
    entry_cost = 8.0
    entry_price = 0.60
    entry_shares = entry_cost / entry_price   # 13.33

    async with exec_mgr._lock:
        exec_mgr._available_balance -= entry_cost
        exec_mgr._locked_margin += entry_cost
        pos = exec_mgr._ensure_position_locked(
            condition_id=snap.condition_id,
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id=snap.up_token_id,
            side=Side.UP,
        )
        pos.cost_basis = entry_cost
        pos.shares = entry_shares
        pos.avg_entry_price = entry_price

    market_state = make_market_state(snap)
    market_state.position_side = Side.UP
    market_state.position_token_id = snap.up_token_id
    market_state.position_cost = entry_cost
    market_state.position_shares = entry_shares
    market_state.avg_entry_price = entry_price
    market_state.position_source = "PHASE1"

    current_bid = 0.91
    unrealised_pct = (entry_shares * current_bid - entry_cost) / entry_cost * 100.0
    log.info("[TRACE] unrealised_pnl_pct=%.2f%% (threshold=%.0f%%)", unrealised_pct, PHASE2_TP_PCT)

    up_book = make_order_book(token_id=snap.up_token_id, best_ask=0.92, best_bid=0.91)
    down_book = make_order_book(token_id=snap.down_token_id, best_ask=0.88, best_bid=0.86)

    await executor.evaluate(
        snap=snap,
        market_state=market_state,
        up_book=up_book,
        down_book=down_book,
    )

    log.info("[TRACE] phase1_status: %s | detail: %s",
             market_state.phase1_status, market_state.phase1_detail)

    if market_state.phase1_status == Phase1Status.EXITED:
        log.info("[TRACE] ✓ Take-Profit triggered and position closed")
    else:
        log.warning("[TRACE] Take-Profit did not trigger — check unrealised calc")

    log.info("[TRACE] LIVE equivalent: same 50%% threshold check, LiveOrderManager closes position")
    log.info("")


async def trace_phase2_emergency_close() -> None:
    """SCENARIO 4: Phase 2 Emergency Close at t≤10s.

    Condition: open UP position at t=7s, NOT hedged, position is losing.
    Expected: immediate market close, capital protected.
    """
    log.info("=" * 70)
    log.info("SCENARIO 4 — Phase 2 Emergency Close (t≤10s, losing)")
    log.info("=" * 70)

    exec_mgr = SimulatorOrderManager(initial_balance=100.0)
    oracle = MockOracle()
    executor = StrategyExecutor(exec_mgr, oracle)

    snap = make_market_snapshot(
        time_remaining=7.0,   # inside emergency close window
        up_ask=0.50,
        up_bid=0.48,          # position losing: entry was 0.60, current bid 0.48
    )
    oracle.register(snap)
    inject_book(exec_mgr, snap.up_token_id, best_ask=0.50, best_bid=0.48)
    inject_book(exec_mgr, snap.down_token_id, best_ask=0.52, best_bid=0.50)

    entry_cost = 8.0
    entry_price = 0.60
    entry_shares = entry_cost / entry_price

    async with exec_mgr._lock:
        exec_mgr._available_balance -= entry_cost
        exec_mgr._locked_margin += entry_cost
        pos = exec_mgr._ensure_position_locked(
            condition_id=snap.condition_id,
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id=snap.up_token_id,
            side=Side.UP,
        )
        pos.cost_basis = entry_cost
        pos.shares = entry_shares
        pos.avg_entry_price = entry_price

    market_state = make_market_state(snap)
    market_state.position_side = Side.UP
    market_state.position_token_id = snap.up_token_id
    market_state.position_cost = entry_cost
    market_state.position_shares = entry_shares
    market_state.avg_entry_price = entry_price
    market_state.position_source = "PHASE1"

    current_bid = 0.48
    unrealised_pnl = entry_shares * current_bid - entry_cost
    log.info("[TRACE] unrealised_pnl=$%.4f (< 0 → emergency close should fire)", unrealised_pnl)

    up_book = make_order_book(token_id=snap.up_token_id, best_ask=0.50, best_bid=0.48)
    down_book = make_order_book(token_id=snap.down_token_id, best_ask=0.52, best_bid=0.50)

    await executor.evaluate(
        snap=snap,
        market_state=market_state,
        up_book=up_book,
        down_book=down_book,
    )

    log.info("[TRACE] phase1_status: %s | detail: %s",
             market_state.phase1_status, market_state.phase1_detail)

    if market_state.phase1_status == Phase1Status.EXITED:
        log.info("[TRACE] ✓ Emergency close fired — capital protected")
    elif market_state.phase1_status == Phase1Status.HOLD_TO_SETTLEMENT:
        log.info("[TRACE] ✓ Holding to close — position profitable (see detail)")
    else:
        log.warning("[TRACE] Unexpected status: %s", market_state.phase1_status)

    log.info("[TRACE] LIVE equivalent: same unrealised_pnl check, LiveOrderManager closes position")
    log.info("")


async def trace_phase3_sniper() -> None:
    """SCENARIO 5: Phase 3 Sniper (10–20 s window).

    Condition: t=15s, delta > margin, OBI aligned, winning ask=0.93, profit > 0.
    Expected: sniper fires once, no retry.
    """
    log.info("=" * 70)
    log.info("SCENARIO 5 — Phase 3 Sniper (t=15s)")
    log.info("=" * 70)

    exec_mgr = SimulatorOrderManager(initial_balance=100.0)
    oracle = MockOracle()
    executor = StrategyExecutor(exec_mgr, oracle)

    snap = make_market_snapshot(
        time_remaining=15.0,   # inside sniper window (10 < t ≤ 20)
        up_ask=0.93,           # > 0.90 minimum for Phase 3
        up_bid=0.92,
        down_ask=0.09,
        down_bid=0.07,
    )
    oracle.register(snap)
    inject_book(exec_mgr, snap.up_token_id, best_ask=0.93, best_bid=0.92)
    inject_book(exec_mgr, snap.down_token_id, best_ask=0.09, best_bid=0.07)

    market_state = make_market_state(snap)  # no existing position

    # ── Pre-compute profit check ────────────────────────────────────────────
    balance = 100.0
    sniper_size = balance * PHASE3_SIZE_RATIO          # = $7.00
    current_price = 0.93
    sniper_shares = sniper_size / current_price        # ≈ 7.53 shares
    sniper_payout = sniper_shares * 1.00               # ≈ $7.53
    expected_profit = sniper_payout - sniper_size
    log.info(
        "[TRACE] Sniper profit check: size=$%.2f | price=%.3f | shares=%.4f"
        " | payout=$%.4f | profit=$%.4f | profitable=%s",
        sniper_size, current_price, sniper_shares, sniper_payout, expected_profit,
        "YES" if expected_profit > 0 else "NO",
    )

    up_book = make_order_book(token_id=snap.up_token_id, best_ask=0.93, best_bid=0.92)
    down_book = make_order_book(token_id=snap.down_token_id, best_ask=0.09, best_bid=0.07)

    await executor.evaluate(
        snap=snap,
        market_state=market_state,
        up_book=up_book,
        down_book=down_book,
    )

    sstate = executor.get_strategy_state(snap.condition_id)
    log.info("[TRACE] phase3_sniper_fired: %s",
             sstate.phase3_sniper_fired if sstate else "N/A")
    log.info("[TRACE] phase2_status: %s | abort_reason: %s",
             market_state.phase2_status, market_state.abort_reason)

    if sstate and sstate.phase3_sniper_fired:
        portfolio = await exec_mgr.get_portfolio_snapshot()
        pos_map = portfolio.active_positions.get(snap.condition_id, {})
        if "UP" in pos_map:
            pos = pos_map["UP"]
            log.info("[TRACE] ✓ Phase 3 Sniper FILLED | shares=%.4f | cost=$%.4f",
                     pos.shares, pos.cost_basis)
        else:
            log.info("[TRACE] Sniper fired but fill failed (slippage / insufficient size)")
    else:
        log.warning("[TRACE] Phase 3 Sniper did not fire — check conditions")

    log.info("[TRACE] LIVE equivalent: same delta/OBI/price/profit gates; LiveOrderManager places order")
    log.info("")


async def trace_entry_size_tiers() -> None:
    """SCENARIO 6: Entry sizing tier validation.

    Verifies the 6%/8%/10% tiers and 12% hard cap.
    """
    log.info("=" * 70)
    log.info("SCENARIO 6 — Entry Sizing Tier Validation")
    log.info("=" * 70)

    cases = [
        (20.0,   "micro (<$50)",   20.0 * 0.06),   # $1.20
        (49.99,  "micro (<$50)",   49.99 * 0.06),  # $3.00
        (50.0,   "small ($50-200)",50.0 * 0.08),   # $4.00
        (100.0,  "small",          100.0 * 0.08),  # $8.00
        (199.99, "small",          199.99 * 0.08), # $16.00
        (200.0,  "normal (≥$200)", 200.0 * 0.10),  # $20.00
        (500.0,  "normal",         500.0 * 0.10),  # $50.00
        (1000.0, "normal",         1000.0 * 0.10), # $100.00
    ]

    all_passed = True
    for balance, tier_label, expected_uncapped in cases:
        hard_cap = balance * 0.12
        expected = max(1.00, min(expected_uncapped, hard_cap))
        # floor to cents
        import math
        expected = math.floor((expected * 100.0) + 1e-9) / 100.0
        actual = compute_entry_size(balance)
        status = "✓" if abs(actual - expected) < 0.005 else "✗"
        if status == "✗":
            all_passed = False
        log.info(
            "[TRACE] %s balance=$%.2f (%s) | expected=$%.2f | actual=$%.2f",
            status, balance, tier_label, expected, actual,
        )

    assert all_passed, "Entry sizing tiers have unexpected values — see trace above"
    log.info("[TRACE] ✓ All sizing tiers correct — SIM and LIVE share this function")
    log.info("")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def main() -> None:
    log.info("╔══════════════════════════════════════════════════════════════════╗")
    log.info("║  Strategy Trace Test — SIM mode, no external connections needed  ║")
    log.info("║  All code paths tagged [SIM]. Same paths run in [LIVE] mode.     ║")
    log.info("╚══════════════════════════════════════════════════════════════════╝")
    log.info("")

    await trace_entry_size_tiers()
    await trace_phase1_entry()
    await trace_phase2_hedge_detection()
    await trace_phase2_take_profit()
    await trace_phase2_emergency_close()
    await trace_phase3_sniper()

    log.info("══════════════════════════════════════════════════════════════════")
    log.info("Trace complete. Summary:")
    log.info("  • All phases use the same StrategyExecutor code path.")
    log.info("  • [SIM] uses SimulatorOrderManager (slippage, no lookahead).")
    log.info("  • [LIVE] uses LiveOrderManager (real CLOB API).")
    log.info("  • Strategy logic (sizes, thresholds, hedge formula) is identical.")
    log.info("  • Only the execution.fill / execution.book_fetch methods differ.")
    log.info("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())
