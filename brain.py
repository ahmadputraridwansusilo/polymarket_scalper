"""
brain.py — BTC Micro-Timeframe Momentum Hedger (strategy engine).

Execution flow per market tick:
  1. Settlement    — resolve expired markets, claim winning tokens.
  2. Phase 1       — Momentum burst entry.
                     Entry gate: oracle delta > SAFE_MARGIN AND OBI signal agrees
                     AND probability model edge >= MIN_EDGE.
                     Sizing: fixed ~77 tokens/bet (spend scales with token price).
                     Exits: take-profit at +15%, OBI-reversal bailout, or
                     hold-to-settlement when bid is high near expiry.
  3. Layered hedges — OTM / perfect / god-tier limit orders placed as the
                     Phase 1 position matures inside the hedge windows.
  4. Phase 2       — Near-expiry TWAP sniper (fires in last 5–15 seconds).
                     Confidence multiplier (1×–3×) scales position with edge.
                     Hard cap: max 10% of total equity per market.

All markets (BTC/ETH/SOL × 5m/15m) are evaluated concurrently via
asyncio.gather, so multiple markets can fire simultaneously (straddle effect).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from enum import Enum
import logging
import math
import random
import time
from typing import Dict, Optional

import config
from calculations import compute_prob_edge, lognormal_win_probability
from executioner import OrderBookSnapshot, PortfolioSnapshot, Side, build_order_manager
from oracle import MarketSnapshot, Oracle

log = logging.getLogger("brain")


class Phase1Status(str, Enum):
    IDLE = "Idle"
    SPAMMING = "Spamming"
    HOLDING = "Holding"
    HOLD_TO_SETTLEMENT = "Hold to Settlement"
    EXITED = "Exited"


class Phase2Status(str, Enum):
    IDLE = "Idle"
    BLOCKED = "Blocked"
    SWEEPING = "Sweeping"
    EXECUTED = "Executed"
    ABORTED = "Aborted"


@dataclass(frozen=True)
class HedgePlan:
    side: Side
    token_id: str
    ask_price: float
    limit_price: float
    payout_coverage_usdc: float
    spend_usdc: float
    target_shares: float
    phase: str


@dataclass(frozen=True)
class Phase2SniperPlan:
    winning_side: Side
    winning_token: str
    winning_ask_price: float
    losing_side: Side
    losing_token: str
    losing_ask_price: float
    total_exposure_usdc: float
    winning_allocation_usdc: float
    insurance_allocation_usdc: float
    insurance_limit_price: float
    chunk_sizes_usdc: tuple[float, ...]


@dataclass(frozen=True)
class PlanDecision:
    plan: Optional[object]
    reason: str = ""


@dataclass
class MarketState:
    condition_id: str
    market_label: str
    asset: str
    timeframe: str
    phase1_status: Phase1Status = Phase1Status.IDLE
    phase2_status: Phase2Status = Phase2Status.IDLE
    settled: bool = False
    mixed_exposure: bool = False
    position_side: Optional[Side] = None
    position_source: str = ""
    position_token_id: str = ""
    position_cost: float = 0.0
    position_shares: float = 0.0
    avg_entry_price: float = 0.0
    anchor_budget_cap: float = 0.0
    phase1_detail: str = "-"
    last_anchor_burst_at: float = 0.0
    last_phase1_obi_poll_at: float = 0.0
    last_exit_at: float = 0.0
    obi_against_polls: int = 0
    hold_to_settlement: bool = False
    straddle_placed: bool = False
    last_obi_value: float = 0.5
    last_obi_signal: str = "NEUTRAL"
    phase2_bullets_fired: int = 0
    phase2_spend: float = 0.0
    phase2_expected_profit: float = 0.0
    abort_reason: str = ""
    last_phase2_block_bucket: str = ""
    last_phase2_block_log_at: float = 0.0
    hedge_detail: str = "-"
    perfect_hedge_placed: bool = False
    otm_hedge_fired: bool = False
    god_hedge_placed: bool = False
    phase2_insurance_placed: bool = False
    phase2_chunk_target: int = 0
    phase2_target_exposure: float = 0.0
    phase2_winning_allocation: float = 0.0
    phase2_insurance_allocation: float = 0.0
    last_settlement_attempt: float = 0.0
    next_settlement_attempt_at: float = 0.0
    settlement_price_at_expiry: float = 0.0
    settlement_winning_side: Optional[Side] = None

    @property
    def has_position(self) -> bool:
        return self.position_shares > 1e-9 and self.position_cost > 1e-9


@dataclass(frozen=True)
class OrderBookImbalance:
    value: float = 0.5
    signal: str = "NEUTRAL"


@dataclass(frozen=True)
class MarketView:
    condition_id: str
    market_label: str
    asset: str
    timeframe: str
    strike_price: float
    oracle_price: float
    delta: float
    margin_text: str
    best_ask_up: float
    best_ask_down: float
    end_time: float
    obi_value: float
    obi_signal: str
    phase: str
    detail: str

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - time.time())


@dataclass(frozen=True)
class DashboardSnapshot:
    prices: Dict[str, float]
    portfolio: PortfolioSnapshot
    markets: list[MarketView]
    oracle_status: str
    wins: int
    losses: int
    win_rate: float
    session_start_balance: float


class Brain:
    def __init__(self, oracle: Oracle) -> None:
        self._oracle = oracle
        self._exec = build_order_manager()
        self._states: Dict[str, MarketState] = {}
        self._phase2_tasks: Dict[str, asyncio.Task[None]] = {}
        self._wins = 0
        self._losses = 0
        self._sync_lock = asyncio.Lock()
        self._session_start_balance = float(
            getattr(self._exec, "total_equity", config.INITIAL_BALANCE)
        )

    @property
    def win_rate(self) -> float:
        total = self._wins + self._losses
        return (self._wins / total * 100.0) if total else 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        log.info("Brain started (DRY_RUN=%s).", config.DRY_RUN)
        last_balance_sync = time.time()

        while True:
            async with self._sync_lock:
                await self._tick()

            if not config.DRY_RUN and hasattr(self._exec, "sync_live_balance"):
                now = time.time()
                if now - last_balance_sync >= 60.0:
                    try:
                        await self._exec.sync_live_balance()
                        last_balance_sync = now
                    except Exception as exc:
                        log.error("[LIVE] Periodic balance sync failed: %s", exc)

            await asyncio.sleep(config.BRAIN_TICK_INTERVAL)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    async def get_dashboard_snapshot(self) -> DashboardSnapshot:
        prices = self._oracle.price_table()
        active_markets = list(self._oracle.active_markets())
        all_markets = list(self._oracle.all_markets())
        states = {
            condition_id: replace(state)
            for condition_id, state in self._states.items()
        }

        portfolio = await self._exec.get_portfolio_snapshot()
        dashboard_markets = self._dashboard_market_snapshots(
            active_markets=active_markets,
            all_markets=all_markets,
            states=states,
        )
        market_obis = await self._fetch_market_obis(
            [snap for snap in dashboard_markets if snap.time_remaining > 0]
        )
        markets = [
            self._market_view(
                snap,
                state=states.get(snap.condition_id),
                obi=market_obis.get(snap.condition_id, OrderBookImbalance()),
            )
            for snap in dashboard_markets
        ]
        return DashboardSnapshot(
            prices=prices,
            portfolio=portfolio,
            markets=markets,
            oracle_status=self._oracle.status_line(),
            wins=self._wins,
            losses=self._losses,
            win_rate=self.win_rate,
            session_start_balance=self._session_start_balance,
        )

    @staticmethod
    def _dashboard_market_snapshots(
        *,
        active_markets: list[MarketSnapshot],
        all_markets: list[MarketSnapshot],
        states: Dict[str, MarketState],
    ) -> list[MarketSnapshot]:
        active_by_key = {
            (snap.asset, snap.timeframe): snap
            for snap in active_markets
        }
        closed_by_key: Dict[tuple[str, str], MarketSnapshot] = {}
        for snap in all_markets:
            if snap.time_remaining > 0:
                continue
            key = (snap.asset, snap.timeframe)
            current = closed_by_key.get(key)
            if current is None or snap.end_time > current.end_time:
                closed_by_key[key] = snap

        dashboard_markets: list[MarketSnapshot] = []
        for asset, timeframe in config.TRACKED_MARKETS:
            snap = active_by_key.get((asset, timeframe))
            if snap is None:
                snap = closed_by_key.get((asset, timeframe))
            if snap is not None:
                dashboard_markets.append(snap)
        return dashboard_markets

    # ------------------------------------------------------------------
    # Per-tick evaluation (called every BRAIN_TICK_INTERVAL seconds)
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        tracked = {snap.condition_id: snap for snap in self._oracle.all_markets()}

        settlement_tasks = []
        for condition_id, state in list(self._states.items()):
            snap = tracked.get(condition_id)
            if not snap or state.settled:
                continue
            if snap.time_remaining <= 0:
                settlement_tasks.append(self._settle_market(snap, state))

        if settlement_tasks:
            await asyncio.gather(*settlement_tasks)

        active_markets = self._oracle.active_markets()
        if not active_markets:
            return

        await asyncio.gather(*[self._evaluate_market(snap) for snap in active_markets])

    async def _evaluate_market(self, snap: MarketSnapshot) -> None:
        state = self._state_for(snap)
        await self._process_limit_orders(snap, state)
        await self._refresh_market_position_state(state)

        if snap.time_remaining <= 0:
            return

        # Straddle fires once at market open (before hedges / Phase 1 / Phase 2).
        await self._maybe_execute_straddle(snap, state)
        await self._refresh_market_position_state(state)

        await self._run_layered_hedges(snap, state)

        if self._phase2_window_open(snap):
            await self.execute_phase_2(snap, state)
            return

        if state.mixed_exposure:
            state.phase1_status = Phase1Status.HOLDING
            label = "Straddle active" if state.straddle_placed else "Layered hedge active"
            state.phase1_detail = f"{label}; holding both sides"
            return

        if state.has_position:
            if state.position_source == "PHASE2":
                state.phase1_status = Phase1Status.HOLDING
                state.phase1_detail = "TWAP sniper owns exposure"
            else:
                await self.execute_phase_1(snap, state)
            return

        if snap.time_remaining > config.PHASE2_WINDOW_START:
            await self.execute_phase_1(snap, state)
            return

        state.phase1_status = Phase1Status.IDLE
        state.phase1_detail = "Waiting for hedge/sniper windows"

    async def _process_limit_orders(self, snap: MarketSnapshot, state: MarketState) -> None:
        fills = await self._exec.process_limit_crosses(
            condition_id=snap.condition_id,
            market_label=state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            best_ask_up=snap.best_ask_up,
            best_ask_down=snap.best_ask_down,
        )
        if fills:
            await self._refresh_market_position_state(state)
            log.info(
                "[%s] Limit hedge fill(s) | %s | count=%d",
                snap.condition_id[:10],
                state.market_label,
                len(fills),
            )

    async def _run_layered_hedges(self, snap: MarketSnapshot, state: MarketState) -> None:
        if snap.time_remaining <= config.PHASE2_WINDOW_END:
            return

        perfect_done = await self._maybe_execute_perfect_hedge(snap, state)
        otm_done = await self._maybe_execute_time_delayed_hedge(snap, state)
        god_done = await self._maybe_execute_god_tier_hedge(snap, state)
        if perfect_done or otm_done or god_done:
            self._update_hedge_detail(state)

    # ------------------------------------------------------------------
    # Phase 1 — Momentum burst
    # ------------------------------------------------------------------

    async def execute_phase_1(self, snap: MarketSnapshot, state: MarketState) -> None:
        up_book, down_book = await asyncio.gather(
            self._book_for_side(snap, Side.UP, snap.up_token_id),
            self._book_for_side(snap, Side.DOWN, snap.down_token_id),
        )
        if up_book is None or down_book is None:
            state.phase1_detail = "Waiting order book"
            return

        obi_value = self._derive_obi_value(snap, up_book, down_book)
        state.last_obi_value = obi_value
        state.last_obi_signal = self._obi_signal(obi_value)

        if state.has_position:
            if state.position_source not in {"PHASE1", ""}:
                state.phase1_status = Phase1Status.HOLDING
                state.phase1_detail = "Late sniper owns exposure"
                return

            anchor_book = up_book if state.position_side == Side.UP else down_book
            if anchor_book is None:
                state.phase1_status = Phase1Status.HOLDING
                state.phase1_detail = "Monitoring legacy core exposure"
                return

            if self._should_hold_to_settlement(snap, state, anchor_book):
                state.phase1_status = Phase1Status.HOLD_TO_SETTLEMENT
                state.phase1_detail = (
                    f"Whale hold | bid={anchor_book.best_bid:.4f}"
                    f" | entry={state.avg_entry_price:.4f}"
                )
                return

            # Exit checks — TP first, then bailout.
            if await self._maybe_take_profit_position(snap, state, anchor_book):
                return
            if await self._maybe_bailout_position(snap, state, anchor_book, obi_value):
                return

            # Continue spamming if oracle + OBI still agree with open position.
            safe, delta_abs, margin = self._oracle_gate(snap)
            bias = self._winning_side_from_oracle(snap)
            obi_side = self._phase1_target_side(obi_value)
            if safe and bias == state.position_side and obi_side == state.position_side:
                if await self._maybe_fire_phase1_burst(
                    snap=snap, state=state, side=obi_side,
                    obi_value=obi_value, up_book=up_book, down_book=down_book,
                ):
                    return

            state.phase1_status = Phase1Status.HOLDING
            state.phase1_detail = (
                f"Holding {state.position_side.value}"
                f" | bid={anchor_book.best_bid:.4f}"
                f" | oracle={'SAFE' if safe else 'WAIT'}"
                f" | OBI={obi_value:.3f}"
            )
            return

        # --- No position — look for entry ---
        state.phase1_status = Phase1Status.IDLE
        state.hold_to_settlement = False
        state.obi_against_polls = 0

        if snap.time_remaining <= config.PHASE2_WINDOW_START:
            state.phase1_detail = "Waiting sniper window"
            return

        if snap.strike_price <= 0:
            state.phase1_detail = "Waiting strike snapshot"
            return

        safe, delta_abs, margin = self._oracle_gate(snap)
        bias = self._winning_side_from_oracle(snap)
        if not safe:
            state.phase1_detail = (
                f"Waiting safe oracle | bias={bias.value if bias else '-'}"
                f" | delta={delta_abs:.2f} <= {margin:.2f}"
            )
            return

        # Spread arb: enter the underpriced side when combined ask < 1 − threshold.
        # This fires before the OBI check — pricing gap is a stronger signal than OBI.
        spread_side = self._spread_arb_side(up_book, down_book)
        if spread_side is not None:
            spread = 1.0 - (up_book.best_ask + down_book.best_ask)
            state.phase1_detail = (
                f"Spread arb {spread_side.value}"
                f" | spread={spread:.3f}"
                f" | OBI={obi_value:.3f}"
            )
            await self._maybe_fire_phase1_burst(
                snap=snap, state=state, side=spread_side,
                obi_value=obi_value, up_book=up_book, down_book=down_book,
            )
            return

        obi_side = self._phase1_target_side(obi_value)
        if obi_side is None:
            state.phase1_detail = (
                f"Waiting OBI signal | bias={bias.value if bias else '-'}"
                f" | delta={delta_abs:.2f} | OBI={obi_value:.3f}"
            )
            return

        if obi_side != bias:
            # OBI and oracle point in opposite directions — wait for alignment.
            state.phase1_detail = (
                f"OBI/oracle diverge | OBI={obi_side.value} oracle={bias.value if bias else '-'}"
                f" | delta={delta_abs:.2f} | OBI={obi_value:.3f}"
            )
            return

        # Oracle gate clear AND OBI agrees → fire burst.
        # All tracked markets (5m + 15m) reach this via asyncio.gather, so both
        # timeframes can fire simultaneously when conditions align (straddle effect).
        await self._maybe_fire_phase1_burst(
            snap=snap, state=state, side=obi_side,
            obi_value=obi_value, up_book=up_book, down_book=down_book,
        )

    # ------------------------------------------------------------------
    # Phase 2 — Near-expiry TWAP sniper
    # ------------------------------------------------------------------

    async def execute_phase_2(self, snap: MarketSnapshot, state: MarketState) -> None:
        if snap.time_remaining <= config.PHASE2_WINDOW_END:
            if state.phase2_bullets_fired > 0:
                state.phase2_status = Phase2Status.EXECUTED
                state.abort_reason = ""
            return

        if snap.time_remaining > config.PHASE2_WINDOW_START:
            state.phase2_status = Phase2Status.IDLE
            state.abort_reason = ""
            return

        existing_task = self._phase2_tasks.get(state.condition_id)
        if existing_task and not existing_task.done():
            state.phase2_status = Phase2Status.SWEEPING
            state.abort_reason = ""
            return
        if state.phase2_status == Phase2Status.ABORTED and state.phase2_bullets_fired > 0:
            return

        if snap.strike_price <= 0:
            self._block_phase2(state, "Strike unavailable")
            return

        safe, delta_abs, margin = self._oracle_gate(snap)
        if not safe:
            self._block_phase2(
                state,
                f"Danger zone | delta={delta_abs:.2f} <= {margin:.2f}",
            )
            return

        winning_side = self._winning_side_from_oracle(snap)
        if winning_side is None:
            self._block_phase2(state, "Oracle price unavailable")
            return

        if state.phase2_chunk_target > 0 and state.phase2_bullets_fired >= state.phase2_chunk_target:
            state.phase2_status = Phase2Status.EXECUTED
            state.abort_reason = ""
            return

        winning_token = snap.up_token_id if winning_side == Side.UP else snap.down_token_id
        losing_side = Side.DOWN if winning_side == Side.UP else Side.UP
        losing_token = snap.down_token_id if winning_side == Side.UP else snap.up_token_id
        winning_book, losing_book = await asyncio.gather(
            self._book_for_side(snap, winning_side, winning_token),
            self._book_for_side(snap, losing_side, losing_token),
        )
        if winning_book is None or losing_book is None:
            self._block_phase2(state, "Winning or losing book unavailable")
            return

        # Price conviction gate — only scale up when the market has already
        # committed (token price >= PHASE2_MIN_WINNING_PRICE, default 0.80).
        # Avoids dumping large capital on a still-uncertain 50/50 outcome.
        if winning_book.best_ask < config.PHASE2_MIN_WINNING_PRICE:
            self._block_phase2(
                state,
                f"Waiting for conviction | ask={winning_book.best_ask:.3f}"
                f" < min={config.PHASE2_MIN_WINNING_PRICE:.2f}",
            )
            return

        # Probability edge gate — only enter when model says Polymarket is underpricing.
        edge, win_prob, vol = self._probability_edge(snap, winning_side, winning_book.best_ask)
        if edge < config.PROB_MODEL_MIN_EDGE:
            self._block_phase2(
                state,
                f"Edge too low | edge={edge:+.3f} < {config.PROB_MODEL_MIN_EDGE:.3f}"
                f" | P(win)={win_prob:.3f} | vol={vol:.2f}",
            )
            log.debug(
                "[%s] Phase 2 blocked by prob model | %s | side=%s | ask=%.4f"
                " | P=%.3f | edge=%+.3f | vol=%.2f",
                snap.condition_id[:10],
                state.market_label,
                winning_side.value,
                winning_book.best_ask,
                win_prob,
                edge,
                vol,
            )
            return

        confidence_mult = self._confidence_multiplier(edge)
        log.debug(
            "[%s] Phase 2 confidence | %s | edge=%+.3f | mult=%.2fx | P(win)=%.3f | vol=%.2f",
            snap.condition_id[:10],
            state.market_label,
            edge,
            confidence_mult,
            win_prob,
            vol,
        )
        decision = await self._build_phase2_plan(
            snap=snap,
            winning_side=winning_side,
            winning_token=winning_token,
            losing_side=losing_side,
            losing_token=losing_token,
            winning_book=winning_book,
            losing_book=losing_book,
            confidence_multiplier=confidence_mult,
        )
        if decision.plan is None:
            self._block_phase2(state, decision.reason)
            return

        plan = decision.plan
        assert isinstance(plan, Phase2SniperPlan)
        state.phase2_target_exposure = plan.total_exposure_usdc
        state.phase2_winning_allocation = plan.winning_allocation_usdc
        state.phase2_insurance_allocation = plan.insurance_allocation_usdc
        state.phase2_chunk_target = len(plan.chunk_sizes_usdc)
        state.phase2_status = Phase2Status.SWEEPING
        state.abort_reason = ""

        if not state.phase2_insurance_placed:
            insurance_order = await self._exec.place_limit_order(
                condition_id=snap.condition_id,
                market_label=state.market_label,
                asset=snap.asset,
                timeframe=snap.timeframe,
                token_id=plan.losing_token,
                side=plan.losing_side,
                price=plan.insurance_limit_price,
                size_usdc=plan.insurance_allocation_usdc,
                phase="PHASE2_INSURANCE",
            )
            if insurance_order is not None:
                state.phase2_insurance_placed = True
                self._update_hedge_detail(state)
                await self._process_limit_orders(snap, state)
                log.info(
                    "[%s] Phase 2 insurance armed | %s | side=%s | size=%.2f | limit=%.4f",
                    snap.condition_id[:10],
                    state.market_label,
                    plan.losing_side.value,
                    plan.insurance_allocation_usdc,
                    plan.insurance_limit_price,
                )

        task = asyncio.create_task(
            self._run_phase2_twap(
                condition_id=snap.condition_id,
                plan=plan,
            )
        )
        self._phase2_tasks[state.condition_id] = task

    async def _run_phase2_twap(
        self,
        *,
        condition_id: str,
        plan: Phase2SniperPlan,
    ) -> None:
        try:
            for idx, chunk_size in enumerate(plan.chunk_sizes_usdc):
                async with self._sync_lock:
                    snap = self._oracle.get_market(condition_id)
                    state = self._states.get(condition_id)
                    if snap is None or state is None:
                        return
                    if not self._phase2_window_open(snap):
                        state.phase2_status = Phase2Status.EXECUTED
                        state.abort_reason = "Sniper window closed"
                        return

                    if snap.strike_price <= 0:
                        self._abort_phase2(state, "Strike unavailable during TWAP")
                        return

                    safe, delta_abs, margin = self._oracle_gate(snap)
                    if not safe:
                        self._abort_phase2(
                            state,
                            f"Oracle drifted back into danger zone | delta={delta_abs:.2f} <= {margin:.2f}",
                        )
                        return

                    winning_side = self._winning_side_from_oracle(snap)
                    if winning_side != plan.winning_side:
                        self._abort_phase2(
                            state,
                            f"Oracle bias flipped to {winning_side.value if winning_side else '?'}",
                        )
                        return

                    winning_book = await self._book_for_side(
                        snap,
                        plan.winning_side,
                        plan.winning_token,
                    )
                    if winning_book is None or winning_book.best_ask <= 0:
                        self._abort_phase2(state, "Winning-side book unavailable during TWAP")
                        return

                    # Always use the live book ask — both SIM and LIVE fetch
                    # winning_book fresh above, so this keeps both modes identical.
                    expected_fill_price = winning_book.best_ask
                    target_shares = chunk_size / expected_fill_price if expected_fill_price > 0 else 0.0
                    fill = await self._exec.execute_taker_buy(
                        condition_id=snap.condition_id,
                        market_label=state.market_label,
                        asset=snap.asset,
                        timeframe=snap.timeframe,
                        token_id=plan.winning_token,
                        side=plan.winning_side,
                        phase="PHASE2_SNIPER",
                        aggressive_price=config.SNIPER_LIMIT_PRICE,
                        expected_fill_price=expected_fill_price,
                        target_shares=target_shares,
                        max_size_usdc=chunk_size,
                    )
                    if fill is None:
                        if state.phase2_bullets_fired == 0:
                            self._abort_phase2(state, "TWAP chunk failed to fill")
                        else:
                            state.abort_reason = "TWAP stopped after partial fills"
                            state.phase2_status = Phase2Status.ABORTED
                        return

                    state.position_source = "PHASE2"
                    state.position_side = plan.winning_side
                    state.position_token_id = plan.winning_token
                    state.phase2_bullets_fired += 1
                    state.phase2_spend += fill.size
                    state.phase2_expected_profit += max(0.0, fill.shares - fill.size)
                    state.phase2_status = Phase2Status.SWEEPING
                    state.abort_reason = ""
                    await self._refresh_market_position_state(state)

                    log.info(
                        "[%s] Phase 2 TWAP chunk | %s | side=%s | chunk=%d/%d | spend=%.2f | fill=%.4f",
                        snap.condition_id[:10],
                        state.market_label,
                        plan.winning_side.value,
                        state.phase2_bullets_fired,
                        len(plan.chunk_sizes_usdc),
                        fill.size,
                        fill.price,
                    )

                if idx < len(plan.chunk_sizes_usdc) - 1:
                    await asyncio.sleep(
                        random.uniform(
                            config.PHASE2_SLEEP_MIN_SECONDS,
                            config.PHASE2_SLEEP_MAX_SECONDS,
                        )
                    )

            async with self._sync_lock:
                state = self._states.get(condition_id)
                if state is not None:
                    state.phase2_status = Phase2Status.EXECUTED
                    state.abort_reason = ""
        finally:
            current_task = self._phase2_tasks.get(condition_id)
            if current_task is asyncio.current_task():
                self._phase2_tasks.pop(condition_id, None)

    # ------------------------------------------------------------------
    # Layered hedges (limit orders placed as Phase 1 position matures)
    # ------------------------------------------------------------------

    async def _maybe_execute_perfect_hedge(
        self,
        snap: MarketSnapshot,
        state: MarketState,
    ) -> bool:
        if (
            state.perfect_hedge_placed
            or not self._window_open(
                snap.time_remaining,
                config.PERFECT_HEDGE_WINDOW_START,
                config.PERFECT_HEDGE_WINDOW_END,
            )
        ):
            return False

        winning_side = self._winning_side_from_oracle(snap)
        if winning_side is None:
            return False

        losing_side = Side.DOWN if winning_side == Side.UP else Side.UP
        losing_token = snap.down_token_id if winning_side == Side.UP else snap.up_token_id
        losing_book = await self._book_for_side(snap, losing_side, losing_token)
        if losing_book is None:
            return False

        decision = await self._build_perfect_hedge_plan(
            snap=snap,
            winning_side=winning_side,
            losing_side=losing_side,
            losing_token=losing_token,
            losing_book=losing_book,
        )
        if decision.plan is None:
            return False

        plan = decision.plan
        assert isinstance(plan, HedgePlan)
        order = await self._exec.place_limit_order(
            condition_id=snap.condition_id,
            market_label=state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=plan.token_id,
            side=plan.side,
            price=plan.limit_price,
            size_usdc=plan.spend_usdc,
            phase=plan.phase,
        )
        if order is None:
            return False

        state.perfect_hedge_placed = True
        self._update_hedge_detail(state)
        await self._process_limit_orders(snap, state)
        log.info(
            "[%s] Perfect hedge armed | %s | side=%s | shares=%.2f | cost=%.2f | ask=%.4f",
            snap.condition_id[:10],
            state.market_label,
            plan.side.value,
            plan.target_shares,
            plan.spend_usdc,
            plan.ask_price,
        )
        return True

    async def _maybe_execute_time_delayed_hedge(
        self,
        snap: MarketSnapshot,
        state: MarketState,
    ) -> bool:
        if (
            state.otm_hedge_fired
            or not self._window_open(
                snap.time_remaining,
                config.OTM_HEDGE_WINDOW_START,
                config.OTM_HEDGE_WINDOW_END,
            )
        ):
            return False

        momentum = self._oracle.price_momentum(snap.asset, lookback_seconds=30.0)
        winning_side = self._winning_side_from_oracle(snap)
        if winning_side is None or momentum == 0:
            return False
        if (momentum > 0 and winning_side != Side.UP) or (momentum < 0 and winning_side != Side.DOWN):
            return False

        losing_side = Side.DOWN if winning_side == Side.UP else Side.UP
        losing_token = snap.down_token_id if winning_side == Side.UP else snap.up_token_id
        losing_book = await self._book_for_side(snap, losing_side, losing_token)
        if losing_book is None or losing_book.best_ask <= 0 or losing_book.best_ask > config.OTM_HEDGE_MAX_ASK:
            return False

        portfolio = await self._exec.get_portfolio_snapshot()
        spend = self._otm_hedge_spend_usdc(
            portfolio.total_equity,
            losing_book.best_ask,
        )
        if spend <= 0:
            return False

        target_shares = spend / losing_book.best_ask if losing_book.best_ask > 0 else 0.0
        fill = await self._exec.execute_taker_buy(
            condition_id=snap.condition_id,
            market_label=state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=losing_token,
            side=losing_side,
            phase="OTM_HEDGE",
            aggressive_price=config.OTM_HEDGE_MAX_ASK,
            expected_fill_price=losing_book.best_ask,
            target_shares=target_shares,
            max_size_usdc=spend,
        )
        if fill is None:
            return False

        state.otm_hedge_fired = True
        await self._refresh_market_position_state(state)
        self._update_hedge_detail(state)
        log.info(
            "[%s] Time-delayed OTM hedge | %s | side=%s | spend=%.2f | ask=%.4f",
            snap.condition_id[:10],
            state.market_label,
            losing_side.value,
            fill.size,
            fill.price,
        )
        return True

    async def _maybe_execute_god_tier_hedge(
        self,
        snap: MarketSnapshot,
        state: MarketState,
    ) -> bool:
        if (
            state.god_hedge_placed
            or not self._window_open(
                snap.time_remaining,
                config.GOD_HEDGE_WINDOW_START,
                config.GOD_HEDGE_WINDOW_END,
            )
        ):
            return False

        winning_side = self._winning_side_from_oracle(snap)
        if winning_side is None:
            return False

        losing_side = Side.DOWN if winning_side == Side.UP else Side.UP
        losing_token = snap.down_token_id if winning_side == Side.UP else snap.up_token_id
        losing_book = await self._book_for_side(snap, losing_side, losing_token)
        if losing_book is None:
            return False

        decision = await self._build_god_tier_hedge_plan(
            snap=snap,
            winning_side=winning_side,
            losing_side=losing_side,
            losing_token=losing_token,
            losing_book=losing_book,
        )
        if decision.plan is None:
            return False

        plan = decision.plan
        assert isinstance(plan, HedgePlan)
        order = await self._exec.place_limit_order(
            condition_id=snap.condition_id,
            market_label=state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=plan.token_id,
            side=plan.side,
            price=plan.limit_price,
            size_usdc=plan.spend_usdc,
            phase=plan.phase,
        )
        if order is None:
            return False

        state.god_hedge_placed = True
        self._update_hedge_detail(state)
        await self._process_limit_orders(snap, state)
        log.info(
            "[%s] God-tier hedge armed | %s | side=%s | shares=%.2f | cost=%.2f | limit=%.4f",
            snap.condition_id[:10],
            state.market_label,
            plan.side.value,
            plan.target_shares,
            plan.spend_usdc,
            plan.limit_price,
        )
        return True

    # ------------------------------------------------------------------
    # Straddle — buy both sides simultaneously at market open
    # ------------------------------------------------------------------

    async def _maybe_execute_straddle(
        self,
        snap: MarketSnapshot,
        state: MarketState,
    ) -> None:
        """
        Place a symmetric UP+DOWN position when the market first becomes active.
        This creates a delta-neutral straddle; Phase 2 then scales into the
        winning side near expiry.  Fires at most once per market.
        """
        if not config.STRADDLE_ENABLED or state.straddle_placed:
            return
        if not (config.STRADDLE_ENTRY_MIN_SECONDS
                <= snap.time_remaining
                <= config.STRADDLE_ENTRY_MAX_SECONDS):
            return
        if not snap.up_token_id or not snap.down_token_id:
            return

        up_book, down_book = await asyncio.gather(
            self._book_for_side(snap, Side.UP, snap.up_token_id),
            self._book_for_side(snap, Side.DOWN, snap.down_token_id),
        )
        if up_book is None or down_book is None:
            return
        if up_book.best_ask <= 0 or down_book.best_ask <= 0:
            return

        target_tokens = config.PHASE1_TARGET_TOKENS
        up_size = max(
            round(target_tokens * up_book.best_ask, 2),
            config.minimum_taker_order_usdc(up_book.best_ask),
        )
        down_size = max(
            round(target_tokens * down_book.best_ask, 2),
            config.minimum_taker_order_usdc(down_book.best_ask),
        )

        portfolio = await self._exec.get_portfolio_snapshot()
        if up_size + down_size > portfolio.available_balance * config.STRADDLE_MAX_BALANCE_RATIO:
            return

        up_fill, down_fill = await asyncio.gather(
            self._exec.execute_taker_buy(
                condition_id=snap.condition_id,
                market_label=state.market_label,
                asset=snap.asset,
                timeframe=snap.timeframe,
                token_id=snap.up_token_id,
                side=Side.UP,
                phase="STRADDLE",
                aggressive_price=self._aggressive_price(up_book, phase="PHASE1"),
                expected_fill_price=up_book.best_ask,
                target_shares=up_size / up_book.best_ask,
                max_size_usdc=up_size,
            ),
            self._exec.execute_taker_buy(
                condition_id=snap.condition_id,
                market_label=state.market_label,
                asset=snap.asset,
                timeframe=snap.timeframe,
                token_id=snap.down_token_id,
                side=Side.DOWN,
                phase="STRADDLE",
                aggressive_price=self._aggressive_price(down_book, phase="PHASE1"),
                expected_fill_price=down_book.best_ask,
                target_shares=down_size / down_book.best_ask,
                max_size_usdc=down_size,
            ),
        )

        # Mark placed even if one side failed — avoid repeated attempts.
        state.straddle_placed = True

        log.info(
            "[%s] Straddle | %s | up_ask=%.4f up_size=%.2f up=%s"
            " | down_ask=%.4f down_size=%.2f down=%s",
            snap.condition_id[:10],
            state.market_label,
            up_book.best_ask, up_size, "ok" if up_fill else "fail",
            down_book.best_ask, down_size, "ok" if down_fill else "fail",
        )

    # ------------------------------------------------------------------
    # Phase 1 execution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spread_arb_side(
        up_book: Optional[OrderBookSnapshot],
        down_book: Optional[OrderBookSnapshot],
    ) -> Optional[Side]:
        """
        Returns the cheaper side when combined ask < 1 − SPREAD_ARB_THRESHOLD.
        A spread gap means one side is underpriced relative to the guaranteed
        $1.00 payout — enter the cheaper side as a value / arb bet.
        Returns None when spread is insufficient or inputs are invalid.
        """
        if not config.SPREAD_ARB_ENABLED:
            return None
        if up_book is None or down_book is None:
            return None
        if up_book.best_ask <= 0 or down_book.best_ask <= 0:
            return None
        spread = 1.0 - (up_book.best_ask + down_book.best_ask)
        if spread < config.SPREAD_ARB_THRESHOLD:
            return None
        return Side.UP if up_book.best_ask <= down_book.best_ask else Side.DOWN

    async def _maybe_fire_phase1_burst(
        self,
        *,
        snap: MarketSnapshot,
        state: MarketState,
        side: Side,
        obi_value: float,
        up_book: OrderBookSnapshot,
        down_book: OrderBookSnapshot,
    ) -> bool:
        if state.has_position and state.position_side != side:
            state.phase1_detail = (
                f"Opposite momentum ignored | holding {state.position_side.value}"
            )
            return False

        if not self._spam_ready(max(state.last_anchor_burst_at, state.last_exit_at)):
            state.phase1_detail = f"Cooldown | OBI={obi_value:.3f}"
            return False

        anchor_book = up_book if side == Side.UP else down_book
        anchor_token = snap.up_token_id if side == Side.UP else snap.down_token_id
        if anchor_book.best_ask <= 0 or anchor_book.best_ask >= 1.0 or not anchor_token:
            state.phase1_detail = "No taker ask on target side"
            return False

        portfolio = await self._exec.get_portfolio_snapshot()
        if state.anchor_budget_cap <= 0:
            state.anchor_budget_cap = self._phase1_anchor_cap_usdc(portfolio.total_equity)

        if state.anchor_budget_cap <= 0:
            state.phase1_detail = "No bankroll available"
            return False

        # Token-based sizing: spend = target_tokens × ask_price.
        # At ask=0.10 → $7.70; at ask=0.90 → $69.30 — naturally scales with conviction.
        # Falls back to min-order USDC if TOKEN mode is disabled or target is zero.
        if config.PHASE1_SIZING_MODE == "TOKEN" and config.PHASE1_TARGET_TOKENS > 0:
            child_size = round(config.PHASE1_TARGET_TOKENS * anchor_book.best_ask, 2)
            child_size = max(child_size, config.minimum_taker_order_usdc(anchor_book.best_ask))
        else:
            child_size = self._phase1_child_order_usdc(anchor_book.best_ask)
        remaining_cap = max(0.0, state.anchor_budget_cap - state.position_cost)
        top_book_usdc = anchor_book.best_ask * anchor_book.best_ask_size
        spend = min(child_size, remaining_cap, portfolio.available_balance, top_book_usdc)

        if spend + 1e-9 < child_size:
            state.phase1_status = Phase1Status.HOLDING if state.has_position else Phase1Status.IDLE
            state.phase1_detail = (
                f"Cap/liquidity block | spend=${state.position_cost:.2f}/{state.anchor_budget_cap:.2f}"
            )
            return False

        fill = await self._exec.execute_taker_buy(
            condition_id=snap.condition_id,
            market_label=state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=anchor_token,
            side=side,
            phase="PHASE1_SPAM",
            aggressive_price=self._aggressive_price(anchor_book, phase="PHASE1"),
            expected_fill_price=anchor_book.best_ask,
            target_shares=spend / anchor_book.best_ask,
            max_size_usdc=spend,
        )
        if fill is None:
            state.phase1_detail = "Burst rejected by execution layer"
            return False

        state.position_source = "PHASE1"
        state.position_side = side
        state.position_token_id = anchor_token
        state.last_anchor_burst_at = time.time()

        await self._refresh_market_position_state(state)
        state.phase1_status = Phase1Status.SPAMMING
        state.phase1_detail = (
            f"Spam {side.value} | OBI={obi_value:.3f}"
            f" | spend=${state.position_cost:.2f}/{state.anchor_budget_cap:.2f}"
        )

        log.info(
            "[%s] Phase 1 burst | %s | side=%s | obi=%.3f | ask=%.4f | spend=%.2f | total=%.2f | cap=%.2f",
            snap.condition_id[:10],
            state.market_label,
            side.value,
            obi_value,
            anchor_book.best_ask,
            fill.size,
            state.position_cost,
            state.anchor_budget_cap,
        )
        return True

    async def _maybe_take_profit_position(
        self,
        snap: MarketSnapshot,
        state: MarketState,
        anchor_book: OrderBookSnapshot,
    ) -> bool:
        if (
            state.position_source != "PHASE1"
            or state.position_side is None
            or state.position_shares <= 0
            or state.avg_entry_price <= 0
            or anchor_book.best_bid <= 0
        ):
            return False

        target_bid = state.avg_entry_price * (1.0 + config.PHASE1_TAKE_PROFIT_RATIO)
        if anchor_book.best_bid + 1e-9 < target_bid:
            return False

        prior_cost = state.position_cost
        prior_shares = state.position_shares
        fill = await self._exec.execute_taker_sell(
            condition_id=snap.condition_id,
            market_label=state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=state.position_token_id,
            side=state.position_side,
            phase="PHASE1_TP",
            target_shares=state.position_shares,
            expected_fill_price=anchor_book.best_bid,
        )
        if fill is None:
            return False

        realized = fill.size - (
            prior_cost * (fill.shares / prior_shares if prior_shares > 0 else 0.0)
        )
        await self._refresh_market_position_state(state)
        state.phase1_status = Phase1Status.EXITED
        state.phase1_detail = f"TP {realized:+.4f}"
        state.last_exit_at = time.time()
        state.anchor_budget_cap = 0.0
        state.obi_against_polls = 0
        state.hold_to_settlement = False

        log.info(
            "[%s] Take profit | %s | side=%s | bid=%.4f | target=%.4f | realized=%+.4f",
            snap.condition_id[:10],
            state.market_label,
            fill.side,
            anchor_book.best_bid,
            target_bid,
            realized,
        )
        return True

    async def _maybe_bailout_position(
        self,
        snap: MarketSnapshot,
        state: MarketState,
        anchor_book: OrderBookSnapshot,
        obi_value: float,
    ) -> bool:
        if (
            state.position_source != "PHASE1"
            or state.position_side is None
            or state.position_shares <= 0
            or state.avg_entry_price <= 0
            or anchor_book.best_bid <= 0
        ):
            return False

        stop_bid = state.avg_entry_price * (1.0 - config.PHASE1_BID_DRAWDOWN_STOP_RATIO)
        if anchor_book.best_bid <= stop_bid + 1e-9:
            return await self._sell_phase1_position(
                snap=snap,
                state=state,
                fill_price=anchor_book.best_bid,
                phase="PHASE1_CUTLOSS",
                reason=(
                    f"Bid drawdown stop | bid={anchor_book.best_bid:.4f}"
                    f" <= {stop_bid:.4f}"
                ),
            )

        if self._obi_poll_ready(state.last_phase1_obi_poll_at):
            state.last_phase1_obi_poll_at = time.time()
            if self._obi_heavily_against(state.position_side, obi_value):
                state.obi_against_polls += 1
            else:
                state.obi_against_polls = 0

        if state.obi_against_polls >= config.PHASE1_OBI_STOP_POLLS:
            return await self._sell_phase1_position(
                snap=snap,
                state=state,
                fill_price=anchor_book.best_bid,
                phase="PHASE1_CUTLOSS",
                reason=(
                    f"Anti-spoof bailout | against={state.obi_against_polls}"
                    f" | OBI={obi_value:.3f}"
                ),
            )

        if state.obi_against_polls > 0:
            state.phase1_detail = (
                f"Against OBI {state.obi_against_polls}/{config.PHASE1_OBI_STOP_POLLS}"
                f" | OBI={obi_value:.3f}"
            )
        return False

    async def _sell_phase1_position(
        self,
        *,
        snap: MarketSnapshot,
        state: MarketState,
        fill_price: float,
        phase: str,
        reason: str,
    ) -> bool:
        if state.position_side is None or state.position_shares <= 0:
            return False

        prior_cost = state.position_cost
        prior_shares = state.position_shares
        fill = await self._exec.execute_taker_sell(
            condition_id=snap.condition_id,
            market_label=state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            token_id=state.position_token_id,
            side=state.position_side,
            phase=phase,
            target_shares=state.position_shares,
            expected_fill_price=fill_price,
        )
        if fill is None:
            return False

        realized = fill.size - (
            prior_cost * (fill.shares / prior_shares if prior_shares > 0 else 0.0)
        )
        await self._refresh_market_position_state(state)
        state.phase1_status = Phase1Status.EXITED
        state.phase1_detail = f"{reason} | realized={realized:+.4f}"
        state.last_exit_at = time.time()
        state.anchor_budget_cap = 0.0
        state.obi_against_polls = 0
        state.hold_to_settlement = False

        log.info(
            "[%s] Phase 1 exit | %s | reason=%s | realized=%+.4f",
            snap.condition_id[:10],
            state.market_label,
            reason,
            realized,
        )
        return True

    # ------------------------------------------------------------------
    # Phase 2 plan builders
    # ------------------------------------------------------------------

    async def _build_phase2_plan(
        self,
        *,
        snap: MarketSnapshot,
        winning_side: Side,
        winning_token: str,
        losing_side: Side,
        losing_token: str,
        winning_book: OrderBookSnapshot,
        losing_book: OrderBookSnapshot,
        confidence_multiplier: float = 1.0,
    ) -> PlanDecision:
        if not winning_token or not losing_token:
            return PlanDecision(plan=None, reason="Missing winning or losing token id")
        if winning_book.best_ask <= 0 or winning_book.best_ask >= 1.0:
            return PlanDecision(
                plan=None,
                reason=f"winning_ask={winning_book.best_ask:.4f} is not < 1.00",
            )
        if losing_book.best_ask <= 0:
            return PlanDecision(plan=None, reason="Losing-side ask is unavailable")

        portfolio = await self._exec.get_portfolio_snapshot()
        base_exposure = self._sniper_target_exposure_usdc(portfolio.available_balance)
        if base_exposure <= 0:
            return PlanDecision(plan=None, reason="No bankroll available for Phase 2")

        # Scale exposure by confidence: higher edge → larger position.
        # Two hard caps applied identically in SIM and LIVE:
        #   1. Per-market cap: max 10% of total equity (PHASE2_MAX_MARKET_EXPOSURE_RATIO)
        #   2. Available balance guard: never exceed 95% of what's liquid
        per_market_cap = portfolio.total_equity * config.PHASE2_MAX_MARKET_EXPOSURE_RATIO
        total_exposure = min(
            base_exposure * confidence_multiplier,
            per_market_cap,
            portfolio.available_balance * 0.95,
        )
        total_exposure = math.floor((total_exposure * 100.0) + 1e-9) / 100.0
        if total_exposure <= 0:
            return PlanDecision(plan=None, reason="No bankroll available for Phase 2")

        winning_allocation = self._sniper_winning_allocation_usdc(total_exposure)
        insurance_allocation = max(0.0, round(total_exposure - winning_allocation, 2))
        min_chunk_usdc = config.minimum_taker_order_usdc(winning_book.best_ask)
        chunk_sizes = self._randomized_chunk_spends(
            winning_allocation,
            min_chunk_usdc,
            seed=f"{snap.condition_id}:{snap.end_time}:{winning_side.value}",
        )
        if not chunk_sizes:
            return PlanDecision(
                plan=None,
                reason=(
                    f"twap_alloc={winning_allocation:.4f} below minimum "
                    f"{config.PHASE2_MIN_BULLETS}x child_size={min_chunk_usdc:.4f}"
                ),
            )

        return PlanDecision(
            plan=Phase2SniperPlan(
                winning_side=winning_side,
                winning_token=winning_token,
                winning_ask_price=winning_book.best_ask,
                losing_side=losing_side,
                losing_token=losing_token,
                losing_ask_price=losing_book.best_ask,
                total_exposure_usdc=total_exposure,
                winning_allocation_usdc=winning_allocation,
                insurance_allocation_usdc=insurance_allocation,
                insurance_limit_price=self._phase2_insurance_limit_price(losing_book.best_ask),
                chunk_sizes_usdc=chunk_sizes,
            )
        )

    async def _build_perfect_hedge_plan(
        self,
        *,
        snap: MarketSnapshot,
        winning_side: Side,
        losing_side: Side,
        losing_token: str,
        losing_book: OrderBookSnapshot,
    ) -> PlanDecision:
        if losing_book.best_ask <= 0 or losing_book.best_ask > config.PERFECT_HEDGE_MAX_ASK:
            return PlanDecision(plan=None, reason="Perfect hedge ask threshold not met")

        portfolio = await self._exec.get_portfolio_snapshot()
        expected_sniper_loss = self._sniper_target_exposure_usdc(portfolio.available_balance)
        if expected_sniper_loss <= 0:
            return PlanDecision(plan=None, reason="No projected sniper risk to hedge")

        existing_coverage = self._existing_losing_coverage_usdc(
            portfolio,
            snap.condition_id,
            losing_side,
        )
        payout_target = max(0.0, expected_sniper_loss - existing_coverage)
        if payout_target <= 0:
            return PlanDecision(plan=None, reason="Existing hedge coverage already sufficient")

        hedge_cost = round(payout_target * losing_book.best_ask, 2)
        if hedge_cost > portfolio.total_equity * config.PERFECT_HEDGE_MAX_EQUITY_RATIO:
            return PlanDecision(
                plan=None,
                reason=(
                    f"perfect hedge cost={hedge_cost:.4f} exceeds "
                    f"{config.PERFECT_HEDGE_MAX_EQUITY_RATIO:.2%} of equity"
                ),
            )

        return PlanDecision(
            plan=HedgePlan(
                side=losing_side,
                token_id=losing_token,
                ask_price=losing_book.best_ask,
                limit_price=losing_book.best_ask,
                payout_coverage_usdc=payout_target,
                spend_usdc=hedge_cost,
                target_shares=payout_target,
                phase="PERFECT_HEDGE",
            )
        )

    async def _build_god_tier_hedge_plan(
        self,
        *,
        snap: MarketSnapshot,
        winning_side: Side,
        losing_side: Side,
        losing_token: str,
        losing_book: OrderBookSnapshot,
    ) -> PlanDecision:
        if losing_book.best_ask <= 0 or losing_book.best_ask > config.GOD_HEDGE_MAX_ASK:
            return PlanDecision(plan=None, reason="God-tier ask threshold not met")

        portfolio = await self._exec.get_portfolio_snapshot()
        current_locked_risk = self._side_cost_basis_usdc(
            portfolio,
            snap.condition_id,
            winning_side,
        )
        projected_sniper_risk = self._sniper_target_exposure_usdc(portfolio.available_balance)
        total_risk_to_hedge = current_locked_risk + projected_sniper_risk
        if total_risk_to_hedge <= 0:
            return PlanDecision(plan=None, reason="No risk on winning side to hedge")

        existing_coverage = self._existing_losing_coverage_usdc(
            portfolio,
            snap.condition_id,
            losing_side,
        )
        payout_target = max(0.0, total_risk_to_hedge - existing_coverage)
        if payout_target <= 0:
            return PlanDecision(plan=None, reason="Existing hedge coverage already sufficient")

        hedge_cost = round(payout_target * losing_book.best_ask, 2)
        if hedge_cost > portfolio.available_balance * config.GOD_HEDGE_MAX_BALANCE_RATIO:
            return PlanDecision(
                plan=None,
                reason=(
                    f"god hedge cost={hedge_cost:.4f} exceeds "
                    f"{config.GOD_HEDGE_MAX_BALANCE_RATIO:.2%} of balance"
                ),
            )

        return PlanDecision(
            plan=HedgePlan(
                side=losing_side,
                token_id=losing_token,
                ask_price=losing_book.best_ask,
                limit_price=losing_book.best_ask,
                payout_coverage_usdc=payout_target,
                spend_usdc=hedge_cost,
                target_shares=payout_target,
                phase="GOD_HEDGE",
            )
        )

    @staticmethod
    def _window_open(time_remaining: float, start: float, end: float) -> bool:
        return end <= time_remaining <= start

    @staticmethod
    def _sniper_target_exposure_usdc(available_balance: float) -> float:
        if available_balance <= 0:
            return 0.0
        raw = min(
            config.MAX_SNIPE_LIMIT,
            available_balance * config.SNIPER_TARGET_BALANCE_RATIO,
        )
        return math.floor((raw * 100.0) + 1e-9) / 100.0

    @staticmethod
    def _sniper_winning_allocation_usdc(total_exposure: float) -> float:
        if total_exposure <= 0:
            return 0.0
        raw = total_exposure * config.SNIPER_WINNING_ALLOCATION_RATIO
        return math.floor((raw * 100.0) + 1e-9) / 100.0

    @staticmethod
    def _phase2_insurance_limit_price(losing_ask: float) -> float:
        bounded = losing_ask if losing_ask > 0 else config.SNIPER_INSURANCE_MAX_PRICE
        bounded = max(config.SNIPER_INSURANCE_MIN_PRICE, bounded)
        bounded = min(config.SNIPER_INSURANCE_MAX_PRICE, bounded)
        return round(bounded, 4)

    @staticmethod
    def _otm_hedge_spend_usdc(total_equity: float, losing_ask: float) -> float:
        if total_equity <= 0 or losing_ask <= 0:
            return 0.0

        ask_floor = 0.01
        ask_ceiling = max(ask_floor, config.OTM_HEDGE_MAX_ASK)
        clamped_ask = min(ask_ceiling, max(ask_floor, losing_ask))
        if math.isclose(ask_ceiling, ask_floor, abs_tol=1e-9):
            collapse_ratio = 1.0
        else:
            collapse_ratio = (ask_ceiling - clamped_ask) / (ask_ceiling - ask_floor)

        min_ratio = min(
            config.OTM_HEDGE_MIN_EQUITY_RATIO,
            config.OTM_HEDGE_MAX_EQUITY_RATIO,
        )
        max_ratio = max(
            config.OTM_HEDGE_MIN_EQUITY_RATIO,
            config.OTM_HEDGE_MAX_EQUITY_RATIO,
        )
        midpoint_ratio = min(
            max(config.OTM_HEDGE_ALLOCATION_RATIO, min_ratio),
            max_ratio,
        )

        if collapse_ratio <= 0.5:
            target_ratio = min_ratio + (
                (midpoint_ratio - min_ratio) * (collapse_ratio / 0.5)
            )
        else:
            target_ratio = midpoint_ratio + (
                (max_ratio - midpoint_ratio) * ((collapse_ratio - 0.5) / 0.5)
            )

        spend = min(config.OTM_HEDGE_MAX_USDC, total_equity * target_ratio)
        return math.floor((spend * 100.0) + 1e-9) / 100.0


    @staticmethod
    def _randomized_chunk_spends(
        total_usdc: float,
        min_chunk_usdc: float,
        *,
        seed: str,
    ) -> tuple[float, ...]:
        total_cents = int(round(max(0.0, total_usdc) * 100))
        min_chunk_cents = int(math.ceil(max(0.0, min_chunk_usdc) * 100.0 - 1e-9))
        if total_cents <= 0 or min_chunk_cents <= 0:
            return ()

        max_chunks = min(config.PHASE2_MAX_BULLETS, total_cents // min_chunk_cents)
        if max_chunks < config.PHASE2_MIN_BULLETS:
            return ()

        rng = random.Random(seed)
        chunk_count = rng.randint(config.PHASE2_MIN_BULLETS, max_chunks)
        extra_cents = total_cents - (chunk_count * min_chunk_cents)
        weights = [rng.uniform(0.5, 1.5) for _ in range(chunk_count)]
        total_weight = sum(weights)
        raw_extra = [extra_cents * (weight / total_weight) for weight in weights]
        extra_alloc = [int(math.floor(value)) for value in raw_extra]
        remainder = extra_cents - sum(extra_alloc)
        if remainder > 0:
            ranked = sorted(
                range(chunk_count),
                key=lambda idx: raw_extra[idx] - extra_alloc[idx],
                reverse=True,
            )
            for idx in ranked[:remainder]:
                extra_alloc[idx] += 1

        spends = tuple((min_chunk_cents + extra_alloc[idx]) / 100.0 for idx in range(chunk_count))
        if round(sum(spends), 2) != round(total_usdc, 2):
            return ()
        return spends

    @staticmethod
    def _side_cost_basis_usdc(
        portfolio: PortfolioSnapshot,
        condition_id: str,
        side: Side,
    ) -> float:
        side_map = portfolio.active_positions.get(condition_id, {})
        position = side_map.get(side.value)
        return position.cost_basis if position is not None else 0.0

    @staticmethod
    def _existing_losing_coverage_usdc(
        portfolio: PortfolioSnapshot,
        condition_id: str,
        side: Side,
    ) -> float:
        side_map = portfolio.active_positions.get(condition_id, {})
        position = side_map.get(side.value)
        return position.shares if position is not None else 0.0

    def _update_hedge_detail(self, state: MarketState) -> None:
        bits: list[str] = []
        if state.perfect_hedge_placed:
            bits.append("Perfect hedge")
        if state.otm_hedge_fired:
            bits.append("OTM hedge")
        if state.god_hedge_placed:
            bits.append("God hedge")
        if state.phase2_insurance_placed:
            bits.append("Sniper insurance")
        state.hedge_detail = " | ".join(bits) if bits else "-"

    async def _refresh_market_position_state(self, state: MarketState) -> PortfolioSnapshot:
        portfolio = await self._exec.get_portfolio_snapshot()
        side_map = portfolio.active_positions.get(state.condition_id, {})
        state.mixed_exposure = len(side_map) > 1

        tracked = None
        if state.position_side is not None:
            tracked = side_map.get(state.position_side.value)
        if tracked is None and side_map:
            tracked = max(side_map.values(), key=lambda position: position.cost_basis)

        if tracked is None:
            self._clear_position_fields(state)
            return portfolio

        state.position_side = Side(tracked.side)
        state.position_token_id = tracked.token_id
        state.position_cost = tracked.cost_basis
        state.position_shares = tracked.shares
        state.avg_entry_price = tracked.avg_entry_price

        if state.mixed_exposure:
            state.position_source = "MIXED"
        elif state.position_source not in {"PHASE1", "PHASE2"}:
            state.position_source = "PHASE1"

        if state.position_source == "PHASE1" and state.anchor_budget_cap <= 0:
            state.anchor_budget_cap = self._phase1_anchor_cap_usdc(
                portfolio.total_equity
            )

        return portfolio

    def _clear_position_fields(self, state: MarketState) -> None:
        state.mixed_exposure = False
        state.position_side = None
        state.position_source = ""
        state.position_token_id = ""
        state.position_cost = 0.0
        state.position_shares = 0.0
        state.avg_entry_price = 0.0

    def _should_hold_to_settlement(
        self,
        snap: MarketSnapshot,
        state: MarketState,
        anchor_book: OrderBookSnapshot,
    ) -> bool:
        if state.hold_to_settlement:
            return True
        if (
            snap.time_remaining < config.PHASE1_HOLD_TO_SETTLEMENT_SECONDS
            and anchor_book.best_bid >= config.PHASE1_HOLD_TO_SETTLEMENT_BID
        ):
            state.hold_to_settlement = True
            log.info(
                "[%s] Hold-to-settlement | %s | side=%s | bid=%.4f",
                snap.condition_id[:10],
                state.market_label,
                state.position_side.value if state.position_side else "-",
                anchor_book.best_bid,
            )
            return True
        return False

    @staticmethod
    def _market_obi(
        up_book: Optional[OrderBookSnapshot],
        down_book: Optional[OrderBookSnapshot],
    ) -> float:
        if up_book is None or down_book is None:
            return 0.5

        up_bid_volume = up_book.bid_volume
        down_bid_volume = down_book.bid_volume
        combined_bid_volume = up_bid_volume + down_bid_volume
        if (
            combined_bid_volume > 0
            and not math.isclose(up_bid_volume, down_bid_volume, abs_tol=1e-9)
        ):
            return up_bid_volume / combined_bid_volume

        up_mid = Brain._book_mid(up_book)
        down_mid = Brain._book_mid(down_book)
        combined_mid = up_mid + down_mid
        if combined_mid > 0:
            return up_mid / combined_mid

        return 0.5

    @staticmethod
    def _derive_obi_value(
        snap: MarketSnapshot,
        up_book: Optional[OrderBookSnapshot],
        down_book: Optional[OrderBookSnapshot],
    ) -> float:
        obi_value = Brain._market_obi(up_book, down_book)
        if abs(obi_value - 0.5) >= 0.05:
            return obi_value
        oracle_fallback = Brain._oracle_delta_obi(snap)
        return oracle_fallback if oracle_fallback is not None else obi_value

    @staticmethod
    def _book_obi(book: Optional[OrderBookSnapshot]) -> float:
        if book is None:
            return 0.5
        bid_volume = book.bid_volume
        ask_volume = book.ask_volume
        total_volume = bid_volume + ask_volume
        if total_volume <= 0:
            return 0.5
        return bid_volume / total_volume

    @staticmethod
    def _book_mid(book: Optional[OrderBookSnapshot]) -> float:
        if book is None:
            return 0.0
        if book.best_bid > 0 and book.best_ask > 0:
            return (book.best_bid + book.best_ask) / 2.0
        return book.best_bid or book.best_ask

    @staticmethod
    def _oracle_delta_obi(snap: MarketSnapshot) -> Optional[float]:
        if snap.binance_live_price <= 0 or snap.strike_price <= 0:
            return None
        margin = config.safe_margin_for(snap.asset)
        if margin <= 0:
            return None
        delta = snap.binance_live_price - snap.strike_price
        ratio = abs(delta) / margin
        if ratio <= 0:
            return None
        offset = min(0.49, 0.20 * ratio)
        return max(0.01, min(0.99, 0.5 + offset if delta > 0 else 0.5 - offset))

    @staticmethod
    def _phase1_child_order_usdc(price: float) -> float:
        return config.minimum_taker_order_usdc(price)

    @staticmethod
    def _phase2_window_open(snap: MarketSnapshot) -> bool:
        return config.PHASE2_WINDOW_END <= snap.time_remaining <= config.PHASE2_WINDOW_START

    @staticmethod
    def _position_cap_usdc(total_balance: float) -> float:
        if total_balance <= 0:
            return 0.0
        raw = total_balance * config.POSITION_MAX_BALANCE_RATIO
        return math.floor((raw * 100.0) + 1e-9) / 100.0

    @staticmethod
    def _phase1_anchor_cap_usdc(total_balance: float) -> float:
        if total_balance <= 0:
            return 0.0
        raw = total_balance * config.PHASE1_MARKET_CAP_RATIO
        return math.floor((raw * 100.0) + 1e-9) / 100.0

    @staticmethod
    def _spam_ready(last_ts: float) -> bool:
        return time.time() - last_ts >= config.PHASE1_BURST_INTERVAL_SECONDS

    @staticmethod
    def _obi_poll_ready(last_ts: float) -> bool:
        return time.time() - last_ts >= config.PHASE1_BURST_INTERVAL_SECONDS

    @staticmethod
    def _phase1_target_side(obi_value: float) -> Optional[Side]:
        if obi_value > config.PHASE1_OBI_UP_THRESHOLD:
            return Side.UP
        if obi_value < config.PHASE1_OBI_DOWN_THRESHOLD:
            return Side.DOWN
        return None

    @staticmethod
    def _obi_heavily_against(position_side: Side, obi_value: float) -> bool:
        if position_side == Side.UP:
            return obi_value < config.PHASE1_OBI_STOP_UP_THRESHOLD
        return obi_value > config.PHASE1_OBI_STOP_DOWN_THRESHOLD

    @staticmethod
    def _obi_signal(obi_value: float) -> str:
        if obi_value > config.PHASE1_OBI_UP_THRESHOLD:
            return "BULLISH UP"
        if obi_value < config.PHASE1_OBI_DOWN_THRESHOLD:
            return "BEARISH DOWN"
        return "NEUTRAL"

    @staticmethod
    def _winning_side_from_oracle(snap: MarketSnapshot) -> Optional[Side]:
        if snap.binance_live_price <= 0 or snap.strike_price <= 0:
            return None
        return Side.UP if snap.binance_live_price >= snap.strike_price else Side.DOWN

    def _oracle_gate(self, snap: MarketSnapshot) -> tuple[bool, float, float]:
        if snap.binance_live_price <= 0 or snap.strike_price <= 0:
            return False, 0.0, config.safe_margin_for(snap.asset)
        delta_abs = abs(snap.binance_live_price - snap.strike_price)
        margin = config.safe_margin_for(snap.asset)
        return delta_abs > margin, delta_abs, margin

    # ------------------------------------------------------------------
    # Probability model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_multiplier(edge: float) -> float:
        """
        Scale Phase 2 exposure based on probability edge magnitude.
        At MIN_EDGE (4%) → 1x baseline.
        Each CONFIDENCE_SCALE (15%) of extra edge → +1x, capped at MAX_MULT (3x).
        Example: edge=0.04 → 1x; edge=0.19 → 2x; edge=0.34 → 3x.
        """
        if edge <= config.PROB_MODEL_MIN_EDGE:
            return 1.0
        extra = (edge - config.PROB_MODEL_MIN_EDGE) / max(config.PROB_MODEL_CONFIDENCE_SCALE, 1e-6)
        return min(1.0 + extra, config.PROB_MODEL_MAX_CONFIDENCE_MULT)

    def _probability_edge(
        self,
        snap: MarketSnapshot,
        winning_side: Side,
        token_ask: float,
    ) -> tuple[float, float, float]:
        """
        Returns (edge, win_probability, ann_volatility).

        edge = model_probability − token_ask_price
        Positive edge → Polymarket underpricing this outcome → enter.
        Negative edge → overpriced → skip.

        When PROB_MODEL_ENABLED=false, always returns (MIN_EDGE, 0.5, 0.0)
        so the gate is bypassed transparently.
        """
        if not config.PROB_MODEL_ENABLED:
            return config.PROB_MODEL_MIN_EDGE, 0.5, 0.0

        vol = self._oracle.realized_volatility(
            snap.asset, config.PROB_MODEL_LOOKBACK_SECONDS
        )
        if vol < 0.01:
            vol = config.PROB_MODEL_FALLBACK_VOL.get(snap.asset, 0.70)

        win_prob = lognormal_win_probability(
            spot=snap.binance_live_price,
            strike=snap.strike_price,
            time_remaining_seconds=snap.time_remaining,
            ann_volatility=vol,
        )
        if winning_side == Side.DOWN:
            win_prob = 1.0 - win_prob

        edge = compute_prob_edge(win_prob, token_ask)
        return edge, win_prob, vol

    def _aggressive_price(self, book: OrderBookSnapshot, *, phase: str) -> float:
        if book.best_ask <= 0:
            return 0.0
        tick = book.tick_size if book.tick_size > 0 else 0.01
        ticks = (
            config.PHASE1_TAKER_PRICE_TICKS
            if phase.startswith("PHASE1")
            else config.PHASE2_TAKER_PRICE_TICKS
        )
        aggressive = book.best_ask + (tick * ticks)
        return min(config.SNIPER_LIMIT_PRICE, round(aggressive, 4))

    async def _book_for_side(
        self,
        snap: MarketSnapshot,
        side: Side,
        token_id: str,
    ) -> Optional[OrderBookSnapshot]:
        if not token_id:
            return None
        fallback_best_ask = snap.best_ask_up if side == Side.UP else snap.best_ask_down
        fallback_best_bid = snap.best_bid_up if side == Side.UP else snap.best_bid_down
        return await self._exec.get_order_book(
            token_id,
            fallback_best_ask=fallback_best_ask,
            fallback_best_bid=fallback_best_bid,
        )

    def _phase1_budget_status(self, state: Optional[MarketState]) -> tuple[float, float]:
        if state and state.anchor_budget_cap > 0:
            anchor_cap = state.anchor_budget_cap
        else:
            anchor_cap = self._phase1_anchor_cap_usdc(self._session_start_balance)
        locked = state.position_cost if state else 0.0
        return locked, anchor_cap

    async def _settle_market(self, snap: MarketSnapshot, state: MarketState) -> None:
        if state.settled:
            return

        task = self._phase2_tasks.pop(snap.condition_id, None)
        if task is not None and not task.done():
            task.cancel()

        await self._refresh_market_position_state(state)

        now = time.time()
        retry_interval = config.SETTLEMENT_RETRY_INTERVAL
        if state.next_settlement_attempt_at and now < state.next_settlement_attempt_at:
            return
        if state.last_settlement_attempt and now - state.last_settlement_attempt < retry_interval:
            return

        has_live_exposure = state.has_position
        if has_live_exposure and state.settlement_winning_side is None:
            if now < snap.end_time + config.SETTLEMENT_INITIAL_CLAIM_DELAY:
                return

        oracle_price = state.settlement_price_at_expiry or snap.binance_live_price
        if oracle_price <= 0:
            return

        if state.settlement_winning_side is None:
            state.settlement_price_at_expiry = oracle_price
            state.settlement_winning_side = (
                Side.UP if oracle_price >= snap.strike_price else Side.DOWN
            )

        winning_side = state.settlement_winning_side
        state.last_settlement_attempt = now
        result = await self._exec.settle_market(
            condition_id=snap.condition_id,
            market_label=state.market_label,
            asset=snap.asset,
            timeframe=snap.timeframe,
            winning_side=winning_side,
        )

        if not result.settled:
            reason_text = result.error or "claim not ready"
            if "result for condition not received yet" in reason_text.lower():
                state.next_settlement_attempt_at = now + config.SETTLEMENT_PENDING_RETRY_INTERVAL
            else:
                state.next_settlement_attempt_at = now + config.SETTLEMENT_RETRY_INTERVAL
            log.info(
                "[%s] Settlement pending | %s | winner=%s | ref_price=%.4f | reason=%s",
                snap.condition_id[:10],
                state.market_label,
                winning_side.value,
                state.settlement_price_at_expiry,
                reason_text,
            )
            return

        if result.realized_pnl > 0:
            self._wins += 1
        elif result.realized_pnl < 0:
            self._losses += 1

        state.settled = True
        state.next_settlement_attempt_at = 0.0

        log.info(
            "[%s] Settled | %s | winner=%s | ref_price=%.4f | realized=%+.4f",
            snap.condition_id[:10],
            state.market_label,
            winning_side.value,
            state.settlement_price_at_expiry,
            result.realized_pnl,
        )

    def _state_for(self, snap: MarketSnapshot) -> MarketState:
        market_label = self._market_label_for(snap)
        if snap.condition_id not in self._states:
            self._states[snap.condition_id] = MarketState(
                condition_id=snap.condition_id,
                market_label=market_label,
                asset=snap.asset,
                timeframe=snap.timeframe,
            )
        else:
            state = self._states[snap.condition_id]
            state.market_label = market_label
            state.asset = snap.asset
            state.timeframe = snap.timeframe
        return self._states[snap.condition_id]

    @staticmethod
    def _market_label_for(snap: MarketSnapshot) -> str:
        end_label = time.strftime("%H:%M", time.localtime(snap.end_time))
        return f"{snap.asset} {snap.timeframe} @{end_label}"

    async def _fetch_market_obis(
        self,
        markets: list[MarketSnapshot],
    ) -> Dict[str, OrderBookImbalance]:
        if not markets:
            return {}
        obis = await asyncio.gather(*[self._fetch_market_obi(snap) for snap in markets])
        return {condition_id: obi for condition_id, obi in obis}

    async def _fetch_market_obi(
        self,
        snap: MarketSnapshot,
    ) -> tuple[str, OrderBookImbalance]:
        try:
            up_book, down_book = await asyncio.gather(
                self._book_for_side(snap, Side.UP, snap.up_token_id),
                self._book_for_side(snap, Side.DOWN, snap.down_token_id),
            )
        except Exception as exc:
            log.debug("[%s] OBI fetch failed: %s", snap.condition_id[:10], exc)
            return snap.condition_id, OrderBookImbalance()

        obi_value = self._derive_obi_value(snap, up_book, down_book)
        return snap.condition_id, OrderBookImbalance(
            value=obi_value,
            signal=self._obi_signal(obi_value),
        )

    def _market_view(
        self,
        snap: MarketSnapshot,
        *,
        state: Optional[MarketState],
        obi: OrderBookImbalance,
    ) -> MarketView:
        if snap.time_remaining <= 0:
            phase, detail = self._closed_market_phase(state)
            locked = state.position_cost if state is not None else 0.0
        else:
            if state is None:
                phase = "Scanning"
                detail = "-"
                locked = 0.0
            else:
                phase, detail = self._phase_text(state)
                locked = state.position_cost

        _, anchor_cap = self._phase1_budget_status(state)
        margin_text = f"Cap: ${anchor_cap:.2f} | Lkd: ${locked:.2f}"

        return MarketView(
            condition_id=snap.condition_id,
            market_label=f"{snap.asset} {snap.timeframe}",
            asset=snap.asset,
            timeframe=snap.timeframe,
            strike_price=snap.strike_price,
            oracle_price=snap.binance_live_price,
            delta=snap.binance_live_price - snap.strike_price,
            margin_text=margin_text,
            best_ask_up=snap.best_ask_up,
            best_ask_down=snap.best_ask_down,
            end_time=snap.end_time,
            obi_value=obi.value,
            obi_signal=obi.signal,
            phase=phase,
            detail=detail,
        )

    @staticmethod
    def _closed_market_phase(state: Optional[MarketState]) -> tuple[str, str]:
        if state is None:
            return "Closed", "Awaiting next market"
        if state.settled:
            return "Settled", "-"
        now = time.time()
        if state.next_settlement_attempt_at and state.next_settlement_attempt_at > now:
            wait_seconds = max(1, int(state.next_settlement_attempt_at - now))
            return "Settlement Pending", f"Retry in {wait_seconds}s"
        if state.last_settlement_attempt > 0:
            return "Settlement Pending", "Claim retry in progress"
        if state.has_position or state.mixed_exposure or state.phase2_bullets_fired > 0:
            return "Closing", "Waiting settlement window"
        if state.hedge_detail and state.hedge_detail != "-":
            return "Closing", state.hedge_detail
        return "Closed", "Awaiting next market"

    def _phase_text(self, state: MarketState) -> tuple[str, str]:
        if state.settled:
            return "Settled", "-"
        if state.mixed_exposure:
            if state.phase2_bullets_fired > 0:
                return (
                    "Mixed + Sniper",
                    (
                        f"Chunks {state.phase2_bullets_fired}/{state.phase2_chunk_target or config.PHASE2_MAX_BULLETS}"
                        f" | EV {state.phase2_expected_profit:+.4f}"
                    ),
                )
            return "Mixed Exposure", state.hedge_detail or state.phase1_detail or "Holding to settlement"
        if state.position_source == "PHASE1" and state.phase1_status == Phase1Status.HOLD_TO_SETTLEMENT:
            return "Hold to Settlement", state.phase1_detail or "Whale hold"
        if state.position_source == "PHASE1" and state.phase1_status == Phase1Status.SPAMMING:
            return "Phase 1 Spam", state.phase1_detail or "-"
        if state.position_source == "PHASE1" and state.has_position:
            return "Phase 1 Hold", state.phase1_detail or "-"
        if state.position_source == "PHASE2" and state.has_position:
            label = (
                "Phase 2 TWAP"
                if state.phase2_status == Phase2Status.SWEEPING
                else "Phase 2 Hold"
            )
            detail = (
                state.abort_reason
                or (
                    f"Chunks {state.phase2_bullets_fired}/{state.phase2_chunk_target or config.PHASE2_MAX_BULLETS}"
                    f" | EV {state.phase2_expected_profit:+.4f}"
                )
            )
            return label, detail
        if state.phase2_status == Phase2Status.BLOCKED:
            return "Phase 2 Blocked", state.abort_reason or "-"
        if state.phase2_status == Phase2Status.ABORTED:
            return "Phase 2 Aborted", state.abort_reason or "-"
        if state.hedge_detail and state.hedge_detail != "-":
            return "Layered Hedge", state.hedge_detail
        if state.phase1_status == Phase1Status.EXITED:
            return "Phase 1 Exited", state.phase1_detail or "-"
        return "Scanning", state.phase1_detail or "-"

    @staticmethod
    def _phase2_reason_bucket(reason: str) -> str:
        if reason.startswith("Danger zone"):
            return "danger-zone"
        if reason.startswith("Strike unavailable"):
            return "strike-unavailable"
        if reason.startswith("twap_alloc="):
            return "twap-cap-vs-min-order"
        if reason == "Oracle price unavailable":
            return "oracle-price-unavailable"
        if reason == "Winning or losing book unavailable":
            return "winning-book-unavailable"
        return reason

    def _block_phase2(self, state: MarketState, reason: str) -> None:
        now = time.time()
        bucket = self._phase2_reason_bucket(reason)
        changed_bucket = (
            state.phase2_status != Phase2Status.BLOCKED
            or state.last_phase2_block_bucket != bucket
        )
        cooled_down = (
            now - state.last_phase2_block_log_at
            >= config.PHASE2_BLOCK_LOG_COOLDOWN_SECONDS
        )

        state.phase2_status = Phase2Status.BLOCKED
        state.abort_reason = reason

        if changed_bucket or cooled_down:
            state.last_phase2_block_bucket = bucket
            state.last_phase2_block_log_at = now
            log.warning(
                "[%s] Phase 2 blocked | %s | %s",
                state.condition_id[:10],
                state.market_label,
                reason,
            )

    def _abort_phase2(self, state: MarketState, reason: str) -> None:
        changed = state.phase2_status != Phase2Status.ABORTED or state.abort_reason != reason
        state.phase2_status = Phase2Status.ABORTED
        state.abort_reason = reason
        if changed:
            log.warning(
                "[%s] Phase 2 aborted | %s | %s",
                state.condition_id[:10],
                state.market_label,
                reason,
            )
