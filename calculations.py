"""
calculations.py — Pure, stateless math helpers for the strategy.

No I/O, no config reads, no side effects.  Every function takes plain numbers
and returns plain numbers so they can be unit-tested trivially.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Phase 1 hedge plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase1LockPlan:
    base_amount: float
    entry_price: float
    entry_shares: float
    hedge_price: float
    current_hedge_shares: float
    current_hedge_cost: float
    hedge_shares_needed: float
    hedge_cost: float
    total_hedge_cost: float
    max_hedge_budget: float
    remaining_hedge_budget: float
    guaranteed_profit: float


def compute_phase1_max_hedge_budget(
    *,
    base_amount: float,
    entry_shares: float,
    target_minimum_profit: float,
    max_acceptable_loss: float = 0.0,
) -> float:
    if base_amount <= 0 or entry_shares <= 0:
        return 0.0
    minimum_guaranteed_profit = (
        -max(0.0, max_acceptable_loss) if max_acceptable_loss > 0 else target_minimum_profit
    )
    return max(0.0, entry_shares - base_amount - minimum_guaranteed_profit)


def compute_phase1_lock_plan(
    *,
    base_amount: float,
    entry_price: float,
    hedge_price: float,
    target_minimum_profit: float,
    entry_shares: float | None = None,
    current_hedge_shares: float = 0.0,
    current_hedge_cost: float = 0.0,
    max_acceptable_loss: float = 0.0,
) -> Phase1LockPlan | None:
    if base_amount <= 0:
        return None
    if entry_shares is None:
        if not 0 < entry_price < 1:
            return None
        entry_shares = base_amount / entry_price
    if entry_shares <= 0:
        return None
    if not 0 < hedge_price < 1:
        return None

    current_hedge_shares = max(0.0, current_hedge_shares)
    current_hedge_cost = max(0.0, current_hedge_cost)
    hedge_shares_needed = max(0.0, entry_shares - current_hedge_shares)
    hedge_cost = hedge_shares_needed * hedge_price
    total_hedge_cost = current_hedge_cost + hedge_cost
    max_hedge_budget = compute_phase1_max_hedge_budget(
        base_amount=base_amount,
        entry_shares=entry_shares,
        target_minimum_profit=target_minimum_profit,
        max_acceptable_loss=max_acceptable_loss,
    )
    remaining_hedge_budget = max(0.0, max_hedge_budget - current_hedge_cost)
    guaranteed_profit = entry_shares - base_amount - total_hedge_cost

    return Phase1LockPlan(
        base_amount=base_amount,
        entry_price=entry_price,
        entry_shares=entry_shares,
        hedge_price=hedge_price,
        current_hedge_shares=current_hedge_shares,
        current_hedge_cost=current_hedge_cost,
        hedge_shares_needed=hedge_shares_needed,
        hedge_cost=hedge_cost,
        total_hedge_cost=total_hedge_cost,
        max_hedge_budget=max_hedge_budget,
        remaining_hedge_budget=remaining_hedge_budget,
        guaranteed_profit=guaranteed_profit,
    )


# ---------------------------------------------------------------------------
# Phase 2 sniper plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase2SniperPlan:
    sniper_risk: float
    winning_ask_price: float
    losing_ask_price: float
    expected_sniper_profit: float
    hedge_shares_needed: float
    hedge_cost: float
    total_outlay: float
    expected_net_profit: float


def compute_phase2_sniper_plan(
    *,
    sniper_risk: float,
    winning_ask_price: float,
    losing_ask_price: float,
) -> Phase2SniperPlan | None:
    if sniper_risk <= 0:
        return None
    if not 0 < winning_ask_price < 1:
        return None
    if not 0 < losing_ask_price < 1:
        return None

    expected_sniper_profit = sniper_risk * (1.0 / winning_ask_price) - sniper_risk
    hedge_shares_needed = sniper_risk / (1.0 - losing_ask_price)
    hedge_cost = hedge_shares_needed * losing_ask_price
    total_outlay = sniper_risk + hedge_cost
    expected_net_profit = expected_sniper_profit - hedge_cost

    return Phase2SniperPlan(
        sniper_risk=sniper_risk,
        winning_ask_price=winning_ask_price,
        losing_ask_price=losing_ask_price,
        expected_sniper_profit=expected_sniper_profit,
        hedge_shares_needed=hedge_shares_needed,
        hedge_cost=hedge_cost,
        total_outlay=total_outlay,
        expected_net_profit=expected_net_profit,
    )


# ---------------------------------------------------------------------------
# Probability model — Level 2 (Lognormal / Black-Scholes)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF.  Uses math.erfc — no external dependency."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def lognormal_win_probability(
    spot: float,
    strike: float,
    time_remaining_seconds: float,
    ann_volatility: float,
) -> float:
    """
    P(S_T > strike) under the zero-drift lognormal model.

    This is the Black-Scholes d2 term for a binary/digital call option —
    the risk-neutral probability that the asset finishes above the strike.

    Returns 0.5 (neutral / no information) when any input is degenerate.
    """
    if spot <= 0 or strike <= 0 or time_remaining_seconds < 1.0 or ann_volatility < 0.001:
        return 0.5
    T = time_remaining_seconds / (365.25 * 24.0 * 3600.0)   # convert to years
    sigma_sqrt_T = ann_volatility * math.sqrt(T)
    if sigma_sqrt_T < 1e-9:
        return 0.5
    d2 = math.log(spot / strike) / sigma_sqrt_T
    return _norm_cdf(d2)


def compute_prob_edge(win_probability: float, token_ask_price: float) -> float:
    """
    Edge = model probability − market implied probability (token ask price).

    Positive → Polymarket is underpricing this outcome → enter.
    Negative → Polymarket is overpricing it → skip.
    """
    if not (0.0 < token_ask_price < 1.0):
        return 0.0
    return win_probability - token_ask_price
