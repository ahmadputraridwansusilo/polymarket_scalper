"""
config.py — Central configuration for the Polymarket Scalper.

Strategy overview (BTC Micro-Timeframe Momentum Hedger):
  Phase 1  — Momentum burst: buy winning side when oracle + OBI agree.
             Fixed-token sizing (~77 tokens/bet) so exposure scales with price.
  Phase 2  — Near-expiry TWAP sniper: commit larger capital when probability
             model confirms a strong edge, scaled by confidence multiplier.
  Hedges   — Layered OTM / perfect / god-tier limit orders protect Phase 1 core.
  Settle   — Hold winning tokens to settlement, merge losing pairs.

All values can be overridden via environment variables or .env file.
Nothing in this file imports from other bot modules.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=False)

# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------
# DRY_RUN=false activates live trading.  Default is simulation (safe).
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower().strip() != "false"

# ---------------------------------------------------------------------------
# Wallet / API credentials  (set via environment — never hard-code)
# ---------------------------------------------------------------------------
PRIVATE_KEY: str = os.getenv("POLY_PRIVATE_KEY", "")
API_KEY: str = os.getenv("POLY_API_KEY", "")
API_SECRET: str = os.getenv("POLY_API_SECRET", "")
API_PASSPHRASE: str = os.getenv(
    "POLY_PASSPHRASE",
    os.getenv("POLY_API_PASSPHRASE", ""),
)
CHAIN_ID: int = int(os.getenv("POLY_CHAIN_ID", "137"))
SIGNATURE_TYPE: int = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))
FUNDER: str = os.getenv("POLY_FUNDER", "").strip()

# Gasless relayer
RELAYER_API_KEY: str = os.getenv("POLY_RELAYER_API_KEY", "")
RELAYER_API_KEY_ADDRESS: str = os.getenv("POLY_RELAYER_API_KEY_ADDRESS", "")

# ---------------------------------------------------------------------------
# Polymarket endpoints
# ---------------------------------------------------------------------------
CLOB_HOST: str = "https://clob.polymarket.com"
GAMMA_HOST: str = "https://gamma-api.polymarket.com"
MARKET_CACHE_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "market_cache.json",
)

# ---------------------------------------------------------------------------
# Binance WebSocket price feeds (sub-100 ms latency, used as oracle reference)
# ---------------------------------------------------------------------------
BINANCE_WS_BASE: str = "wss://data-stream.binance.vision/ws"
BINANCE_STREAMS: dict[str, str] = {
    "BTC": f"{BINANCE_WS_BASE}/btcusdt@aggTrade",
    "ETH": f"{BINANCE_WS_BASE}/ethusdt@aggTrade",
    "SOL": f"{BINANCE_WS_BASE}/solusdt@aggTrade",
}
BINANCE_DNS_CACHE_TTL_SECONDS: float = float(
    os.getenv("BINANCE_DNS_CACHE_TTL_SECONDS", "900")
)
BINANCE_TICK_STALE_AFTER_SECONDS: float = float(
    os.getenv("BINANCE_TICK_STALE_AFTER_SECONDS", "15")
)

# ---------------------------------------------------------------------------
# Market discovery — only 5m and 15m BTC/ETH/SOL Up-or-Down markets
# ---------------------------------------------------------------------------
TRACKED_ASSETS: tuple[str, ...] = ("BTC", "ETH", "SOL")
ALLOWED_TIMEFRAMES: tuple[str, ...] = ("5m", "15m")
TRACKED_MARKETS: tuple[tuple[str, str], ...] = (
    ("BTC", "5m"),
    ("BTC", "15m"),
    ("ETH", "5m"),
    ("ETH", "15m"),
    ("SOL", "5m"),
    ("SOL", "15m"),
)
DISCOVERY_MAX_TIME_REMAINING_S: float = 900.0
DISCOVERY_MAX_PER_ASSET: int = 2

# ---------------------------------------------------------------------------
# Order minimums
# ---------------------------------------------------------------------------
MIN_ORDER_USDC: float = 1.00
PASSIVE_MIN_ORDER_USDC: float = float(os.getenv("PASSIVE_MIN_ORDER_USDC", "1.00"))
MIN_ORDER_SHARES: float = float(os.getenv("MIN_ORDER_SHARES", "5"))
TAKER_MIN_BUFFER_RATIO: float = float(os.getenv("TAKER_MIN_BUFFER_RATIO", "1.05"))
TAKER_ORDER_TYPE: str = os.getenv("TAKER_ORDER_TYPE", "IOC").upper()
SIMULATED_TOP_BOOK_SIZE: float = 10_000.0
SIMULATED_TICK_SIZE: float = 0.001

# ---------------------------------------------------------------------------
# Phase 1 — Momentum burst (entry, spam, exit)
# ---------------------------------------------------------------------------
# Fixed-token sizing: each burst targets PHASE1_TARGET_TOKENS tokens.
# At ask=0.10 → $7.70 spend; at ask=0.90 → $69.30 spend (scales with price).
# Set PHASE1_SIZING_MODE=USDC to revert to min-order USDC behaviour.
PHASE1_SIZING_MODE: str = os.getenv("PHASE1_SIZING_MODE", "TOKEN").upper().strip()
PHASE1_TARGET_TOKENS: float = float(os.getenv("PHASE1_TARGET_TOKENS", "77"))

# Entry / spam timing
PHASE1_BURST_INTERVAL_SECONDS: float = float(os.getenv("PHASE1_BURST_INTERVAL_SECONDS", "1.5"))

# Position size cap: max fraction of total equity per market
POSITION_MAX_BALANCE_RATIO: float = float(
    os.getenv("POSITION_MAX_BALANCE_RATIO", os.getenv("PHASE1_MARKET_CAP_RATIO", "0.05"))
)
PHASE1_MARKET_CAP_RATIO: float = POSITION_MAX_BALANCE_RATIO  # alias

# OBI thresholds for entry signal
PHASE1_OBI_UP_THRESHOLD: float = float(os.getenv("PHASE1_OBI_UP_THRESHOLD", "0.70"))
PHASE1_OBI_DOWN_THRESHOLD: float = float(os.getenv("PHASE1_OBI_DOWN_THRESHOLD", "0.30"))

# Exit thresholds
PHASE1_TAKE_PROFIT_RATIO: float = float(os.getenv("PHASE1_TAKE_PROFIT_RATIO", "0.15"))
PHASE1_HOLD_TO_SETTLEMENT_SECONDS: float = float(
    os.getenv("PHASE1_HOLD_TO_SETTLEMENT_SECONDS", "120")
)
PHASE1_HOLD_TO_SETTLEMENT_BID: float = float(
    os.getenv("PHASE1_HOLD_TO_SETTLEMENT_BID", "0.85")
)
PHASE1_OBI_STOP_UP_THRESHOLD: float = float(
    os.getenv("PHASE1_OBI_STOP_UP_THRESHOLD", "0.25")
)
PHASE1_OBI_STOP_DOWN_THRESHOLD: float = float(
    os.getenv("PHASE1_OBI_STOP_DOWN_THRESHOLD", "0.75")
)
PHASE1_OBI_STOP_POLLS: int = int(os.getenv("PHASE1_OBI_STOP_POLLS", "3"))
PHASE1_BID_DRAWDOWN_STOP_RATIO: float = float(
    os.getenv("PHASE1_BID_DRAWDOWN_STOP_RATIO", "0.35")
)
PHASE1_TAKER_PRICE_TICKS: int = 1

# ---------------------------------------------------------------------------
# Layered hedge windows (limit orders placed as Phase 1 position matures)
# ---------------------------------------------------------------------------
PERFECT_HEDGE_WINDOW_START: int = int(os.getenv("PERFECT_HEDGE_WINDOW_START", "120"))
PERFECT_HEDGE_WINDOW_END: int = int(os.getenv("PERFECT_HEDGE_WINDOW_END", "60"))
OTM_HEDGE_WINDOW_START: int = int(os.getenv("OTM_HEDGE_WINDOW_START", "45"))
OTM_HEDGE_WINDOW_END: int = int(os.getenv("OTM_HEDGE_WINDOW_END", "20"))
GOD_HEDGE_WINDOW_START: int = int(os.getenv("GOD_HEDGE_WINDOW_START", "45"))
GOD_HEDGE_WINDOW_END: int = int(os.getenv("GOD_HEDGE_WINDOW_END", "15"))

PERFECT_HEDGE_MAX_ASK: float = float(os.getenv("PERFECT_HEDGE_MAX_ASK", "0.15"))
PERFECT_HEDGE_MAX_EQUITY_RATIO: float = float(
    os.getenv("PERFECT_HEDGE_MAX_EQUITY_RATIO", "0.02")
)
OTM_HEDGE_MAX_ASK: float = float(os.getenv("OTM_HEDGE_MAX_ASK", "0.15"))
OTM_HEDGE_MIN_EQUITY_RATIO: float = float(
    os.getenv("OTM_HEDGE_MIN_EQUITY_RATIO", "0.005")
)
OTM_HEDGE_MAX_EQUITY_RATIO: float = float(
    os.getenv("OTM_HEDGE_MAX_EQUITY_RATIO", "0.01")
)
OTM_HEDGE_ALLOCATION_RATIO: float = float(
    os.getenv("OTM_HEDGE_ALLOCATION_RATIO", "0.0075")
)
OTM_HEDGE_MAX_USDC: float = float(os.getenv("OTM_HEDGE_MAX_USDC", "10.0"))
GOD_HEDGE_MAX_ASK: float = float(os.getenv("GOD_HEDGE_MAX_ASK", "0.03"))
GOD_HEDGE_MAX_BALANCE_RATIO: float = float(
    os.getenv("GOD_HEDGE_MAX_BALANCE_RATIO", "0.01")
)

SAFE_MARGIN: Dict[str, float] = {
    "BTC": float(os.getenv("SAFE_MARGIN_BTC", os.getenv("BTC_SAFE_MARGIN", "15.0"))),
    "ETH": float(os.getenv("SAFE_MARGIN_ETH", os.getenv("ETH_SAFE_MARGIN", "3.0"))),
    "SOL": float(os.getenv("SAFE_MARGIN_SOL", os.getenv("SOL_SAFE_MARGIN", "0.35"))),
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Initial straddle — buy both UP and DOWN simultaneously when market opens.
# Creates a neutral delta position; Phase 2 then scales into the winning side.
# ---------------------------------------------------------------------------
STRADDLE_ENABLED: bool = os.getenv("STRADDLE_ENABLED", "true").lower().strip() == "true"
# Only straddle within this time window (seconds remaining).
STRADDLE_ENTRY_MIN_SECONDS: float = float(os.getenv("STRADDLE_ENTRY_MIN_SECONDS", "120"))
STRADDLE_ENTRY_MAX_SECONDS: float = float(os.getenv("STRADDLE_ENTRY_MAX_SECONDS", "900"))
# Max total spend (both sides) as fraction of available balance.
STRADDLE_MAX_BALANCE_RATIO: float = float(os.getenv("STRADDLE_MAX_BALANCE_RATIO", "0.20"))

# ---------------------------------------------------------------------------
# Spread arb — enter the cheaper side when combined ask < 1 - threshold.
# Example: up=0.44 + down=0.50 = 0.94 → spread=0.06 > 0.05 → buy UP.
# ---------------------------------------------------------------------------
SPREAD_ARB_ENABLED: bool = os.getenv("SPREAD_ARB_ENABLED", "true").lower().strip() == "true"
SPREAD_ARB_THRESHOLD: float = float(os.getenv("SPREAD_ARB_THRESHOLD", "0.05"))

# ---------------------------------------------------------------------------
# Phase 2 — Near-expiry TWAP sniper
# ---------------------------------------------------------------------------
# Extended to 60 s so the sniper can act on early high-conviction signals.
# Phase 2 only fires when winning side ask >= PHASE2_MIN_WINNING_PRICE (0.80).
PHASE2_WINDOW_START: int = int(os.getenv("PHASE2_WINDOW_START", "60"))
PHASE2_WINDOW_END: int = int(os.getenv("PHASE2_WINDOW_END", "5"))
# Minimum token price on the winning side before committing Phase 2 capital.
# Prevents large bets when the market is still uncertain (< 80 % confidence).
PHASE2_MIN_WINNING_PRICE: float = float(os.getenv("PHASE2_MIN_WINNING_PRICE", "0.80"))

# TWAP chunk count
PHASE2_MIN_BULLETS: int = int(os.getenv("PHASE2_MIN_BULLETS", "3"))
PHASE2_MAX_BULLETS: int = int(os.getenv("PHASE2_MAX_BULLETS", "5"))
PHASE2_SLEEP_MIN_SECONDS: float = float(os.getenv("PHASE2_SLEEP_MIN_SECONDS", "1.0"))
PHASE2_SLEEP_MAX_SECONDS: float = float(os.getenv("PHASE2_SLEEP_MAX_SECONDS", "2.5"))
PHASE2_BLOCK_LOG_COOLDOWN_SECONDS: float = float(
    os.getenv("PHASE2_BLOCK_LOG_COOLDOWN_SECONDS", "5")
)
PHASE2_TAKER_PRICE_TICKS: int = 1

# Sniper sizing
SNIPER_LIMIT_PRICE: float = 1.00
MAX_SNIPE_LIMIT: float = float(os.getenv("MAX_SNIPE_LIMIT", "500.00"))
# Hard cap: never risk more than this fraction of total equity in one market.
# Applies in SIM and LIVE identically (10% = max loss per market).
PHASE2_MAX_MARKET_EXPOSURE_RATIO: float = float(
    os.getenv("PHASE2_MAX_MARKET_EXPOSURE_RATIO", "0.10")
)
SNIPER_TARGET_BALANCE_RATIO: float = float(
    os.getenv("SNIPER_TARGET_BALANCE_RATIO", "0.15")
)
SNIPER_WINNING_ALLOCATION_RATIO: float = float(
    os.getenv("SNIPER_WINNING_ALLOCATION_RATIO", "0.95")
)
SNIPER_INSURANCE_ALLOCATION_RATIO: float = float(
    os.getenv("SNIPER_INSURANCE_ALLOCATION_RATIO", "0.05")
)
SNIPER_INSURANCE_MIN_PRICE: float = float(
    os.getenv("SNIPER_INSURANCE_MIN_PRICE", "0.05")
)
SNIPER_INSURANCE_MAX_PRICE: float = float(
    os.getenv("SNIPER_INSURANCE_MAX_PRICE", "0.15")
)

# ---------------------------------------------------------------------------
# Probability model — Level 2 (Lognormal / Black-Scholes)
# ---------------------------------------------------------------------------
# Compares real-market probability (from Binance price + vol) against
# Polymarket's implied probability (token ask price).  Only enters when edge
# (model_prob − token_price) >= PROB_MODEL_MIN_EDGE.
#
# Set PROB_MODEL_ENABLED=false to revert to pure oracle-gate + OBI behaviour.
PROB_MODEL_ENABLED: bool = os.getenv("PROB_MODEL_ENABLED", "true").lower().strip() == "true"
PROB_MODEL_LOOKBACK_SECONDS: float = float(os.getenv("PROB_MODEL_LOOKBACK_SECONDS", "300"))
PROB_MODEL_MIN_EDGE: float = float(os.getenv("PROB_MODEL_MIN_EDGE", "0.04"))
PROB_MODEL_FALLBACK_VOL: Dict[str, float] = {
    "BTC": float(os.getenv("PROB_MODEL_FALLBACK_VOL_BTC", "0.65")),
    "ETH": float(os.getenv("PROB_MODEL_FALLBACK_VOL_ETH", "0.85")),
    "SOL": float(os.getenv("PROB_MODEL_FALLBACK_VOL_SOL", "1.05")),
}

# Phase 2 confidence multiplier: scales exposure when edge is large.
# Each CONFIDENCE_SCALE above MIN_EDGE adds +1x, capped at MAX_MULT.
# edge=0.04 → 1x  |  edge=0.19 → 2x  |  edge=0.34 → 3x (cap)
PROB_MODEL_CONFIDENCE_SCALE: float = float(os.getenv("PROB_MODEL_CONFIDENCE_SCALE", "0.15"))
PROB_MODEL_MAX_CONFIDENCE_MULT: float = float(os.getenv("PROB_MODEL_MAX_CONFIDENCE_MULT", "3.0"))

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
ORACLE_POLL_INTERVAL: float = 0.5
BRAIN_TICK_INTERVAL: float = 0.25
DASHBOARD_REFRESH: float = 1.0
SETTLEMENT_RETRY_INTERVAL: float = 10.0
SETTLEMENT_INITIAL_CLAIM_DELAY: float = float(os.getenv("SETTLEMENT_INITIAL_CLAIM_DELAY", "45"))
SETTLEMENT_PENDING_RETRY_INTERVAL: float = float(os.getenv("SETTLEMENT_PENDING_RETRY_INTERVAL", "30"))

# Starting balance for simulation mode
INITIAL_BALANCE: float = 10000.0

# ---------------------------------------------------------------------------
# Polygon RPC endpoints (for settlement claims)
# ---------------------------------------------------------------------------
POLYGON_RPCS = [
    "https://polygon-mainnet.g.alchemy.com/v2/2JqxFLIxr_2NAULIMnQsE",
    "https://polygon.llamarpc.com",
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.network",
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://matic-mainnet.chainstacklabs.com",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def minimum_taker_order_usdc(price: float) -> float:
    """Minimum USDC spend for a taker order at the given token price."""
    if price <= 0:
        return 0.0
    raw = max(MIN_ORDER_USDC, MIN_ORDER_SHARES * price * TAKER_MIN_BUFFER_RATIO)
    return math.ceil((raw * 100.0) - 1e-9) / 100.0


def safe_margin_for(asset: str) -> float:
    """Minimum absolute delta (spot vs strike) required before entering."""
    return SAFE_MARGIN.get(asset, SAFE_MARGIN["SOL"])


# ---------------------------------------------------------------------------
# Legacy / unused — kept for .env backward compatibility only
# ---------------------------------------------------------------------------
@dataclass
class MarketConfig:
    condition_id: str
    asset: str
    timeframe: str
    up_token_id: str = ""
    down_token_id: str = ""
    strike_price: float = 0.0
    end_time: float = 0.0


TARGET_MARKETS: list[MarketConfig] = []
