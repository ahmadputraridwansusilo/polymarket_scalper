"""
oracle.py — The Oracle (Data Fetcher)

Responsibilities:
  • Maintains a live, in-memory price table for BTC / ETH / SOL via
    Binance Mini-Ticker WebSocket streams (sub-100 ms latency).
  • Polls the Polymarket Gamma REST API to discover and refresh
    target markets (strike price, token IDs, time remaining).
  • Exposes a single shared OracleState dataclass consumed by the Brain.
"""

from __future__ import annotations

import asyncio
from collections import deque
import datetime as dt
import json
import logging
import socket
import ssl
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional
from urllib.parse import urlsplit

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosedError

import config

log = logging.getLogger("oracle")


# ---------------------------------------------------------------------------
# Shared state (read by Brain, written only by Oracle tasks)
# ---------------------------------------------------------------------------
@dataclass
class LivePrice:
    price: float = 0.0
    ts: float    = 0.0     # UNIX timestamp of last update


@dataclass
class MarketSnapshot:
    condition_id: str
    asset: str
    timeframe: str
    up_token_id: str
    down_token_id: str
    strike_price: float
    event_start_time: float
    end_time: float        # UNIX epoch
    best_ask_up: float   = 0.99
    best_bid_up: float   = 0.01
    best_ask_down: float = 0.99
    best_bid_down: float = 0.01
    binance_live_price: float = 0.0

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - time.time())


@dataclass
class OracleState:
    prices: Dict[str, LivePrice] = field(
        default_factory=lambda: {
            "BTC": LivePrice(),
            "ETH": LivePrice(),
            "SOL": LivePrice(),
        }
    )
    markets: Dict[str, MarketSnapshot] = field(default_factory=dict)
    # condition_id -> snapshot


@dataclass(frozen=True)
class OracleStatus:
    gamma_ok: bool
    gamma_error: str
    gamma_last_success_at: float
    binance_connected: Dict[str, bool]
    binance_errors: Dict[str, str]
    using_cached_markets: bool
    cache_loaded_at: float


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------
class Oracle:
    # ---------------------------------------------------------------------------
    # STRICT 5m / 15m lockdown
    #
    # Only slugs matching EXACTLY one of these patterns are accepted:
    #   btc-updown-5m-{unix_ts}
    #   btc-updown-15m-{unix_ts}
    #   eth-updown-5m-{unix_ts}  ...etc
    #
    # No fallback to daily, hourly, or any other timeframe — ever.
    # If no valid markets exist the bot waits silently.
    # ---------------------------------------------------------------------------

    # Valid slug prefixes: (asset_key, timeframe) → asset label
    # Built from TRACKED_ASSETS × ALLOWED_TIMEFRAMES so config drives everything.
    _VALID_SLUG_PREFIXES: dict[str, tuple[str, str]] = {
        f"{asset.lower()}-updown-{tf}-": (asset, tf)
        for asset in ("BTC", "ETH", "SOL")
        for tf in ("5m", "15m")
    }
    # e.g. "btc-updown-5m-" → ("BTC", "5m")
    #      "eth-updown-15m-" → ("ETH", "15m")

    def __init__(self) -> None:
        self.state = OracleState()
        self._session: Optional[aiohttp.ClientSession] = None
        # Cached pagination offset so we skip past already-expired markets
        # on every poll without re-scanning thousands of rows each time.
        self._discovery_offset: int = 0
        self._gamma_last_error: str = ""
        self._gamma_last_success_at: float = 0.0
        self._binance_connected: Dict[str, bool] = {
            asset: False for asset in config.TRACKED_ASSETS
        }
        self._binance_last_error: Dict[str, str] = {
            asset: "not connected yet" for asset in config.TRACKED_ASSETS
        }
        self._binance_host_ipv4_cache: Dict[str, list[str]] = {}
        self._binance_host_ipv4_cached_at: Dict[str, float] = {}
        self._using_cached_markets: bool = False
        self._cache_loaded_at: float = 0.0
        self._price_history: Dict[str, Deque[tuple[float, float]]] = {
            asset: deque(maxlen=4096)
            for asset in config.TRACKED_ASSETS
        }
        self._cached_market_strikes: Dict[str, float] = {}
        self._refresh_cached_market_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_price(self, asset: str) -> float:
        lp = self.state.prices.get(asset)
        return lp.price if lp else 0.0

    def get_market(self, condition_id: str) -> Optional[MarketSnapshot]:
        return self.state.markets.get(condition_id)

    def all_markets(self) -> list[MarketSnapshot]:
        return list(self.state.markets.values())

    def realized_volatility(self, asset: str, lookback_seconds: float = 300.0) -> float:
        """
        Annualized realized volatility from recent Binance tick history.
        Uses sum-of-squared log-returns normalised by elapsed time (no mean subtraction
        for efficiency — mean ≈ 0 at short horizons).
        Returns 0.0 when fewer than 10 observations are available.
        """
        import math
        history = self._price_history.get(asset)
        if not history or len(history) < 10:
            return 0.0

        now = history[-1][0]
        cutoff = now - max(lookback_seconds, 10.0)
        window = [(ts, p) for ts, p in history if ts >= cutoff and p > 0]
        if len(window) < 10:
            # Not enough recent ticks — use full history as fallback
            window = [(ts, p) for ts, p in history if p > 0]
        if len(window) < 10:
            return 0.0

        sum_r2 = 0.0
        sum_dt = 0.0
        for i in range(1, len(window)):
            dt = window[i][0] - window[i - 1][0]
            if dt <= 0 or window[i - 1][1] <= 0:
                continue
            lr = math.log(window[i][1] / window[i - 1][1])
            sum_r2 += lr * lr
            sum_dt += dt

        if sum_dt < 1.0:
            return 0.0

        variance_per_second = sum_r2 / sum_dt
        return math.sqrt(max(0.0, variance_per_second * 365.25 * 24.0 * 3600.0))

    def price_momentum(self, asset: str, lookback_seconds: float = 30.0) -> float:
        history = self._price_history.get(asset)
        if not history:
            return 0.0

        latest_ts, latest_price = history[-1]
        cutoff = latest_ts - max(1.0, lookback_seconds)
        anchor_price = history[0][1]
        for ts, price in history:
            anchor_price = price
            if ts >= cutoff:
                break
        return latest_price - anchor_price

    def invalidate_market(self, condition_id: str) -> None:
        """Remove a market from the state immediately (e.g. orderbook closed error)."""
        removed = self.state.markets.pop(condition_id, None)
        if removed:
            log.info(
                "EVICTED expired market: %s %s %s",
                condition_id[:12], removed.asset, removed.timeframe,
            )

    def active_markets(self) -> list[MarketSnapshot]:
        """
        Return exactly one market per (asset, timeframe) pair — the one expiring
        soonest.  Only 5m and 15m markets within the time window are considered.
        Result is ordered by config.TRACKED_MARKETS so the dashboard rows are stable.
        """
        # Best candidate per (asset, timeframe) key
        best: Dict[tuple[str, str], MarketSnapshot] = {}
        for snap in self.state.markets.values():
            tr = snap.time_remaining
            if tr <= 0 or tr > config.DISCOVERY_MAX_TIME_REMAINING_S:
                continue
            if snap.timeframe not in config.ALLOWED_TIMEFRAMES:
                continue
            key = (snap.asset, snap.timeframe)
            if key not in best or tr < best[key].time_remaining:
                best[key] = snap

        # Return in the order defined by TRACKED_MARKETS so rows don't jump around
        result: list[MarketSnapshot] = []
        for asset, tf in config.TRACKED_MARKETS:
            snap = best.get((asset, tf))
            if snap:
                result.append(snap)
        return result

    def price_table(self) -> Dict[str, float]:
        return {
            asset: live_price.price
            for asset, live_price in self.state.prices.items()
        }

    def status_snapshot(self) -> OracleStatus:
        return OracleStatus(
            gamma_ok=(self._gamma_last_error == ""),
            gamma_error=self._gamma_last_error,
            gamma_last_success_at=self._gamma_last_success_at,
            binance_connected=dict(self._binance_connected),
            binance_errors=dict(self._binance_last_error),
            using_cached_markets=self._using_cached_markets,
            cache_loaded_at=self._cache_loaded_at,
        )

    def status_line(self) -> str:
        if self._gamma_last_error:
            gamma_status = "Gamma DNS/NET FAIL"
        elif self._gamma_last_success_at > 0:
            gamma_status = "Gamma OK"
        else:
            gamma_status = "Gamma booting"

        binance_status = self._binance_status_line()

        cache_status = "CACHE" if self._using_cached_markets else "LIVE"
        return f"{gamma_status} | {binance_status} | {cache_status}"

    def _binance_status_line(self) -> str:
        now = time.time()
        fresh_assets: list[str] = []
        missing_assets: list[str] = []
        had_live_ticks = False

        for asset in config.TRACKED_ASSETS:
            live_price = self.state.prices.get(asset)
            last_tick_at = live_price.ts if live_price else 0.0
            if last_tick_at > 0:
                had_live_ticks = True
            if last_tick_at > 0 and (
                now - last_tick_at <= config.BINANCE_TICK_STALE_AFTER_SECONDS
            ):
                fresh_assets.append(asset)
            else:
                missing_assets.append(asset)

        if len(fresh_assets) == len(config.TRACKED_ASSETS):
            return "Binance OK"

        if fresh_assets:
            return f"Binance partial ({','.join(missing_assets)})"

        if any(self._binance_connected.values()):
            return "Binance warming"

        if had_live_ticks:
            return "Binance reconnecting"

        meaningful_errors = [
            err
            for err in self._binance_last_error.values()
            if err and err != "not connected yet"
        ]
        if meaningful_errors and all(
            self._looks_like_network_error(err) for err in meaningful_errors
        ):
            return "Binance DNS/NET FAIL"
        if meaningful_errors:
            return "Binance feed down"
        return "Binance booting"

    @staticmethod
    def _looks_like_network_error(error: str) -> bool:
        lowered = error.lower()
        indicators = (
            "nodename nor servname",
            "name or service not known",
            "temporary failure in name resolution",
            "connect call failed",
            "connection refused",
            "network is unreachable",
            "host is unreachable",
            "no route to host",
            "timed out",
        )
        return any(indicator in lowered for indicator in indicators)

    # ------------------------------------------------------------------
    # Entry point — run both feed tasks concurrently
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._session = aiohttp.ClientSession()
        try:
            await asyncio.gather(
                self._binance_ws_loop(),
                self._polymarket_poll_loop(),
            )
        finally:
            await self._session.close()

    # ------------------------------------------------------------------
    # Binance aggTrade WebSocket — three independent per-asset connections
    #
    # Why this works when stream.binance.com:9443 does not:
    #   • Host:  data-stream.binance.vision  (Binance public data mirror)
    #            This is the same base URL used by the existing working bot.
    #            The main stream.binance.com host is blocked at ISP level on
    #            this network for both port 9443 and 443.
    #   • Port:  443 (standard WSS — no explicit port in URL).
    #   • IPv4:  socket.AF_INET passed to websockets.connect() so asyncio
    #            never picks an IPv6 address from DNS.
    #   • SSL:   built from certifi's CA bundle; falls back to system store.
    #
    # aggTrade message format (direct stream — no envelope wrapper):
    #   {
    #     "e": "aggTrade",
    #     "E": 1713000000000,   <- event time ms
    #     "s": "BTCUSDT",      <- symbol
    #     "p": "84321.50",     <- trade price  ← we use this
    #     "q": "0.01200",
    #     "T": 1713000000000,
    #     "m": false
    #   }
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        """Return a verified SSL context, preferring certifi's CA bundle."""
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
            log.debug("SSL: certifi CA bundle.")
        except ImportError:
            ctx = ssl.create_default_context()
            log.debug("SSL: system default CA bundle (certifi not installed).")
        return ctx

    async def _resolve_ipv4_hosts(self, hostname: str, port: int) -> list[str]:
        cached = list(self._binance_host_ipv4_cache.get(hostname, []))
        cached_at = self._binance_host_ipv4_cached_at.get(hostname, 0.0)
        if cached and (
            time.time() - cached_at <= config.BINANCE_DNS_CACHE_TTL_SECONDS
        ):
            return cached

        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(
                hostname,
                port,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            if cached:
                log.warning(
                    "Binance DNS resolve failed for %s: %s — reusing cached IPv4 %s",
                    hostname,
                    exc,
                    ",".join(cached),
                )
                return cached
            log.warning("Binance DNS resolve failed for %s: %s", hostname, exc)
            return []

        ipv4_hosts: list[str] = []
        for info in infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            ip = sockaddr[0]
            if ip not in ipv4_hosts:
                ipv4_hosts.append(ip)

        if ipv4_hosts:
            self._binance_host_ipv4_cache[hostname] = ipv4_hosts
            self._binance_host_ipv4_cached_at[hostname] = time.time()

        return ipv4_hosts

    async def _binance_connect_targets(
        self, url: str
    ) -> list[tuple[Optional[str], Optional[int], str]]:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return [(None, None, url)]

        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        resolved_ipv4 = await self._resolve_ipv4_hosts(hostname, port)
        cached_ipv4 = self._binance_host_ipv4_cache.get(hostname, [])

        targets: list[tuple[Optional[str], Optional[int], str]] = []
        seen: set[str] = set()
        for ip in [*resolved_ipv4, *cached_ipv4]:
            if not ip or ip in seen:
                continue
            seen.add(ip)
            targets.append((ip, port, f"{url} via {ip}"))

        if not targets:
            targets.append((None, None, url))

        return targets

    async def _binance_ws_loop(self) -> None:
        """Launch one feed task per asset and wait for all of them."""
        ssl_ctx = self._build_ssl_context()
        await asyncio.gather(
            *[
                self._binance_asset_feed(asset, url, ssl_ctx)
                for asset, url in config.BINANCE_STREAMS.items()
            ]
        )

    async def _binance_asset_feed(
        self, asset: str, url: str, ssl_ctx: ssl.SSLContext
    ) -> None:
        """
        Maintain a single reconnecting aggTrade WebSocket for one asset.
        Mirrors the _subscribe() pattern from the working polymarket_bot.
        """
        backoff = 1.0
        while True:
            targets = await self._binance_connect_targets(url)
            for override_host, override_port, label in targets:
                try:
                    log.info("Connecting Binance %s feed → %s", asset, label)
                    connect_kwargs = {
                        "ssl": ssl_ctx,
                        "family": socket.AF_INET,   # never try IPv6
                        "ping_interval": 20,
                        "ping_timeout": 10,
                        "close_timeout": 5,
                        "max_size": 2 ** 20,
                    }
                    if override_host is not None:
                        connect_kwargs["host"] = override_host
                    if override_port is not None:
                        connect_kwargs["port"] = override_port

                    async with websockets.connect(url, **connect_kwargs) as ws:
                        backoff = 1.0
                        self._binance_connected[asset] = True
                        self._binance_last_error[asset] = ""
                        log.info(
                            "Binance %s feed connected (IPv4, %s).",
                            asset,
                            override_host or "data-stream.binance.vision",
                        )
                        async for raw in ws:
                            await self._handle_binance_tick(asset, raw)

                except ConnectionClosedError as exc:
                    self._binance_connected[asset] = False
                    self._binance_last_error[asset] = str(exc)
                    log.warning(
                        "Binance %s WS closed via %s: %s — reconnect in %.0fs",
                        asset,
                        override_host or "DNS",
                        exc,
                        backoff,
                    )
                except OSError as exc:
                    # Catches ECONNREFUSED [Errno 61], ETIMEDOUT [Errno 60], etc.
                    self._binance_connected[asset] = False
                    self._binance_last_error[asset] = str(exc)
                    log.error(
                        "Binance %s OS error via %s: %s — reconnect in %.0fs",
                        asset,
                        override_host or "DNS",
                        exc,
                        backoff,
                    )
                except Exception as exc:
                    self._binance_connected[asset] = False
                    self._binance_last_error[asset] = str(exc)
                    log.error(
                        "Binance %s WS error via %s: %s — reconnect in %.0fs",
                        asset,
                        override_host or "DNS",
                        exc,
                        backoff,
                    )

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _handle_binance_tick(self, asset: str, raw: str) -> None:
        """
        Parse a Binance aggTrade message and update the in-memory price.
        The price field is msg["p"] (string) — same as the working bot.
        """
        try:
            msg   = json.loads(raw)
            price = float(msg["p"])
            now = time.time()

            lp = self.state.prices[asset]
            lp.price = price
            lp.ts    = now
            self._binance_connected[asset] = True
            self._binance_last_error[asset] = ""
            history = self._price_history.setdefault(asset, deque(maxlen=4096))
            history.append((now, price))
            cutoff = now - (config.DISCOVERY_MAX_TIME_REMAINING_S + 1800.0)
            while history and history[0][0] < cutoff:
                history.popleft()

            for snap in self.state.markets.values():
                if snap.asset == asset:
                    snap.binance_live_price = price

            log.debug("Binance tick | %s = %.4f", asset, price)

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.debug("Bad Binance tick (%s): %s | raw=%r", asset, exc, raw[:120])

    # ------------------------------------------------------------------
    # Polymarket Gamma REST poll
    # ------------------------------------------------------------------
    async def _polymarket_poll_loop(self) -> None:
        while True:
            try:
                found = await self._refresh_markets()
            except Exception as exc:
                log.warning("Market poll failed: %s", exc)
                found = False

            if found:
                await asyncio.sleep(config.ORACLE_POLL_INTERVAL)
            else:
                # No live 5m/15m markets right now — wait 30s before next scan.
                msg = "[DISCOVERY] No active 5m/15m markets found. Waiting for next cycle..."
                log.info(msg)
                await asyncio.sleep(30)

    # ------------------------------------------------------------------
    # Polymarket Gamma REST — market discovery
    #
    # Why the old approach failed:
    #   • `active=true` returns historical markets, not live ones.
    #   • Filtering for "5 min"/"15 min" in the question text never matched
    #     because actual market titles are like "Bitcoin Up or Down - April 5,
    #     7:30PM-7:35PM ET" (no "5 min" substring).
    #   • The token IDs were looked up from a `tokens` sub-array that doesn't
    #     exist — they live in the `clobTokenIds` JSON string field.
    #
    # Correct approach:
    #   1. Fetch /markets sorted by endDate ascending (closed=false).
    #   2. Fast-skip batches whose LAST endDate is already in the past,
    #      advancing the cached offset so subsequent polls are O(1).
    #   3. In batches that span the current time, filter by slug prefix:
    #        btc-updown-5m-{ts} / eth-updown-5m-{ts} / sol-updown-5m-{ts}
    #        btc-updown-15m-{ts} / eth-updown-15m-{ts} / sol-updown-15m-{ts}
    #      The trailing {ts} IS the UNIX close timestamp — no date parsing needed.
    #   4. Pre-created markets exist up to ~2 hours ahead, so keep scanning
    #      until we've collected 2 full future-bearing batches.
    # ------------------------------------------------------------------

    async def _refresh_markets(self) -> bool:
        """
        Scan Gamma API for active 5m/15m up-or-down markets.

        Hard rules (zero exceptions):
          • Slug MUST match exactly: {asset}-updown-{5m|15m}-{ts}
          • time_remaining MUST be in (0, DISCOVERY_MAX_TIME_REMAINING_S]
          • No daily, hourly, or any other timeframe — ever.

        Returns True if at least one valid market is currently live.
        """
        assert self._session is not None

        # ── Prune expired / oversized markets ─────────────────────────────────
        now_ts = time.time()
        expiry_retention = max(
            config.DISCOVERY_MAX_TIME_REMAINING_S,
            config.SETTLEMENT_INITIAL_CLAIM_DELAY
            + config.SETTLEMENT_PENDING_RETRY_INTERVAL
            + config.SETTLEMENT_RETRY_INTERVAL,
        )
        to_drop = [
            cid for cid, s in self.state.markets.items()
            if s.end_time <= now_ts - expiry_retention
            or s.end_time > now_ts + config.DISCOVERY_MAX_TIME_REMAINING_S
            or s.timeframe not in config.ALLOWED_TIMEFRAMES
        ]
        for cid in to_drop:
            self.state.markets.pop(cid, None)
        if to_drop:
            log.debug("DISCOVERY pruned %d market(s) (expired/disallowed).", len(to_drop))

        # ── Manual override ────────────────────────────────────────────────────
        if config.TARGET_MARKETS:
            for mc in config.TARGET_MARKETS:
                if mc.timeframe in config.ALLOWED_TIMEFRAMES:
                    await self._fetch_market_by_condition(mc.condition_id, mc.asset, mc.timeframe)
            self._using_cached_markets = False
            return bool(self.active_markets())

        # ── Paginated Gamma scan ───────────────────────────────────────────────
        max_end_ts     = now_ts + config.DISCOVERY_MAX_TIME_REMAINING_S
        BATCH          = 500
        url            = f"{config.GAMMA_HOST}/markets"
        offset         = max(0, self._discovery_offset - BATCH)
        found_count    = 0
        future_batches = 0

        while True:
            params = {
                "closed":    "false",
                "active":    "true",
                "limit":     BATCH,
                "offset":    offset,
                "order":     "endDate",
                "ascending": "true",
            }
            try:
                async with self._session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        self._gamma_last_error = f"HTTP {resp.status}"
                        log.warning("Gamma /markets HTTP %s (offset=%d)", resp.status, offset)
                        break
                    batch: list = await resp.json(content_type=None)
                    self._gamma_last_error = ""
                    self._gamma_last_success_at = time.time()
                    self._using_cached_markets = False
            except Exception as exc:
                self._gamma_last_error = str(exc)
                log.warning("Gamma /markets fetch error: %s", exc)
                break

            if not batch or not isinstance(batch, list):
                break

            last_end_str: str = ""
            for item in reversed(batch):
                if isinstance(item, dict):
                    last_end_str = item.get("endDate", "") or item.get("endDateIso", "")
                    break
            last_end_ts = self._parse_iso_ts(last_end_str)

            # Skip entire batches that are fully expired.
            if last_end_ts <= now_ts:
                offset                 += BATCH
                self._discovery_offset  = offset
                log.debug("DISCOVERY skip: offset=%d all expired", offset)
                continue

            # Stop once batch exceeds our 15-minute window — nothing further matters.
            if last_end_ts > max_end_ts and future_batches >= 1:
                break

            for mkt in batch:
                if not isinstance(mkt, dict):
                    continue
                end_ts = self._parse_iso_ts(mkt.get("endDate", "") or mkt.get("endDateIso", ""))
                tr     = end_ts - now_ts

                # ── HARD FILTER 1: time window ─────────────────────────────
                if tr <= 0 or tr > config.DISCOVERY_MAX_TIME_REMAINING_S:
                    continue

                # ── HARD FILTER 2: slug must be exactly 5m or 15m updown ──
                slug = mkt.get("slug", "").lower()
                matched = None
                for prefix, (asset, tf) in self._VALID_SLUG_PREFIXES.items():
                    if slug.startswith(prefix):
                        matched = (asset, tf)
                        break

                if matched is None:
                    continue   # not a 5m/15m updown market — skip silently

                asset, tf = matched
                self._ingest_gamma_market(mkt, asset=asset, market_type="updown")
                found_count += 1

            future_batches += 1
            if future_batches >= 2 or len(batch) < BATCH:
                break
            offset += BATCH

        if found_count:
            self._using_cached_markets = False
            log.info(
                "DISCOVERY: +%d valid 5m/15m market(s) | total live=%d",
                found_count, len(self.state.markets),
            )
            self._log_locked_markets(now_ts)
            self._save_market_cache()
        elif self._gamma_last_error:
            loaded = self._load_market_cache()
            if loaded:
                self._using_cached_markets = True
                log.warning(
                    "[DISCOVERY] Using %d cached market(s) because Gamma is unavailable.",
                    loaded,
                )

        # When running from cache, keep the slower retry cadence for Gamma.
        return bool(self.active_markets()) and not self._using_cached_markets

    async def _fetch_market_by_condition(
        self, condition_id: str, asset: str, timeframe: str
    ) -> None:
        assert self._session is not None
        url = f"{config.GAMMA_HOST}/markets/{condition_id}"
        async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return
            mkt = await resp.json(content_type=None)
        self._ingest_gamma_market(mkt, asset=asset, market_type="manual")


    def _log_locked_markets(self, now_ts: float) -> None:
        """
        Print a clean summary of every market the bot is currently tracking.
        Runs at the end of each discovery poll so the user can see what is live.
        Only prints when there is at least one tracked market.
        """
        active = [s for s in self.state.markets.values() if s.time_remaining > 0]
        if not active:
            return

        active.sort(key=lambda s: (s.asset, s.end_time))
        lines = ["[DISCOVERY] Locked-on markets:"]
        for snap in active:
            tr  = snap.time_remaining
            hrs = int(tr // 3600)
            mn  = int((tr % 3600) // 60)
            sc  = int(tr % 60)
            if hrs:
                tr_str = f"{hrs}h {mn}m"
            else:
                tr_str = f"{mn}m {sc}s"
            lines.append(
                f"  [{snap.asset} {snap.timeframe}]  cid={snap.condition_id[:12]}…"
                f"  up={snap.up_token_id[:10]}…  down={snap.down_token_id[:10]}…"
                f"  resolves_in={tr_str}"
            )
        summary = "\n".join(lines)
        log.info(summary)

    def _save_market_cache(self) -> None:
        active = [
            snap for snap in self.state.markets.values()
            if 0 < snap.time_remaining <= config.DISCOVERY_MAX_TIME_REMAINING_S
        ]
        if not active:
            return
        payload = {
            "saved_at": time.time(),
            "markets": [
                {
                    "condition_id": snap.condition_id,
                    "asset": snap.asset,
                    "timeframe": snap.timeframe,
                    "up_token_id": snap.up_token_id,
                    "down_token_id": snap.down_token_id,
                    "strike_price": snap.strike_price,
                    "event_start_time": snap.event_start_time,
                    "end_time": snap.end_time,
                    "best_ask_up": snap.best_ask_up,
                    "best_bid_up": snap.best_bid_up,
                    "best_ask_down": snap.best_ask_down,
                    "best_bid_down": snap.best_bid_down,
                }
                for snap in active
            ],
        }
        tmp_path = f"{config.MARKET_CACHE_FILE}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            import os
            os.replace(tmp_path, config.MARKET_CACHE_FILE)
            self._cached_market_strikes = {
                snap.condition_id: snap.strike_price
                for snap in active
                if snap.strike_price > 0
            }
        except Exception as exc:
            log.debug("Market cache save skipped: %s", exc)

    def _refresh_cached_market_index(self) -> None:
        self._cached_market_strikes = {}
        try:
            with open(config.MARKET_CACHE_FILE, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return

        markets = payload.get("markets") if isinstance(payload, dict) else None
        if not isinstance(markets, list):
            return

        now_ts = time.time()
        for item in markets:
            if not isinstance(item, dict):
                continue
            condition_id = str(item.get("condition_id") or "")
            strike_price = float(item.get("strike_price") or 0.0)
            end_time = float(item.get("end_time") or 0.0)
            if not condition_id or strike_price <= 0:
                continue
            if end_time > 0 and end_time < now_ts - config.DISCOVERY_MAX_TIME_REMAINING_S:
                continue
            self._cached_market_strikes[condition_id] = strike_price

    def _load_market_cache(self) -> int:
        try:
            with open(config.MARKET_CACHE_FILE, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return 0

        now_ts = time.time()
        loaded = 0
        self._cached_market_strikes = {}
        markets = payload.get("markets") if isinstance(payload, dict) else None
        if not isinstance(markets, list):
            return 0

        for item in markets:
            if not isinstance(item, dict):
                continue
            end_time = float(item.get("end_time") or 0.0)
            timeframe = str(item.get("timeframe") or "")
            if (
                end_time <= now_ts
                or end_time > now_ts + config.DISCOVERY_MAX_TIME_REMAINING_S
                or timeframe not in config.ALLOWED_TIMEFRAMES
            ):
                continue

            condition_id = str(item.get("condition_id") or "")
            asset = str(item.get("asset") or "")
            if not condition_id or not asset:
                continue
            strike_price = float(item.get("strike_price") or 0.0)
            if strike_price > 0:
                self._cached_market_strikes[condition_id] = strike_price
            event_start_time = float(item.get("event_start_time") or 0.0)
            if event_start_time <= 0:
                tf_seconds = 300.0 if timeframe == "5m" else 900.0 if timeframe == "15m" else 0.0
                event_start_time = end_time - tf_seconds if tf_seconds > 0 else 0.0

            cached = MarketSnapshot(
                condition_id=condition_id,
                asset=asset,
                timeframe=timeframe,
                up_token_id=str(item.get("up_token_id") or ""),
                down_token_id=str(item.get("down_token_id") or ""),
                strike_price=strike_price,
                event_start_time=event_start_time,
                end_time=end_time,
                best_ask_up=float(item.get("best_ask_up") or 0.99),
                best_bid_up=float(item.get("best_bid_up") or 0.01),
                best_ask_down=float(item.get("best_ask_down") or 0.99),
                best_bid_down=float(item.get("best_bid_down") or 0.01),
                binance_live_price=self.get_price(asset),
            )
            self.state.markets[condition_id] = cached
            loaded += 1

        if loaded:
            self._cache_loaded_at = now_ts
        return loaded

    def _resolve_market_strike(
        self,
        *,
        condition_id: str,
        asset: str,
        event_start_time: float,
        existing: Optional[MarketSnapshot],
        gamma_strike: float = 0.0,
    ) -> float:
        if existing and existing.strike_price > 0:
            return existing.strike_price
        if gamma_strike > 0:
            return gamma_strike
        cached_strike = self._cached_market_strikes.get(condition_id, 0.0)
        if cached_strike > 0:
            return cached_strike
        if event_start_time <= 0 or event_start_time > time.time():
            return 0.0
        anchored = self._historical_price_for(asset, event_start_time)
        return anchored if anchored > 0 else 0.0

    @staticmethod
    def _extract_gamma_strike(mkt: dict) -> float:
        for key in (
            "strikePrice",
            "strike_price",
            "strike",
            "referencePrice",
            "reference_price",
            "openPrice",
            "open_price",
        ):
            value = mkt.get(key)
            try:
                strike = float(value)
            except (TypeError, ValueError):
                continue
            if strike > 0:
                return strike
        return 0.0

    def _historical_price_for(self, asset: str, event_start_time: float) -> float:
        history = self._price_history.get(asset)
        if not history:
            return 0.0

        best_price = 0.0
        best_distance = float("inf")
        for ts, price in history:
            distance = abs(ts - event_start_time)
            if distance < best_distance:
                best_distance = distance
                best_price = price

        return best_price if best_distance <= 90.0 else 0.0

    def _ingest_gamma_market(
        self,
        mkt: dict,
        *,
        asset: str = "",
        market_type: str = "updown",
    ) -> None:
        """
        Parse one Gamma market dict and upsert it into OracleState.

        asset       — pre-identified by the caller ("BTC"/"ETH"/"SOL")
        market_type — "updown" or "daily"

        Slug format for up-or-down markets:
          {asset}-updown-{timeframe}-{unix_close_ts}
          e.g.  btc-updown-5m-1775431800  /  eth-updown-1h-1775438000

        clobTokenIds: JSON string  '["up_token_id", "down_token_id"]'
          outcomes[0] = "Up"  → token[0]
          outcomes[1] = "Down" → token[1]
        """
        slug: str = mkt.get("slug", "").lower()
        cid:  str = mkt.get("conditionId", "")
        if not cid:
            return

        # ── Identify asset if not supplied ───────────────────────────────────
        if not asset:
            for prefix_key, a in self._UPDOWN_PREFIXES.items():
                if slug.startswith(f"{prefix_key}-"):
                    asset = a
                    break
        if not asset:
            return

        # ── Extract and validate timeframe from slug ─────────────────────────
        # Slug format: btc-updown-{tf}-{unix_ts}  e.g. btc-updown-5m-1775431800
        timeframe = ""
        if "-updown-" in slug:
            after_updown = slug.split("-updown-", 1)[1]   # "5m-1775431800"
            parts = after_updown.split("-")
            if parts and not parts[0].isdigit():
                timeframe = parts[0]                       # "5m" / "15m"

        # Hard reject: only 5m and 15m are allowed — no exceptions.
        if timeframe not in config.ALLOWED_TIMEFRAMES:
            log.debug(
                "DISCOVERY reject %s — timeframe=%r not in ALLOWED_TIMEFRAMES",
                slug, timeframe,
            )
            return

        # ── Close timestamp ───────────────────────────────────────────────────
        # Gamma's endDate is more reliable than slug suffixes for live countdowns.
        end_time = self._parse_iso_ts(mkt.get("endDate", "") or mkt.get("endDateIso", ""))
        if end_time <= 0:
            suffix = slug.rsplit("-", 1)[-1]
            if suffix.isdigit():
                end_time = float(suffix)
        if end_time <= 0:
            return
        event_start_time = self._parse_iso_ts(
            mkt.get("eventStartTime", "") or mkt.get("startDate", "") or mkt.get("startDateIso", "")
        )
        if event_start_time <= 0:
            tf_seconds = 300.0 if timeframe == "5m" else 900.0 if timeframe == "15m" else 0.0
            event_start_time = end_time - tf_seconds if tf_seconds > 0 else 0.0

        # ── Token IDs ─────────────────────────────────────────────────────────
        raw_toks = mkt.get("clobTokenIds", "[]")
        try:
            toks: list = json.loads(raw_toks) if isinstance(raw_toks, str) else (raw_toks or [])
        except json.JSONDecodeError:
            toks = []

        up_token   = str(toks[0]) if len(toks) > 0 else ""
        down_token = str(toks[1]) if len(toks) > 1 else ""

        # Skip if both token IDs are missing — cannot trade without them.
        if not up_token and not down_token:
            log.debug("DISCOVERY skip %s — no clobTokenIds", slug)
            return

        # ── Prices ────────────────────────────────────────────────────────────
        raw_prices = mkt.get("outcomePrices") or []
        try:
            up_mid   = float(raw_prices[0]) if raw_prices else 0.50
            down_mid = float(raw_prices[1]) if len(raw_prices) > 1 else round(1.0 - up_mid, 4)
        except (ValueError, TypeError):
            up_mid = 0.50
            down_mid = 0.50

        HALF_SPREAD   = 0.01
        best_ask_up   = min(up_mid   + HALF_SPREAD, 0.99)
        best_bid_up   = max(up_mid   - HALF_SPREAD, 0.01)
        best_ask_down = min(down_mid + HALF_SPREAD, 0.99)
        best_bid_down = max(down_mid - HALF_SPREAD, 0.01)

        # ── Strike price ──────────────────────────────────────────────────────
        existing = self.state.markets.get(cid)
        gamma_strike = self._extract_gamma_strike(mkt)
        strike = self._resolve_market_strike(
            condition_id=cid,
            asset=asset,
            event_start_time=event_start_time,
            existing=existing,
            gamma_strike=gamma_strike,
        )

        snap = MarketSnapshot(
            condition_id       = cid,
            asset              = asset,
            timeframe          = timeframe,
            up_token_id        = up_token,
            down_token_id      = down_token,
            strike_price       = strike,
            event_start_time   = event_start_time,
            end_time           = end_time,
            best_ask_up        = best_ask_up,
            best_bid_up        = best_bid_up,
            best_ask_down      = best_ask_down,
            best_bid_down      = best_bid_down,
            binance_live_price = self.get_price(asset),
        )

        if existing:
            existing.best_ask_up        = snap.best_ask_up
            existing.best_bid_up        = snap.best_bid_up
            existing.best_ask_down      = snap.best_ask_down
            existing.best_bid_down      = snap.best_bid_down
            existing.binance_live_price = snap.binance_live_price
            existing.event_start_time   = snap.event_start_time
            if not existing.strike_price and snap.strike_price:
                existing.strike_price = snap.strike_price
        else:
            self.state.markets[cid] = snap
            tr = snap.time_remaining
            hrs, rem = divmod(int(tr), 3600)
            mn, sc   = divmod(rem, 60)
            tr_str   = f"{hrs}h {mn}m {sc}s" if hrs else f"{mn}m {sc}s"
            question = mkt.get("question", "") or mkt.get("title", slug)
            msg = (
                f"[DISCOVERY] NEW market locked → [{asset} {timeframe} | {market_type.upper()}]\n"
                f"  Question   : {question}\n"
                f"  cid        : {cid}\n"
                f"  up_token   : {up_token}\n"
                f"  down_token : {down_token}\n"
                f"  strike     : {strike:.4f}\n"
                f"  resolves_in: {tr_str}"
            )
            log.info(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_iso_ts(s: str) -> float:
        """Parse an ISO-8601 UTC string to a UNIX float.  Returns 0.0 on failure."""
        if not s:
            return 0.0
        try:
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
