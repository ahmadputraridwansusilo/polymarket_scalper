import json
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import config
from oracle import MarketSnapshot, Oracle


class OracleStrikeCacheTests(unittest.TestCase):
    def test_resolve_market_strike_reuses_cached_condition_on_restart(self) -> None:
        original_cache_file = config.MARKET_CACHE_FILE
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_file = Path(tmpdir) / "market_cache.json"
                config.MARKET_CACHE_FILE = str(cache_file)
                cache_file.write_text(
                    json.dumps(
                        {
                            "saved_at": time.time(),
                            "markets": [
                                {
                                    "condition_id": "cid-1",
                                    "asset": "BTC",
                                    "timeframe": "5m",
                                    "strike_price": 68724.0,
                                    "end_time": time.time() + 120.0,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                oracle = Oracle()

                self.assertEqual(
                    oracle._resolve_market_strike(
                        condition_id="cid-1",
                        asset="BTC",
                        event_start_time=0.0,
                        existing=None,
                    ),
                    68724.0,
                )
        finally:
            config.MARKET_CACHE_FILE = original_cache_file

    def test_ingest_gamma_market_prefers_gamma_strike_and_enddate(self) -> None:
        oracle = Oracle()
        future_end = time.time() + 240.0
        market = {
            "conditionId": "cid-15m",
            "slug": "btc-updown-15m-1",
            "endDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(future_end)),
            "eventStartTime": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(future_end - 900.0),
            ),
            "clobTokenIds": json.dumps(["up-token", "down-token"]),
            "outcomePrices": ["0.62", "0.38"],
            "strikePrice": "71297.54",
        }

        oracle._ingest_gamma_market(market, asset="BTC", market_type="updown")

        snap = oracle.state.markets["cid-15m"]
        self.assertEqual(snap.timeframe, "15m")
        self.assertAlmostEqual(snap.strike_price, 71297.54)
        self.assertGreater(snap.end_time, time.time() + 200.0)
        self.assertLess(snap.end_time, time.time() + 280.0)


class _FakeGammaResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, content_type=None):
        return []


class _FakeGammaSession:
    def get(self, *args, **kwargs):
        return _FakeGammaResponse()


class OracleExpiryRetentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_markets_keeps_recently_closed_market_for_settlement(self) -> None:
        oracle = Oracle()
        oracle._session = _FakeGammaSession()
        now = time.time()

        recent_closed = {
            "condition_id": "recent",
            "asset": "BTC",
            "timeframe": "5m",
            "up_token_id": "up",
            "down_token_id": "down",
            "strike_price": 68000.0,
            "event_start_time": now - 305.0,
            "end_time": now - 5.0,
            "binance_live_price": 68010.0,
        }
        stale_closed = {
            "condition_id": "stale",
            "asset": "ETH",
            "timeframe": "5m",
            "up_token_id": "up",
            "down_token_id": "down",
            "strike_price": 2100.0,
            "event_start_time": now - config.DISCOVERY_MAX_TIME_REMAINING_S - 305.0,
            "end_time": now - config.DISCOVERY_MAX_TIME_REMAINING_S - 5.0,
            "binance_live_price": 2101.0,
        }

        oracle.state.markets = {
            "recent": MarketSnapshot(**recent_closed),
            "stale": MarketSnapshot(**stale_closed),
        }

        await oracle._refresh_markets()

        self.assertIn("recent", oracle.state.markets)
        self.assertNotIn("stale", oracle.state.markets)


class OracleStatusLineTests(unittest.TestCase):
    def test_status_line_uses_reconnecting_after_stale_ticks(self) -> None:
        oracle = Oracle()
        now = time.time()
        oracle._gamma_last_success_at = now

        for asset in config.TRACKED_ASSETS:
            oracle.state.prices[asset].price = 1.0
            oracle.state.prices[asset].ts = (
                now - config.BINANCE_TICK_STALE_AFTER_SECONDS - 1.0
            )
            oracle._binance_connected[asset] = False
            oracle._binance_last_error[asset] = (
                "[Errno 8] nodename nor servname provided, or not known"
            )

        status = oracle.status_line()

        self.assertIn("Gamma OK", status)
        self.assertIn("Binance reconnecting", status)
        self.assertNotIn("Binance DNS/NET FAIL", status)

    def test_status_line_uses_dns_fail_before_first_tick(self) -> None:
        oracle = Oracle()
        oracle._gamma_last_success_at = time.time()

        for asset in config.TRACKED_ASSETS:
            oracle._binance_connected[asset] = False
            oracle._binance_last_error[asset] = (
                "[Errno 8] nodename nor servname provided, or not known"
            )

        self.assertIn("Binance DNS/NET FAIL", oracle.status_line())


class OracleBinanceDnsCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_binance_connect_targets_reuse_cached_ipv4_on_dns_failure(self) -> None:
        oracle = Oracle()
        hostname = "data-stream.binance.vision"
        cached_ip = "203.0.113.10"
        oracle._binance_host_ipv4_cache[hostname] = [cached_ip]
        oracle._binance_host_ipv4_cached_at[hostname] = (
            time.time() - config.BINANCE_DNS_CACHE_TTL_SECONDS - 1.0
        )

        class FailingLoop:
            async def getaddrinfo(self, *args, **kwargs):
                raise socket.gaierror(
                    8, "nodename nor servname provided, or not known"
                )

        with patch("oracle.asyncio.get_running_loop", return_value=FailingLoop()):
            targets = await oracle._binance_connect_targets(
                config.BINANCE_STREAMS["BTC"]
            )

        self.assertEqual(targets[0][0], cached_ip)
        self.assertEqual(targets[0][1], 443)
        self.assertIn(cached_ip, targets[0][2])


if __name__ == "__main__":
    unittest.main()
