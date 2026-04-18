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


class _FakeHttpResponse:
    def __init__(self, status: int, payload) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, content_type=None):
        return self._payload


class _RoutingSession:
    def __init__(self, gamma_payload: list[dict], book_payloads: dict[str, tuple[int, dict]]) -> None:
        self._gamma_payload = gamma_payload
        self._book_payloads = book_payloads

    def get(self, url, *args, **kwargs):
        if url.endswith("/markets"):
            return _FakeHttpResponse(200, self._gamma_payload)
        if url.endswith("/book"):
            token_id = (kwargs.get("params") or {}).get("token_id", "")
            status, payload = self._book_payloads.get(
                token_id,
                (404, {"error": "No orderbook exists for the requested token id"}),
            )
            return _FakeHttpResponse(status, payload)
        raise AssertionError(f"Unexpected URL: {url}")


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

    async def test_refresh_markets_drops_market_without_live_clob_book(self) -> None:
        oracle = Oracle()
        future_end = time.time() + 240.0
        market = {
            "conditionId": "cid-live-check",
            "slug": "btc-updown-5m-1",
            "endDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(future_end)),
            "eventStartTime": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(future_end - 300.0),
            ),
            "clobTokenIds": json.dumps(["up-missing", "down-ok"]),
            "outcomePrices": ["0.52", "0.48"],
            "strikePrice": "68000",
        }
        oracle._session = _RoutingSession(
            gamma_payload=[market],
            book_payloads={
                "up-missing": (404, {"error": "No orderbook exists for the requested token id"}),
                "down-ok": (
                    200,
                    {
                        "bids": [{"price": "0.47", "size": "100"}],
                        "asks": [{"price": "0.49", "size": "100"}],
                    },
                ),
            },
        )

        await oracle._refresh_markets()

        self.assertNotIn("cid-live-check", oracle.state.markets)
        self.assertEqual(oracle.active_markets(), [])
        self.assertEqual(oracle._clob_validation_cache["up-missing"][0], "missing")

    def test_active_markets_skips_cached_missing_orderbook_tokens(self) -> None:
        oracle = Oracle()
        now = time.time()
        oracle.state.markets = {
            "cid": MarketSnapshot(
                condition_id="cid",
                asset="BTC",
                timeframe="5m",
                up_token_id="up-missing",
                down_token_id="down-ok",
                strike_price=68000.0,
                event_start_time=now - 120.0,
                end_time=now + 120.0,
                binance_live_price=68010.0,
            )
        }
        oracle._clob_validation_cache["up-missing"] = ("missing", now)

        self.assertEqual(oracle.active_markets(), [])

    def test_active_markets_skips_placeholder_price_shapes(self) -> None:
        oracle = Oracle()
        now = time.time()
        oracle.state.markets = {
            "cid": MarketSnapshot(
                condition_id="cid",
                asset="BTC",
                timeframe="5m",
                up_token_id="up",
                down_token_id="down",
                strike_price=68000.0,
                event_start_time=now - 120.0,
                end_time=now + 120.0,
                best_ask_up=0.99,
                best_bid_up=0.01,
                best_ask_down=0.99,
                best_bid_down=0.01,
                binance_live_price=68010.0,
            )
        }

        self.assertEqual(oracle.active_markets(), [])


class OracleExtractStrikeTests(unittest.TestCase):
    """Unit tests for _extract_gamma_strike — including text parsing."""

    def _strike(self, mkt: dict) -> float:
        return Oracle._extract_gamma_strike(mkt)

    # ── Priority 1: direct numeric fields ─────────────────────────────────────

    def test_numeric_field_strikePrice(self) -> None:
        self.assertAlmostEqual(self._strike({"strikePrice": "84320.50"}), 84320.50)

    def test_numeric_field_referencePrice(self) -> None:
        self.assertAlmostEqual(self._strike({"referencePrice": "1234.0"}), 1234.0)

    def test_numeric_field_openPrice(self) -> None:
        self.assertAlmostEqual(self._strike({"openPrice": 127.5}), 127.5)

    # ── Priority 2: question / description text ────────────────────────────────

    def test_question_text_btc_above(self) -> None:
        mkt = {"question": "Will BTC be above $84,320 at 7:35PM ET?"}
        self.assertAlmostEqual(self._strike(mkt), 84320.0)

    def test_question_text_with_decimal(self) -> None:
        mkt = {"question": "Will ETH be above $1,234.56 at close?"}
        self.assertAlmostEqual(self._strike(mkt), 1234.56)

    def test_question_text_no_commas(self) -> None:
        mkt = {"question": "Will SOL close above $127.50?"}
        self.assertAlmostEqual(self._strike(mkt), 127.50)

    def test_description_text_fallback(self) -> None:
        mkt = {"description": "Strike price is $84320 for this market."}
        self.assertAlmostEqual(self._strike(mkt), 84320.0)

    def test_question_takes_priority_over_description(self) -> None:
        mkt = {
            "question": "Will BTC be above $84,320?",
            "description": "Reference: $80,000",
        }
        self.assertAlmostEqual(self._strike(mkt), 84320.0)

    # ── Priority 3: outcomes list ──────────────────────────────────────────────

    def test_outcomes_list_above_string(self) -> None:
        mkt = {"outcomes": json.dumps(["Above $84,320", "Below $84,320"])}
        self.assertAlmostEqual(self._strike(mkt), 84320.0)

    def test_outcomes_list_plain_names_no_match(self) -> None:
        # "Up" / "Down" outcomes carry no dollar amount — should return 0
        mkt = {"outcomes": json.dumps(["Up", "Down"])}
        self.assertEqual(self._strike(mkt), 0.0)

    # ── Direct-field takes priority over text ─────────────────────────────────

    def test_numeric_field_beats_question_text(self) -> None:
        mkt = {
            "strikePrice": "84320.0",
            "question": "Will BTC be above $99,999?",
        }
        self.assertAlmostEqual(self._strike(mkt), 84320.0)

    # ── No strike anywhere ────────────────────────────────────────────────────

    def test_returns_zero_when_no_strike(self) -> None:
        mkt = {"question": "Will BTC go up or down?", "outcomePrices": ["0.52", "0.48"]}
        self.assertEqual(self._strike(mkt), 0.0)


class OracleResolveStrikePriorityTests(unittest.TestCase):
    """gamma_strike must override provisional/existing strike."""

    def test_gamma_beats_existing_provisional(self) -> None:
        oracle = Oracle()
        # Simulate existing snapshot with provisional Binance-derived strike
        existing = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=83000.0,   # wrong provisional value
            event_start_time=time.time() - 60,
            end_time=time.time() + 240,
        )
        # Gamma now correctly provides the real strike via text parse
        result = oracle._resolve_market_strike(
            condition_id="cid",
            asset="BTC",
            event_start_time=existing.event_start_time,
            existing=existing,
            gamma_strike=84320.0,
        )
        self.assertAlmostEqual(result, 84320.0)

    def test_existing_used_when_no_gamma_strike(self) -> None:
        oracle = Oracle()
        existing = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=84320.0,
            event_start_time=time.time() - 60,
            end_time=time.time() + 240,
        )
        result = oracle._resolve_market_strike(
            condition_id="cid",
            asset="BTC",
            event_start_time=existing.event_start_time,
            existing=existing,
            gamma_strike=0.0,
        )
        self.assertAlmostEqual(result, 84320.0)


class OracleIngestWithTextStrikeTests(unittest.TestCase):
    """Integration: _ingest_gamma_market picks up strike from question text."""

    def test_ingest_uses_question_text_when_no_numeric_field(self) -> None:
        oracle = Oracle()
        future_end = time.time() + 240.0
        market = {
            "conditionId": "cid-text-strike",
            "slug": "btc-updown-5m-1",
            "endDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(future_end)),
            "eventStartTime": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(future_end - 300.0),
            ),
            "clobTokenIds": json.dumps(["up-tok", "down-tok"]),
            "outcomePrices": ["0.55", "0.45"],
            # No strikePrice field — strike is embedded in question
            "question": "Will BTC be above $84,320 at 7:35PM ET?",
        }
        oracle._ingest_gamma_market(market, asset="BTC", market_type="updown")
        snap = oracle.state.markets.get("cid-text-strike")
        self.assertIsNotNone(snap)
        self.assertAlmostEqual(snap.strike_price, 84320.0)

    def test_ingest_gamma_strike_updates_existing_provisional(self) -> None:
        oracle = Oracle()
        future_end = time.time() + 240.0
        # Pre-populate with a wrong provisional strike (from Binance estimate)
        oracle.state.markets["cid-update"] = MarketSnapshot(
            condition_id="cid-update",
            asset="BTC",
            timeframe="5m",
            up_token_id="up-tok",
            down_token_id="down-tok",
            strike_price=83000.0,   # wrong provisional
            event_start_time=future_end - 300.0,
            end_time=future_end,
        )
        market = {
            "conditionId": "cid-update",
            "slug": "btc-updown-5m-1",
            "endDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(future_end)),
            "eventStartTime": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(future_end - 300.0),
            ),
            "clobTokenIds": json.dumps(["up-tok", "down-tok"]),
            "outcomePrices": ["0.55", "0.45"],
            "strikePrice": "84320.0",   # correct strike now in Gamma
        }
        oracle._ingest_gamma_market(market, asset="BTC", market_type="updown")
        snap = oracle.state.markets["cid-update"]
        self.assertAlmostEqual(snap.strike_price, 84320.0)


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
