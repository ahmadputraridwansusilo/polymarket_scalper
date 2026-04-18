import time
import unittest

import config
from brain import Brain, MarketState
from executioner import (
    BookLevel,
    LiveOrderManager,
    OrderBookSnapshot,
    PortfolioSnapshot,
    Side,
    SimulatorOrderManager,
)
from oracle import MarketSnapshot


class _DummyExec:
    def __init__(
        self,
        available_balance: float = 49.0,
        total_equity: float | None = None,
        active_positions: dict | None = None,
    ) -> None:
        self._available_balance = available_balance
        self._total_equity = available_balance if total_equity is None else total_equity
        self._active_positions = active_positions or {}

    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            available_balance=self._available_balance,
            locked_margin=0.0,
            realized_pnl=0.0,
            total_equity=self._total_equity,
            current_balance=self._total_equity,
            active_positions=self._active_positions,
            open_orders=[],
            recent_chunks=[],
        )


class BrainHelperTests(unittest.TestCase):
    def _brain(
        self,
        available_balance: float = 49.0,
        total_equity: float | None = None,
    ) -> Brain:
        brain = Brain.__new__(Brain)
        brain._exec = _DummyExec(
            available_balance=available_balance,
            total_equity=total_equity,
        )
        brain._oracle = type("_OracleStub", (), {})()
        brain._phase2_tasks = {}
        brain._states = {}
        brain._session_start_balance = config.INITIAL_BALANCE
        brain._wins = 0
        brain._losses = 0
        return brain

    def test_scaled_entry_amount_uses_initial_balance_as_reference(self) -> None:
        brain = self._brain()

        self.assertAlmostEqual(
            brain._scaled_entry_amount(
                total_equity=49.0,
                available_balance=49.0,
                ask_price=0.20,
            ),
            3.0,
        )
        self.assertAlmostEqual(
            brain._scaled_entry_amount(
                total_equity=98.0,
                available_balance=98.0,
                ask_price=0.20,
            ),
            6.0,
        )

    def test_obi_side_uses_safe_margin_band(self) -> None:
        self.assertEqual(Brain._obi_side(0.61), Side.UP)
        self.assertEqual(Brain._obi_side(0.39), Side.DOWN)
        self.assertIsNone(Brain._obi_side(0.50))

    def test_market_obi_falls_back_to_price_skew_when_books_are_symmetric(self) -> None:
        up_book = OrderBookSnapshot(
            token_id="up",
            bids=[BookLevel(price=0.79, size=10000.0)],
            asks=[BookLevel(price=0.81, size=10000.0)],
            tick_size=0.01,
        )
        down_book = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.19, size=10000.0)],
            asks=[BookLevel(price=0.21, size=10000.0)],
            tick_size=0.01,
        )

        self.assertAlmostEqual(Brain._market_obi(up_book, down_book), 0.80)

    def test_derive_obi_uses_oracle_delta_when_books_are_flat(self) -> None:
        snap = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 60.0,
            end_time=time.time() + 120.0,
            best_ask_up=0.51,
            best_bid_up=0.49,
            best_ask_down=0.51,
            best_bid_down=0.49,
            binance_live_price=68020.0,
        )
        flat_up = OrderBookSnapshot(
            token_id="up",
            bids=[BookLevel(price=0.49, size=10000.0)],
            asks=[BookLevel(price=0.51, size=10000.0)],
            tick_size=0.01,
        )
        flat_down = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.49, size=10000.0)],
            asks=[BookLevel(price=0.51, size=10000.0)],
            tick_size=0.01,
        )

        self.assertGreaterEqual(Brain._derive_obi_value(snap, flat_up, flat_down), 0.60)

    def test_oracle_gate_requires_delta_beyond_margin(self) -> None:
        brain = self._brain()
        margin = config.safe_margin_for("BTC") * config.SAFE_MARGIN_DELTA

        at_margin = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 60.0,
            end_time=time.time() + 30.0,
            binance_live_price=68000.0 + margin,
        )
        beyond_margin = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 60.0,
            end_time=time.time() + 30.0,
            binance_live_price=68000.0 + margin + 0.5,
        )

        safe, delta_abs, returned_margin = brain._oracle_gate(at_margin)
        self.assertFalse(safe)
        self.assertAlmostEqual(delta_abs, margin)
        self.assertAlmostEqual(returned_margin, margin)

        safe, delta_abs, returned_margin = brain._oracle_gate(beyond_margin)
        self.assertTrue(safe)
        self.assertAlmostEqual(delta_abs, margin + 0.5)
        self.assertAlmostEqual(returned_margin, margin)

    def test_build_hedge_plan_finds_locked_hedge_for_cheap_opposite_side(self) -> None:
        brain = self._brain()
        state = MarketState(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            position_side=Side.UP,
            position_token_id="up",
            position_cost=5.0,
            position_shares=6.17,
            avg_entry_price=0.81,
            up_token_id="up",
            up_cost=5.0,
            up_shares=6.17,
            up_avg_entry_price=0.81,
        )
        hedge_book = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.06, size=100.0)],
            asks=[BookLevel(price=0.07, size=100.0)],
            tick_size=0.01,
        )

        plan = brain._build_hedge_plan(
            state=state,
            hedge_side=Side.DOWN,
            hedge_book=hedge_book,
            max_hedge_amount=10.0,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertGreaterEqual(plan.entry_net, -1e-6)
        self.assertGreaterEqual(plan.hedge_net, -1e-6)
        self.assertAlmostEqual(plan.hedge_amount, 1.0)

    def test_dual_take_profit_sells_both_when_both_sides_are_in_profit(self) -> None:
        brain = self._brain()
        state = MarketState(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            up_cost=3.0,
            up_shares=6.0,
            down_cost=1.2,
            down_shares=12.0,
        )
        up_book = OrderBookSnapshot(
            token_id="up",
            bids=[BookLevel(price=0.70, size=100.0)],
            asks=[BookLevel(price=0.71, size=100.0)],
            tick_size=0.01,
        )
        down_book = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.15, size=100.0)],
            asks=[BookLevel(price=0.16, size=100.0)],
            tick_size=0.01,
        )

        self.assertEqual(
            brain._dual_take_profit_candidates(
                state=state,
                up_book=up_book,
                down_book=down_book,
            ),
            (Side.UP, Side.DOWN),
        )


class BrainDashboardSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_snapshot_keeps_recently_closed_market_visible(self) -> None:
        closed_snap = MarketSnapshot(
            condition_id="cid-close",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 305.0,
            end_time=time.time() - 5.0,
            best_ask_up=0.99,
            best_bid_up=0.01,
            best_ask_down=0.01,
            best_bid_down=0.99,
            binance_live_price=68025.0,
        )

        oracle = type(
            "_OracleStub",
            (),
            {
                "price_table": staticmethod(lambda: {"BTC": 68025.0, "ETH": 0.0, "SOL": 0.0}),
                "active_markets": staticmethod(lambda: []),
                "all_markets": staticmethod(lambda: [closed_snap]),
                "status_line": staticmethod(lambda: "Gamma OK | Binance OK | LIVE"),
            },
        )()

        brain = Brain.__new__(Brain)
        brain._oracle = oracle
        brain._exec = _DummyExec(available_balance=49.0, total_equity=49.0)
        brain._states = {
            "cid-close": MarketState(
                condition_id="cid-close",
                market_label="BTC 5m",
                asset="BTC",
                timeframe="5m",
                up_shares=10.0,
                up_cost=6.0,
            )
        }
        brain._phase2_tasks = {}
        brain._wins = 0
        brain._losses = 0
        brain._session_start_balance = 49.0

        snapshot = await brain.get_dashboard_snapshot()

        self.assertEqual(len(snapshot.markets), 1)
        self.assertEqual(snapshot.markets[0].phase, "Closing")
        self.assertEqual(snapshot.markets[0].detail, "Waiting settlement window")


class ExecutionerTakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_order_book_uses_short_ttl_cache(self) -> None:
        manager = LiveOrderManager.__new__(LiveOrderManager)
        manager._sdk_ready = True
        manager._order_book_cache = {}
        manager._tick_size_cache = {}

        class _Level:
            def __init__(self, price: float, size: float) -> None:
                self.price = price
                self.size = size

        summary = type(
            "_Summary",
            (),
            {
                "bids": [_Level(0.49, 100.0)],
                "asks": [_Level(0.51, 120.0)],
                "tick_size": 0.01,
                "last_trade_price": 0.50,
            },
        )()
        calls: list[str] = []

        class _Client:
            def get_order_book(self, token_id: str):
                calls.append(token_id)
                return summary

        manager._client = _Client()

        async def fake_sdk_read_call(fn):
            return fn()

        manager._sdk_read_call = fake_sdk_read_call

        first = await manager.get_order_book("token-1")
        second = await manager.get_order_book("token-1")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(first.best_ask, 0.51)
        self.assertAlmostEqual(second.best_bid, 0.49)

    async def test_simulator_taker_buy_rejects_below_dynamic_taker_minimum(self) -> None:
        manager = SimulatorOrderManager(initial_balance=20.0)

        rejected = await manager.execute_taker_buy(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="up",
            side=Side.UP,
            phase="ENTRY",
            aggressive_price=0.80,
            expected_fill_price=0.80,
            target_shares=5.0,
            max_size_usdc=4.00,
        )
        self.assertIsNone(rejected)

        filled = await manager.execute_taker_buy(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="up",
            side=Side.UP,
            phase="ENTRY",
            aggressive_price=0.80,
            expected_fill_price=0.80,
            target_shares=5.25,
            max_size_usdc=4.20,
        )
        self.assertIsNotNone(filled)

    async def test_simulator_settlement_redeems_penny_hedge_at_one_dollar(self) -> None:
        manager = SimulatorOrderManager(initial_balance=400.0)

        main_fill = await manager.execute_taker_buy(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="up",
            side=Side.UP,
            phase="ENTRY",
            aggressive_price=1.0,
            expected_fill_price=1.0,
            target_shares=300.0,
            max_size_usdc=300.0,
        )
        self.assertIsNotNone(main_fill)

        hedge_order = await manager.place_limit_order(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="down",
            side=Side.DOWN,
            phase="GOD_HEDGE",
            price=0.03,
            size_usdc=3.0,
        )
        self.assertIsNotNone(hedge_order)

        fills = await manager.process_limit_crosses(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            best_ask_up=1.0,
            best_ask_down=0.01,
        )
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].price, 0.01)
        self.assertAlmostEqual(fills[0].shares, 300.0)

        result = await manager.settle_market(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            winning_side=Side.DOWN,
        )
        self.assertAlmostEqual(result.payout, 300.0)
        self.assertAlmostEqual(result.realized_pnl, -3.0)


if __name__ == "__main__":
    unittest.main()
