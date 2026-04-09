import time
import unittest

from brain import Brain, MarketState, Phase1Status
from executioner import (
    BookLevel,
    OrderBookSnapshot,
    PortfolioSnapshot,
    PositionSnapshot,
    Side,
    SimulatorOrderManager,
)
from oracle import MarketSnapshot


class _DummyExec:
    def __init__(
        self,
        available_balance: float = 10.0,
        total_equity: float | None = None,
        active_positions: dict | None = None,
    ) -> None:
        self._available_balance = available_balance
        self._total_equity = (
            available_balance if total_equity is None else total_equity
        )
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
        available_balance: float = 10.0,
        total_equity: float | None = None,
    ) -> Brain:
        brain = Brain.__new__(Brain)
        brain._exec = _DummyExec(
            available_balance=available_balance,
            total_equity=total_equity,
        )
        brain._oracle = type("_OracleStub", (), {"price_momentum": staticmethod(lambda asset, lookback_seconds=30.0: 0.0)})()
        brain._phase2_tasks = {}
        brain._session_start_balance = 45.0
        return brain

    def test_phase1_child_order_matches_polymarket_taker_minimum(self) -> None:
        self.assertAlmostEqual(Brain._phase1_child_order_usdc(0.19), 1.00)
        self.assertAlmostEqual(Brain._phase1_child_order_usdc(0.31), 1.63)
        self.assertAlmostEqual(Brain._phase1_child_order_usdc(0.80), 4.20)

    def test_phase1_anchor_cap_uses_five_percent_of_total_balance(self) -> None:
        self.assertAlmostEqual(Brain._phase1_anchor_cap_usdc(45.0), 2.25)
        self.assertAlmostEqual(Brain._phase1_anchor_cap_usdc(12.34), 0.61)

    def test_phase1_target_side_uses_obi_thresholds(self) -> None:
        self.assertEqual(Brain._phase1_target_side(0.71), Side.UP)
        self.assertEqual(Brain._phase1_target_side(0.29), Side.DOWN)
        self.assertIsNone(Brain._phase1_target_side(0.50))

    def test_phase2_reason_bucket_normalizes_numeric_noise(self) -> None:
        self.assertEqual(
            Brain._phase2_reason_bucket("Danger zone | delta=0.14 < 0.35"),
            "danger-zone",
        )
        self.assertEqual(
            Brain._phase2_reason_bucket(
                "twap_alloc=3.6900 below minimum 3x child_size=5.2000"
            ),
            "twap-cap-vs-min-order",
        )

    def test_market_obi_falls_back_to_price_skew_when_simulated_books_are_symmetric(self) -> None:
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

        self.assertGreater(Brain._derive_obi_value(snap, flat_up, flat_down), 0.70)

    def test_anti_spoof_thresholds_flip_against_position(self) -> None:
        self.assertTrue(Brain._obi_heavily_against(Side.UP, 0.24))
        self.assertFalse(Brain._obi_heavily_against(Side.UP, 0.26))
        self.assertTrue(Brain._obi_heavily_against(Side.DOWN, 0.76))
        self.assertFalse(Brain._obi_heavily_against(Side.DOWN, 0.74))

    def test_hold_to_settlement_becomes_sticky(self) -> None:
        brain = self._brain()
        state = MarketState(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            position_side=Side.UP,
            position_source="PHASE1",
            position_cost=6.0,
            position_shares=10.0,
            avg_entry_price=0.60,
        )
        snap = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=100000.0,
            event_start_time=time.time() - 180.0,
            end_time=time.time() + 119.0,
            binance_live_price=100050.0,
        )
        high_bid_book = OrderBookSnapshot(
            token_id="up",
            bids=[BookLevel(price=0.86, size=50.0)],
            asks=[BookLevel(price=0.87, size=50.0)],
            tick_size=0.01,
        )

        self.assertTrue(brain._should_hold_to_settlement(snap, state, high_bid_book))
        self.assertTrue(state.hold_to_settlement)

        later_snap = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=100000.0,
            event_start_time=time.time() - 180.0,
            end_time=time.time() + 15.0,
            binance_live_price=100050.0,
        )
        lower_bid_book = OrderBookSnapshot(
            token_id="up",
            bids=[BookLevel(price=0.70, size=50.0)],
            asks=[BookLevel(price=0.71, size=50.0)],
            tick_size=0.01,
        )

        self.assertTrue(brain._should_hold_to_settlement(later_snap, state, lower_bid_book))

    def test_oracle_gate_requires_delta_strictly_beyond_safe_margin(self) -> None:
        brain = self._brain()
        at_margin = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 60.0,
            end_time=time.time() + 30.0,
            binance_live_price=68015.0,
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
            binance_live_price=68015.5,
        )

        safe, delta_abs, margin = brain._oracle_gate(at_margin)
        self.assertFalse(safe)
        self.assertAlmostEqual(delta_abs, 15.0)
        self.assertAlmostEqual(margin, 15.0)

        safe, delta_abs, margin = brain._oracle_gate(beyond_margin)
        self.assertTrue(safe)
        self.assertAlmostEqual(delta_abs, 15.5)
        self.assertAlmostEqual(margin, 15.0)


class BrainPhase2PlanTests(unittest.IsolatedAsyncioTestCase):
    def _brain(
        self,
        available_balance: float = 10.0,
        total_equity: float | None = None,
        active_positions: dict | None = None,
    ) -> Brain:
        brain = Brain.__new__(Brain)
        brain._exec = _DummyExec(
            available_balance=available_balance,
            total_equity=total_equity,
            active_positions=active_positions,
        )
        brain._oracle = type("_OracleStub", (), {"price_momentum": staticmethod(lambda asset, lookback_seconds=30.0: 0.0)})()
        brain._phase2_tasks = {}
        brain._session_start_balance = 45.0
        return brain

    def _snap(self) -> MarketSnapshot:
        return MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 60.0,
            end_time=time.time() + 12.0,
            best_ask_up=0.94,
            best_bid_up=0.93,
            best_ask_down=0.08,
            best_bid_down=0.07,
            binance_live_price=68040.0,
        )


class BrainDashboardSnapshotTests(unittest.IsolatedAsyncioTestCase):
    def _brain(
        self,
        available_balance: float = 10.0,
        total_equity: float | None = None,
        active_positions: dict | None = None,
    ) -> Brain:
        brain = Brain.__new__(Brain)
        brain._exec = _DummyExec(
            available_balance=available_balance,
            total_equity=total_equity,
            active_positions=active_positions,
        )
        brain._oracle = type(
            "_OracleStub",
            (),
            {"price_momentum": staticmethod(lambda asset, lookback_seconds=30.0: 0.0)},
        )()
        brain._phase2_tasks = {}
        brain._session_start_balance = 45.0
        return brain

    def _snap(self) -> MarketSnapshot:
        return MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 60.0,
            end_time=time.time() + 12.0,
            best_ask_up=0.94,
            best_bid_up=0.93,
            best_ask_down=0.08,
            best_bid_down=0.07,
            binance_live_price=68040.0,
        )

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
                "price_momentum": staticmethod(lambda asset, lookback_seconds=30.0: 0.0),
            },
        )()

        brain = Brain.__new__(Brain)
        brain._oracle = oracle
        brain._exec = _DummyExec(available_balance=100.0, total_equity=100.0)
        brain._states = {
            "cid-close": MarketState(
                condition_id="cid-close",
                market_label="BTC 5m",
                asset="BTC",
                timeframe="5m",
                position_side=Side.UP,
                position_source="PHASE2",
                position_cost=15.0,
                position_shares=16.0,
                avg_entry_price=0.9375,
            )
        }
        brain._phase2_tasks = {}
        brain._wins = 0
        brain._losses = 0
        brain._session_start_balance = 100.0

        snapshot = await brain.get_dashboard_snapshot()

        self.assertEqual(len(snapshot.markets), 1)
        self.assertEqual(snapshot.markets[0].phase, "Closing")
        self.assertEqual(snapshot.markets[0].detail, "Waiting settlement window")

    async def test_phase2_plan_builds_asymmetric_twap_chunks(self) -> None:
        brain = self._brain(available_balance=1000.0, total_equity=1000.0)
        snap = self._snap()
        winning_book = OrderBookSnapshot(
            token_id="up",
            bids=[BookLevel(price=0.93, size=100.0)],
            asks=[BookLevel(price=0.94, size=100.0)],
            tick_size=0.001,
        )
        losing_book = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.07, size=100.0)],
            asks=[BookLevel(price=0.08, size=100.0)],
            tick_size=0.001,
        )

        decision = await brain._build_phase2_plan(
            snap=snap,
            winning_side=Side.UP,
            winning_token="up",
            losing_side=Side.DOWN,
            losing_token="down",
            winning_book=winning_book,
            losing_book=losing_book,
        )

        self.assertIsNotNone(decision.plan)
        plan = decision.plan
        assert plan is not None
        self.assertAlmostEqual(plan.total_exposure_usdc, 150.0)
        self.assertAlmostEqual(plan.winning_allocation_usdc, 142.5)
        self.assertAlmostEqual(plan.insurance_allocation_usdc, 7.5)
        self.assertGreaterEqual(len(plan.chunk_sizes_usdc), 3)
        self.assertLessEqual(len(plan.chunk_sizes_usdc), 5)
        self.assertAlmostEqual(sum(plan.chunk_sizes_usdc), 142.5)
        self.assertLess(plan.chunk_fill_prices[0], plan.chunk_fill_prices[-1])

    async def test_phase2_plan_rejects_when_twap_cannot_support_three_chunks(self) -> None:
        brain = self._brain(available_balance=30.0, total_equity=200.0)
        snap = self._snap()
        winning_book = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.79, size=100.0)],
            asks=[BookLevel(price=0.80, size=100.0)],
            tick_size=0.01,
        )
        losing_book = OrderBookSnapshot(
            token_id="up",
            bids=[BookLevel(price=0.19, size=100.0)],
            asks=[BookLevel(price=0.20, size=100.0)],
            tick_size=0.01,
        )

        decision = await brain._build_phase2_plan(
            snap=snap,
            winning_side=Side.DOWN,
            winning_token="down",
            losing_side=Side.UP,
            losing_token="up",
            winning_book=winning_book,
            losing_book=losing_book,
        )

        self.assertIsNone(decision.plan)
        self.assertIn("twap_alloc=", decision.reason)
        self.assertIn("child_size=", decision.reason)

    async def test_perfect_hedge_plan_matches_expected_sniper_loss(self) -> None:
        brain = self._brain(available_balance=1000.0, total_equity=1000.0)
        snap = self._snap()
        losing_book = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.09, size=100.0)],
            asks=[BookLevel(price=0.10, size=100.0)],
            tick_size=0.01,
        )

        decision = await brain._build_perfect_hedge_plan(
            snap=snap,
            winning_side=Side.UP,
            losing_side=Side.DOWN,
            losing_token="down",
            losing_book=losing_book,
        )

        self.assertIsNotNone(decision.plan)
        plan = decision.plan
        assert plan is not None
        self.assertAlmostEqual(plan.target_shares, 150.0)
        self.assertAlmostEqual(plan.spend_usdc, 15.0)

    async def test_god_tier_hedge_plan_offsets_existing_risk_plus_projected_sniper(self) -> None:
        active_positions = {
            "cid": {
                "UP": PositionSnapshot(
                    condition_id="cid",
                    market_label="BTC 5m",
                    asset="BTC",
                    timeframe="5m",
                    token_id="up",
                    side="UP",
                    shares=100.0,
                    cost_basis=100.0,
                    avg_entry_price=1.0,
                )
            }
        }
        brain = self._brain(
            available_balance=1000.0,
            total_equity=1100.0,
            active_positions=active_positions,
        )
        snap = self._snap()
        losing_book = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.02, size=100.0)],
            asks=[BookLevel(price=0.03, size=100.0)],
            tick_size=0.01,
        )

        decision = await brain._build_god_tier_hedge_plan(
            snap=snap,
            winning_side=Side.UP,
            losing_side=Side.DOWN,
            losing_token="down",
            losing_book=losing_book,
        )

        self.assertIsNotNone(decision.plan)
        plan = decision.plan
        assert plan is not None
        self.assertAlmostEqual(plan.target_shares, 250.0)
        self.assertAlmostEqual(plan.spend_usdc, 7.5)

    async def test_otm_hedge_micro_allocation_scales_inside_insurance_band(self) -> None:
        self.assertAlmostEqual(Brain._otm_hedge_spend_usdc(1000.0, 0.15), 5.0)
        self.assertAlmostEqual(Brain._otm_hedge_spend_usdc(1000.0, 0.08), 7.5)
        self.assertAlmostEqual(Brain._otm_hedge_spend_usdc(1000.0, 0.01), 10.0)

    async def test_phase1_no_longer_places_proactive_anchor_orders(self) -> None:
        brain = self._brain(available_balance=1000.0, total_equity=1000.0)
        snap = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 60.0,
            end_time=time.time() + 90.0,
            best_ask_up=0.88,
            best_bid_up=0.87,
            best_ask_down=0.12,
            best_bid_down=0.11,
            binance_live_price=68040.0,
        )
        up_book = OrderBookSnapshot(
            token_id="up",
            bids=[BookLevel(price=0.87, size=100.0)],
            asks=[BookLevel(price=0.88, size=100.0)],
            tick_size=0.001,
        )
        down_book = OrderBookSnapshot(
            token_id="down",
            bids=[BookLevel(price=0.11, size=100.0)],
            asks=[BookLevel(price=0.12, size=100.0)],
            tick_size=0.001,
        )
        burst_calls: list[str] = []

        async def fake_book(
            _snap: MarketSnapshot,
            side: Side,
            _token_id: str,
        ) -> OrderBookSnapshot:
            return up_book if side == Side.UP else down_book

        async def fake_burst(**_kwargs) -> bool:
            burst_calls.append("burst")
            return True

        brain._book_for_side = fake_book
        brain._maybe_fire_phase1_burst = fake_burst
        state = MarketState(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
        )

        await brain.execute_phase_1(snap, state)

        self.assertEqual(burst_calls, [])
        self.assertEqual(state.phase1_status, Phase1Status.IDLE)
        self.assertIn("Waiting hedge/sniper stack", state.phase1_detail)


class ExecutionerTakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_simulator_limit_crosses_fill_phase2_insurance_orders(self) -> None:
        manager = SimulatorOrderManager(initial_balance=20.0)

        order = await manager.place_limit_order(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="down",
            side=Side.DOWN,
            price=0.15,
            size_usdc=3.0,
            phase="PHASE2_INSURANCE",
        )
        self.assertIsNotNone(order)

        fills = await manager.process_limit_crosses(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            best_ask_up=0.94,
            best_ask_down=0.12,
        )
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].price, 0.12)

    async def test_simulator_taker_buy_rejects_below_dynamic_taker_minimum(self) -> None:
        manager = SimulatorOrderManager(initial_balance=20.0)

        rejected = await manager.execute_taker_buy(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="up",
            side=Side.UP,
            phase="PHASE1_SPAM",
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
            phase="PHASE1_SPAM",
            aggressive_price=0.80,
            expected_fill_price=0.80,
            target_shares=5.25,
            max_size_usdc=4.20,
        )
        self.assertIsNotNone(filled)

    async def test_simulator_taker_sell_realizes_profit_and_clears_position(self) -> None:
        manager = SimulatorOrderManager(initial_balance=20.0)

        buy_fill = await manager.execute_taker_buy(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="up",
            side=Side.UP,
            phase="PHASE1_SPAM",
            aggressive_price=0.50,
            expected_fill_price=0.50,
            target_shares=10.0,
            max_size_usdc=5.0,
        )
        self.assertIsNotNone(buy_fill)

        sell_fill = await manager.execute_taker_sell(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="up",
            side=Side.UP,
            phase="PHASE1_TP",
            target_shares=10.0,
            expected_fill_price=0.60,
        )
        self.assertIsNotNone(sell_fill)

        snapshot = await manager.get_portfolio_snapshot()
        self.assertAlmostEqual(snapshot.realized_pnl, 1.0)
        self.assertAlmostEqual(snapshot.available_balance, 21.0)
        self.assertAlmostEqual(snapshot.locked_margin, 0.0)
        self.assertEqual(snapshot.active_positions, {})

    async def test_simulator_settlement_redeems_penny_hedge_at_one_dollar(self) -> None:
        manager = SimulatorOrderManager(initial_balance=400.0)

        main_fill = await manager.execute_taker_buy(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            token_id="up",
            side=Side.UP,
            phase="PHASE2_SNIPER",
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
            price=0.03,
            size_usdc=3.0,
            phase="GOD_HEDGE",
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
