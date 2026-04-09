import time
import unittest

from brain import Brain, MarketState
from executioner import PortfolioSnapshot, Side
from oracle import MarketSnapshot


class BrainPhase2OverlapTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_window_prioritizes_phase2_over_phase1_position(self) -> None:
        brain = Brain.__new__(Brain)
        brain._states = {}
        brain._phase2_tasks = {}
        brain._wins = 0
        brain._losses = 0

        state = MarketState(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            position_side=Side.UP,
            position_source="PHASE1",
            position_token_id="up",
            position_cost=2.68,
            position_shares=5.25,
            avg_entry_price=0.51,
        )
        snap = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 120.0,
            end_time=time.time() + 12.0,
            binance_live_price=68040.0,
        )

        calls: list[str] = []

        async def fake_refresh(_state: MarketState) -> PortfolioSnapshot:
            return PortfolioSnapshot(
                available_balance=10.0,
                locked_margin=2.68,
                realized_pnl=0.0,
                total_equity=12.68,
                current_balance=12.68,
                active_positions={},
                open_orders=[],
                recent_chunks=[],
            )

        async def fake_phase1(_snap: MarketSnapshot, _state: MarketState) -> None:
            calls.append("phase1")

        async def fake_phase2(_snap: MarketSnapshot, _state: MarketState) -> None:
            calls.append("phase2")

        async def noop(*_args, **_kwargs) -> None:
            return None

        brain._state_for = lambda _snap: state
        brain._process_limit_orders = noop
        brain._run_layered_hedges = noop
        brain._refresh_market_position_state = fake_refresh
        brain.execute_phase_1 = fake_phase1
        brain.execute_phase_2 = fake_phase2

        await brain._evaluate_market(snap)

        self.assertEqual(calls, ["phase2"])

    async def test_mixed_exposure_can_continue_phase2_sniper(self) -> None:
        brain = Brain.__new__(Brain)
        brain._states = {}
        brain._phase2_tasks = {}
        brain._wins = 0
        brain._losses = 0

        state = MarketState(
            condition_id="cid",
            market_label="BTC 5m",
            asset="BTC",
            timeframe="5m",
            mixed_exposure=True,
            position_side=Side.UP,
            position_source="PHASE2",
            position_token_id="up",
            position_cost=2.68,
            position_shares=5.25,
            avg_entry_price=0.51,
            phase2_bullets_fired=1,
        )
        snap = MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 120.0,
            end_time=time.time() + 12.0,
            binance_live_price=68040.0,
        )

        calls: list[str] = []

        async def fake_refresh(_state: MarketState) -> PortfolioSnapshot:
            return PortfolioSnapshot(
                available_balance=10.0,
                locked_margin=2.68,
                realized_pnl=0.0,
                total_equity=12.68,
                current_balance=12.68,
                active_positions={},
                open_orders=[],
                recent_chunks=[],
            )

        async def fake_phase2(_snap: MarketSnapshot, _state: MarketState) -> None:
            calls.append("phase2")

        async def noop(*_args, **_kwargs) -> None:
            return None

        brain._state_for = lambda _snap: state
        brain._process_limit_orders = noop
        brain._run_layered_hedges = noop
        brain._refresh_market_position_state = fake_refresh
        brain.execute_phase_2 = fake_phase2

        await brain._evaluate_market(snap)

        self.assertEqual(calls, ["phase2"])


if __name__ == "__main__":
    unittest.main()
