import time
import unittest

from brain import Brain, MarketState
from executioner import BookLevel, OrderBookSnapshot, PortfolioSnapshot, Side
from oracle import MarketSnapshot


class BrainRoutingTests(unittest.IsolatedAsyncioTestCase):
    def _snap(self, *, seconds_left: float = 90.0) -> MarketSnapshot:
        return MarketSnapshot(
            condition_id="cid",
            asset="BTC",
            timeframe="5m",
            up_token_id="up",
            down_token_id="down",
            strike_price=68000.0,
            event_start_time=time.time() - 120.0,
            end_time=time.time() + seconds_left,
            best_ask_up=0.60,
            best_bid_up=0.59,
            best_ask_down=0.40,
            best_bid_down=0.39,
            binance_live_price=68040.0,
        )

    def _book(self, bid: float, ask: float, token_id: str) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            token_id=token_id,
            bids=[BookLevel(price=bid, size=100.0)],
            asks=[BookLevel(price=ask, size=100.0)],
            tick_size=0.01,
        )

    async def test_late_window_skips_new_entry_and_enters_hold_mode(self) -> None:
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
        )
        snap = self._snap(seconds_left=30.0)
        portfolio = PortfolioSnapshot(
            available_balance=49.0,
            locked_margin=0.0,
            realized_pnl=0.0,
            total_equity=49.0,
            current_balance=49.0,
            active_positions={},
            open_orders=[],
            recent_chunks=[],
        )
        calls: list[str] = []

        async def noop(*_args, **_kwargs):
            return None

        async def fake_refresh(_state: MarketState) -> PortfolioSnapshot:
            return portfolio

        async def fake_book(_snap: MarketSnapshot, side: Side, _token: str) -> OrderBookSnapshot:
            return self._book(0.59, 0.60, "up") if side == Side.UP else self._book(0.39, 0.40, "down")

        async def fake_enter(**_kwargs) -> None:
            calls.append("entry")

        brain._state_for = lambda _snap: state
        brain._process_limit_orders = noop
        brain._refresh_market_position_state = fake_refresh
        brain._book_for_side = fake_book
        brain._maybe_enter_position = fake_enter

        await brain._evaluate_market(snap)

        self.assertEqual(calls, [])
        self.assertEqual(state.mode, "Hold Mode")

    async def test_dual_position_routes_to_dual_monitor_before_entry_logic(self) -> None:
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
            up_shares=6.0,
            up_cost=3.0,
            down_shares=12.0,
            down_cost=1.2,
        )
        snap = self._snap(seconds_left=90.0)
        portfolio = PortfolioSnapshot(
            available_balance=44.8,
            locked_margin=4.2,
            realized_pnl=0.0,
            total_equity=49.0,
            current_balance=49.0,
            active_positions={},
            open_orders=[],
            recent_chunks=[],
        )
        calls: list[str] = []

        async def noop(*_args, **_kwargs):
            return None

        async def fake_refresh(_state: MarketState) -> PortfolioSnapshot:
            return portfolio

        async def fake_book(_snap: MarketSnapshot, side: Side, _token: str) -> OrderBookSnapshot:
            return self._book(0.70, 0.71, "up") if side == Side.UP else self._book(0.15, 0.16, "down")

        async def fake_dual(**_kwargs) -> None:
            calls.append("dual")

        async def fake_enter(**_kwargs) -> None:
            calls.append("entry")

        brain._state_for = lambda _snap: state
        brain._process_limit_orders = noop
        brain._refresh_market_position_state = fake_refresh
        brain._book_for_side = fake_book
        brain._monitor_dual_position = fake_dual
        brain._maybe_enter_position = fake_enter

        await brain._evaluate_market(snap)

        self.assertEqual(calls, ["dual"])


if __name__ == "__main__":
    unittest.main()
