import unittest
import time

from rich.console import Console

import config
from brain import DashboardSnapshot, MarketView
from dashboard import _build_market_table, _build_stats, _missing_market_reason, _resolve_dashboard_render_mode
from executioner import PortfolioSnapshot


class DashboardRenderModeTests(unittest.TestCase):
    def test_auto_mode_falls_back_to_static_for_dumb_terminal(self) -> None:
        self.assertEqual(
            _resolve_dashboard_render_mode(
                term_name="dumb",
                text_dashboard=False,
                requested_mode="auto",
            ),
            "static",
        )

    def test_live_override_is_respected(self) -> None:
        self.assertEqual(
            _resolve_dashboard_render_mode(
                term_name="dumb",
                text_dashboard=False,
                requested_mode="live",
            ),
            "live",
        )

    def test_text_flag_wins_over_live_capability(self) -> None:
        self.assertEqual(
            _resolve_dashboard_render_mode(
                term_name="xterm-256color",
                text_dashboard=True,
                requested_mode="auto",
            ),
            "text",
        )


class DashboardMarketTableTests(unittest.TestCase):
    @staticmethod
    def _portfolio() -> PortfolioSnapshot:
        return PortfolioSnapshot(
            available_balance=45.0,
            locked_margin=0.0,
            realized_pnl=0.0,
            total_equity=45.0,
            current_balance=45.0,
            active_positions={},
            open_orders=[],
            recent_chunks=[],
        )

    @staticmethod
    def _market(asset: str, timeframe: str, *, delta: float = 10.0) -> MarketView:
        return MarketView(
            condition_id=f"{asset}-{timeframe}",
            market_label=f"{asset} {timeframe}",
            asset=asset,
            timeframe=timeframe,
            strike_price=68000.0 if asset == "BTC" else 2100.0 if asset == "ETH" else 80.0,
            oracle_price=68010.0 if asset == "BTC" else 2103.0 if asset == "ETH" else 80.35,
            delta=delta,
            margin_text="Cap: $2.25 | Lkd: $0.00",
            best_ask_up=0.51,
            best_ask_down=0.49,
            end_time=time.time() + 60.0,
            obi_value=0.75,
            obi_signal="BULLISH UP",
            phase="Phase 2 Blocked",
            detail="Danger zone | delta=10.00 < 15.00",
        )

    @classmethod
    def _snapshot(
        cls,
        *,
        markets: list[MarketView] | None = None,
        oracle_status: str = "Gamma OK | Binance OK | LIVE",
    ) -> DashboardSnapshot:
        return DashboardSnapshot(
            prices={"BTC": 68010.0, "ETH": 2100.0, "SOL": 80.0},
            portfolio=cls._portfolio(),
            markets=[] if markets is None else markets,
            oracle_status=oracle_status,
            wins=0,
            losses=0,
            win_rate=0.0,
            session_start_balance=45.0,
        )

    @staticmethod
    def _render(snapshot: DashboardSnapshot, *, width: int) -> str:
        console = Console(width=width, record=True, force_terminal=True)
        console.print(_build_market_table(snapshot, terminal_width=width))
        return console.export_text()

    def test_compact_market_table_keeps_delta_visible(self) -> None:
        snapshot = self._snapshot(markets=[self._market("BTC", "5m")])
        rendered = self._render(snapshot, width=80)

        self.assertIn("Delta", rendered)
        self.assertIn("+10.00", rendered)
        self.assertIn("BTC 5m", rendered)

    def test_market_table_keeps_assets_visible_at_140_columns(self) -> None:
        snapshot = self._snapshot(
            markets=[
                self._market(asset, timeframe)
                for asset, timeframe in config.TRACKED_MARKETS
            ]
        )
        rendered = self._render(snapshot, width=140)

        self.assertIn("Mkt", rendered)
        self.assertIn("Status", rendered)
        for asset, timeframe in config.TRACKED_MARKETS:
            self.assertIn(f"{asset} {timeframe}", rendered)

    def test_market_table_renders_placeholders_for_missing_rows(self) -> None:
        snapshot = self._snapshot(
            markets=[
                self._market("BTC", "5m"),
                self._market("SOL", "5m"),
            ]
        )
        rendered = self._render(snapshot, width=140)

        for asset, timeframe in config.TRACKED_MARKETS:
            self.assertIn(f"{asset} {timeframe}", rendered)
        self.assertIn("No live market", rendered)

    def test_missing_market_reason_for_gamma_failure(self) -> None:
        self.assertEqual(
            _missing_market_reason("Gamma DNS/NET FAIL | Binance OK | LIVE"),
            "Gamma unavailable",
        )

    def test_missing_market_reason_for_cache_mode(self) -> None:
        self.assertEqual(
            _missing_market_reason("Gamma OK | Binance OK | CACHE"),
            "Cached / no fresh live market",
        )

    def test_missing_market_reason_for_booting(self) -> None:
        self.assertEqual(
            _missing_market_reason("Gamma booting | Binance booting | LIVE"),
            "Waiting for discovery",
        )

    def test_missing_market_reason_for_normal_live_status(self) -> None:
        self.assertEqual(
            _missing_market_reason("Gamma OK | Binance OK | LIVE"),
            "No live market",
        )


class DashboardStatsTests(unittest.TestCase):
    def test_stats_panel_renders_portfolio_values_in_single_table(self) -> None:
        snapshot = DashboardSnapshot(
            prices={"BTC": 0.0, "ETH": 0.0, "SOL": 0.0},
            portfolio=PortfolioSnapshot(
                available_balance=63.45,
                locked_margin=36.55,
                realized_pnl=0.0,
                total_equity=100.0,
                current_balance=100.0,
                active_positions={},
                open_orders=[],
                recent_chunks=[],
            ),
            markets=[],
            oracle_status="Gamma OK | Binance OK | LIVE",
            wins=0,
            losses=0,
            win_rate=0.0,
            session_start_balance=100.0,
        )

        console = Console(width=180, record=True, force_terminal=True)
        console.print(_build_stats(snapshot))
        rendered = console.export_text()

        self.assertIn("Available", rendered)
        self.assertIn("USDC", rendered)
        self.assertIn("$63.45", rendered)
        self.assertIn("$36.55", rendered)
        self.assertIn("$100.00", rendered)


if __name__ == "__main__":
    unittest.main()
