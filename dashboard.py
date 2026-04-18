"""
dashboard.py — live Rich dashboard for the micro-order taker strategy.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import sys
import time
from typing import TYPE_CHECKING

import os

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import config

if TYPE_CHECKING:
    from brain import Brain, DashboardSnapshot
    from executioner import ChunkExecution, PositionSnapshot
    from oracle import Oracle

_TERM_NAME = os.getenv("TERM", "").lower().strip()
_TERM_IS_DUMB = _TERM_NAME in {"", "dumb", "unknown"}
_STDOUT_IS_TTY = sys.stdout.isatty()
_STDERR_IS_TTY = sys.stderr.isatty()
_RICH_LIVE_CAPABLE = _STDOUT_IS_TTY and _STDERR_IS_TTY and not _TERM_IS_DUMB

console = Console(
    force_terminal=True if _RICH_LIVE_CAPABLE else False,
    force_interactive=True if _RICH_LIVE_CAPABLE else False,
)
TEXT_DASHBOARD: bool = os.getenv("TEXT_DASHBOARD", "false").lower().strip() == "true"
DASHBOARD_RENDER_MODE: str = os.getenv(
    "DASHBOARD_RENDER_MODE", "auto"
).lower().strip()


def _resolve_dashboard_render_mode(
    *,
    term_name: str | None = None,
    text_dashboard: bool | None = None,
    requested_mode: str | None = None,
    stdout_is_tty: bool | None = None,
    stderr_is_tty: bool | None = None,
) -> str:
    term = (_TERM_NAME if term_name is None else term_name).lower().strip()
    text_mode = TEXT_DASHBOARD if text_dashboard is None else text_dashboard
    mode = DASHBOARD_RENDER_MODE if requested_mode is None else requested_mode
    mode = mode.lower().strip()
    stdout_tty = _STDOUT_IS_TTY if stdout_is_tty is None else stdout_is_tty
    stderr_tty = _STDERR_IS_TTY if stderr_is_tty is None else stderr_is_tty

    if text_mode or mode == "text":
        return "text"
    if mode == "static":
        return "static"
    if mode == "live":
        return "live"
    if not stdout_tty or not stderr_tty:
        return "text"
    if term in {"", "dumb", "unknown"}:
        return "static"
    return "live"


def _fmt_usdc(value: float, sign: bool = False) -> str:
    prefix = "+" if sign and value > 0 else ""
    return f"{prefix}${value:,.2f}"


def _fmt_time(seconds: float) -> str:
    if seconds <= 0:
        return "expired"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def _fmt_delta(delta: float) -> str:
    if delta > 0:
        color = "bright_green"
    elif delta < 0:
        color = "bright_red"
    else:
        color = "white"
    return f"[{color}]{delta:+.4f}[/{color}]"


def _fmt_market_delta(delta: float, strike_price: float, precision: int) -> str:
    if strike_price <= 0:
        return "-"
    return f"{delta:+.{precision}f}"


def _fmt_obi(value: float, signal: str) -> str:
    if "UP" in signal:
        color = "bright_green"
    elif "DOWN" in signal:
        color = "bright_red"
    else:
        color = "white"
    return f"[{color}]{value:.2f} [{signal}][/{color}]"


def _compact_signal(signal: str) -> str:
    if "UP" in signal:
        return "UP"
    if "DOWN" in signal:
        return "DOWN"
    return "NEUT"


def _compact_status(phase: str, detail: str, max_len: int = 22) -> str:
    detail = detail.strip()
    if detail in {"", "-"}:
        raw = phase
    elif phase in detail:
        raw = detail
    else:
        raw = f"{phase} | {detail}"
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 1].rstrip() + "…"


def _missing_market_reason(oracle_status: str) -> str:
    normalized = oracle_status.lower().strip()
    if "gamma booting" in normalized:
        return "Waiting for discovery"
    if normalized.endswith("cache") or " | cache" in normalized:
        return "Cached / no fresh live market"
    if "gamma dns/net fail" in normalized:
        return "Gamma unavailable"
    return "No live market"


def _tracked_market_views(
    snapshot: "DashboardSnapshot",
) -> list[tuple[str, str, "MarketView | None"]]:
    live_markets = {
        (market.asset, market.timeframe): market
        for market in snapshot.markets
    }
    return [
        (asset, timeframe, live_markets.get((asset, timeframe)))
        for asset, timeframe in config.TRACKED_MARKETS
    ]


def _build_header(snapshot: "DashboardSnapshot") -> Panel:
    mode = "[bold yellow]DRY RUN[/bold yellow]" if config.DRY_RUN else "[bold red]LIVE[/bold red]"
    prices = snapshot.prices
    header = (
        "POLYMARKET MICRO-TAKER"
        f" | {mode}"
        f" | BTC {prices.get('BTC', 0.0):,.2f}"
        f" | ETH {prices.get('ETH', 0.0):,.2f}"
        f" | SOL {prices.get('SOL', 0.0):,.4f}"
        f" | {snapshot.oracle_status}"
        f" | {datetime.now().strftime('%H:%M:%S')}"
    )
    return Panel(header, border_style="bright_blue")


def _build_stats(snapshot: "DashboardSnapshot") -> Panel:
    portfolio = snapshot.portfolio
    realized = portfolio.realized_pnl
    realized_pct = (
        realized / snapshot.session_start_balance * 100.0
        if snapshot.session_start_balance
        else 0.0
    )
    total_trades = snapshot.wins + snapshot.losses
    win_rate = f"{snapshot.win_rate:.1f}% ({snapshot.wins}/{snapshot.losses})" if total_trades else "-"
    table = Table(
        box=box.SIMPLE_HEAVY,
        expand=True,
        show_header=False,
        pad_edge=False,
    )
    for _ in range(5):
        table.add_column(justify="center", no_wrap=True, ratio=1)

    table.add_row(
        "[bold]Available\nUSDC[/bold]",
        "[bold]Locked\nExposure[/bold]",
        "[bold]Total\nEquity[/bold]",
        "[bold]Realized\nPnL[/bold]",
        "[bold]Win\nRate[/bold]",
    )
    table.add_row(
        _fmt_usdc(portfolio.available_balance),
        _fmt_usdc(portfolio.locked_margin),
        _fmt_usdc(portfolio.total_equity),
        f"{_fmt_usdc(realized, sign=True)} ({realized_pct:+.2f}%)",
        win_rate,
    )
    return Panel(table, title="Portfolio", border_style="bright_blue")


def _build_market_table(
    snapshot: "DashboardSnapshot",
    *,
    terminal_width: int | None = None,
) -> Panel:
    _ = terminal_width
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Mkt", justify="left", width=7, no_wrap=True)
    table.add_column("Oracle", justify="right", width=9, no_wrap=True)
    table.add_column("Delta", justify="right", width=9, no_wrap=True)
    table.add_column("OBI", justify="left", width=7, no_wrap=True)
    table.add_column("Status", justify="left")
    table.add_column("Time", justify="right", width=5, no_wrap=True)

    missing_reason = _missing_market_reason(snapshot.oracle_status)
    for asset, timeframe, market in _tracked_market_views(snapshot):
        if market is None:
            table.add_row(
                f"{asset} {timeframe}",
                "-",
                "-",
                "-",
                missing_reason,
                "-",
            )
            continue

        precision = 4 if market.asset == "SOL" else 2
        table.add_row(
            f"{market.asset} {market.timeframe}",
            f"{market.oracle_price:,.{precision}f}",
            _fmt_market_delta(market.delta, market.strike_price, precision),
            f"{market.obi_value:.2f} {_compact_signal(market.obi_signal)}",
            _compact_status(market.phase, market.detail, max_len=30),
            _fmt_time(market.time_remaining),
        )

    return Panel(table, title="Parallel Markets", border_style="bright_blue")


def _flatten_positions(snapshot: "DashboardSnapshot") -> list["PositionSnapshot"]:
    positions: list["PositionSnapshot"] = []
    for side_map in snapshot.portfolio.active_positions.values():
        positions.extend(side_map.values())
    positions.sort(key=lambda position: (position.asset, position.timeframe, position.side))
    return positions


def _build_positions(snapshot: "DashboardSnapshot") -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Market", justify="left")
    table.add_column("Side", justify="center", width=6)
    table.add_column("Shares", justify="right")
    table.add_column("Avg Entry", justify="right")
    table.add_column("Cost Basis", justify="right")

    rows = _flatten_positions(snapshot)
    for position in rows[:8]:
        precision = 4 if position.asset == "SOL" else 2
        side_color = "bright_green" if position.side == "UP" else "bright_red"
        table.add_row(
            position.market_label,
            f"[{side_color}]{position.side}[/{side_color}]",
            f"{position.shares:,.4f}",
            f"{position.avg_entry_price:.4f}",
            f"{position.cost_basis:,.{precision}f}",
        )

    for _ in range(max(0, 8 - len(rows))):
        table.add_row("", "", "", "", "")

    return Panel(table, title="Open Positions", border_style="bright_blue")


def _build_chunks(snapshot: "DashboardSnapshot") -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Time", width=8)
    table.add_column("Action", justify="center", width=12)
    table.add_column("Market", justify="left")
    table.add_column("Side", justify="center", width=6)
    table.add_column("Phase", justify="center", width=12)
    table.add_column("Size", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Shares", justify="right")

    rows: list["ChunkExecution"] = snapshot.portfolio.recent_chunks[:10]
    for chunk in rows:
        side_color = "bright_green" if chunk.side == "UP" else "bright_red"
        if chunk.side not in {"UP", "DOWN"}:
            side_color = "white"
        table.add_row(
            datetime.fromtimestamp(chunk.timestamp).strftime("%H:%M:%S"),
            chunk.action,
            chunk.market_label,
            f"[{side_color}]{chunk.side}[/{side_color}]",
            chunk.phase,
            _fmt_usdc(chunk.size_usdc),
            f"${chunk.price:.3f}" if chunk.price else "-",
            f"{chunk.shares:,.4f}" if chunk.shares else "-",
        )

    for _ in range(max(0, 10 - len(rows))):
        table.add_row("", "", "", "", "", "", "", "")

    return Panel(table, title="Last 10 Executed Chunks", border_style="bright_blue")


class Dashboard:
    def __init__(self, oracle: "Oracle", brain: "Brain") -> None:
        self._oracle = oracle
        self._brain = brain
        self._last_snapshot: "DashboardSnapshot | None" = None

    def _make_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stats", size=7),
            Layout(name="markets", size=11),
            Layout(name="lower"),
        )
        layout["lower"].split_row(
            Layout(name="positions"),
            Layout(name="chunks"),
        )
        return layout

    def _render(self, layout: Layout, snapshot: "DashboardSnapshot") -> None:
        layout["header"].update(_build_header(snapshot))
        layout["stats"].update(_build_stats(snapshot))
        layout["markets"].update(_build_market_table(snapshot))
        layout["positions"].update(_build_positions(snapshot))
        layout["chunks"].update(_build_chunks(snapshot))

    def generate_table(self, snapshot: "DashboardSnapshot") -> Layout:
        layout = self._make_layout()
        self._render(layout, snapshot)
        return layout

    @staticmethod
    def _snapshot_timeout() -> float:
        return max(2.0, config.DASHBOARD_REFRESH * 3.0)

    async def _refresh_snapshot(self) -> None:
        snapshot = await asyncio.wait_for(
            self._brain.get_dashboard_snapshot(),
            timeout=self._snapshot_timeout(),
        )
        self._last_snapshot = snapshot

    def _print_static_snapshot(self, snapshot: "DashboardSnapshot") -> None:
        if console.is_terminal:
            console.clear(home=True)
        console.print(self.generate_table(snapshot))

    async def _run_static_dashboard(self, *, announce_fallback: bool = False) -> None:
        if announce_fallback:
            console.print(
                "[yellow]Dashboard fallback:[/yellow] static mode (terminal does not support Rich Live cleanly)."
            )
        refresh_task: asyncio.Task | None = None
        while True:
            loop_start = time.monotonic()
            if refresh_task is None or refresh_task.done():
                if refresh_task is not None:
                    try:
                        refresh_task.result()
                    except (asyncio.TimeoutError, Exception) as exc:
                        console.print(f"[yellow]Snapshot refresh failed:[/yellow] {type(exc).__name__}: {exc}")
                refresh_task = asyncio.create_task(self._refresh_snapshot())
            self._print_static_snapshot(self._last_snapshot)
            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.0, config.DASHBOARD_REFRESH - elapsed))

    async def _run_text_dashboard(self, *, announce_fallback: bool = False) -> None:
        if announce_fallback:
            console.print(
                "[yellow]Dashboard fallback:[/yellow] text mode (terminal does not support Rich Live cleanly)."
            )
        refresh_task: asyncio.Task | None = None
        while True:
            loop_start = time.monotonic()
            if refresh_task is None or refresh_task.done():
                if refresh_task is not None:
                    try:
                        refresh_task.result()
                    except (asyncio.TimeoutError, Exception) as exc:
                        console.print(f"[yellow]Snapshot refresh failed:[/yellow] {type(exc).__name__}: {exc}")
                refresh_task = asyncio.create_task(self._refresh_snapshot())
            if console.is_terminal:
                console.clear(home=True)
            self._print_text_snapshot(self._last_snapshot)
            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.0, config.DASHBOARD_REFRESH - elapsed))

    def _print_text_snapshot(self, snapshot: "DashboardSnapshot") -> None:
        portfolio = snapshot.portfolio
        lines = [
            (
                "POLYMARKET MICRO-TAKER"
                f" | BTC {snapshot.prices.get('BTC', 0.0):,.2f}"
                f" | ETH {snapshot.prices.get('ETH', 0.0):,.2f}"
                f" | SOL {snapshot.prices.get('SOL', 0.0):,.4f}"
                f" | {snapshot.oracle_status}"
            ),
            (
                f"Bal {_fmt_usdc(portfolio.available_balance)}"
                f" | Locked {_fmt_usdc(portfolio.locked_margin)}"
                f" | Eq {_fmt_usdc(portfolio.total_equity)}"
                f" | PnL {_fmt_usdc(portfolio.realized_pnl, sign=True)}"
            ),
        ]
        missing_reason = _missing_market_reason(snapshot.oracle_status)
        for asset, timeframe, market in _tracked_market_views(snapshot):
            if market is None:
                lines.append(
                    f"{asset} {timeframe} | delta - | OBI - | {missing_reason}"
                )
                continue
            lines.append(
                (
                    f"{market.asset} {market.timeframe}"
                    f" | delta {_fmt_market_delta(market.delta, market.strike_price, 4)}"
                    f" | OBI {market.obi_value:.2f} {market.obi_signal}"
                    f" | {market.phase}"
                    f" | {market.detail}"
                )
            )
        recent_chunks = snapshot.portfolio.recent_chunks[:3]
        if recent_chunks:
            lines.append("Recent fills:")
            for chunk in recent_chunks:
                lines.append(
                    (
                        f" {datetime.fromtimestamp(chunk.timestamp).strftime('%H:%M:%S')}"
                        f" | {chunk.action}"
                        f" | {chunk.market_label}"
                        f" | {chunk.side}"
                        f" | {chunk.phase}"
                        f" | {_fmt_usdc(chunk.size_usdc)}"
                        f" @ {chunk.price:.3f}"
                    )
                )
        positions = _flatten_positions(snapshot)[:3]
        if positions:
            lines.append("Open positions:")
            for position in positions:
                lines.append(
                    (
                        f" {position.market_label}"
                        f" | {position.side}"
                        f" | shares {position.shares:,.4f}"
                        f" | avg {position.avg_entry_price:.4f}"
                        f" | cost {_fmt_usdc(position.cost_basis)}"
                    )
                )
        console.print("\n".join(lines))

    async def live_dashboard(self) -> None:
        console.print("[bold cyan]Starting dashboard...[/bold cyan]")
        while self._last_snapshot is None:
            try:
                self._last_snapshot = await self._brain.get_dashboard_snapshot()
            except Exception as exc:
                console.print(f"[yellow]Waiting for first snapshot:[/yellow] {exc}")
                await asyncio.sleep(0.1)

        render_mode = _resolve_dashboard_render_mode()
        if render_mode == "static":
            await self._run_static_dashboard(
                announce_fallback=(not TEXT_DASHBOARD and _TERM_IS_DUMB)
            )
            return
        if render_mode == "text":
            await self._run_text_dashboard(announce_fallback=False)
            return

        try:
            with Live(
                self.generate_table(self._last_snapshot),
                console=console,
                screen=False,
                auto_refresh=False,
                refresh_per_second=1,
                vertical_overflow="crop",
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
                refresh_task: asyncio.Task | None = None
                while True:
                    loop_start = time.monotonic()

                    # Start a new snapshot refresh whenever the previous one finishes.
                    # This runs in the background so the display updates every second
                    # even when live-mode network calls (OBI fetches, balance sync) are slow.
                    if refresh_task is None or refresh_task.done():
                        if refresh_task is not None:
                            try:
                                refresh_task.result()
                            except (asyncio.TimeoutError, Exception):
                                pass
                        refresh_task = asyncio.create_task(self._refresh_snapshot())

                    # Always re-render so datetime.now() in the header is current.
                    live.update(self.generate_table(self._last_snapshot), refresh=True)
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0.0, config.DASHBOARD_REFRESH - elapsed))
        except Exception as exc:
            console.print(f"[yellow]Rich Live fallback:[/yellow] {exc}")
            await self._run_static_dashboard(announce_fallback=False)

    async def run(self) -> None:
        await self.live_dashboard()
