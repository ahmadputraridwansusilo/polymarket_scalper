"""
main.py — Entry point.

Starts the oracle, brain, and dashboard concurrently. By default, runtime logs
go only to scalper.log so Rich Live keeps full control of the terminal.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import config  # triggers load_dotenv(dotenv_path=..., override=False) inside config.py

from brain import Brain
from dashboard import Dashboard
from oracle import Oracle

LOG_CONSOLE: bool = os.getenv("LOG_CONSOLE", "false").lower().strip() == "true"


# ---------------------------------------------------------------------------
# Logging — redirect everything to file so Rich's Live display stays clean.
# A StreamHandler is added only when LOG_CONSOLE=true.
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_polyscalper_logging_configured", False):
        return

    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler("scalper.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logging.getLogger("websockets.client").setLevel(logging.WARNING)
    logging.getLogger("websockets.server").setLevel(logging.WARNING)

    if LOG_CONSOLE:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)

    root._polyscalper_logging_configured = True


log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    _configure_logging()
    print(
        f"Starting Polymarket Scalper | DRY_RUN={config.DRY_RUN} | log=scalper.log",
        flush=True,
    )
    log.info(
        "Startup config | dry_run=%s | signature_type=%s | funder=%s | api_key=%s",
        config.DRY_RUN,
        config.SIGNATURE_TYPE,
        config.FUNDER or "(unset)",
        "loaded" if config.API_KEY else "missing",
    )

    if not config.DRY_RUN:
        if not config.PRIVATE_KEY:
            print("ERROR: LIVE mode requires POLY_PRIVATE_KEY. Aborting.", file=sys.stderr)
            return
        log.warning("LIVE TRADING — real capital at risk.")

    oracle    = Oracle()
    brain     = Brain(oracle)
    dashboard = Dashboard(oracle, brain)

    # In LIVE mode: fetch the real wallet balance NOW, before the dashboard starts.
    # This is a hard failure — if we can't read the balance, we do NOT proceed.
    if not config.DRY_RUN:
        await brain._exec.sync_live_balance()
        portfolio = await brain._exec.get_portfolio_snapshot()
        brain._session_start_balance = portfolio.total_equity
        log.info("[LIVE] Session start balance: $%.2f", brain._session_start_balance)

    log.info("Starting Polymarket Scalper (DRY_RUN=%s).", config.DRY_RUN)

    try:
        await asyncio.gather(
            oracle.run(),
            brain.run(),
            dashboard.run(),
        )
    except asyncio.CancelledError:
        log.info("Shutdown requested.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass   # dashboard's Live context already cleaned up the terminal
