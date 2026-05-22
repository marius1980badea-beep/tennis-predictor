#!/usr/bin/env python3
"""Standalone script to perform the initial data load.

This is the script you run ONCE to populate the database from scratch.
After this, n8n will handle incremental updates.

Usage:
    python scripts/initial_data_load.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src importable when running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from tennis_predictor.cli import cli

console = Console()


def main() -> None:
    """Run the full initial data load for ATP + WTA, 2000-2024."""
    console.print(
        "\n[bold cyan]═══════════════════════════════════════════════[/bold cyan]"
    )
    console.print("[bold cyan]  TENNIS PREDICTOR - INITIAL DATA LOAD[/bold cyan]")
    console.print(
        "[bold cyan]═══════════════════════════════════════════════[/bold cyan]\n"
    )

    console.print(
        "[yellow]This will:[/yellow]\n"
        "  1. Clone Sackmann ATP + WTA repos (~500MB each)\n"
        "  2. Load all players (~80,000 total)\n"
        "  3. Load all matches 2000-2024 (~700,000 total)\n"
        "  4. Load all match statistics\n\n"
        "[yellow]Expected duration: 15-45 minutes (depending on network)[/yellow]\n"
        "[yellow]Database disk usage: ~300-400 MB[/yellow]\n"
    )

    if not console.input("\n[bold]Continue? [y/N]:[/bold] ").lower().startswith("y"):
        console.print("[red]Cancelled.[/red]")
        sys.exit(0)

    # Run via Click CLI for consistent behavior
    cli(["load-data", "--tour", "BOTH"], standalone_mode=False)


if __name__ == "__main__":
    main()
