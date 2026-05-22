"""Build Elo ratings from historical matches.

Processes all matches in chronological order, updating Elo ratings after each.
The final state can be saved to DB as a snapshot, or used directly for predictions.

CRITICAL: For valid backtest, this script processes matches strictly in time order.
Never use future information to predict past matches.

Usage:
    python -m tennis_predictor.models.build_elo --tour ATP --end-date 2024-12-31
    python -m tennis_predictor.models.build_elo --tour WTA --end-date 2024-12-31 --save
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sqlalchemy import text

from tennis_predictor.data.storage import get_session
from tennis_predictor.logging_config import get_logger, setup_logging
from tennis_predictor.models.elo import EloConfig, update_ratings
from tennis_predictor.models.elo_manager import EloStateManager

console = Console()
logger = get_logger(__name__)


def build_elo_from_matches(
    tour: str,
    end_date: date,
    config: EloConfig | None = None,
    start_date: date | None = None,
) -> EloStateManager:
    """Process all matches up to end_date and return final Elo state.

    Args:
        tour: 'ATP' or 'WTA'
        end_date: Process matches with match_date <= end_date (inclusive)
        config: EloConfig to use (defaults if None)
        start_date: Optional lower bound (default: process from earliest match)

    Returns:
        EloStateManager with final Elo state for all players
    """
    config = config or EloConfig()
    manager = EloStateManager(config=config)

    # Count matches first for progress bar
    where_clauses = ["m.tour = :tour", "m.match_date <= :end_date"]
    params = {"tour": tour, "end_date": end_date}
    if start_date:
        where_clauses.append("m.match_date >= :start_date")
        params["start_date"] = start_date
    where_sql = " AND ".join(where_clauses)

    with get_session() as session:
        count_result = session.execute(
            text(f"SELECT COUNT(*) FROM matches m WHERE {where_sql}"),
            params,
        )
        total_matches = count_result.scalar() or 0

    if total_matches == 0:
        console.print(f"[red]No matches found for {tour} up to {end_date}[/red]")
        return manager

    console.print(
        f"[cyan]Processing {total_matches:,} {tour} matches "
        f"up to {end_date}...[/cyan]"
    )

    # Stream matches in chronological order
    # Join with tournaments to get level for K-factor adjustment
    query_sql = text(f"""
        SELECT m.match_date, m.surface, m.winner_id, m.loser_id,
               m.score, m.retirement, m.walkover,
               t.level AS tournament_level
        FROM matches m
        JOIN tournaments t ON m.tournament_id = t.tournament_id
        WHERE {where_sql}
          AND m.surface IS NOT NULL
        ORDER BY m.match_date ASC, m.match_id ASC
    """)

    matches_processed = 0
    matches_skipped = 0

    with Progress(
        TextColumn("[bold blue]Building Elo:"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("elo", total=total_matches)

        with get_session() as session:
            result = session.execute(query_sql, params)
            for row in result:
                # Skip if surface not one we model
                if row.surface not in ("Hard", "Clay", "Grass", "Carpet"):
                    matches_skipped += 1
                    progress.update(task, advance=1)
                    continue

                winner_state = manager.get_state(row.winner_id)
                loser_state = manager.get_state(row.loser_id)

                update_ratings(
                    winner_state=winner_state,
                    loser_state=loser_state,
                    surface=row.surface,
                    match_date=row.match_date,
                    score=row.score,
                    tournament_level=row.tournament_level,
                    is_retirement=row.retirement or False,
                    is_walkover=row.walkover or False,
                    config=config,
                )

                matches_processed += 1
                progress.update(task, advance=1)

    console.print(
        f"[green]✓[/green] Processed {matches_processed:,} matches, "
        f"skipped {matches_skipped:,}"
    )
    console.print(f"[green]✓[/green] Tracking {manager.num_players():,} players")

    return manager


def show_top_players(manager: EloStateManager, tour: str) -> None:
    """Display top 10 players per surface as a sanity check."""
    console.print()

    for surface in ("Overall", "Hard", "Clay", "Grass"):
        console.print(f"[bold cyan]Top 10 {tour} on {surface}:[/bold cyan]")

        top = manager.top_players(surface=surface, n=10, min_matches=20)

        if not top:
            console.print(f"  [yellow]No players with sufficient matches[/yellow]")
            console.print()
            continue

        # Resolve player names from DB
        player_ids = [pid for pid, _, _ in top]
        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT player_id, name_full, country_code
                    FROM players
                    WHERE player_id = ANY(:ids)
                """),
                {"ids": player_ids},
            )
            name_map = {
                row.player_id: (row.name_full, row.country_code)
                for row in result
            }

        table = Table(show_header=True, header_style="bold")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Player", style="cyan")
        table.add_column("Country", width=8)
        table.add_column("Elo", justify="right", style="green")
        table.add_column("Matches", justify="right", style="dim")

        for i, (player_id, rating, matches) in enumerate(top, 1):
            name, country = name_map.get(player_id, (player_id, "?"))
            table.add_row(
                str(i),
                name,
                country or "?",
                f"{rating:.0f}",
                str(matches),
            )

        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Elo ratings from historical matches"
    )
    parser.add_argument(
        "--tour",
        choices=["ATP", "WTA"],
        default="ATP",
        help="Which tour to process",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date(2024, 12, 31),
        help="Process matches up to this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Optional: only process matches from this date onward",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save final state to elo_ratings table",
    )
    parser.add_argument(
        "--algorithm-version",
        default="elo_v1_surface",
        help="Algorithm identifier when saving",
    )
    parser.add_argument(
        "--show-top",
        action="store_true",
        default=True,
        help="Display top 10 players per surface at end",
    )

    args = parser.parse_args()

    setup_logging()

    console.print(
        f"[bold]Building Elo for {args.tour}, "
        f"{'from ' + str(args.start_date) + ' ' if args.start_date else ''}"
        f"up to {args.end_date}[/bold]\n"
    )

    manager = build_elo_from_matches(
        tour=args.tour,
        end_date=args.end_date,
        start_date=args.start_date,
    )

    if args.show_top and manager.num_players() > 0:
        show_top_players(manager, args.tour)

    if args.save:
        console.print(f"[cyan]Saving Elo snapshot to database...[/cyan]")
        rows_saved = manager.save_to_db(
            rating_date=args.end_date,
            algorithm_version=args.algorithm_version,
        )
        console.print(f"[green]✓[/green] Saved {rows_saved} rating rows")


if __name__ == "__main__":
    main()
