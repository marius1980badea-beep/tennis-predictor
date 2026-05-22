"""Command-line interface for tennis-predictor.

Usage:
    tennis-predictor health-check
    tennis-predictor load-data --tour ATP --start-year 2000 --end-year 2024
    tennis-predictor load-data --tour WTA --start-year 2000 --end-year 2024
    tennis-predictor stats
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from tennis_predictor.config import get_settings
from tennis_predictor.data.loaders.sackmann import SackmannLoader
from tennis_predictor.data.storage import get_session, get_supabase_client
from tennis_predictor.logging_config import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


@click.group()
@click.version_option(package_name="tennis-predictor")
def cli() -> None:
    """Tennis Predictor - Match prediction system for sports betting analysis."""
    setup_logging()


@cli.command()
def health_check() -> None:
    """Verify all external services are reachable."""
    console.print("[bold cyan]Running health checks...[/bold cyan]\n")

    all_ok = True

    # Check 1: Configuration loaded
    try:
        settings = get_settings()
        console.print("[green]✓[/green] Configuration loaded")
        console.print(f"  Environment: {settings.app.environment}")
        console.print(f"  Supabase URL: {settings.supabase.url}")
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
        all_ok = False
        sys.exit(1)

    # Check 2: Supabase REST API
    try:
        supabase = get_supabase_client()
        # Simple query to verify connection
        result = supabase.table("bookmakers").select("bookmaker_code").limit(1).execute()
        console.print(
            f"[green]✓[/green] Supabase REST API reachable "
            f"(bookmakers table has {len(result.data)} sample row)"
        )
    except Exception as e:
        console.print(f"[red]✗[/red] Supabase REST API error: {e}")
        all_ok = False

    # Check 3: Direct DB connection
    try:
        with get_session() as session:
            result = session.execute(text("SELECT version()")).scalar()
            console.print(f"[green]✓[/green] Direct DB connection OK")
            console.print(f"  PostgreSQL: {result[:50]}...")
    except Exception as e:
        console.print(f"[red]✗[/red] Direct DB connection error: {e}")
        all_ok = False

    # Check 4: Required tables exist
    try:
        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """)
            )
            tables = [row[0] for row in result]
            expected = {
                "players", "tournaments", "matches", "match_stats",
                "bookmakers", "historical_odds", "elo_ratings",
                "model_versions", "predictions", "backtest_runs",
                "player_features", "data_ingestion_log",
            }
            missing = expected - set(tables)
            if missing:
                console.print(f"[red]✗[/red] Missing tables: {missing}")
                all_ok = False
            else:
                console.print(f"[green]✓[/green] All 12 required tables present")
    except Exception as e:
        console.print(f"[red]✗[/red] Table check error: {e}")
        all_ok = False

    if all_ok:
        console.print("\n[bold green]All checks passed! System is ready.[/bold green]")
    else:
        console.print("\n[bold red]Some checks failed. Review errors above.[/bold red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--tour",
    type=click.Choice(["ATP", "WTA", "BOTH"], case_sensitive=False),
    default="BOTH",
    help="Which tour to load",
)
@click.option("--start-year", type=int, default=None, help="Start year (default from env)")
@click.option("--end-year", type=int, default=None, help="End year (default from env)")
@click.option("--skip-sync", is_flag=True, help="Skip git pull/clone (use existing local data)")
@click.option("--players-only", is_flag=True, help="Only load players, skip matches")
def load_data(
    tour: str,
    start_year: int | None,
    end_year: int | None,
    skip_sync: bool,
    players_only: bool,
) -> None:
    """Load tennis data from Sackmann repositories into Supabase."""
    settings = get_settings()
    start_year = start_year or settings.data_load.load_start_year
    end_year = end_year or settings.data_load.load_end_year

    tours_to_load = ["ATP", "WTA"] if tour.upper() == "BOTH" else [tour.upper()]

    console.print(
        f"[bold cyan]Loading data for {tours_to_load} "
        f"({start_year}-{end_year})[/bold cyan]\n"
    )

    for current_tour in tours_to_load:
        console.print(f"\n[bold]== {current_tour} ==[/bold]")
        loader = SackmannLoader(tour=current_tour)  # type: ignore[arg-type]

        # Step 1: sync repo
        if not skip_sync:
            console.print(f"Syncing Sackmann {current_tour} repository...")
            try:
                loader.sync_repo()
                console.print(f"[green]✓[/green] Repository synced")
            except Exception as e:
                console.print(f"[red]✗[/red] Sync failed: {e}")
                continue

        # Step 2: load players
        console.print("Loading players...")
        try:
            result = loader.load_players()
            console.print(
                f"[green]✓[/green] Players: {result.rows_inserted} inserted, "
                f"{result.rows_updated} updated ({result.duration_seconds:.1f}s)"
            )
        except Exception as e:
            console.print(f"[red]✗[/red] Player load failed: {e}")
            continue

        if players_only:
            continue

        # Step 3: load matches per year
        console.print(f"Loading matches for years {start_year}-{end_year}...")
        results = loader.load_year_range(start_year, end_year)

        # Summary table
        table = Table(title=f"{current_tour} Match Load Summary")
        table.add_column("Year", style="cyan", no_wrap=True)
        table.add_column("Processed", justify="right")
        table.add_column("Inserted", justify="right", style="green")
        table.add_column("Skipped", justify="right", style="yellow")
        table.add_column("Errors", justify="right", style="red")
        table.add_column("Time (s)", justify="right")

        for i, result in enumerate(results):
            year = start_year + i
            table.add_row(
                str(year),
                str(result.rows_processed),
                str(result.rows_inserted),
                str(result.rows_skipped),
                str(result.rows_errored),
                f"{result.duration_seconds:.1f}",
            )

        console.print(table)

        total_inserted = sum(r.rows_inserted for r in results)
        total_errored = sum(r.rows_errored for r in results)
        console.print(
            f"\n[bold]Total: {total_inserted} matches loaded, "
            f"{total_errored} errors[/bold]"
        )


@cli.command()
def stats() -> None:
    """Show database statistics."""
    console.print("[bold cyan]Database Statistics[/bold cyan]\n")

    queries = {
        "Players (ATP)": "SELECT COUNT(*) FROM players WHERE tour='ATP'",
        "Players (WTA)": "SELECT COUNT(*) FROM players WHERE tour='WTA'",
        "Tournaments (ATP)": "SELECT COUNT(*) FROM tournaments WHERE tour='ATP'",
        "Tournaments (WTA)": "SELECT COUNT(*) FROM tournaments WHERE tour='WTA'",
        "Matches (ATP)": "SELECT COUNT(*) FROM matches WHERE tour='ATP'",
        "Matches (WTA)": "SELECT COUNT(*) FROM matches WHERE tour='WTA'",
        "Match stats rows": "SELECT COUNT(*) FROM match_stats",
        "Historical odds": "SELECT COUNT(*) FROM historical_odds",
        "Elo ratings (snapshots)": "SELECT COUNT(*) FROM elo_ratings",
        "Predictions": "SELECT COUNT(*) FROM predictions",
        "Backtest runs": "SELECT COUNT(*) FROM backtest_runs",
    }

    table = Table()
    table.add_column("Entity", style="cyan")
    table.add_column("Count", justify="right", style="green")

    with get_session() as session:
        for label, query in queries.items():
            try:
                count = session.execute(text(query)).scalar() or 0
                table.add_row(label, f"{count:,}")
            except Exception as e:
                table.add_row(label, f"[red]ERR: {e}[/red]")

    console.print(table)

    # Date range
    console.print("\n[bold]Match date ranges:[/bold]")
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT tour,
                       MIN(match_date) as earliest,
                       MAX(match_date) as latest,
                       COUNT(*) as total
                FROM matches
                GROUP BY tour
                ORDER BY tour
            """)
        )
        for row in result:
            console.print(
                f"  {row.tour}: {row.earliest} → {row.latest} ({row.total:,} matches)"
            )


@cli.command()
@click.option("--last", type=int, default=10, help="Show last N entries")
def ingestion_log(last: int) -> None:
    """Show recent data ingestion log entries."""
    console.print(f"[bold cyan]Last {last} ingestion log entries[/bold cyan]\n")

    table = Table()
    table.add_column("Time", style="cyan")
    table.add_column("Source", style="white")
    table.add_column("Status")
    table.add_column("Processed", justify="right")
    table.add_column("Inserted", justify="right", style="green")
    table.add_column("Errors", justify="right", style="red")

    with get_session() as session:
        result = session.execute(
            text("""
                SELECT started_at, source, status,
                       rows_processed, rows_inserted, rows_errored
                FROM data_ingestion_log
                ORDER BY started_at DESC
                LIMIT :limit
            """),
            {"limit": last},
        )
        for row in result:
            status_styled = (
                f"[green]{row.status}[/green]"
                if row.status == "completed"
                else f"[yellow]{row.status}[/yellow]"
            )
            table.add_row(
                row.started_at.strftime("%Y-%m-%d %H:%M"),
                row.source,
                status_styled,
                str(row.rows_processed or 0),
                str(row.rows_inserted or 0),
                str(row.rows_errored or 0),
            )

    console.print(table)


if __name__ == "__main__":
    cli()
