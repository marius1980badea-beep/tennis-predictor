"""CLI: load historical odds from tennis-data.co.uk into ``historical_odds_raw``.

Examples:

    # Single year, dry run (parse only, do not touch DB)
    python -m tennis_predictor.data.loaders.load_odds --year 2024 --tour ATP --dry-run

    # Full Pinnacle-coverage backtest range
    python -m tennis_predictor.data.loaders.load_odds --years 2003-2024 --tour ATP

    # WTA, comma list
    python -m tennis_predictor.data.loaders.load_odds --years 2007,2010,2015 --tour WTA

    # Force re-download (ignore cache)
    python -m tennis_predictor.data.loaders.load_odds --year 2024 --tour ATP --force-download

Output is idempotent: the staging table has a uniqueness constraint on the
natural identifier, so re-running for the same year inserts only new rows.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import click
from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from tennis_predictor.data.loaders.tennis_data_uk import (
    EARLIEST_YEAR, OddsRow, YearLoadReport, load_year,
)
from tennis_predictor.data.storage.db import get_engine

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_year_spec(spec: str) -> list[int]:
    """Parse ``--years`` spec: ``"2003-2024"`` or ``"2020,2021,2024"`` or mix."""
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d{4})-(\d{4})", part)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if end < start:
                raise click.BadParameter(f"range {part!r} ends before it starts")
            years.extend(range(start, end + 1))
        elif part.isdigit() and len(part) == 4:
            years.append(int(part))
        else:
            raise click.BadParameter(f"unrecognised year token: {part!r}")
    return sorted(set(years))


# ---------------------------------------------------------------------------
# DB insertion
# ---------------------------------------------------------------------------

INSERT_SQL = text("""
    INSERT INTO historical_odds_raw (
        source, source_year, tour, match_date, tournament_name,
        series_or_tier, court, surface, round, best_of,
        winner_name, loser_name, winner_rank, loser_rank, comment,
        bookmaker_code, winner_odds, loser_odds,
        winner_implied_prob, loser_implied_prob, vig
    ) VALUES (
        :source, :source_year, :tour, :match_date, :tournament_name,
        :series_or_tier, :court, :surface, :round, :best_of,
        :winner_name, :loser_name, :winner_rank, :loser_rank, :comment,
        :bookmaker_code, :winner_odds, :loser_odds,
        :winner_implied_prob, :loser_implied_prob, :vig
    )
    ON CONFLICT (source, source_year, tour, match_date,
                 tournament_name, winner_name, loser_name, bookmaker_code)
    DO NOTHING
""")


def _row_to_params(r: OddsRow) -> dict:
    return {
        "source": r.source,
        "source_year": r.source_year,
        "tour": r.tour,
        "match_date": r.match_date,
        "tournament_name": r.tournament_name,
        "series_or_tier": r.series_or_tier,
        "court": r.court,
        "surface": r.surface,
        "round": r.round,
        "best_of": r.best_of,
        "winner_name": r.winner_name,
        "loser_name": r.loser_name,
        "winner_rank": r.winner_rank,
        "loser_rank": r.loser_rank,
        "comment": r.comment,
        "bookmaker_code": r.bookmaker_code,
        "winner_odds": r.winner_odds,
        "loser_odds": r.loser_odds,
        "winner_implied_prob": r.winner_implied_prob,
        "loser_implied_prob": r.loser_implied_prob,
        "vig": r.vig,
    }


def insert_odds_rows(rows: Iterable[OddsRow], *, batch_size: int = 1000) -> int:
    """Insert OddsRow records into ``historical_odds_raw``.

    Uses ``ON CONFLICT DO NOTHING`` for idempotent re-runs. Returns the
    number of new rows actually inserted (excludes duplicates).
    """
    payload = [_row_to_params(r) for r in rows]
    if not payload:
        return 0

    engine = get_engine()
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(payload), batch_size):
            batch = payload[i:i + batch_size]
            result = conn.execute(INSERT_SQL, batch)
            # rowcount reports actually-affected rows (excludes DO NOTHING skips)
            inserted += result.rowcount or 0
    return inserted


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def render_reports(reports: list[YearLoadReport]) -> None:
    """Render a Rich summary table for the CLI output."""
    table = Table(title="tennis-data.co.uk load summary")
    table.add_column("Tour")
    table.add_column("Year",      justify="right")
    table.add_column("Raw rows",  justify="right")
    table.add_column("Skipped",   justify="right")
    table.add_column("Odds rows", justify="right")
    table.add_column("Pinnacle",  justify="right", style="bold green")
    table.add_column("Bet365",    justify="right")
    table.add_column("Avg",       justify="right")
    table.add_column("Vig warns", justify="right", style="yellow")

    for r in reports:
        table.add_row(
            r.tour, str(r.year),
            f"{r.raw_matches:,}",
            f"{r.skipped_matches:,}",
            f"{r.odds_rows:,}",
            f"{r.pinnacle_rows:,}",
            f"{r.bet365_rows:,}",
            f"{r.avg_rows:,}",
            f"{r.vig_warnings:,}",
        )

    # Totals row
    if len(reports) > 1:
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]", "",
            f"{sum(r.raw_matches for r in reports):,}",
            f"{sum(r.skipped_matches for r in reports):,}",
            f"{sum(r.odds_rows for r in reports):,}",
            f"{sum(r.pinnacle_rows for r in reports):,}",
            f"{sum(r.bet365_rows for r in reports):,}",
            f"{sum(r.avg_rows for r in reports):,}",
            f"{sum(r.vig_warnings for r in reports):,}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Click entry point
# ---------------------------------------------------------------------------

@click.command(name="load-odds")
@click.option("--year", type=int,
              help="Single year, e.g. --year 2024")
@click.option("--years", "year_spec",
              help="Range or comma list, e.g. --years 2003-2024 or --years 2020,2022")
@click.option("--tour", type=click.Choice(["ATP", "WTA"]), required=True,
              help="Which tour to load")
@click.option("--cache-dir", type=click.Path(path_type=Path),
              default=Path("data/odds/raw"),
              help="Where to cache downloaded Excel files (default: data/odds/raw)")
@click.option("--force-download", is_flag=True,
              help="Re-download files even if already cached")
@click.option("--dry-run", is_flag=True,
              help="Parse + validate only; do not touch the database")
def load_odds_cli(
    year: int | None,
    year_spec: str | None,
    tour: str,
    cache_dir: Path,
    force_download: bool,
    dry_run: bool,
) -> None:
    """Download annual files from tennis-data.co.uk and stage them into the DB."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Argument validation -----------------------------------------------------
    if (year is None) == (year_spec is None):
        raise click.UsageError("Specify exactly one of --year or --years")
    years = [year] if year is not None else parse_year_spec(year_spec or "")

    earliest = EARLIEST_YEAR[tour]
    too_early = [y for y in years if y < earliest]
    if too_early:
        raise click.UsageError(
            f"{tour} odds available from {earliest}; cannot load {too_early}"
        )

    # Load each year ----------------------------------------------------------
    console.print(
        f"[bold]Loading {tour} odds[/bold] for "
        f"{len(years)} year(s): {years[0]}{('-' + str(years[-1])) if len(years) > 1 else ''}"
    )
    all_rows: list[OddsRow] = []
    reports: list[YearLoadReport] = []
    for y in years:
        try:
            rows, report = load_year(
                year=y, tour=tour, cache_dir=cache_dir,
                force_download=force_download,
            )
        except Exception as exc:  # pragma: no cover -- IO-bound, hard to test
            console.print(f"[red]Failed to load {tour} {y}: {exc}[/red]")
            continue
        all_rows.extend(rows)
        reports.append(report)
        console.print(
            f"  [cyan]{y}[/cyan]: {report.odds_rows:,} odds rows from "
            f"{report.raw_matches - report.skipped_matches:,} matches "
            f"(skipped {report.skipped_matches:,})"
        )

    render_reports(reports)

    # Insert or dry-run -------------------------------------------------------
    if dry_run:
        console.print(f"[yellow]--dry-run: parsed {len(all_rows):,} rows; "
                      "nothing inserted into DB[/yellow]")
        return

    console.print(f"\n[bold]Inserting {len(all_rows):,} rows into "
                  "historical_odds_raw...[/bold]")
    inserted = insert_odds_rows(all_rows)
    skipped = len(all_rows) - inserted
    console.print(
        f"[green]Inserted {inserted:,} new rows[/green]"
        + (f" ([dim]skipped {skipped:,} duplicates[/dim])" if skipped else "")
    )


if __name__ == "__main__":
    load_odds_cli()
