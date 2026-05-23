"""CLI: match ``historical_odds_raw`` to ``matches`` via player-ID pivot.

Usage:

    # Small-batch dry run (start here!)
    python -m tennis_predictor.data.matching.cli --tour ATP --limit 200 --dry-run

    # Full dry run before committing to UPDATE
    python -m tennis_predictor.data.matching.cli --tour ATP --dry-run

    # Real run
    python -m tennis_predictor.data.matching.cli --tour ATP

    # Wider date window (default ±14 days handles tournament-start dates)
    python -m tennis_predictor.data.matching.cli --tour ATP --date-window 21

    # Stricter player resolution
    python -m tennis_predictor.data.matching.cli --tour ATP --fuzzy-threshold 0.90

The matcher is idempotent: already-matched rows are skipped on re-runs.
"""

from __future__ import annotations

import logging

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from tennis_predictor.data.matching.odds_match import (
    MatchReport, PlayerResolver, _bucket_label, resolve_via_player_ids,
)
from tennis_predictor.data.storage.db import get_engine

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

SQL_UNMATCHED_IDENTITIES = text("""
    SELECT
        tour,
        source_year,
        match_date,
        tournament_name,
        winner_name,
        loser_name,
        COUNT(*) AS row_count
    FROM historical_odds_raw
    WHERE match_id IS NULL
      AND tour = :tour
    GROUP BY tour, source_year, match_date, tournament_name, winner_name, loser_name
    ORDER BY match_date
""")

# Players for the given tour. The compact-name index will be built from
# these. We pull all players because we don't know upfront which IDs will
# appear in any given row.
SQL_PLAYERS = text("""
    SELECT player_id, name_full
    FROM players
    WHERE tour = :tour
""")

# All matches for the tour as a single DataFrame. We expect ~50k for ATP
# (12 years × ~3000 matches/year). At ~150 bytes/row that's ~7 MB, easily
# fits in memory.
SQL_MATCHES = text("""
    SELECT
        m.match_id,
        m.winner_id,
        m.loser_id,
        m.match_date,
        t.name AS tournament_name
    FROM matches m
    JOIN tournaments t ON m.tournament_id = t.tournament_id
    WHERE m.tour = :tour
""")

SQL_UPDATE_MATCHED = text("""
    UPDATE historical_odds_raw
    SET match_id         = :match_id,
        match_confidence = :confidence,
        matched_at       = NOW()
    WHERE tour             = :tour
      AND match_date       = :match_date
      AND tournament_name  = :tournament_name
      AND winner_name      = :winner_name
      AND loser_name       = :loser_name
      AND match_id IS NULL
""")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_unmatched_identities(engine, tour: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(SQL_UNMATCHED_IDENTITIES, conn, params={"tour": tour})
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


def load_players(engine, tour: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(SQL_PLAYERS, conn, params={"tour": tour})
    return df


def load_matches(engine, tour: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(SQL_MATCHES, conn, params={"tour": tour})
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def perform_matching(
    engine,
    tour: str,
    fuzzy_threshold: float,
    date_window_days: int,
    min_confidence: float,
    limit: int | None,
    dry_run: bool,
) -> MatchReport:
    report = MatchReport(tour=tour)

    # 1) Load identities to match (deduplicated by match identity)
    console.print(f"[bold]Loading unmatched identities for {tour}...[/bold]")
    identities = load_unmatched_identities(engine, tour)
    if limit is not None:
        identities = identities.head(limit)
    report.total_unique_identities = len(identities)
    console.print(f"  Found [cyan]{len(identities):,}[/cyan] unique identities")

    if identities.empty:
        console.print("[yellow]Nothing to match.[/yellow]")
        return report

    # 2) Build player resolver
    console.print(f"[bold]Building player lookup for {tour}...[/bold]")
    players_df = load_players(engine, tour)
    resolver = PlayerResolver(players_df, fuzzy_threshold=fuzzy_threshold)
    console.print(f"  Indexed [cyan]{len(resolver):,}[/cyan] players")

    # 3) Load all matches for this tour (one big query, in-memory after)
    console.print(f"[bold]Loading matches for {tour}...[/bold]")
    matches_df = load_matches(engine, tour)
    console.print(f"  Loaded [cyan]{len(matches_df):,}[/cyan] matches")
    if matches_df.empty:
        console.print(f"[red]No matches in `matches` for tour={tour}.[/red]")
        return report

    # 4) Resolve each identity
    console.print(f"[bold]Matching...[/bold]")
    updates: list[dict] = []

    for _, row in identities.iterrows():
        result = resolve_via_player_ids(
            td_winner=row["winner_name"],
            td_loser=row["loser_name"],
            td_tournament=row["tournament_name"],
            td_date=row["match_date"],
            matches_df=matches_df,
            player_resolver=resolver,
            date_window_days=date_window_days,
            min_confidence=min_confidence,
        )

        if result is None:
            # Classify the failure mode by re-running the player resolution
            w_hit = resolver.resolve(row["winner_name"])
            l_hit = resolver.resolve(row["loser_name"])
            if w_hit is None or l_hit is None:
                report.unmatched_player_resolve += 1
            else:
                # Players resolved but no Sackmann match found
                composite = (w_hit.confidence + l_hit.confidence) / 2
                if composite < min_confidence:
                    report.unmatched_below_confidence += 1
                else:
                    report.unmatched_no_match_in_window += 1
            continue

        report.matched += 1
        if result.winner_resolution == "fuzzy":
            report.fuzzy_winner_resolutions += 1
        if result.loser_resolution == "fuzzy":
            report.fuzzy_loser_resolutions += 1
        bucket = _bucket_label(result.confidence)
        report.confidence_buckets[bucket] = report.confidence_buckets.get(bucket, 0) + 1

        updates.append({
            "tour":            row["tour"],
            "match_date":      row["match_date"],
            "tournament_name": row["tournament_name"],
            "winner_name":     row["winner_name"],
            "loser_name":      row["loser_name"],
            "match_id":        result.match_id,
            "confidence":      result.confidence,
        })

    # 5) Apply updates
    if dry_run:
        console.print(
            f"[yellow]--dry-run: identified {len(updates):,} matchable identities; "
            "no UPDATE.[/yellow]"
        )
    elif updates:
        console.print(f"[bold]Applying {len(updates):,} UPDATEs...[/bold]")
        with engine.begin() as conn:
            for i in range(0, len(updates), 500):
                batch = updates[i:i + 500]
                result = conn.execute(SQL_UPDATE_MATCHED, batch)
                report.rows_updated += result.rowcount or 0

    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render_report(report: MatchReport, dry_run: bool) -> None:
    table = Table(title=f"Fuzzy-match report ({report.tour})")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Unique identities (input)",
                  f"{report.total_unique_identities:,}")
    table.add_row("[green]Matched[/green]",
                  f"{report.matched:,}")

    table.add_section()
    table.add_row("[dim]Failure breakdown[/dim]", "")
    table.add_row("  Player(s) couldn't be resolved",
                  f"{report.unmatched_player_resolve:,}")
    table.add_row("  Players OK, no match in date window",
                  f"{report.unmatched_no_match_in_window:,}")
    table.add_row("  Composite confidence below threshold",
                  f"{report.unmatched_below_confidence:,}")

    if not dry_run:
        table.add_section()
        table.add_row(
            "[bold]historical_odds_raw rows updated[/bold]",
            f"{report.rows_updated:,}",
        )

    if report.matched > 0:
        table.add_section()
        table.add_row("[dim]Confidence distribution[/dim]", "")
        for label, n in report.confidence_buckets.items():
            if n > 0:
                table.add_row(f"  {label}", f"{n:,}")
        table.add_section()
        table.add_row("[dim]Fuzzy resolutions (vs exact)[/dim]", "")
        table.add_row("  Winner via fuzzy", f"{report.fuzzy_winner_resolutions:,}")
        table.add_row("  Loser via fuzzy", f"{report.fuzzy_loser_resolutions:,}")

    console.print(table)

    if report.total_unique_identities > 0:
        rate = 100.0 * report.matched / report.total_unique_identities
        colour = "green" if rate >= 95 else "yellow" if rate >= 80 else "red"
        console.print(f"\n[bold {colour}]Match rate: {rate:.2f}%[/bold {colour}]")


# ---------------------------------------------------------------------------
# Click entry point
# ---------------------------------------------------------------------------

@click.command(name="match-odds")
@click.option("--tour", type=click.Choice(["ATP", "WTA"]), required=True)
@click.option("--fuzzy-threshold", type=click.FloatRange(0.0, 1.0), default=0.85,
              show_default=True,
              help="Minimum fuzzy score for player-name resolution")
@click.option("--date-window", type=int, default=14, show_default=True,
              help="Search Sackmann matches within +/- N days of odds row date "
                   "(Sackmann uses tournament start date; tennis-data uses "
                   "actual match date)")
@click.option("--min-confidence", type=click.FloatRange(0.0, 1.0), default=0.70,
              show_default=True,
              help="Minimum composite confidence to accept a match")
@click.option("--limit", type=int, default=None,
              help="Process only first N identities")
@click.option("--dry-run", is_flag=True,
              help="Do not UPDATE; report only")
def match_odds_cli(
    tour: str,
    fuzzy_threshold: float,
    date_window: int,
    min_confidence: float,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Link historical_odds_raw rows to Sackmann matches via player-ID pivot."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    engine = get_engine()
    console.print(
        f"[bold]Fuzzy matching v2:[/bold] tour={tour}, "
        f"player_threshold={fuzzy_threshold:.2f}, "
        f"date_window=±{date_window}d, "
        f"min_confidence={min_confidence:.2f}, "
        f"limit={limit or '∞'}, dry_run={dry_run}"
    )

    report = perform_matching(
        engine=engine,
        tour=tour,
        fuzzy_threshold=fuzzy_threshold,
        date_window_days=date_window,
        min_confidence=min_confidence,
        limit=limit,
        dry_run=dry_run,
    )
    render_report(report, dry_run=dry_run)


if __name__ == "__main__":
    match_odds_cli()
