"""Live match prediction CLI.

Supports both Elo v1 (default) and v2 (with --version elo_v2_surface).

Usage:
    # Use v1 (baseline):
    python -m tennis_predictor.interface.predict "Sinner" "Alcaraz" Hard

    # Use v2 (with improvements):
    python -m tennis_predictor.interface.predict "Sinner" "Alcaraz" Hard --version elo_v2_surface
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import text

from tennis_predictor.data.storage import get_session
from tennis_predictor.logging_config import setup_logging

console = Console()

Surface = Literal["Hard", "Clay", "Grass", "Carpet"]


def find_player(name_query: str, tour: str, threshold: float = 0.3) -> list[dict]:
    """Find players matching the query using trigram similarity."""
    query = text("""
        SELECT player_id, name_full, country_code,
               extensions.similarity(name_full, :q) AS sim
        FROM players
        WHERE tour = :tour
          AND extensions.similarity(name_full, :q) > :threshold
        ORDER BY sim DESC
        LIMIT 10
    """)

    with get_session() as session:
        result = session.execute(
            query, {"q": name_query, "tour": tour, "threshold": threshold},
        )
        return [
            {
                "player_id": row.player_id,
                "name_full": row.name_full,
                "country_code": row.country_code,
                "similarity": float(row.sim),
            }
            for row in result
        ]


def resolve_player(name_query: str, tour: str) -> dict | None:
    matches = find_player(name_query, tour)
    if not matches:
        console.print(f"[red]No player found matching '{name_query}'[/red]")
        return None

    if len(matches) == 1 or matches[0]["similarity"] > 0.9:
        return matches[0]

    console.print(f"\n[yellow]Multiple matches for '{name_query}':[/yellow]")
    table = Table()
    table.add_column("#", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Country", style="green")
    table.add_column("Match Score", justify="right", style="dim")

    for i, m in enumerate(matches[:5], 1):
        table.add_row(
            str(i), m["name_full"], m["country_code"] or "?",
            f"{m['similarity']:.2f}",
        )
    console.print(table)

    choice = console.input("\nChoose [1-5] (or Enter for #1): ").strip()
    if not choice:
        return matches[0]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(matches):
            return matches[idx]
    except ValueError:
        pass

    return None


def predict(
    player_a_query: str,
    player_b_query: str,
    surface: Surface,
    tour: str,
    algorithm_version: str = "elo_v1_surface",
) -> None:
    """Predict a match using either v1 or v2 Elo (auto-detected from version name)."""
    is_v2 = "v2" in algorithm_version

    player_a = resolve_player(player_a_query, tour)
    if not player_a:
        sys.exit(1)
    player_b = resolve_player(player_b_query, tour)
    if not player_b:
        sys.exit(1)

    with get_session() as session:
        result = session.execute(
            text("""
                SELECT MAX(rating_date) AS latest
                FROM elo_ratings
                WHERE algorithm_version = :v
            """),
            {"v": algorithm_version},
        )
        latest_date = result.scalar()

    if latest_date is None:
        console.print(
            f"[red]No Elo snapshot found for '{algorithm_version}'. "
            f"Run build_elo (or compare_v1_v2 for v2) first.[/red]"
        )
        sys.exit(1)

    console.print(f"\n[dim]Using {algorithm_version} snapshot from {latest_date}[/dim]")

    # Load appropriate manager + config
    if is_v2:
        from tennis_predictor.models.elo_manager_v2 import EloStateManagerV2
        from tennis_predictor.models.elo_v2 import EloConfigV2, predict_match_probability
        config = EloConfigV2()
        manager = EloStateManagerV2.load_from_db(
            rating_date=latest_date,
            algorithm_version=algorithm_version,
            config=config,
        )
    else:
        from tennis_predictor.models.elo_manager import EloStateManager
        from tennis_predictor.models.elo import EloConfig, predict_match_probability
        config = EloConfig()
        manager = EloStateManager.load_from_db(
            rating_date=latest_date,
            algorithm_version=algorithm_version,
            config=config,
        )

    state_a = manager.get_state(player_a["player_id"])
    state_b = manager.get_state(player_b["player_id"])

    p_a_wins = predict_match_probability(state_a, state_b, surface, config=config)
    p_b_wins = 1 - p_a_wins

    name_a = player_a["name_full"]
    name_b = player_b["name_full"]

    rating_a_surface = state_a.get_rating(surface, config)
    rating_b_surface = state_b.get_rating(surface, config)
    rating_a_overall = state_a.get_rating("Overall", config)
    rating_b_overall = state_b.get_rating("Overall", config)
    matches_a_surface = state_a.get_matches_played(surface)
    matches_b_surface = state_b.get_matches_played(surface)

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{name_a}[/bold cyan]  vs  [bold magenta]{name_b}[/bold magenta]\n"
            f"[dim]Surface: {surface}  |  Model: {algorithm_version}[/dim]",
            border_style="white",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Player", style="cyan")
    table.add_column(f"Elo ({surface})", justify="right", style="green")
    table.add_column("Matches", justify="right", style="dim")
    table.add_column("Elo (Overall)", justify="right", style="yellow")
    table.add_column("Win Probability", justify="right", style="bold")

    table.add_row(
        name_a, f"{rating_a_surface:.0f}", str(matches_a_surface),
        f"{rating_a_overall:.0f}", f"{p_a_wins:.1%}",
    )
    table.add_row(
        name_b, f"{rating_b_surface:.0f}", str(matches_b_surface),
        f"{rating_b_overall:.0f}", f"{p_b_wins:.1%}",
    )
    console.print(table)

    fair_odds_a = 1 / p_a_wins
    fair_odds_b = 1 / p_b_wins
    console.print()
    console.print(
        f"[bold]Fair odds (no vig):[/bold]  "
        f"{name_a}: [green]{fair_odds_a:.2f}[/green]   "
        f"{name_b}: [green]{fair_odds_b:.2f}[/green]"
    )

    console.print(
        "\n[dim]To check for value bets, compare these with bookmaker odds. "
        "A bookmaker offering odds > the fair odds implies positive expected value.[/dim]"
    )

    warnings = []
    if matches_a_surface < 10:
        warnings.append(
            f"[yellow]⚠ {name_a} has only {matches_a_surface} matches on {surface}[/yellow]"
        )
    if matches_b_surface < 10:
        warnings.append(
            f"[yellow]⚠ {name_b} has only {matches_b_surface} matches on {surface}[/yellow]"
        )
    if warnings:
        console.print()
        for w in warnings:
            console.print(w)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a tennis match using Elo ratings"
    )
    parser.add_argument("player_a", help="First player name (fuzzy match)")
    parser.add_argument("player_b", help="Second player name (fuzzy match)")
    parser.add_argument(
        "surface", choices=["Hard", "Clay", "Grass", "Carpet"],
    )
    parser.add_argument("--tour", choices=["ATP", "WTA"], default="ATP")
    parser.add_argument(
        "--version",
        default="elo_v1_surface",
        help="Elo algorithm version: elo_v1_surface (baseline) or elo_v2_surface (improved)",
    )

    args = parser.parse_args()
    setup_logging()
    predict(
        player_a_query=args.player_a,
        player_b_query=args.player_b,
        surface=args.surface,
        tour=args.tour,
        algorithm_version=args.version,
    )


if __name__ == "__main__":
    main()
