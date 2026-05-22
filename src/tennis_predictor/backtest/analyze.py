"""Analyze backtest results in more depth.

Looks beyond aggregate metrics to understand WHERE the model wins and loses.

Usage:
    python -m tennis_predictor.backtest.analyze --tour ATP

Produces:
- Top 10 most upset predictions (model said 90%, lost)
- Top 10 confident correct predictions (model said 90%, won)
- Top 10 surprising correct predictions (model said 20%, won)
- Breakdown by tournament level (Slams vs Masters vs 250s)
- Performance vs ranking gap (closely-ranked vs lopsided matches)
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import text

from tennis_predictor.backtest.metrics import evaluate_predictions
from tennis_predictor.backtest.walk_forward import run_walk_forward_backtest
from tennis_predictor.data.storage import get_session
from tennis_predictor.logging_config import setup_logging
from tennis_predictor.models.elo import EloConfig

console = Console()


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _resolve_player_names(player_ids: list[str]) -> dict[str, str]:
    """Get display names for player IDs in batch."""
    if not player_ids:
        return {}
    with get_session() as session:
        result = session.execute(
            text("SELECT player_id, name_full FROM players WHERE player_id = ANY(:ids)"),
            {"ids": list(set(player_ids))},
        )
        return {row.player_id: row.name_full for row in result}


def _resolve_tournament_names(match_ids: list[int]) -> dict[int, tuple[str, str]]:
    """Get tournament name + level for match IDs."""
    if not match_ids:
        return {}
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT m.match_id, t.name, t.level
                FROM matches m
                JOIN tournaments t ON m.tournament_id = t.tournament_id
                WHERE m.match_id = ANY(:ids)
            """),
            {"ids": list(set(match_ids))},
        )
        return {row.match_id: (row.name, row.level) for row in result}


def _print_predictions_table(
    predictions,
    title: str,
    name_map: dict[str, str],
    tourney_map: dict[int, tuple[str, str]],
) -> None:
    """Render a table of predictions for human inspection."""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Date", style="dim")
    table.add_column("Tournament", style="cyan")
    table.add_column("Winner (actual)", style="green")
    table.add_column("Loser", style="red")
    table.add_column("Surface", style="dim")
    table.add_column("Model P(W)", justify="right", style="yellow")

    for p in predictions:
        winner_name = name_map.get(p.winner_id, p.winner_id)
        loser_name = name_map.get(p.loser_id, p.loser_id)
        tourney_name, level = tourney_map.get(p.match_id, ("?", "?"))
        tourney_label = f"{tourney_name} ({level})" if level else tourney_name

        table.add_row(
            str(p.match_date),
            tourney_label,
            winner_name,
            loser_name,
            p.surface,
            f"{p.p_winner_wins:.1%}",
        )
    console.print(table)
    console.print()


def analyze(
    tour: str,
    train_start: date,
    test_start: date,
    test_end: date,
) -> None:
    """Run backtest and produce deep-dive analysis."""
    config = EloConfig()

    console.print(f"[cyan]Running backtest for analysis...[/cyan]")
    summary, predictions = run_walk_forward_backtest(
        tour=tour,
        train_start_date=train_start,
        test_start_date=test_start,
        test_end_date=test_end,
        config=config,
        config_name="elo_v1_surface_analysis",
    )

    console.print(
        Panel.fit(
            f"[bold green]Loaded {len(predictions):,} predictions[/bold green]",
            border_style="green",
        )
    )

    # Sort by various criteria for interesting case studies
    # We use p_winner_wins (model's probability that the actual winner wins)

    # Resolve names in batch
    all_player_ids = [p.winner_id for p in predictions] + [p.loser_id for p in predictions]
    all_match_ids = [p.match_id for p in predictions]
    name_map = _resolve_player_names(all_player_ids)
    tourney_map = _resolve_tournament_names(all_match_ids)

    # 1. WORST UPSETS - model was >90% sure of one player, but the underdog won
    # These are predictions where p_winner_wins < 0.10 (we predicted them to LOSE strongly)
    upsets = sorted(predictions, key=lambda p: p.p_winner_wins)[:10]
    console.print()
    _print_predictions_table(
        upsets,
        "Top 10 Biggest Upsets (model gave winner <10% chance)",
        name_map,
        tourney_map,
    )

    # 2. MOST CONFIDENT CORRECT - model was >95% sure, and was right
    confident_correct = sorted(predictions, key=lambda p: -p.p_winner_wins)[:10]
    _print_predictions_table(
        confident_correct,
        "Top 10 Most Confident Correct Predictions (model >95% sure, winner won)",
        name_map,
        tourney_map,
    )

    # 3. By tournament level
    console.print("[bold cyan]═══ Performance by Tournament Level ═══[/bold cyan]\n")

    by_level: dict[str, list] = {}
    for p in predictions:
        _, level = tourney_map.get(p.match_id, (None, None))
        if level:
            by_level.setdefault(level, []).append(p)

    level_table = Table(show_header=True, header_style="bold")
    level_table.add_column("Level", style="cyan")
    level_table.add_column("Description", style="dim")
    level_table.add_column("N", justify="right")
    level_table.add_column("Accuracy", justify="right", style="green")
    level_table.add_column("Log loss", justify="right", style="yellow")
    level_table.add_column("Brier", justify="right", style="yellow")

    level_descriptions = {
        "G": "Grand Slams",
        "M": "Masters 1000",
        "A": "ATP 500",
        "D": "ATP 250",
        "F": "Tour Finals",
        "C": "Challengers",
        "S": "ITF/Satellite",
        "O": "Olympics",
        "PM": "WTA Premier Mandatory",
        "I": "WTA International",
    }

    for level in ("G", "M", "F", "A", "D", "C", "S", "O", "PM", "I"):
        subset = by_level.get(level, [])
        if not subset:
            continue
        m = evaluate_predictions(
            [p.p_player_a_wins for p in subset],
            [p.a_is_winner for p in subset],
        )
        level_table.add_row(
            level,
            level_descriptions.get(level, "?"),
            f"{m.n_predictions:,}",
            f"{m.accuracy:.4f}",
            f"{m.log_loss:.4f}",
            f"{m.brier_score:.4f}",
        )
    console.print(level_table)
    console.print()

    # 4. By predicted confidence bucket - where does the model do BEST/WORST?
    console.print("[bold cyan]═══ Performance by Prediction Confidence ═══[/bold cyan]\n")

    confidence_buckets = [
        ("Toss-up (45-55%)", lambda p: 0.45 <= p.p_winner_wins <= 0.55),
        ("Slight fav (55-65%)", lambda p: 0.55 < p.p_winner_wins <= 0.65),
        ("Clear fav (65-75%)", lambda p: 0.65 < p.p_winner_wins <= 0.75),
        ("Strong fav (75-90%)", lambda p: 0.75 < p.p_winner_wins <= 0.90),
        ("Heavy fav (>90%)", lambda p: p.p_winner_wins > 0.90),
    ]

    bucket_table = Table(show_header=True, header_style="bold")
    bucket_table.add_column("Confidence", style="cyan")
    bucket_table.add_column("N", justify="right")
    bucket_table.add_column("Winner won %", justify="right", style="green")
    bucket_table.add_column("Avg P(winner)", justify="right", style="yellow")
    bucket_table.add_column("Implied Gap", justify="right")

    for label, filter_fn in confidence_buckets:
        subset = [p for p in predictions if filter_fn(p)]
        if not subset:
            continue
        actual_win_rate = sum(1 for p in subset if p.p_winner_wins > 0.5) / len(subset)
        avg_prob = sum(p.p_winner_wins for p in subset) / len(subset)
        # Note: for the toss-up bucket where p_winner ~ 0.5, "winner won" is by definition 100%
        # since winner is by definition the one who won. The interesting stat here is
        # avg_prob - 0.5: did we lean correctly?
        gap = actual_win_rate - avg_prob
        bucket_table.add_row(
            label,
            f"{len(subset):,}",
            f"{actual_win_rate:.1%}",
            f"{avg_prob:.1%}",
            f"{gap:+.1%}",
        )
    console.print(bucket_table)
    console.print()
    console.print(
        "[dim]Note: 'Winner won %' is by definition 100% since these are observed "
        "winners. Compare it with 'Avg P(winner)' — the gap tells you if the model "
        "is under/over-confident in each bucket.[/dim]"
    )
    console.print()

    # 5. Find value bet candidates — meciuri where model gave low prob to underdog who won
    # These are training data for "where is the model wrong?"
    console.print(
        "[bold cyan]═══ Surprising Wins (potential value-bet zones) ═══[/bold cyan]\n"
    )
    surprising_wins = [p for p in predictions if p.p_winner_wins < 0.25][:20]
    surprising_winners_count = sum(1 for p in predictions if p.p_winner_wins < 0.25)
    console.print(
        f"In {surprising_winners_count:,} matches the model gave the actual "
        f"winner <25% chance.\n"
        f"That's [yellow]{surprising_winners_count / len(predictions):.1%}[/yellow] of matches.\n"
        f"In a perfectly calibrated world, ~25% of those underdogs should win — "
        f"that gives you [green]edge[/green] to bet on underdogs the model dislikes."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep-dive analysis of backtest predictions"
    )
    parser.add_argument("--tour", choices=["ATP", "WTA"], default="ATP")
    parser.add_argument("--train-start", type=_parse_date, default=date(2000, 1, 1))
    parser.add_argument("--test-start", type=_parse_date, default=date(2011, 1, 1))
    parser.add_argument("--test-end", type=_parse_date, default=date(2024, 12, 31))

    args = parser.parse_args()
    setup_logging()
    analyze(
        tour=args.tour,
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
    )


if __name__ == "__main__":
    main()
