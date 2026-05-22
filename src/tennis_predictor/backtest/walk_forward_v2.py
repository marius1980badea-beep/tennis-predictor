"""Walk-forward backtest engine for Elo v2.

Identical control flow to v1 walk_forward.py, but uses v2 Elo state +
update_ratings + predict_match_probability.

We keep them separate to make A/B comparison fully reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date

from tennis_predictor.backtest.metrics import EvaluationMetrics, evaluate_predictions
from tennis_predictor.backtest.walk_forward import (
    BacktestPrediction,
    BacktestRunSummary,
)
from tennis_predictor.models.elo_v2 import (
    EloConfigV2,
    predict_match_probability,
    update_ratings,
)
from tennis_predictor.models.elo_manager_v2 import EloStateManagerV2


def _get_console():
    from rich.console import Console
    return Console()


def _get_logger():
    try:
        from tennis_predictor.logging_config import get_logger
        return get_logger(__name__)
    except ImportError:
        import logging
        return logging.getLogger(__name__)


def _iter_matches_in_range(
    tour: str,
    start_date: date | None,
    end_date: date | None,
) -> Iterator:
    from sqlalchemy import text
    from tennis_predictor.data.storage import get_session

    where_clauses = ["m.tour = :tour", "m.surface IS NOT NULL"]
    params: dict = {"tour": tour}

    if start_date:
        where_clauses.append("m.match_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_clauses.append("m.match_date <= :end_date")
        params["end_date"] = end_date

    query = text(f"""
        SELECT m.match_id, m.match_date, m.tour, m.surface,
               m.winner_id, m.loser_id,
               m.score, m.retirement, m.walkover,
               t.level AS tournament_level
        FROM matches m
        JOIN tournaments t ON m.tournament_id = t.tournament_id
        WHERE {" AND ".join(where_clauses)}
        ORDER BY m.match_date ASC, m.match_id ASC
    """)

    with get_session() as session:
        result = session.execute(query, params)
        yield from result


def _count_matches(
    tour: str,
    start_date: date | None,
    end_date: date | None,
) -> int:
    from sqlalchemy import text
    from tennis_predictor.data.storage import get_session

    where_clauses = ["tour = :tour", "surface IS NOT NULL"]
    params: dict = {"tour": tour}
    if start_date:
        where_clauses.append("match_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_clauses.append("match_date <= :end_date")
        params["end_date"] = end_date

    with get_session() as session:
        result = session.execute(
            text(f"SELECT COUNT(*) FROM matches WHERE {' AND '.join(where_clauses)}"),
            params,
        )
        return result.scalar() or 0


def run_walk_forward_backtest_v2(
    tour: str,
    train_start_date: date,
    test_start_date: date,
    test_end_date: date,
    config: EloConfigV2 | None = None,
    config_name: str = "elo_v2_surface",
    random_seed: int = 42,
    show_progress: bool = True,
) -> tuple[BacktestRunSummary, list[BacktestPrediction]]:
    """Walk-forward backtest using v2 Elo. Same interface as v1."""
    config = config or EloConfigV2()
    rng = random.Random(random_seed)
    manager = EloStateManagerV2(config=config)

    from rich.progress import (
        BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    console = _get_console()

    # Phase 1: Warmup
    console.print(
        f"[bold cyan]Phase 1: Warmup[/bold cyan] - "
        f"Building Elo v2 from {train_start_date} to {test_start_date}..."
    )
    warmup_count = _count_matches(
        tour=tour, start_date=train_start_date, end_date=test_start_date,
    )
    console.print(f"  Warmup matches: {warmup_count:,}")

    warmup_processed = 0
    with Progress(
        TextColumn("[bold yellow]Warmup:"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"), TimeElapsedColumn(),
        console=console, disable=not show_progress,
    ) as progress:
        task = progress.add_task("warmup", total=warmup_count)
        for row in _iter_matches_in_range(
            tour=tour, start_date=train_start_date, end_date=test_start_date,
        ):
            if row.match_date >= test_start_date:
                continue
            winner = manager.get_state(row.winner_id)
            loser = manager.get_state(row.loser_id)
            update_ratings(
                winner_state=winner, loser_state=loser,
                surface=row.surface, match_date=row.match_date,
                score=row.score, tournament_level=row.tournament_level,
                is_retirement=row.retirement or False,
                is_walkover=row.walkover or False,
                config=config,
            )
            warmup_processed += 1
            progress.update(task, advance=1)

    console.print(
        f"[green]✓[/green] Warmup complete. "
        f"Tracking {manager.num_players():,} players.\n"
    )

    # Phase 2: Test
    console.print(
        f"[bold cyan]Phase 2: Test[/bold cyan] - "
        f"Predicting + evaluating from {test_start_date} to {test_end_date}..."
    )
    test_count = _count_matches(
        tour=tour, start_date=test_start_date, end_date=test_end_date,
    )
    console.print(f"  Test matches: {test_count:,}")

    predictions: list[BacktestPrediction] = []
    skipped_walkover = 0

    with Progress(
        TextColumn("[bold green]Backtest:"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"), TimeElapsedColumn(),
        TextColumn("•"), TimeRemainingColumn(),
        console=console, disable=not show_progress,
    ) as progress:
        task = progress.add_task("backtest", total=test_count)
        for row in _iter_matches_in_range(
            tour=tour, start_date=test_start_date, end_date=test_end_date,
        ):
            if row.walkover:
                skipped_walkover += 1
                progress.update(task, advance=1)
                continue

            winner = manager.get_state(row.winner_id)
            loser = manager.get_state(row.loser_id)

            # PREDICT first - using v2 (which applies time decay internally)
            p_winner = predict_match_probability(
                winner, loser, row.surface,
                config=config,
                prediction_date=row.match_date,
            )

            if rng.random() < 0.5:
                p_a = p_winner
                a_is_winner = 1
            else:
                p_a = 1.0 - p_winner
                a_is_winner = 0

            predictions.append(BacktestPrediction(
                match_date=row.match_date, match_id=row.match_id,
                tour=row.tour, surface=row.surface,
                winner_id=row.winner_id, loser_id=row.loser_id,
                p_winner_wins=p_winner, p_player_a_wins=p_a,
                a_is_winner=a_is_winner,
            ))

            update_ratings(
                winner_state=winner, loser_state=loser,
                surface=row.surface, match_date=row.match_date,
                score=row.score, tournament_level=row.tournament_level,
                is_retirement=row.retirement or False,
                is_walkover=False, config=config,
            )

            progress.update(task, advance=1)

    console.print(
        f"[green]✓[/green] Generated {len(predictions):,} predictions "
        f"(skipped {skipped_walkover} walkovers)\n"
    )

    # Aggregate metrics
    overall_metrics = evaluate_predictions(
        [p.p_player_a_wins for p in predictions],
        [p.a_is_winner for p in predictions],
    )

    per_surface_metrics = {}
    for surface in ("Hard", "Clay", "Grass", "Carpet"):
        subset = [p for p in predictions if p.surface == surface]
        if subset:
            per_surface_metrics[surface] = evaluate_predictions(
                [p.p_player_a_wins for p in subset],
                [p.a_is_winner for p in subset],
            )

    per_year_metrics = {}
    years = sorted({p.match_date.year for p in predictions})
    for year in years:
        subset = [p for p in predictions if p.match_date.year == year]
        if subset:
            per_year_metrics[year] = evaluate_predictions(
                [p.p_player_a_wins for p in subset],
                [p.a_is_winner for p in subset],
            )

    summary = BacktestRunSummary(
        config_name=config_name, tour=tour,
        train_start_date=train_start_date, train_end_date=test_start_date,
        test_start_date=test_start_date, test_end_date=test_end_date,
        total_train_matches=warmup_processed,
        total_predictions=len(predictions),
        overall_metrics=overall_metrics,
        per_surface_metrics=per_surface_metrics,
        per_year_metrics=per_year_metrics,
    )

    return summary, predictions, manager
