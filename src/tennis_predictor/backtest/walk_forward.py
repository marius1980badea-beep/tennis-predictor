"""Walk-forward backtest engine for Elo predictions.

Critical principle: we predict each match using ONLY information available
BEFORE that match was played. The Elo state evolves match by match.

Workflow:
1. Iterate through matches chronologically (sorted by date, then match_id)
2. For each match:
   a. PREDICT first using current Elo state (no future leakage)
   b. Store the prediction
   c. THEN update Elo state based on actual outcome
3. At end, aggregate predictions into metrics

A subtle but critical detail: we predict on the SURFACE-SPECIFIC effective
rating (which blends Overall for new players). This must match what
update_ratings would have used.

Predictions are stored in two forms:
- 'p_winner_wins': model's probability that the eventual winner wins
  (always corresponds to the actual outcome of 1.0)
- 'p_random_side': probability for a randomly chosen side, with the
  corresponding 0/1 outcome — this is what we evaluate on, since
  evaluating only on 'p_winner_wins' would inflate accuracy artificially.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date

from tennis_predictor.backtest.metrics import EvaluationMetrics, evaluate_predictions
from tennis_predictor.models.elo import EloConfig, predict_match_probability, update_ratings
from tennis_predictor.models.elo_manager import EloStateManager


def _get_console():
    """Lazy import of rich console."""
    from rich.console import Console
    return Console()


def _get_logger():
    """Lazy import of structlog."""
    try:
        from tennis_predictor.logging_config import get_logger
        return get_logger(__name__)
    except ImportError:
        import logging
        return logging.getLogger(__name__)


@dataclass
class BacktestPrediction:
    """A single prediction made during backtest, with outcome."""

    match_date: date
    match_id: int
    tour: str
    surface: str

    winner_id: str
    loser_id: str

    # Model's predicted probability that the eventual winner wins
    p_winner_wins: float

    # Randomized prediction for unbiased evaluation:
    # We flip a coin to choose "player A" (either winner or loser),
    # store P(A wins) and outcome (1 if A=winner, 0 if A=loser)
    p_player_a_wins: float
    a_is_winner: int  # 1 or 0


@dataclass
class BacktestRunSummary:
    """Aggregated results of a complete backtest run."""

    config_name: str
    tour: str

    train_start_date: date
    train_end_date: date
    test_start_date: date
    test_end_date: date

    total_train_matches: int
    total_predictions: int

    overall_metrics: EvaluationMetrics
    per_surface_metrics: dict[str, EvaluationMetrics] = field(default_factory=dict)
    per_year_metrics: dict[int, EvaluationMetrics] = field(default_factory=dict)

    def pretty_print(self) -> str:
        """Format full report."""
        lines = [
            f"Backtest Configuration: {self.config_name}",
            f"Tour: {self.tour}",
            f"Train period: {self.train_start_date} → {self.train_end_date} "
            f"({self.total_train_matches:,} matches)",
            f"Test period:  {self.test_start_date} → {self.test_end_date}",
            f"Predictions:  {self.total_predictions:,}",
            "",
            "═══ Overall Performance ═══",
            self.overall_metrics.pretty_print(),
        ]
        if self.per_surface_metrics:
            lines += ["", "═══ Per-Surface ═══"]
            for surface in ("Hard", "Clay", "Grass"):
                if surface in self.per_surface_metrics:
                    m = self.per_surface_metrics[surface]
                    lines.append(
                        f"{surface:<6} N={m.n_predictions:>6,}  "
                        f"acc={m.accuracy:.4f}  ll={m.log_loss:.4f}  "
                        f"brier={m.brier_score:.4f}  ece={m.calibration_error:.4f}"
                    )
        if self.per_year_metrics:
            lines += ["", "═══ Per Year ═══"]
            for year in sorted(self.per_year_metrics.keys()):
                m = self.per_year_metrics[year]
                lines.append(
                    f"{year}  N={m.n_predictions:>6,}  "
                    f"acc={m.accuracy:.4f}  ll={m.log_loss:.4f}"
                )
        return "\n".join(lines)


def _iter_matches_in_range(
    tour: str,
    start_date: date | None,
    end_date: date | None,
) -> Iterator:
    """Stream matches in chronological order from DB.

    Yields one row at a time to avoid loading all matches into memory.
    """
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
    """Count matches in range (for progress bar)."""
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


def run_walk_forward_backtest(
    tour: str,
    train_start_date: date,
    test_start_date: date,
    test_end_date: date,
    config: EloConfig | None = None,
    config_name: str = "elo_v1_surface",
    random_seed: int = 42,
    show_progress: bool = True,
) -> tuple[BacktestRunSummary, list[BacktestPrediction]]:
    """Run walk-forward backtest for Elo predictions.

    Process:
    1. Build initial Elo state from train_start_date to (test_start_date - 1).
       This is the "warmup" period - no predictions evaluated here.
    2. From test_start_date to test_end_date: predict each match BEFORE
       updating Elo, then update.

    Args:
        tour: 'ATP' or 'WTA'
        train_start_date: Where to start building Elo (warmup begins)
        test_start_date: First match where we record predictions
        test_end_date: Last match where we record predictions
        config: EloConfig (defaults if None)
        config_name: Identifier for this run (saved to backtest_runs)
        random_seed: Seed for randomizing side choice (reproducibility)
        show_progress: Display progress bar

    Returns:
        (summary, predictions) - summary aggregates metrics,
        predictions is the full list (useful for further analysis)
    """
    config = config or EloConfig()
    rng = random.Random(random_seed)
    manager = EloStateManager(config=config)

    # Lazy imports for runtime-only deps
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    console = _get_console()
    logger = _get_logger()

    # Phase 1: Warmup - build Elo without recording predictions
    console.print(
        f"[bold cyan]Phase 1: Warmup[/bold cyan] - "
        f"Building Elo from {train_start_date} to {test_start_date}..."
    )
    warmup_count = _count_matches(
        tour=tour,
        start_date=train_start_date,
        end_date=test_start_date,
    )
    console.print(f"  Warmup matches: {warmup_count:,}")

    warmup_processed = 0
    with Progress(
        TextColumn("[bold yellow]Warmup:"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        disable=not show_progress,
    ) as progress:
        task = progress.add_task("warmup", total=warmup_count)
        for row in _iter_matches_in_range(
            tour=tour,
            start_date=train_start_date,
            end_date=test_start_date,
        ):
            # Skip the boundary date if it's the same as test_start to avoid
            # using its matches twice
            if row.match_date >= test_start_date:
                continue
            winner = manager.get_state(row.winner_id)
            loser = manager.get_state(row.loser_id)
            update_ratings(
                winner_state=winner,
                loser_state=loser,
                surface=row.surface,
                match_date=row.match_date,
                score=row.score,
                tournament_level=row.tournament_level,
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

    # Phase 2: Test - predict, then update
    console.print(
        f"[bold cyan]Phase 2: Test[/bold cyan] - "
        f"Predicting + evaluating from {test_start_date} to {test_end_date}..."
    )
    test_count = _count_matches(
        tour=tour,
        start_date=test_start_date,
        end_date=test_end_date,
    )
    console.print(f"  Test matches: {test_count:,}")

    predictions: list[BacktestPrediction] = []
    skipped_walkover = 0

    with Progress(
        TextColumn("[bold green]Backtest:"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        disable=not show_progress,
    ) as progress:
        task = progress.add_task("backtest", total=test_count)
        for row in _iter_matches_in_range(
            tour=tour,
            start_date=test_start_date,
            end_date=test_end_date,
        ):
            if row.walkover:
                # Walkovers contain no real outcome information
                skipped_walkover += 1
                progress.update(task, advance=1)
                continue

            winner = manager.get_state(row.winner_id)
            loser = manager.get_state(row.loser_id)

            # STEP 1: PREDICT (using current Elo, no future leakage)
            p_winner = predict_match_probability(
                winner, loser, row.surface, config=config
            )

            # Randomize side choice for unbiased evaluation
            # Half the time we treat "player A" as the eventual winner,
            # half the time as the eventual loser. This prevents the metrics
            # from being inflated by always predicting the actual winner.
            if rng.random() < 0.5:
                # A = winner
                p_a = p_winner
                a_is_winner = 1
            else:
                # A = loser; flip probability
                p_a = 1.0 - p_winner
                a_is_winner = 0

            predictions.append(BacktestPrediction(
                match_date=row.match_date,
                match_id=row.match_id,
                tour=row.tour,
                surface=row.surface,
                winner_id=row.winner_id,
                loser_id=row.loser_id,
                p_winner_wins=p_winner,
                p_player_a_wins=p_a,
                a_is_winner=a_is_winner,
            ))

            # STEP 2: UPDATE Elo state with actual outcome
            update_ratings(
                winner_state=winner,
                loser_state=loser,
                surface=row.surface,
                match_date=row.match_date,
                score=row.score,
                tournament_level=row.tournament_level,
                is_retirement=row.retirement or False,
                is_walkover=False,  # we already filtered walkovers
                config=config,
            )

            progress.update(task, advance=1)

    console.print(
        f"[green]✓[/green] Generated {len(predictions):,} predictions "
        f"(skipped {skipped_walkover} walkovers)\n"
    )

    # Compute metrics
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
        config_name=config_name,
        tour=tour,
        train_start_date=train_start_date,
        train_end_date=test_start_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        total_train_matches=warmup_processed,
        total_predictions=len(predictions),
        overall_metrics=overall_metrics,
        per_surface_metrics=per_surface_metrics,
        per_year_metrics=per_year_metrics,
    )

    return summary, predictions


def save_backtest_to_db(
    summary: BacktestRunSummary,
    notes: str | None = None,
) -> int:
    """Persist a backtest run summary to backtest_runs table.

    Args:
        summary: BacktestRunSummary from run_walk_forward_backtest
        notes: Optional human-readable notes

    Returns:
        The backtest_id assigned by the database
    """
    import json

    from sqlalchemy import text
    from tennis_predictor.data.storage import get_session

    logger = _get_logger()

    config_json = {
        "tour": summary.tour,
        "train_start": str(summary.train_start_date),
        "test_start": str(summary.test_start_date),
        "test_end": str(summary.test_end_date),
        "notes": notes,
    }

    # First ensure the model version exists
    ensure_model_sql = text("""
        INSERT INTO model_versions (
            model_version_id, model_type, description,
            training_data_start, training_data_end, is_active
        ) VALUES (
            :version_id, 'elo', :description,
            :train_start, :train_end, FALSE
        )
        ON CONFLICT (model_version_id) DO NOTHING
    """)

    insert_sql = text("""
        INSERT INTO backtest_runs (
            model_version_id, run_name,
            test_start_date, test_end_date,
            total_matches, total_predictions,
            accuracy, log_loss, brier_score, calibration_error,
            config, started_at, completed_at, status
        ) VALUES (
            :model_version_id, :run_name,
            :test_start_date, :test_end_date,
            :total_matches, :total_predictions,
            :accuracy, :log_loss, :brier_score, :calibration_error,
            :config, NOW(), NOW(), 'completed'
        )
        RETURNING backtest_id
    """)

    with get_session() as session:
        session.execute(ensure_model_sql, {
            "version_id": summary.config_name,
            "description": f"Surface-adjusted Elo for {summary.tour}",
            "train_start": summary.train_start_date,
            "train_end": summary.train_end_date,
        })

        result = session.execute(insert_sql, {
            "model_version_id": summary.config_name,
            "run_name": f"{summary.tour} {summary.test_start_date}→{summary.test_end_date}",
            "test_start_date": summary.test_start_date,
            "test_end_date": summary.test_end_date,
            "total_matches": summary.total_predictions,
            "total_predictions": summary.total_predictions,
            "accuracy": round(summary.overall_metrics.accuracy, 5),
            "log_loss": round(summary.overall_metrics.log_loss, 5),
            "brier_score": round(summary.overall_metrics.brier_score, 5),
            "calibration_error": round(summary.overall_metrics.calibration_error, 5),
            "config": json.dumps(config_json),
        })
        backtest_id = result.scalar()

    logger.info("backtest_saved", backtest_id=backtest_id)
    return backtest_id
