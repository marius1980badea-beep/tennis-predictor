"""CLI: simulate ROI of value bets identified from backtest predictions vs Pinnacle.

Prerequisite (already done in Phase 3.3):
  - Migration 011_create_backtest_predictions.sql applied
  - Backtest re-run with --save to populate backtest_predictions
  - Phase 3.2 fuzzy matching populated historical_odds_raw.match_id

This CLI evaluates 4 strategies in parallel:
  1. flat_all_favorites:  bet 1u on every match where we are favorite (no edge filter)
  2. flat_value:          bet 1u only on value bets (edge ≥ min_edge, prob ≥ min_prob, odds ≥ min_odds)
  3. kelly_1/4 on value:  fractional Kelly (0.25 * full Kelly) on value bets only
  4. kelly_1/8 on value:  more conservative

For each match in `backtest_predictions` (winner side from model's perspective)
we also evaluate the LOSER SIDE: prob = 1 - predicted_prob_winner, odds = loser_odds.
That gives us up to 2 candidate bets per match.

Usage:

    # Latest run, default thresholds (edge 5%, prob 55%, odds 1.60)
    python -m tennis_predictor.backtest.roi_analysis --tour ATP

    # Custom thresholds
    python -m tennis_predictor.backtest.roi_analysis --tour ATP \\
        --min-edge 0.07 --min-prob 0.55 --min-odds 1.60

    # Export equity curves CSV (one row per bet across all strategies)
    python -m tennis_predictor.backtest.roi_analysis --tour ATP \\
        --output-csv data/roi_equity.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from tennis_predictor.backtest.clv import (
    DEFAULT_MIN_EDGE, DEFAULT_MIN_ODDS, DEFAULT_MIN_PROB,
    ValueBetCriteria, compute_edge, is_value_bet,
)
from tennis_predictor.backtest.roi import (
    Bet, SimulationResult, roi_by_attribute,
    simulate_flat_staking, simulate_kelly_staking,
)
from tennis_predictor.data.storage import get_session

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# SQL queries (same structure as clv_analysis.py)
# ---------------------------------------------------------------------------

SQL_LATEST_RUN = text("""
    SELECT br.backtest_id, br.model_version_id, br.run_name, br.completed_at
    FROM backtest_runs br
    WHERE (CAST(:version AS TEXT) IS NULL
           OR br.model_version_id = CAST(:version AS TEXT))
      AND EXISTS (
          SELECT 1 FROM backtest_predictions bp
          WHERE bp.backtest_run_id = br.backtest_id
            AND (CAST(:tour AS TEXT) IS NULL
                 OR bp.tour = CAST(:tour AS TEXT))
      )
    ORDER BY br.completed_at DESC NULLS LAST
    LIMIT 1
""")

SQL_RUN_BY_ID = text("""
    SELECT br.backtest_id, br.model_version_id, br.run_name, br.completed_at
    FROM backtest_runs br
    WHERE br.backtest_id = :id
""")

SQL_LOAD_FOR_ROI = text("""
    SELECT
        bp.prediction_id,
        bp.match_id,
        bp.predicted_prob_winner,
        bp.was_correct,
        bp.surface,
        bp.tournament_level,
        bp.tour,
        m.match_date,
        hor.winner_odds          AS pinnacle_winner_odds,
        hor.loser_odds           AS pinnacle_loser_odds,
        hor.winner_implied_prob  AS pinnacle_winner_implied,
        hor.loser_implied_prob   AS pinnacle_loser_implied
    FROM backtest_predictions bp
    JOIN matches m
        ON bp.match_id = m.match_id
    JOIN historical_odds_raw hor
        ON hor.match_id = bp.match_id
       AND hor.bookmaker_code = 'PS'
    WHERE bp.backtest_run_id = :run_id
      AND (CAST(:tour AS TEXT) IS NULL OR bp.tour = CAST(:tour AS TEXT))
    ORDER BY m.match_date, bp.match_id
""")


# ---------------------------------------------------------------------------
# Run resolution
# ---------------------------------------------------------------------------

def resolve_run(
    run_id: int | None,
    tour: str | None,
    version: str | None,
) -> tuple[int, str, str]:
    """Return (backtest_id, model_version_id, run_name)."""
    with get_session() as session:
        if run_id is not None:
            row = session.execute(SQL_RUN_BY_ID, {"id": run_id}).first()
            if row is None:
                raise click.ClickException(
                    f"No backtest_runs row with backtest_id={run_id}"
                )
            return int(row.backtest_id), row.model_version_id, row.run_name

        row = session.execute(
            SQL_LATEST_RUN, {"tour": tour, "version": version},
        ).first()

    if row is None:
        raise click.ClickException(
            f"No backtest_runs match tour={tour}, version={version} "
            "with non-empty backtest_predictions."
        )
    return int(row.backtest_id), row.model_version_id, row.run_name


def load_predictions_with_odds(run_id: int, tour: str | None) -> pd.DataFrame:
    """Load predictions JOINed with Pinnacle odds for one backtest run."""
    with get_session() as session:
        df = pd.read_sql(
            SQL_LOAD_FOR_ROI, session.connection(),
            params={"run_id": run_id, "tour": tour},
        )
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
        df["year"] = df["match_date"].apply(lambda d: d.year)
    return df


# ---------------------------------------------------------------------------
# Bet construction
#
# For each match row, we generate 0, 1 or 2 candidate bets:
#   - Winner side: prob = predicted_prob_winner,    odds = pinnacle_winner_odds, won = True
#   - Loser side:  prob = 1 - predicted_prob_winner, odds = pinnacle_loser_odds,  won = False
# Then we filter by selection_strategy (see below).
# ---------------------------------------------------------------------------

def _row_to_two_sided_bets(row) -> list[Bet]:
    """Generate the two candidate Bet objects (winner side, loser side) for a match.

    NOTE: 'won' on the loser-side bet is always False (the actual loser, by definition,
    lost the match). 'won' on the winner-side bet is always True.
    """
    p_winner = float(row.predicted_prob_winner)
    p_loser = 1.0 - p_winner

    common = {
        "surface":          row.surface,
        "tournament_level": row.tournament_level,
        "year":             int(row.year),
        "match_id":         int(row.match_id) if pd.notna(row.match_id) else None,
    }

    winner_bet = Bet(
        predicted_prob=p_winner,
        decimal_odds=float(row.pinnacle_winner_odds),
        won=True,
        **common,
    )
    loser_bet = Bet(
        predicted_prob=p_loser,
        decimal_odds=float(row.pinnacle_loser_odds),
        won=False,
        **common,
    )
    return [winner_bet, loser_bet]


def build_bets_all_favorites(df: pd.DataFrame) -> list[Bet]:
    """Naive strategy: bet on whichever side the model thinks is favorite (>50%).

    Used to quantify the "loss to vig" baseline when ignoring edge filters.
    """
    bets: list[Bet] = []
    for row in df.itertuples(index=False):
        sides = _row_to_two_sided_bets(row)
        # Pick the side with the higher model prob
        favorite = max(sides, key=lambda b: b.predicted_prob)
        if favorite.predicted_prob > 0.5:
            bets.append(favorite)
    return bets


def build_bets_value(df: pd.DataFrame, criteria: ValueBetCriteria) -> list[Bet]:
    """Selective strategy: bet on EITHER side if it passes the value-bet filter."""
    bets: list[Bet] = []
    for row in df.itertuples(index=False):
        for side in _row_to_two_sided_bets(row):
            if is_value_bet(side.predicted_prob, side.decimal_odds, criteria):
                bets.append(side)
    return bets


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_money(x: float, initial: float = 1000.0) -> str:
    """Format bankroll values with EUR symbol (decoupled from any real currency)."""
    return f"€{x:,.2f}"


def render_strategy_table(
    results: list[tuple[str, list[Bet], SimulationResult]],
    initial_bankroll: float,
) -> None:
    """Compact comparison table across strategies."""
    table = Table(title="Strategy Comparison", show_lines=True)
    table.add_column("Strategy", style="bold")
    table.add_column("Bets", justify="right")
    table.add_column("Win rate", justify="right")
    table.add_column("Staked", justify="right")
    table.add_column("Profit", justify="right")
    table.add_column("ROI", justify="right")
    table.add_column("Bankroll", justify="right")
    table.add_column("Max DD", justify="right")

    for label, _bets, sim in results:
        roi_str = f"{sim.roi * 100:+.2f}%"
        roi_color = "green" if sim.roi > 0 else "red" if sim.roi < -0.02 else "yellow"
        profit_color = roi_color

        table.add_row(
            label,
            f"{sim.n_bets:,}",
            f"{sim.win_rate * 100:.1f}%",
            _format_money(sim.total_staked),
            f"[{profit_color}]{_format_money(sim.profit)}[/{profit_color}]",
            f"[{roi_color}]{roi_str}[/{roi_color}]",
            _format_money(sim.final_bankroll),
            f"{sim.max_drawdown * 100:.1f}%",
        )

    console.print(table)


def render_value_breakdown(bets: list[Bet], dim: str, label: str) -> None:
    """Per-subslice flat-staking ROI for value bets."""
    rows = roi_by_attribute(bets, dim)
    if not rows:
        return

    table = Table(title=f"Flat-1u ROI on value bets, by {label}")
    table.add_column(label, style="bold")
    table.add_column("Bets", justify="right")
    table.add_column("Win rate", justify="right")
    table.add_column("Staked", justify="right")
    table.add_column("Profit", justify="right")
    table.add_column("ROI", justify="right")

    for r in rows:
        if r.n_bets < 25:
            continue  # ignore noisy small samples
        roi_color = "green" if r.roi > 0 else "red" if r.roi < -0.02 else "yellow"
        table.add_row(
            r.name,
            f"{r.n_bets:,}",
            f"{r.win_rate * 100:.1f}%",
            f"€{r.total_staked:,.2f}",
            f"[{roi_color}]€{r.profit:+,.2f}[/{roi_color}]",
            f"[{roi_color}]{r.roi * 100:+.2f}%[/{roi_color}]",
        )

    console.print(table)


def write_equity_csv(
    results: list[tuple[str, list[Bet], SimulationResult]],
    path: Path,
) -> None:
    """Long-format CSV: one row per (strategy, bet_index) with bankroll trace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, _bets, sim in results:
        for i, equity in enumerate(sim.equity_curve, start=1):
            rows.append({
                "strategy":      label,
                "bet_index":     i,
                "bankroll":      equity,
                "growth_factor": equity / sim.initial_bankroll if sim.initial_bankroll else 0,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Click entry
# ---------------------------------------------------------------------------

@click.command(name="roi-analyze")
@click.option("--run-id", type=int, default=None,
              help="Specific backtest_id; otherwise use latest matching")
@click.option("--tour", type=click.Choice(["ATP", "WTA"]), default=None,
              help="Filter to a tour")
@click.option("--version", "model_version", type=str, default=None,
              help="Filter to a model_version_id (e.g. elo_v1_surface)")
@click.option("--min-edge", type=float, default=DEFAULT_MIN_EDGE,
              show_default=True)
@click.option("--min-prob", type=float, default=DEFAULT_MIN_PROB,
              show_default=True)
@click.option("--min-odds", type=float, default=DEFAULT_MIN_ODDS,
              show_default=True)
@click.option("--initial-bankroll", type=float, default=1000.0, show_default=True,
              help="Starting bankroll for simulation (€)")
@click.option("--max-bet-pct", type=float, default=0.05, show_default=True,
              help="Max stake as fraction of bankroll (Kelly safety cap)")
@click.option("--output-csv", type=click.Path(path_type=Path), default=None,
              help="Optional path to write equity-curve CSV")
def roi_analyze_cli(
    run_id: int | None,
    tour: str | None,
    model_version: str | None,
    min_edge: float,
    min_prob: float,
    min_odds: float,
    initial_bankroll: float,
    max_bet_pct: float,
    output_csv: Optional[Path],
) -> None:
    """ROI simulation of staking strategies on backtest predictions vs Pinnacle."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    criteria = ValueBetCriteria(
        min_edge=min_edge, min_prob=min_prob, min_odds=min_odds,
    )

    backtest_id, version_resolved, run_name = resolve_run(
        run_id, tour, model_version,
    )
    console.print(
        f"[bold]ROI analysis[/bold] backtest_id={backtest_id} "
        f"({run_name} / {version_resolved})\n"
        f"Tour filter: {tour or 'all'}  |  "
        f"value criteria: edge≥{min_edge:.2f}, prob≥{min_prob:.2f}, "
        f"odds≥{min_odds:.2f}\n"
        f"Initial bankroll: €{initial_bankroll:,.2f}  |  "
        f"Kelly safety cap: {max_bet_pct:.1%} per bet"
    )

    df = load_predictions_with_odds(backtest_id, tour)
    if df.empty:
        console.print("[red]No predictions JOINable with Pinnacle for this run.[/red]")
        return

    console.print(f"  Loaded [cyan]{len(df):,}[/cyan] predictions with Pinnacle odds")

    # Build bet sets for each strategy
    bets_all_favorites = build_bets_all_favorites(df)
    bets_value = build_bets_value(df, criteria)

    console.print(
        f"  All-favorites bets: [cyan]{len(bets_all_favorites):,}[/cyan]"
        f"  |  Value bets: [cyan]{len(bets_value):,}[/cyan] "
        f"({len(bets_value) / max(len(df) * 2, 1) * 100:.2f}% of candidates)"
    )
    console.print()

    # Run the four simulations
    sim_flat_all = simulate_flat_staking(
        bets_all_favorites, unit=1.0, initial_bankroll=initial_bankroll,
    )
    sim_flat_value = simulate_flat_staking(
        bets_value, unit=1.0, initial_bankroll=initial_bankroll,
    )
    sim_kelly_quarter = simulate_kelly_staking(
        bets_value, fraction=0.25,
        initial_bankroll=initial_bankroll, max_bet_pct=max_bet_pct,
    )
    sim_kelly_eighth = simulate_kelly_staking(
        bets_value, fraction=0.125,
        initial_bankroll=initial_bankroll, max_bet_pct=max_bet_pct,
    )

    results = [
        ("flat 1u: all favorites",  bets_all_favorites, sim_flat_all),
        ("flat 1u: value bets",     bets_value,         sim_flat_value),
        ("Kelly 1/4: value bets",   bets_value,         sim_kelly_quarter),
        ("Kelly 1/8: value bets",   bets_value,         sim_kelly_eighth),
    ]

    render_strategy_table(results, initial_bankroll)
    console.print()

    # Per-subslice breakdown ONLY on value bets (where we'd actually bet)
    if bets_value:
        render_value_breakdown(bets_value, "surface",          "Surface")
        console.print()
        render_value_breakdown(bets_value, "tournament_level", "Level")
        console.print()
        render_value_breakdown(bets_value, "year",             "Year")

    # Headline verdict
    console.print()
    if sim_flat_value.roi > 0.02:
        console.print(
            f"[bold green]Value-bet flat ROI {sim_flat_value.roi * 100:+.2f}%[/bold green]"
            " — promising. Investigate WHICH subslices drive it before celebrating."
        )
    elif sim_flat_value.roi > -0.01:
        console.print(
            f"[yellow]Value-bet flat ROI {sim_flat_value.roi * 100:+.2f}%[/yellow]"
            " — near breakeven. Edge probably real but consumed by vig."
        )
    else:
        console.print(
            f"[red]Value-bet flat ROI {sim_flat_value.roi * 100:+.2f}%[/red]"
            " — strategy loses money. Consistent with the CLV finding (model "
            "under-priced by Pinnacle). Look for any positive subslices above; "
            "if none, direct Pinnacle match-winner betting is closed."
        )

    if output_csv is not None:
        write_equity_csv(results, output_csv)
        console.print(f"\n[green]Wrote equity-curve CSV to {output_csv}[/green]")


if __name__ == "__main__":
    roi_analyze_cli()
