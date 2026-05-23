"""CLI: compute CLV vs Pinnacle and identify value bets from backtest predictions.

Prerequisite: ``backtest_predictions`` table must be populated. See
README for how to wire your existing backtest to save predictions.

Usage:

    # Default analysis: latest run for given tour + model_version
    python -m tennis_predictor.backtest.clv_analysis --tour ATP

    # Specific backtest_run_id
    python -m tennis_predictor.backtest.clv_analysis --run-id 3

    # Custom value-bet thresholds
    python -m tennis_predictor.backtest.clv_analysis --tour ATP \\
        --min-edge 0.07 --min-prob 0.55 --min-odds 1.60

    # Export per-prediction CSV
    python -m tennis_predictor.backtest.clv_analysis --tour ATP \\
        --output-csv data/clv_atp.csv

The analysis JOINs:
    backtest_predictions  (predicted_prob_winner)
    historical_odds_raw   (Pinnacle winner_odds / winner_implied_prob)
    matches               (winner_id - to determine which side of the
                          Pinnacle quote corresponds to the model's pick)
"""

from __future__ import annotations

import csv
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
    ValueBetCriteria, compute_clv, compute_edge, is_value_bet,
    summarise_clv,
)
from tennis_predictor.data.storage.db import get_engine

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# The key JOIN. For each backtest prediction:
#   - Pull the matched Pinnacle row from historical_odds_raw (one per match)
#   - Pinnacle's winner_odds = odds on the actual winner of the match.
#     This is critical: when comparing CLV, both sides of the comparison
#     refer to the SAME side of the match (the winner).
#
# Notes:
#   - ``bookmaker_code = 'PS'`` filters to Pinnacle.
#   - We use INNER JOIN so unmatched predictions (no Pinnacle row) are dropped.
#   - ``predicted_prob_winner`` is the model's probability for the actual
#     winner. ``winner_implied_prob`` is Pinnacle's. So CLV is directly
#     comparable: (predicted - implied) / implied.

SQL_LOAD_FOR_CLV = text("""
    SELECT
        bp.prediction_id,
        bp.match_id,
        bp.predicted_prob_winner,
        bp.was_correct,
        bp.surface,
        bp.tournament_level,
        bp.model_version,
        m.match_date,
        hor.winner_odds          AS pinnacle_winner_odds,
        hor.winner_implied_prob  AS pinnacle_winner_implied,
        hor.loser_odds           AS pinnacle_loser_odds,
        hor.vig                  AS pinnacle_vig
    FROM backtest_predictions bp
    JOIN matches m
        ON bp.match_id = m.match_id
    JOIN historical_odds_raw hor
        ON hor.match_id = bp.match_id
       AND hor.bookmaker_code = 'PS'
    WHERE bp.backtest_run_id = :run_id
    ORDER BY m.match_date
""")

SQL_LATEST_RUN_FOR = text("""
    SELECT br.run_id, br.model_version, br.tour, br.created_at
    FROM backtest_runs br
    WHERE (:tour IS NULL OR br.tour = :tour)
      AND (:version IS NULL OR br.model_version = :version)
    ORDER BY br.created_at DESC
    LIMIT 1
""")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_run_id(
    engine,
    run_id: int | None,
    tour: str | None,
    version: str | None,
) -> tuple[int, str, str]:
    """Return (run_id, tour, model_version). Raises if nothing found."""
    if run_id is not None:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT run_id, model_version, tour FROM backtest_runs "
                     "WHERE run_id = :id"),
                {"id": run_id},
            ).first()
        if row is None:
            raise click.ClickException(f"No backtest_run with run_id={run_id}")
        return int(row.run_id), row.tour, row.model_version

    with engine.connect() as conn:
        row = conn.execute(
            SQL_LATEST_RUN_FOR,
            {"tour": tour, "version": version},
        ).first()
    if row is None:
        raise click.ClickException(
            f"No backtest_runs match tour={tour}, version={version}. "
            "Did you run the backtest with prediction-saving enabled?"
        )
    return int(row.run_id), row.tour, row.model_version


def load_predictions_with_odds(engine, run_id: int) -> pd.DataFrame:
    """Load predictions JOINed with Pinnacle odds for one backtest run."""
    with engine.connect() as conn:
        df = pd.read_sql(SQL_LOAD_FOR_CLV, conn, params={"run_id": run_id})
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Per-prediction CLV + value-bet enrichment
# ---------------------------------------------------------------------------

def enrich_with_clv(df: pd.DataFrame, criteria: ValueBetCriteria) -> pd.DataFrame:
    """Add per-row columns: clv, edge, is_value_bet."""
    if df.empty:
        return df
    df = df.copy()
    df["clv"] = df.apply(
        lambda r: compute_clv(
            float(r["predicted_prob_winner"]),
            float(r["pinnacle_winner_implied"]),
        ),
        axis=1,
    )
    df["edge"] = df.apply(
        lambda r: compute_edge(
            float(r["predicted_prob_winner"]),
            float(r["pinnacle_winner_odds"]),
        ),
        axis=1,
    )
    df["is_value_bet"] = df.apply(
        lambda r: is_value_bet(
            float(r["predicted_prob_winner"]),
            float(r["pinnacle_winner_odds"]),
            criteria,
        ),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _year(d) -> int:
    return d.year if hasattr(d, "year") else int(str(d)[:4])


def render_overall(df: pd.DataFrame, tour: str, version: str) -> None:
    stats = summarise_clv(df["clv"].tolist(), df["is_value_bet"].tolist())

    table = Table(title=f"CLV Analysis — {tour} / {version}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Predictions analysed", f"{stats.n_predictions:,}")
    table.add_row("Mean CLV", f"{stats.mean_clv * 100:+.2f}%")
    table.add_row("Median CLV", f"{stats.median_clv * 100:+.2f}%")
    table.add_row("% with positive CLV",
                  f"{stats.pct_positive_clv:.2f}%")
    table.add_row("% with CLV > 2% (significant)",
                  f"{stats.pct_significantly_positive:.2f}%")
    table.add_section()
    table.add_row("[bold]Value bets identified[/bold]", f"{stats.n_value_bets:,}")
    table.add_row("Value bet rate", f"{stats.value_bet_rate:.2f}%")

    console.print(table)

    # Headline verdict
    if stats.mean_clv > 0.005:  # >0.5%
        console.print(
            "\n[bold green]Positive mean CLV detected.[/bold green] "
            "Model would have beaten Pinnacle closing on average over this run."
        )
    elif stats.mean_clv > -0.005:
        console.print(
            "\n[yellow]Mean CLV near zero.[/yellow] Model is line-with-market; "
            "no systematic edge OR loss vs Pinnacle."
        )
    else:
        console.print(
            "\n[red]Negative mean CLV.[/red] Model is being out-priced by Pinnacle "
            "on average. Treat backtest accuracy with appropriate scepticism."
        )


def render_by_dimension(df: pd.DataFrame, dim: str, dim_label: str) -> None:
    table = Table(title=f"CLV by {dim_label}")
    table.add_column(dim_label)
    table.add_column("N", justify="right")
    table.add_column("Mean CLV", justify="right")
    table.add_column("% Positive", justify="right")
    table.add_column("Value Bets", justify="right")

    grouped = df.groupby(dim, dropna=False)
    for key, sub in grouped:
        if len(sub) < 50:  # too few to be informative
            continue
        mean = sub["clv"].mean()
        pct_pos = 100 * (sub["clv"] > 0).mean()
        n_vb = int(sub["is_value_bet"].sum())
        table.add_row(
            str(key) if key else "(unknown)",
            f"{len(sub):,}",
            f"{mean * 100:+.2f}%",
            f"{pct_pos:.1f}%",
            f"{n_vb:,}",
        )
    console.print(table)


def write_per_prediction_csv(df: pd.DataFrame, path: Path) -> None:
    """Dump enriched predictions to CSV for offline analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "match_id", "match_date", "surface", "tournament_level",
        "predicted_prob_winner", "pinnacle_winner_implied",
        "pinnacle_winner_odds", "pinnacle_vig",
        "clv", "edge", "is_value_bet", "was_correct",
    ]
    df[cols].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Click entry
# ---------------------------------------------------------------------------

@click.command(name="clv-analyze")
@click.option("--run-id", type=int, default=None,
              help="Specific backtest_run_id; otherwise use latest matching")
@click.option("--tour", type=click.Choice(["ATP", "WTA"]), default=None,
              help="Filter to a tour (used only when --run-id absent)")
@click.option("--version", "model_version", type=str, default=None,
              help="Filter to a model version (used only when --run-id absent)")
@click.option("--min-edge", type=float, default=DEFAULT_MIN_EDGE,
              show_default=True)
@click.option("--min-prob", type=float, default=DEFAULT_MIN_PROB,
              show_default=True)
@click.option("--min-odds", type=float, default=DEFAULT_MIN_ODDS,
              show_default=True)
@click.option("--output-csv", type=click.Path(path_type=Path), default=None,
              help="Optional path to write per-prediction enriched CSV")
def clv_analyze_cli(
    run_id: int | None,
    tour: str | None,
    model_version: str | None,
    min_edge: float,
    min_prob: float,
    min_odds: float,
    output_csv: Optional[Path],
) -> None:
    """Closing Line Value analysis on stored backtest predictions."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    criteria = ValueBetCriteria(min_edge=min_edge, min_prob=min_prob,
                                min_odds=min_odds)
    engine = get_engine()

    # Resolve which run to analyse
    run_id, tour_resolved, version_resolved = resolve_run_id(
        engine, run_id, tour, model_version,
    )
    console.print(
        f"[bold]CLV analysis[/bold] for run_id={run_id} "
        f"({tour_resolved} / {version_resolved}), "
        f"value-bet criteria: edge≥{min_edge:.2f}, prob≥{min_prob:.2f}, "
        f"odds≥{min_odds:.2f}"
    )

    # Load
    df = load_predictions_with_odds(engine, run_id)
    if df.empty:
        console.print(
            f"[red]No predictions JOINable with Pinnacle for run_id={run_id}.[/red]"
        )
        console.print(
            "Possible causes: backtest didn't save predictions, or matches "
            "weren't yet linked to historical_odds_raw (Phase 3.2)."
        )
        return

    console.print(f"  Loaded [cyan]{len(df):,}[/cyan] predictions with Pinnacle odds")

    # Enrich
    df = enrich_with_clv(df, criteria)

    # Add year column for time-series analysis
    df["year"] = df["match_date"].apply(_year)

    # Render
    render_overall(df, tour_resolved, version_resolved)
    console.print()
    render_by_dimension(df, "surface", "Surface")
    console.print()
    render_by_dimension(df, "tournament_level", "Level")
    console.print()
    render_by_dimension(df, "year", "Year")

    if output_csv is not None:
        write_per_prediction_csv(df, output_csv)
        console.print(f"\n[green]Wrote per-prediction CSV to {output_csv}[/green]")


if __name__ == "__main__":
    clv_analyze_cli()
