"""CLI: compute CLV vs Pinnacle and identify value bets from backtest predictions.

Prerequisite:
  1. Apply migration 011_create_backtest_predictions.sql
  2. Modify walk_forward.py + run_backtest.py per PATCH instructions
  3. Re-run backtest with --save flag to populate backtest_predictions
  4. THEN run this analysis

Usage:

    # Latest run for given tour
    python -m tennis_predictor.backtest.clv_analysis --tour ATP

    # Specific backtest_id
    python -m tennis_predictor.backtest.clv_analysis --run-id 3

    # Custom value-bet thresholds
    python -m tennis_predictor.backtest.clv_analysis --tour ATP \\
        --min-edge 0.07 --min-prob 0.55 --min-odds 1.60

    # Export per-prediction CSV
    python -m tennis_predictor.backtest.clv_analysis --tour ATP \\
        --output-csv data/clv_atp.csv
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
    ValueBetCriteria, compute_clv, compute_edge, is_value_bet,
    summarise_clv,
)
from tennis_predictor.data.storage import get_session

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# SQL queries (match real backtest_runs schema: backtest_id, model_version_id)
# ---------------------------------------------------------------------------

SQL_LATEST_RUN = text("""
    SELECT br.backtest_id, br.model_version_id, br.run_name, br.completed_at
    FROM backtest_runs br
    WHERE (:version IS NULL OR br.model_version_id = :version)
      AND EXISTS (
          SELECT 1 FROM backtest_predictions bp
          WHERE bp.backtest_run_id = br.backtest_id
            AND (:tour IS NULL OR bp.tour = :tour)
      )
    ORDER BY br.completed_at DESC NULLS LAST
    LIMIT 1
""")

SQL_RUN_BY_ID = text("""
    SELECT br.backtest_id, br.model_version_id, br.run_name, br.completed_at
    FROM backtest_runs br
    WHERE br.backtest_id = :id
""")

# For each backtest prediction, find the Pinnacle row for that match.
# Pinnacle's winner_implied_prob is the implied probability of the actual
# winner -- same convention as our predicted_prob_winner -- so CLV is
# directly comparable.
SQL_LOAD_FOR_CLV = text("""
    SELECT
        bp.prediction_id,
        bp.match_id,
        bp.predicted_prob_winner,
        bp.was_correct,
        bp.surface,
        bp.tournament_level,
        bp.tour,
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
      AND (:tour IS NULL OR bp.tour = :tour)
    ORDER BY m.match_date
""")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_run(
    run_id: int | None,
    tour: str | None,
    version: str | None,
) -> tuple[int, str, str]:
    """Return (backtest_id, model_version_id, run_name).

    Raises click.ClickException if nothing matches.
    """
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
            "with non-empty backtest_predictions. "
            "Did you re-run the backtest with --save after applying the "
            "walk_forward.py patch?"
        )
    return int(row.backtest_id), row.model_version_id, row.run_name


def load_predictions_with_odds(run_id: int, tour: str | None) -> pd.DataFrame:
    """Load predictions JOINed with Pinnacle odds for one backtest run."""
    with get_session() as session:
        df = pd.read_sql(
            SQL_LOAD_FOR_CLV, session.connection(),
            params={"run_id": run_id, "tour": tour},
        )
    if not df.empty:
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Per-prediction CLV + value bet enrichment
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


def render_overall(df: pd.DataFrame, run_name: str, version: str) -> None:
    stats = summarise_clv(df["clv"].tolist(), df["is_value_bet"].tolist())

    table = Table(title=f"CLV Analysis — {run_name} / {version}")
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
    if stats.mean_clv > 0.005:
        console.print(
            "\n[bold green]Positive mean CLV detected.[/bold green] "
            "Model would have beaten Pinnacle closing on average."
        )
    elif stats.mean_clv > -0.005:
        console.print(
            "\n[yellow]Mean CLV near zero.[/yellow] Line-with-market; "
            "no systematic edge or loss vs Pinnacle."
        )
    else:
        console.print(
            "\n[red]Negative mean CLV.[/red] Model is being out-priced by "
            "Pinnacle on average. Treat backtest accuracy with scepticism."
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
        if len(sub) < 50:
            continue
        mean = sub["clv"].mean()
        pct_pos = 100 * (sub["clv"] > 0).mean()
        n_vb = int(sub["is_value_bet"].sum())
        table.add_row(
            str(key) if key is not None else "(unknown)",
            f"{len(sub):,}",
            f"{mean * 100:+.2f}%",
            f"{pct_pos:.1f}%",
            f"{n_vb:,}",
        )
    console.print(table)


def write_per_prediction_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "match_id", "match_date", "surface", "tournament_level", "tour",
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
@click.option("--output-csv", type=click.Path(path_type=Path), default=None,
              help="Optional path to write per-prediction CSV")
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

    backtest_id, version_resolved, run_name = resolve_run(
        run_id, tour, model_version,
    )
    console.print(
        f"[bold]CLV analysis[/bold] backtest_id={backtest_id} "
        f"({run_name} / {version_resolved})\n"
        f"Tour filter: {tour or 'all'}  |  "
        f"value-bet criteria: edge≥{min_edge:.2f}, prob≥{min_prob:.2f}, "
        f"odds≥{min_odds:.2f}"
    )

    df = load_predictions_with_odds(backtest_id, tour)
    if df.empty:
        console.print(
            f"[red]No predictions JOINable with Pinnacle for backtest_id="
            f"{backtest_id}, tour={tour}.[/red]"
        )
        console.print(
            "Possible causes:\n"
            "  - backtest didn't save predictions (run_backtest --save?)\n"
            "  - matches not linked to historical_odds_raw (Phase 3.2)\n"
            "  - this run is for a different tour than --tour"
        )
        return

    console.print(f"  Loaded [cyan]{len(df):,}[/cyan] predictions with Pinnacle odds")

    df = enrich_with_clv(df, criteria)
    df["year"] = df["match_date"].apply(_year)

    render_overall(df, run_name, version_resolved)
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
