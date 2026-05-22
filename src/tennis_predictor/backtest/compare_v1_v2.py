"""A/B comparison: Elo v1 (baseline) vs Elo v2 (with improvements).

Runs both backtests on identical data and prints a side-by-side comparison.

Usage:
    python -m tennis_predictor.backtest.compare_v1_v2 --tour ATP

This is the rigorous way to validate model improvements:
- Same train/test split
- Same data
- Same random seed
- Only the model differs
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from rich.console import Console
from rich.table import Table

from tennis_predictor.backtest.walk_forward import run_walk_forward_backtest
from tennis_predictor.backtest.walk_forward_v2 import run_walk_forward_backtest_v2
from tennis_predictor.logging_config import setup_logging
from tennis_predictor.models.elo import EloConfig
from tennis_predictor.models.elo_v2 import EloConfigV2

console = Console()


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B comparison: Elo v1 vs v2")
    parser.add_argument("--tour", choices=["ATP", "WTA"], default="ATP")
    parser.add_argument("--train-start", type=_parse_date, default=date(2000, 1, 1))
    parser.add_argument("--test-start", type=_parse_date, default=date(2011, 1, 1))
    parser.add_argument("--test-end", type=_parse_date, default=date(2024, 12, 31))
    parser.add_argument(
        "--save-v2",
        action="store_true",
        help="Save v2 Elo snapshot to DB after backtest",
    )

    args = parser.parse_args()
    setup_logging()

    # Run v1 (baseline)
    console.print("[bold yellow]═══ Running Elo v1 (baseline) ═══[/bold yellow]\n")
    config_v1 = EloConfig()
    summary_v1, _ = run_walk_forward_backtest(
        tour=args.tour,
        train_start_date=args.train_start,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        config=config_v1,
        config_name="elo_v1_surface",
        random_seed=42,
    )

    # Run v2
    console.print(
        "\n[bold green]═══ Running Elo v2 (with improvements) ═══[/bold green]\n"
    )
    config_v2 = EloConfigV2()
    summary_v2, _, manager_v2 = run_walk_forward_backtest_v2(
        tour=args.tour,
        train_start_date=args.train_start,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        config=config_v2,
        config_name="elo_v2_surface",
        random_seed=42,
    )

    # Comparison table
    console.print()
    console.print("[bold]╔═══════════════════════════════════════════════════╗[/bold]")
    console.print("[bold]║          A/B COMPARISON: v1 vs v2                 ║[/bold]")
    console.print("[bold]╚═══════════════════════════════════════════════════╝[/bold]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("v1 (baseline)", justify="right")
    table.add_column("v2 (improved)", justify="right")
    table.add_column("Δ change", justify="right")
    table.add_column("Better?", justify="center")

    m1 = summary_v1.overall_metrics
    m2 = summary_v2.overall_metrics

    # Accuracy (higher better)
    delta_acc = m2.accuracy - m1.accuracy
    table.add_row(
        "Accuracy",
        f"{m1.accuracy:.4f}",
        f"{m2.accuracy:.4f}",
        f"{delta_acc:+.4f}",
        "✓" if delta_acc > 0 else "✗",
    )

    # Log loss (lower better)
    delta_ll = m2.log_loss - m1.log_loss
    table.add_row(
        "Log loss",
        f"{m1.log_loss:.4f}",
        f"{m2.log_loss:.4f}",
        f"{delta_ll:+.4f}",
        "✓" if delta_ll < 0 else "✗",
    )

    # Brier (lower better)
    delta_brier = m2.brier_score - m1.brier_score
    table.add_row(
        "Brier score",
        f"{m1.brier_score:.4f}",
        f"{m2.brier_score:.4f}",
        f"{delta_brier:+.4f}",
        "✓" if delta_brier < 0 else "✗",
    )

    # Calibration error (lower better)
    delta_ece = m2.calibration_error - m1.calibration_error
    table.add_row(
        "Calibration error",
        f"{m1.calibration_error:.4f}",
        f"{m2.calibration_error:.4f}",
        f"{delta_ece:+.4f}",
        "✓" if delta_ece < 0 else "✗",
    )

    console.print(table)
    console.print()

    # Per-surface comparison
    console.print("[bold cyan]Per-Surface Accuracy[/bold cyan]")
    surface_table = Table(show_header=True, header_style="bold")
    surface_table.add_column("Surface", style="cyan")
    surface_table.add_column("v1", justify="right")
    surface_table.add_column("v2", justify="right")
    surface_table.add_column("Δ", justify="right")

    for surface in ("Hard", "Clay", "Grass"):
        if surface in summary_v1.per_surface_metrics and surface in summary_v2.per_surface_metrics:
            v1 = summary_v1.per_surface_metrics[surface].accuracy
            v2 = summary_v2.per_surface_metrics[surface].accuracy
            delta = v2 - v1
            surface_table.add_row(
                surface,
                f"{v1:.4f}",
                f"{v2:.4f}",
                f"[{'green' if delta > 0 else 'red'}]{delta:+.4f}[/]",
            )

    console.print(surface_table)
    console.print()

    # Per-year comparison (recent years only - to see if v2 catches the post-Big-3 decline)
    console.print("[bold cyan]Per-Year Accuracy (2020-2024)[/bold cyan]")
    year_table = Table(show_header=True, header_style="bold")
    year_table.add_column("Year", style="cyan")
    year_table.add_column("v1", justify="right")
    year_table.add_column("v2", justify="right")
    year_table.add_column("Δ", justify="right")

    for year in (2020, 2021, 2022, 2023, 2024):
        if year in summary_v1.per_year_metrics and year in summary_v2.per_year_metrics:
            v1 = summary_v1.per_year_metrics[year].accuracy
            v2 = summary_v2.per_year_metrics[year].accuracy
            delta = v2 - v1
            year_table.add_row(
                str(year),
                f"{v1:.4f}",
                f"{v2:.4f}",
                f"[{'green' if delta > 0 else 'red'}]{delta:+.4f}[/]",
            )

    console.print(year_table)
    console.print()

    # Save v2 if requested
    if args.save_v2:
        console.print("[cyan]Saving v2 Elo snapshot...[/cyan]")
        rows = manager_v2.save_to_db(
            rating_date=args.test_end,
            algorithm_version="elo_v2_surface",
        )
        console.print(f"[green]✓[/green] Saved {rows} v2 rating rows")

        # Also save to backtest_runs
        from tennis_predictor.backtest.walk_forward import save_backtest_to_db
        bt_id = save_backtest_to_db(summary_v2)
        console.print(f"[green]✓[/green] Saved backtest run as id={bt_id}")

    # Verdict
    n_better = sum([delta_acc > 0, delta_ll < 0, delta_brier < 0, delta_ece < 0])
    console.print()
    if n_better >= 3:
        console.print("[bold green]VERDICT: v2 is an improvement (>=3 of 4 metrics better)[/bold green]")
    elif n_better == 2:
        console.print("[bold yellow]VERDICT: v2 is a mixed result (2 of 4 metrics better)[/bold yellow]")
    else:
        console.print("[bold red]VERDICT: v2 is WORSE (<=1 of 4 metrics better)[/bold red]")
        console.print("[red]→ Stick with v1, or revisit v2 design[/red]")


if __name__ == "__main__":
    main()
