"""CLI script to run an Elo backtest and display results.

Usage:
    python -m tennis_predictor.backtest.run_backtest --tour ATP \
        --train-start 2000-01-01 \
        --test-start 2011-01-01 \
        --test-end 2024-12-31 \
        --save

Recommendations:
- Warmup of ~10 years (2000-2010) allows Elo to stabilize
- Test period 2011-2024 gives 14 years of out-of-sample evaluation
- Use --save to persist run to backtest_runs table
- With --save, per-prediction rows are ALSO persisted to backtest_predictions
  for Phase 3.3 CLV analysis.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from rich.console import Console
from rich.table import Table

from tennis_predictor.backtest.walk_forward import (
    run_walk_forward_backtest,
    save_backtest_predictions,
    save_backtest_to_db,
)
from tennis_predictor.logging_config import setup_logging
from tennis_predictor.models.elo import EloConfig

console = Console()


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest of Elo predictions"
    )
    parser.add_argument("--tour", choices=["ATP", "WTA"], default="ATP")
    parser.add_argument("--train-start", type=_parse_date, default=date(2000, 1, 1))
    parser.add_argument("--test-start", type=_parse_date, default=date(2011, 1, 1))
    parser.add_argument("--test-end", type=_parse_date, default=date(2024, 12, 31))
    parser.add_argument(
        "--config-name",
        default="elo_v1_surface",
        help="Identifier saved with the run",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to backtest_runs table AND per-prediction rows "
             "to backtest_predictions (for CLV analysis)",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional notes saved alongside the run",
    )

    args = parser.parse_args()
    setup_logging()

    config = EloConfig()

    summary, predictions = run_walk_forward_backtest(
        tour=args.tour,
        train_start_date=args.train_start,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        config=config,
        config_name=args.config_name,
    )

    # Display full report
    console.print()
    console.print("[bold]═══════════════════════════════════════════════[/bold]")
    console.print("[bold green]            BACKTEST RESULTS                    [/bold green]")
    console.print("[bold]═══════════════════════════════════════════════[/bold]")
    console.print()
    console.print(summary.pretty_print())
    console.print()

    # Calibration table
    if summary.overall_metrics.calibration_bins:
        console.print("[bold cyan]Calibration Reliability[/bold cyan]")
        console.print("(For predictions in each probability bucket, "
                      "do they actually win at the expected rate?)")
        console.print()

        cal_table = Table(show_header=True, header_style="bold")
        cal_table.add_column("Bucket", justify="center")
        cal_table.add_column("Mean Predicted", justify="right")
        cal_table.add_column("Mean Actual", justify="right")
        cal_table.add_column("Gap", justify="right")
        cal_table.add_column("N", justify="right", style="dim")

        for mean_pred, mean_actual, count in summary.overall_metrics.calibration_bins:
            bucket = f"{mean_pred:.0%}"
            gap = mean_pred - mean_actual
            gap_color = "red" if abs(gap) > 0.05 else "green"
            cal_table.add_row(
                bucket,
                f"{mean_pred:.4f}",
                f"{mean_actual:.4f}",
                f"[{gap_color}]{gap:+.4f}[/{gap_color}]",
                f"{count:,}",
            )
        console.print(cal_table)
        console.print()

    # Benchmarks comparison
    console.print("[bold cyan]Benchmark Comparison[/bold cyan]")
    console.print(
        "  Random guessing:        accuracy=50%, log_loss=0.693, brier=0.250"
    )
    console.print(
        "  Tennis literature Elo:  accuracy~68%, log_loss~0.59,  brier~0.21"
    )
    console.print(
        "  Pinnacle closing line:  accuracy~70%, log_loss~0.55,  brier~0.19"
    )
    console.print()

    m = summary.overall_metrics
    if m.accuracy > 0.65:
        console.print(f"[green]✓ Accuracy {m.accuracy:.2%} - in the expected range for surface Elo[/green]")
    else:
        console.print(f"[yellow]⚠ Accuracy {m.accuracy:.2%} - below typical Elo benchmarks[/yellow]")

    if m.log_loss < 0.65:
        console.print(f"[green]✓ Log loss {m.log_loss:.4f} - good calibration[/green]")
    else:
        console.print(f"[yellow]⚠ Log loss {m.log_loss:.4f} - could be more calibrated[/yellow]")

    if args.save:
        console.print()
        console.print("[cyan]Saving backtest summary to database...[/cyan]")
        backtest_id = save_backtest_to_db(summary, notes=args.notes)
        console.print(f"[green]✓[/green] Saved as backtest_id={backtest_id}")

        console.print("[cyan]Saving per-prediction rows for CLV analysis...[/cyan]")
        n_inserted = save_backtest_predictions(
            backtest_id=backtest_id,
            model_version=args.config_name,
            predictions=predictions,
        )
        console.print(
            f"[green]✓[/green] Saved {n_inserted:,} predictions "
            f"to backtest_predictions"
        )


if __name__ == "__main__":
    main()
