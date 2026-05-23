# Commit 3.3 — Complete file replacements

Just overwrite these 6 files in your repo. No edits, no patches.

## Files in this ZIP

| ZIP path | Repo path | Action |
|---|---|---|
| `migrations/011_create_backtest_predictions.sql` | same | **APPLY** in Supabase |
| `src/tennis_predictor/backtest/walk_forward.py`  | same | **OVERWRITE** |
| `src/tennis_predictor/backtest/run_backtest.py`  | same | **OVERWRITE** |
| `src/tennis_predictor/backtest/clv.py`           | same | **NEW FILE** |
| `src/tennis_predictor/backtest/clv_analysis.py`  | same | **NEW FILE** |
| `tests/unit/test_clv.py`                         | same | **NEW FILE** |

## Steps

### 1) If you already applied a previous version of migration 011, DROP it

```sql
DROP TABLE IF EXISTS backtest_predictions;
```

If you never applied it, skip this.

### 2) Apply the corrected migration

```cmd
psql %DATABASE_URL% -f migrations/011_create_backtest_predictions.sql
```

Or via Supabase SQL editor.

### 3) Overwrite the 5 Python files

Just copy from this ZIP into the same paths in your repo.

### 4) Verify with unit tests

```cmd
.venv\Scripts\activate.bat
python -m pytest tests/unit/test_clv.py -v
```

Expected: **40 passed**.

### 5) Re-run the backtest (creates new backtest_id)

```cmd
python -m tennis_predictor.backtest.run_backtest --tour ATP ^
    --train-start 2000-01-01 ^
    --test-start 2011-01-01 ^
    --test-end 2024-12-31 ^
    --save
```

The new `--save` path now also writes per-prediction rows. At the end you
should see:

```
[cyan]Saving backtest summary to database...[/cyan]
[green]✓[/green] Saved as backtest_id=5
[cyan]Saving per-prediction rows for CLV analysis...[/cyan]
[green]✓[/green] Saved 39,245 predictions to backtest_predictions
```

### 6) Quick verify in SQL

```sql
SELECT br.backtest_id, br.model_version_id, br.run_name,
       COUNT(bp.prediction_id) AS predictions_stored
FROM backtest_runs br
LEFT JOIN backtest_predictions bp ON bp.backtest_run_id = br.backtest_id
GROUP BY 1, 2, 3
ORDER BY br.backtest_id DESC
LIMIT 5;
```

Should show the newest backtest with ~39,245 predictions_stored.

### 7) Run CLV analysis

```cmd
python -m tennis_predictor.backtest.clv_analysis --tour ATP
```

Paste the output here.

## What changed vs your existing code

### `walk_forward.py`
- `BacktestPrediction` dataclass: added `tournament_level: str | None` field
- Loop: passes `tournament_level=row.tournament_level` when appending predictions
- New function `save_backtest_predictions(backtest_id, model_version, predictions)`
  that bulk-inserts to `backtest_predictions` with `ON CONFLICT DO NOTHING`
- Everything else unchanged (warmup, test loop, metrics, save_backtest_to_db)

### `run_backtest.py`
- Import added: `save_backtest_predictions`
- In `--save` block, after `save_backtest_to_db`, calls
  `save_backtest_predictions` and prints the count

### `clv.py` (new)
Pure math: `compute_clv`, `compute_edge`, `is_value_bet`, `ValueBetCriteria`,
`summarise_clv`. 40 unit tests in `tests/unit/test_clv.py`.

### `clv_analysis.py` (new)
Click CLI that JOINs `backtest_predictions` + `historical_odds_raw` + `matches`,
computes per-prediction CLV/edge/value_bet, renders 4 Rich tables.

### Migration
Creates `backtest_predictions` with FK to `backtest_runs(backtest_id)`,
denormalized `tour` and `surface` for fast filtering, unique constraint on
`(backtest_run_id, match_id)` for idempotent re-runs.

## Common errors

**`AttributeError: 'Row' object has no attribute 'tournament_level'`**
→ `_iter_matches_in_range` already SELECTs `t.level AS tournament_level`, so
this shouldn't happen. If it does, you're running an old version somehow.

**`column "tour" of relation "backtest_predictions" violates not-null`**
→ A prediction was generated with `tour=None`. Look at how `row.tour` is set
in `_iter_matches_in_range` — should always be set since matches.tour is
NOT NULL.

**`relation "backtest_predictions" does not exist`**
→ Step 2 failed. Re-run the migration.

**CLV analysis says "No predictions JOINable with Pinnacle"**
→ Check Phase 3.2 ran:
```sql
SELECT COUNT(*) FROM historical_odds_raw
WHERE match_id IS NOT NULL AND bookmaker_code = 'PS';
-- should be ~25,000+
```
