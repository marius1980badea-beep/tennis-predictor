# Commit 3.3 — CLV Calculator + Value Bet Detection

**Goal**: Determine retrospectively whether the Elo model has commercial edge
by comparing every backtest prediction to its Pinnacle closing line.

This commit delivers ~80% of Phase 3.3:
- ✅ CLV math (pure functions, 40 unit tests)
- ✅ Value bet detection logic
- ✅ Schema migration for `backtest_predictions` table
- ✅ Analysis CLI (reads from DB, produces Rich reports + optional CSV)
- ⏳ **Persistence integration**: small modification needed in your
      `walk_forward.py` to populate `backtest_predictions`. **See section
      "Integration with walk_forward" below.**

---

## Files in ZIP

| Path | Action |
|---|---|
| `migrations/011_create_backtest_predictions.sql` | NEW migration |
| `src/tennis_predictor/backtest/clv.py` | NEW pure math |
| `src/tennis_predictor/backtest/clv_analysis.py` | NEW analysis CLI |
| `tests/unit/test_clv.py` | NEW 40 tests |
| `COMMIT_3.3_README.md` | This file |

---

## Integration steps

### 1) Apply migration

```sql
-- in Supabase SQL editor, or psql:
\i migrations/011_create_backtest_predictions.sql
```

Verify:
```sql
\d backtest_predictions
SELECT COUNT(*) FROM backtest_predictions;  -- should be 0
```

### 2) Copy Python files into the repo

Standard locations:
```
src/tennis_predictor/backtest/clv.py            (NEW)
src/tennis_predictor/backtest/clv_analysis.py   (NEW)
tests/unit/test_clv.py                          (NEW)
```

### 3) Run unit tests

```cmd
.venv\Scripts\activate.bat
python -m pytest tests/unit/test_clv.py -v
```

Expected: **40 passed**.

### 4) Integration with walk_forward (manual step needed)

The schema for `backtest_predictions` table expects per-prediction rows. The
existing `walk_forward.py` produces predictions internally but only stores
aggregate metrics in `backtest_runs`. You need to modify it to also save the
individual predictions.

**The minimal change**: inside your walk-forward main loop, after each
prediction is computed for a match, append a row to a list. After the loop
ends (just before computing aggregate metrics), do a bulk INSERT into
`backtest_predictions`.

#### What `walk_forward.py` needs to expose per prediction

Each prediction should produce a dictionary like:

```python
{
    "backtest_run_id":       <run_id from backtest_runs>,
    "match_id":              <match_id from matches>,
    "model_version":         "elo_v1_surface",
    "predicted_prob_winner": 0.6234,   # P(actual winner) per model
    "was_correct":           True,     # i.e. predicted_prob_winner > 0.5
    "surface":               "Hard",
    "tournament_level":      "G",      # from matches.level
}
```

**Critical convention**: `predicted_prob_winner` is the probability the
model assigned to the player who ACTUALLY won. This matches how
`metrics.py` computes log loss. If your model predicts P(player A wins) and
B wins, `predicted_prob_winner = 1 - P(A wins)`.

#### Suggested code pattern

```python
# in walk_forward.py, somewhere in the main loop:

predictions_to_insert = []
for match in matches_in_test_period:
    # ... existing Elo update + prediction logic ...
    p_winner = compute_predicted_prob_for_winner(match, current_elo_state)

    predictions_to_insert.append({
        "match_id":              match.match_id,
        "model_version":         model_version,
        "predicted_prob_winner": p_winner,
        "was_correct":           p_winner > 0.5,
        "surface":               match.surface,
        "tournament_level":      match.level,
    })

# After loop, bulk insert (set backtest_run_id once you know it)
def save_predictions(engine, run_id, rows):
    if not rows:
        return
    for r in rows:
        r["backtest_run_id"] = run_id
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO backtest_predictions (
                backtest_run_id, match_id, model_version,
                predicted_prob_winner, was_correct,
                surface, tournament_level
            ) VALUES (
                :backtest_run_id, :match_id, :model_version,
                :predicted_prob_winner, :was_correct,
                :surface, :tournament_level
            )
            ON CONFLICT (backtest_run_id, match_id) DO NOTHING
        """), rows)
```

The `ON CONFLICT DO NOTHING` makes re-runs idempotent. Inserts ~40k rows in
under 30 seconds via Supabase pooler.

#### Re-run the backtest

```cmd
python -m tennis_predictor.backtest.run_backtest \
  --tour ATP --train-start 2000-01-01 --test-start 2011-01-01 \
  --test-end 2024-12-31 --save
```

After this, `SELECT COUNT(*) FROM backtest_predictions` should match the
"39,245 predictions evaluated" from your blueprint.

### 5) Run CLV analysis

```cmd
python -m tennis_predictor.backtest.clv_analysis --tour ATP
```

Or for a specific run:
```cmd
python -m tennis_predictor.backtest.clv_analysis --run-id 3
```

With per-prediction CSV export:
```cmd
python -m tennis_predictor.backtest.clv_analysis --tour ATP \
  --output-csv data/clv_atp_v1.csv
```

---

## What the analysis tells you

**Mean CLV** is the headline number. Interpretation:

| Mean CLV | Verdict |
|---|---|
| > +1% | Strong signal of model edge vs Pinnacle |
| 0 to +1% | Marginal positive; statistically promising |
| -1% to 0 | Line-with-market; no edge but also no systematic loss |
| -1% to -3% | Mild negative edge; comparable to soft-book vig |
| < -3% | Model is being out-priced; concerning |

**Realistic expectation for a strong Elo baseline**: mean CLV near zero or
slightly positive on Pinnacle. Elo isn't expected to beat Pinnacle on average
— Pinnacle aggregates sharp action. The real question is whether we have
positive CLV in SPECIFIC SLICES (e.g. clay underdogs, 250-level tournaments,
post-injury favorites) which Faza 4 will exploit.

**Value bet rate** at default criteria (5% edge, 55% prob, 1.60 odds) should
be ~3-7% of predictions for a competent model. Too many (>15%) suggests
model overconfidence. Too few (<1%) suggests model is essentially mirror of
Pinnacle.

---

## Output structure

The CLI produces 4 Rich tables:

1. **Overall** — n, mean/median CLV, % positive, value bet count
2. **By Surface** — Hard / Clay / Grass / Carpet breakdown
3. **By Level** — G(Grand Slam), M(Masters), A(500), D(250), etc.
4. **By Year** — 2011-2024 time series

Optional CSV (`--output-csv`) has 12 columns including per-row CLV, edge,
is_value_bet flag — useful for further analysis in Pandas / Excel.

---

## Test coverage

```
40 passed in 0.06s
```

Categories:
- **implied_prob** (5) — odds-to-probability conversion
- **compute_clv** (8) — perfect agreement, positive, negative, edge cases
- **compute_edge** (6) — fair markets, value, negative edge
- **ValueBetCriteria** (6) — validation of threshold inputs
- **is_value_bet** (7) — all three filters individually + combined
- **summarise_clv** (6) — aggregation, edge cases
- **Consistency** (2) — CLV and edge agree at known vig levels

---

## Known limitations

1. **Only PS (Pinnacle) used as benchmark**. Could be extended to MAX
   (best-across-books) or AVG (mean-of-books) in a future iteration.

2. **Player roles**: we currently match `predicted_prob_winner` against
   `pinnacle_winner_implied_prob`. Both refer to the actual winner of the
   match, so the comparison is apples-to-apples. But if your walk_forward
   for any reason swaps the convention, the sign of CLV would flip. The
   migration's column comments document the expected convention to prevent
   this drift.

3. **Vig is NOT subtracted from CLV**. CLV here is the raw probability
   gap. If you want "true" model accuracy adjusted for Pinnacle's margin,
   subtract `vig / 2` (since vig is split symmetrically between sides).

4. **No bankroll simulation** (yet). That's Commit 3.4: hypothetical ROI
   with flat / Kelly staking, drawdown analysis, etc.

---

## Next: Commit 3.4

Once 3.3 runs cleanly and you have a verdict on mean CLV:
- Simulate flat staking, fractional Kelly 1/4, Kelly 1/8 on value bets only
- Compute ROI per surface, year, tournament level
- Max drawdown, Sharpe / Sortino ratio
- The hard-truth answer to "would this have made money 2011-2024?"

---

**Effort estimate from kickoff: ~25 min.**
**Real**: ~40 min for code + tests. Integration into walk_forward is
your work (~15-30 min depending on how walk_forward is structured).
