-- ============================================================================
-- Phase 3.3: per-prediction storage for backtest runs
--
-- The existing `backtest_runs` table only stores aggregate metrics (accuracy,
-- log loss, Brier, etc.). For CLV analysis we need each individual prediction
-- so we can JOIN with `historical_odds_raw` on match_id and compute:
--   - CLV = (predicted_prob - pinnacle_implied_prob) / pinnacle_implied_prob
--   - Edge = predicted_prob * pinnacle_decimal_odds - 1
--   - Value bet flag (edge >= min_edge AND prob >= min_prob AND odds >= min_odds)
--
-- One row per (backtest_run_id, match_id). The `predicted_prob_winner` is
-- the model's probability assigned to the player who ACTUALLY won. This
-- is the convention used by `metrics.py` for log-loss / Brier calculation.
--
-- The corresponding probability for the actual loser is implicit (1.0 -
-- predicted_prob_winner) for binary outcomes.
-- ============================================================================

CREATE TABLE IF NOT EXISTS backtest_predictions (
  prediction_id    BIGSERIAL PRIMARY KEY,

  -- Backtest run this prediction belongs to. FK so deleting a run cascades.
  backtest_run_id  BIGINT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,

  -- The match being predicted. NULL only for synthetic test fixtures.
  match_id         BIGINT REFERENCES matches(match_id),

  -- Model version that produced this prediction (denormalised for fast filtering)
  model_version    TEXT NOT NULL,

  -- The probability the model assigned to the actual winner (0-1).
  -- E.g. if model predicted P(A wins) = 0.6 and A won, this is 0.6.
  -- If A won but model predicted P(A wins) = 0.3, this is 0.3.
  -- This matches the convention in walk_forward.py for log loss calculation.
  predicted_prob_winner NUMERIC(7, 6) NOT NULL
    CHECK (predicted_prob_winner BETWEEN 0 AND 1),

  -- Whether the model's argmax prediction was correct (predicted_prob_winner > 0.5)
  was_correct      BOOLEAN NOT NULL,

  -- Surface this prediction was made on (denormalised from matches table for
  -- fast surface-level CLV stratification)
  surface          TEXT CHECK (surface IN ('Hard', 'Clay', 'Grass', 'Carpet')),

  -- Tournament level (G/M/A/D/F/C/S/PM/I/O) - denormalised for slicing
  tournament_level TEXT,

  -- When this row was inserted (for audit trail)
  predicted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- One prediction per (run, match) - re-running the same backtest replaces
  UNIQUE (backtest_run_id, match_id)
);

-- Indexes for the analysis queries we'll run --------------------------------

-- Quick filter by model_version (avoids touching backtest_runs)
CREATE INDEX IF NOT EXISTS idx_backtest_predictions_version
  ON backtest_predictions (model_version, backtest_run_id);

-- Foreign-table JOINs from historical_odds_raw use match_id
CREATE INDEX IF NOT EXISTS idx_backtest_predictions_match
  ON backtest_predictions (match_id);

-- Surface-level CLV slicing (analysis often groups by surface)
CREATE INDEX IF NOT EXISTS idx_backtest_predictions_surface
  ON backtest_predictions (backtest_run_id, surface);

-- RLS: backend-only access ---------------------------------------------------
ALTER TABLE backtest_predictions ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE backtest_predictions IS
  'Per-prediction output from walk_forward backtest runs (Phase 3.3). '
  'Used by clv_analysis.py to JOIN with historical_odds_raw and compute '
  'Closing Line Value vs Pinnacle.';

COMMENT ON COLUMN backtest_predictions.predicted_prob_winner IS
  'Probability the MODEL assigned to the player who actually won. Matches '
  'the convention in metrics.py log_loss/Brier calculations.';

COMMENT ON COLUMN backtest_predictions.was_correct IS
  'TRUE iff predicted_prob_winner > 0.5 (model picked the right player at '
  'argmax). This is a derived field but stored for query speed.';
