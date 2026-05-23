-- ============================================================================
-- Phase 3.3: per-prediction storage for backtest runs
--
-- One row per (backtest_run_id, match_id). JOINed with historical_odds_raw
-- in clv_analysis to compute CLV vs Pinnacle.
--
-- v2 corrections (after seeing real walk_forward.py / run_backtest.py):
--   - FK references backtest_runs(backtest_id), NOT a fictional run_id
--   - model_version stored as TEXT (matches model_versions.model_version_id)
--   - tour denormalized (backtest_runs has no tour column; it lives in config JSON)
-- ============================================================================

CREATE TABLE IF NOT EXISTS backtest_predictions (
  prediction_id     BIGSERIAL PRIMARY KEY,

  -- FK to backtest_runs PRIMARY KEY (`backtest_id`, not `run_id`)
  backtest_run_id   BIGINT NOT NULL REFERENCES backtest_runs(backtest_id) ON DELETE CASCADE,

  -- The match being predicted
  match_id          BIGINT REFERENCES matches(match_id),

  -- Model identifier (matches backtest_runs.model_version_id; denormalised
  -- here so analysis queries can filter without joining backtest_runs)
  model_version     TEXT NOT NULL,

  -- Tour denormalised from the match (NOT from backtest_runs, which has
  -- no tour column). Lets us slice CLV analysis without a tournaments JOIN.
  tour              TEXT NOT NULL CHECK (tour IN ('ATP', 'WTA')),

  -- Probability the model assigned to the actual winner (0..1).
  -- This matches BacktestPrediction.p_winner_wins in walk_forward.py.
  predicted_prob_winner NUMERIC(7, 6) NOT NULL
    CHECK (predicted_prob_winner BETWEEN 0 AND 1),

  -- True iff predicted_prob_winner > 0.5 (model picked the actual winner)
  was_correct       BOOLEAN NOT NULL,

  -- Surface (denormalised from matches for fast filtering)
  surface           TEXT CHECK (surface IN ('Hard', 'Clay', 'Grass', 'Carpet')),

  -- Tournament level from tournaments.level (G/M/A/D/F/C/S/PM/I/O)
  tournament_level  TEXT,

  predicted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- One prediction per (run, match). Re-running a backtest is a no-op
  -- via ON CONFLICT DO NOTHING in the insert.
  UNIQUE (backtest_run_id, match_id)
);

-- Indexes for analysis -------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_backtest_predictions_run_tour
  ON backtest_predictions (backtest_run_id, tour);

CREATE INDEX IF NOT EXISTS idx_backtest_predictions_match
  ON backtest_predictions (match_id);

CREATE INDEX IF NOT EXISTS idx_backtest_predictions_surface
  ON backtest_predictions (backtest_run_id, surface);

-- RLS
ALTER TABLE backtest_predictions ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE backtest_predictions IS
  'Per-prediction output from walk_forward backtest runs (Phase 3.3). '
  'Populated by save_backtest_predictions() in walk_forward.py. '
  'JOINed with historical_odds_raw in clv_analysis.py.';

COMMENT ON COLUMN backtest_predictions.predicted_prob_winner IS
  'P(actual winner wins) from the model. Matches BacktestPrediction.p_winner_wins.';
