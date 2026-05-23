-- ============================================================================
-- Phase 3.1: staging table for historical odds from tennis-data.co.uk
--
-- One row per (match, bookmaker) combination, captured exactly as parsed from
-- the source Excel file. Player names are raw strings ("Djokovic N."). Phase
-- 3.2 will fuzzy-match these against our `matches`/`players` tables and
-- populate `match_id`. Phase 3.3+ will then JOIN this table when computing CLV.
--
-- We keep this as a staging layer (rather than inserting into `historical_odds`
-- directly) so that we can iterate on the fuzzy-matching algorithm without
-- re-downloading and re-parsing the raw Excel data.
-- ============================================================================

CREATE TABLE IF NOT EXISTS historical_odds_raw (
  raw_id BIGSERIAL PRIMARY KEY,

  -- Source provenance ---------------------------------------------------------
  source           TEXT       NOT NULL DEFAULT 'tennis-data.co.uk',
  source_year      SMALLINT   NOT NULL CHECK (source_year BETWEEN 2000 AND 2100),
  tour             TEXT       NOT NULL CHECK (tour IN ('ATP', 'WTA')),

  -- Match identity (raw strings, to be fuzzy-matched in Phase 3.2) ------------
  match_date       DATE       NOT NULL,
  tournament_name  TEXT       NOT NULL,
  series_or_tier   TEXT,                    -- "Series" for ATP, "Tier" for WTA
  court            TEXT,                    -- Outdoor / Indoor
  surface          TEXT,                    -- Hard / Clay / Grass / Carpet
  round            TEXT,
  best_of          SMALLINT,
  winner_name      TEXT       NOT NULL,     -- e.g. "Djokovic N."
  loser_name       TEXT       NOT NULL,
  winner_rank      INTEGER,
  loser_rank       INTEGER,
  comment          TEXT,                    -- "Completed" / "Retired" / "Walkover"

  -- Bookmaker odds (one row per bookmaker per match) -------------------------
  bookmaker_code        TEXT          NOT NULL REFERENCES bookmakers(bookmaker_code),
  winner_odds           NUMERIC(8, 3) NOT NULL CHECK (winner_odds >= 1.0),
  loser_odds            NUMERIC(8, 3) NOT NULL CHECK (loser_odds  >= 1.0),
  winner_implied_prob   NUMERIC(7, 6) NOT NULL CHECK (winner_implied_prob BETWEEN 0 AND 1),
  loser_implied_prob    NUMERIC(7, 6) NOT NULL CHECK (loser_implied_prob  BETWEEN 0 AND 1),
  vig                   NUMERIC(7, 6) NOT NULL,
  -- Vig can be slightly negative for the "MAX" pseudo-bookmaker (arb across
  -- multiple soft books). No CHECK constraint to allow this.

  -- Match linkage (populated in Phase 3.2) -----------------------------------
  match_id          BIGINT REFERENCES matches(match_id),
  matched_at        TIMESTAMPTZ,
  match_confidence  NUMERIC(5, 4) CHECK (match_confidence BETWEEN 0 AND 1),

  loaded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Idempotency: re-running the loader for the same year is a no-op on rows
-- already loaded. Uniqueness key matches the natural identifier in the source.
ALTER TABLE historical_odds_raw
  ADD CONSTRAINT uq_historical_odds_raw_natural_key
  UNIQUE (source, source_year, tour, match_date,
          tournament_name, winner_name, loser_name, bookmaker_code);

-- Lookup index for fuzzy matching in Phase 3.2 ------------------------------
CREATE INDEX IF NOT EXISTS idx_historical_odds_raw_match_lookup
  ON historical_odds_raw (tour, match_date, winner_name, loser_name);

-- Partial index for unmatched rows (fast scans during 3.2 matching)
CREATE INDEX IF NOT EXISTS idx_historical_odds_raw_unmatched
  ON historical_odds_raw (tour, source_year)
  WHERE match_id IS NULL;

-- Forward lookup once matching is done
CREATE INDEX IF NOT EXISTS idx_historical_odds_raw_match_id
  ON historical_odds_raw (match_id)
  WHERE match_id IS NOT NULL;

-- Bookmaker scans (e.g. "get all Pinnacle rows for 2023 ATP")
CREATE INDEX IF NOT EXISTS idx_historical_odds_raw_bookmaker
  ON historical_odds_raw (bookmaker_code, tour, source_year);

-- Row Level Security: backend-only access (service_role bypasses RLS) -------
ALTER TABLE historical_odds_raw ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE historical_odds_raw IS
  'Staging table for tennis-data.co.uk historical odds (Phase 3.1). '
  'Phase 3.2 populates match_id via fuzzy matching against `matches`. '
  'Phase 3.3 JOINs this table to compute CLV vs model predictions.';

COMMENT ON COLUMN historical_odds_raw.bookmaker_code IS
  'PS=Pinnacle (benchmark), B365=Bet365, plus 9 other books. '
  'Also MAX (best across books) and AVG (mean across books) as pseudo-bookmakers.';

COMMENT ON COLUMN historical_odds_raw.match_confidence IS
  'Fuzzy match score (0-1) from pg_trgm similarity, populated in Phase 3.2.';
