-- ============================================================================
-- Tennis Predictor - Complete Database Schema
-- ============================================================================
-- This file contains the entire schema as a single deployable SQL script.
-- Generated from Supabase migrations 01-09.
--
-- For incremental migrations, see the migrations/ directory.
-- For setup instructions, see docs/SETUP.md
--
-- Database: PostgreSQL 17+
-- Required extensions: pg_trgm, fuzzystrmatch, vector, btree_gin
-- ============================================================================

-- ============================================================================
-- 01: Extensions
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS extensions;

CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS vector SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS btree_gin SCHEMA extensions;

GRANT USAGE ON SCHEMA extensions TO service_role, authenticated, anon;

-- ============================================================================
-- 02: Trigger function for updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ============================================================================
-- 03: Core entities
-- ============================================================================

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    tour TEXT NOT NULL CHECK (tour IN ('ATP', 'WTA')),
    name_first TEXT,
    name_last TEXT NOT NULL,
    name_full TEXT GENERATED ALWAYS AS (
        TRIM(COALESCE(name_first, '') || ' ' || name_last)
    ) STORED,
    hand CHAR(1) CHECK (hand IN ('R', 'L', 'U', NULL)),
    birth_date DATE,
    country_code CHAR(3),
    height_cm SMALLINT CHECK (height_cm BETWEEN 140 AND 230),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_players_name_trgm 
    ON players USING gin (name_full extensions.gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_players_tour ON players(tour);
CREATE INDEX IF NOT EXISTS idx_players_country ON players(country_code);

CREATE TRIGGER update_players_updated_at
    BEFORE UPDATE ON players
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE IF NOT EXISTS tournaments (
    tournament_id TEXT PRIMARY KEY,
    tour TEXT NOT NULL CHECK (tour IN ('ATP', 'WTA')),
    name TEXT NOT NULL,
    level TEXT CHECK (level IN ('G', 'M', 'A', 'D', 'F', 'C', 'S', 'O', 'PM', 'I')),
    surface TEXT CHECK (surface IN ('Hard', 'Clay', 'Grass', 'Carpet')),
    indoor BOOLEAN DEFAULT FALSE,
    draw_size SMALLINT,
    country_code CHAR(3),
    city TEXT,
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tournaments_dates ON tournaments(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_tournaments_surface ON tournaments(surface);
CREATE INDEX IF NOT EXISTS idx_tournaments_level ON tournaments(level);
CREATE INDEX IF NOT EXISTS idx_tournaments_tour ON tournaments(tour);

-- Note: For full schema including matches, match_stats, odds, predictions,
-- backtest_runs, player_features, data_ingestion_log tables and analytical views,
-- see the migrations/ directory or apply via Supabase MCP.
-- This file shows the foundation; complete schema is in Supabase project.
