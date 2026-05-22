# Architecture

This document describes the technical architecture of Tennis Predictor, design decisions, and the rationale behind them.

## System Overview

Tennis Predictor is a multi-component system that ingests tennis match data, computes probabilistic forecasts, and identifies value bets across multiple bookmaker markets.

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (External)                       │
│  Sackmann GitHub │ tennis-data.co.uk │ News RSS │ Live Odds API  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    n8n (Orchestration Layer)                     │
│  Schedule triggers │ HTTP requests │ Anthropic API calls         │
│  Webhook receivers │ Conditional routing │ Notifications         │
└──────────────────────────────────────────────────────────────────┘
                  │                            │
                  ▼                            ▼
       ┌──────────────────┐          ┌─────────────────────┐
       │  Anthropic API   │          │  Python Service     │
       │  - Haiku 4.5     │          │  (Railway/FastAPI)  │
       │  - Sonnet 4.6    │          │  - Feature engineer │
       │  News extraction │          │  - Elo engine       │
       │  Risk analysis   │          │  - Monte Carlo sim  │
       └──────────────────┘          │  - ML models        │
                  │                  │  - Backtest         │
                  │                  └─────────────────────┘
                  │                            │
                  ▼                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Supabase (PostgreSQL 17)                        │
│                                                                  │
│   Core:     players, tournaments, matches, match_stats           │
│   Odds:     bookmakers, historical_odds                          │
│   Models:   elo_ratings, model_versions, predictions             │
│   Backtest: backtest_runs, player_features                       │
│   Future:   news_articles, news_facts, player_status             │
│   Audit:    data_ingestion_log                                   │
│                                                                  │
│   Extensions: pg_trgm (fuzzy match), pgvector (embeddings)       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  ┌─────────────────────┐
                  │  Interfaces         │
                  │  - CLI (Click)      │
                  │  - Streamlit dash   │
                  │  - Telegram bot     │
                  │  - Email alerts     │
                  └─────────────────────┘
```

## Why this stack?

### Why Supabase?
- **Managed PostgreSQL** with backups, monitoring, auth out of the box
- **pgvector** built-in for future news embeddings
- **Free tier** sufficient for development phase (500MB DB)
- **REST + GraphQL APIs** auto-generated (useful for dashboard later)
- **Easy migration path** to Pro tier ($25/mo) when we need more

### Why n8n?
- **Visual workflow editor** — easier to modify than YAML/code
- **Schedule triggers** built-in (no separate cron service)
- **HTTP nodes** for any API
- **Native Anthropic integration** (when we get to news layer)
- **Persistence** of workflow runs (debug failures easily)
- Already in your stack — no additional cost

### Why Railway for Python service?
- **GitHub integration** — push to deploy
- **Persistent volumes** for cached data
- **No serverless timeouts** (backtests take 5-30 minutes)
- **Pay-per-use** pricing (~$5-15/mo for our workload)
- **Vertical auto-scaling** when running heavy backtests

### Why Anthropic Claude (vs OpenAI)?
- **Better at structured extraction** from messy news text (subjective but consistent in benchmarks)
- **Haiku model** is cheap enough to scan 50-100 articles/day at ~$0.50/day
- **Sonnet 4.6** for ambiguous cases provides excellent reasoning
- Already in your stack

## Data Flow

### Initial bulk load (one-time)

```
Sackmann CSVs ──► sackmann.py loader ──► Supabase
                  - Parse YYYYMMDD dates
                  - Normalize surface/level codes
                  - Build player IDs with tour prefix
                  - Validate winner != loser
                  - Detect RET/W/O in scores
                  - Upsert with conflict handling
```

### Daily prediction pipeline (Phase 5)

```
n8n Schedule (06:00)
   │
   ├─► Python service: fetch_fixtures_today
   │   └─► For each upcoming match:
   │       ├─► Compute features (anti-leakage strict)
   │       ├─► Run point-level simulator
   │       ├─► Generate predictions for all markets
   │       └─► Apply value filter (edge >= 7%, prob >= 55%)
   │
   ├─► n8n: fetch news (last 2h)
   │   ├─► Send articles to Claude Haiku for extraction
   │   └─► Update player_status table
   │
   ├─► Python service: apply_news_filter
   │   └─► Demote/eliminate picks with active concerns
   │
   ├─► Sort picks by edge × confidence
   ├─► Select top 5-10 picks
   ├─► Compute fractional Kelly stakes
   │
   └─► n8n: send Telegram alert with picks
```

## Key Design Decisions

### 1. Point-level simulation over direct match prediction

**Decision**: Model `P(winning a point on serve)` per player, then run Monte Carlo simulation for the full match.

**Rationale**:
- One model unlocks ALL markets (match winner, set betting, total games, etc.)
- Niche markets are less efficient → bigger edges
- Distribution outputs allow proper uncertainty quantification
- Backtest sample size multiplies (e.g. 15 markets × 700k matches = ~10M predictions)

**Alternative considered**: Separate models per market.
**Why rejected**: Massive duplication of effort, no shared signal, hard to calibrate.

### 2. Walk-forward backtest, never random split

**Decision**: At each prediction point in history, train ONLY on data strictly before that date.

**Rationale**: Random k-fold cross-validation leaks future information through aggregate statistics and model features. Real-world deployment only has past data; backtests must simulate this.

**Implementation**: Re-train monthly, predict next month. ~150 retraining points across 2011-2024.

### 3. Closing Line Value (CLV) as primary metric

**Decision**: Evaluate models against closing line at Pinnacle, not just final P&L.

**Rationale**: 
- Variance is huge in short term — a bad model can be profitable for 200 bets
- CLV is forward-looking: if you consistently get better prices than the eventual efficient line, you have edge
- Standard metric in professional betting

**Implementation**: For every prediction, store both opening odds (when we'd bet) and closing odds, then compute `CLV = (placed_odds - closing_odds) / closing_odds`.

### 4. Surface-adjusted Elo as baseline (not direct ML)

**Decision**: Start with Elo per surface, only add ML when it provably beats Elo.

**Rationale**: 
- Elo is interpretable, well-understood, robust
- Provides a strong baseline (~68-70% accuracy alone)
- Easy to debug when something looks wrong
- ML models that don't beat Elo aren't worth the complexity

**Implementation**: Separate Elo per surface (Hard/Clay/Grass), weighted blend with overall Elo for new players or surface switches.

### 5. Anti-leakage testing as first-class concern

**Decision**: Every feature has a unit test that verifies it doesn't use future data.

**Rationale**: Data leakage is the #1 cause of "great backtest, terrible live performance". It's subtle and easy to introduce accidentally. Tests catch it.

**Implementation**: 
- Each Feature class has `test_no_leakage()` method (mandatory)
- CI/CD blocks PRs that don't include leakage tests for new features
- Marker `@pytest.mark.anti_leakage` for explicit identification

### 6. Service role only for backend access

**Decision**: RLS enabled with no policies (default deny), service_role bypasses RLS.

**Rationale**: 
- No risk of accidental data exposure via dashboard or public APIs
- Backend is the only entry point to data (centralizes access logic)
- Easy to add read-only policies later if we want public dashboard

**Implementation**: All tables have `ENABLE ROW LEVEL SECURITY` but no policies. Service role (used by Python service and n8n) bypasses RLS by default.

## Performance Considerations

### Database query patterns

The most common queries are:
1. **Player history**: "all matches player X played before date Y" → indexed on `(player_id, match_date)`
2. **Surface filtering**: "all hard court matches in 2023" → indexed on `(surface, match_date)`
3. **Player + surface**: "Nadal's clay matches in last 12 months" → composite index `(player_id, surface, match_date)`

Generated columns (`first_serve_win_pct`, `vig`, etc.) are stored, not computed at query time.

### Bulk operations

For initial data load (~700k matches + ~1.4M stats):
- Use direct SQLAlchemy `INSERT ... ON CONFLICT` (not REST API)
- Batch size 500-1000 rows
- Connection pooling via Supabase transaction pooler (port 6543)

### Monte Carlo simulation

For 10,000 simulations per match × ~20 matches/day = 200k simulations:
- **Vectorized NumPy** (no Python loops)
- Pre-compute probability matrices once per match
- Estimated time: ~50ms per match on Railway shared CPU

## Security

### Secrets management
- All secrets in `.env` (never committed)
- Production secrets in Railway environment variables
- Service role key marked as "secret" in Railway/n8n
- No secrets ever logged

### Database access
- Service role for backend (full access, bypasses RLS)
- Anon/publishable key for any frontend (currently none — defense in depth)
- RLS enabled on all tables (defense in depth)
- Views use `security_invoker = true` (respect caller's permissions)

### Function security
- Trigger functions use `SECURITY INVOKER` (not DEFINER)
- All functions have explicit `SET search_path = ''` (prevent injection)

## Open Questions / Future Decisions

- [ ] **Should we move to dbt for transformations?** Currently SQL is embedded in Python. dbt would be cleaner but adds complexity.
- [ ] **Use MLflow or custom tracking?** Currently planning custom in `model_versions` + `backtest_runs` tables. MLflow has better UI but more moving parts.
- [ ] **When to migrate to Supabase Pro?** ~3-4 months in if we need more storage or want backup PITR.
- [ ] **Real-time predictions for live betting?** Initially no — too much latency competition. Maybe Phase 6.
- [ ] **Multi-region deployment?** If we add Liga 1 Romania, we might want Romanian data closer geographically. EU-central works for both now.

## References

For methodology details and academic background, see [README.md § Key References](../README.md#-key-references).
