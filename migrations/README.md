# Database Migrations

These SQL files are the source of truth for the database schema.
They were applied to the Supabase project `tennis-predictor` (id: `bjjqnqxyfzgkgnkwlgsc`) on 2026-05-22.

## Applying migrations

### Option 1: Via Supabase Dashboard
1. Go to Supabase Dashboard > SQL Editor
2. Run each file in numerical order (01, 02, 03, ...)

### Option 2: Via Supabase CLI
```bash
supabase db push
```

### Option 3: Via psql
```bash
for f in migrations/*.sql; do
  psql "$DATABASE_URL" -f "$f"
done
```

## Migrations applied

| # | Name | Purpose |
|---|------|---------|
| 01 | enable_extensions | Enable pg_trgm, fuzzystrmatch, vector, btree_gin |
| 02 | core_entities_players_tournaments | Players + tournaments tables |
| 03 | matches_and_stats | Matches + per-player stats |
| 04 | historical_odds | Bookmakers + odds tables |
| 05 | elo_ratings_and_predictions | Elo ratings + model versions + predictions |
| 06 | backtest_and_features | Backtest runs + features cache + ingestion log |
| 07 | analytical_views | Convenience views |
| 08 | row_level_security | Enable RLS on all tables |
| 09 | fix_security_advisors | security_invoker on views, search_path on funcs, move extensions |

## Notes

- All tables use Row Level Security (RLS) enabled with no policies = "deny all" 
  for anon/authenticated. Service role (backend) bypasses RLS automatically.
- Views use `security_invoker = true` (run with caller's permissions, not creator's).
- Extensions are installed in `extensions` schema, not `public`.
