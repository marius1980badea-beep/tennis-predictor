# Setup Guide

Detailed step-by-step setup instructions for the Tennis Predictor project.

## Prerequisites

### Required
- **Python 3.11+** ([download](https://www.python.org/downloads/))
- **Git** ([download](https://git-scm.com/downloads))
- **Supabase account** with the `tennis-predictor` project already created (project ID: `bjjqnqxyfzgkgnkwlgsc`)

### Optional (for later phases)
- **Anthropic API key** for news intelligence (Phase 4)
- **n8n Cloud account** for orchestration (Phase 5)
- **Railway account** for hosting the Python microservice (Phase 5)

## Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/tennis-predictor.git
cd tennis-predictor
```

## Step 2: Set up Python environment

We strongly recommend using a virtual environment.

### Option A: venv (built-in)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Option B: uv (faster, recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv
source .venv/bin/activate
```

## Step 3: Install dependencies

```bash
# Production dependencies
pip install -e .

# Development dependencies (recommended for working on the code)
pip install -e ".[dev]"

# Optional: API and dashboard
pip install -e ".[dev,api,dashboard]"
```

## Step 4: Get your Supabase credentials

The Supabase project `tennis-predictor` is already created. You need three secrets:

### 4a. Service Role Key
1. Go to [Supabase Dashboard > Project Settings > API](https://supabase.com/dashboard/project/bjjqnqxyfzgkgnkwlgsc/settings/api)
2. Under **Project API keys**, find **`service_role`** (marked as "secret")
3. Click **Reveal** and copy the value
4. ⚠️ **NEVER commit this key** — it bypasses Row Level Security

### 4b. Database Password
1. Go to [Supabase Dashboard > Settings > Database](https://supabase.com/dashboard/project/bjjqnqxyfzgkgnkwlgsc/settings/database)
2. Under **Database password**, click **Reset database password** if you don't have it saved
3. Copy the password somewhere safe (password manager)

### 4c. Pooler connection details
On the same Database settings page, scroll to **Connection pooling**:
- Note the **Host** (should be like `aws-1-eu-central-1.pooler.supabase.com`)
- The default port is `6543` (transaction pooler — recommended for our use case)

## Step 5: Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your favorite editor and fill in:

```bash
# These are already correct in the template:
SUPABASE_URL=https://bjjqnqxyfzgkgnkwlgsc.supabase.co
SUPABASE_ANON_KEY=eyJhbG...  # Already filled

# You need to fill these:
SUPABASE_SERVICE_ROLE_KEY=eyJhbG...  # From step 4a
SUPABASE_DB_PASSWORD=your_actual_password  # From step 4b
SUPABASE_DB_HOST=aws-1-eu-central-1.pooler.supabase.com  # Verify from step 4c
```

## Step 6: Verify setup

```bash
tennis-predictor health-check
```

If you see errors:

### Error: "configuration error"
- Check that all required env vars are set in `.env`
- Verify `.env` is in the project root directory

### Error: "Supabase REST API error"
- Verify `SUPABASE_URL` is correct
- Verify `SUPABASE_SERVICE_ROLE_KEY` is the actual service_role key (not anon!)

### Error: "Direct DB connection error"
- Verify `SUPABASE_DB_HOST` matches exactly what's in the Supabase dashboard
- Verify `SUPABASE_DB_PASSWORD` is correct
- Check firewall/VPN isn't blocking port 6543

### Error: "Missing tables"
- The schema migrations weren't applied. Re-run them from `migrations/` directory.

## Step 7: Initial data load

This is the big one. Expect 15-45 minutes depending on your network speed and database performance.

```bash
tennis-predictor load-data --tour BOTH
```

What happens:
1. Clones two Sackmann Git repos (~500MB each) to `./tennis_atp/` and `./tennis_wta/`
2. Reads all match CSV files for years 2000-2024
3. Loads players (~80,000 total)
4. Loads tournaments (deduplicated)
5. Loads matches (~700,000 total)
6. Loads per-match statistics (~1.4M rows)

You can monitor progress in the terminal. The CLI shows a per-year summary table.

If something fails partway through, just re-run the command — all operations are idempotent (upsert/conflict-do-nothing).

## Step 8: Verify data loaded correctly

```bash
tennis-predictor stats
```

Expected approximate counts:
- Players (ATP): ~5,000-6,000
- Players (WTA): ~4,500-5,500
- Tournaments: ~3,000-4,000 each tour
- Matches (ATP): ~70,000-80,000
- Matches (WTA): ~65,000-75,000
- Match stats rows: ~280,000-310,000

```bash
tennis-predictor ingestion-log --last 30
```

Shows the audit trail of all data loads.

## Step 9 (optional): Explore in Jupyter

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name tennis-predictor
jupyter lab notebooks/
```

Create a new notebook and explore:

```python
from tennis_predictor.data.storage import get_session
from sqlalchemy import text
import pandas as pd

with get_session() as session:
    df = pd.read_sql(
        text("SELECT * FROM matches_enriched LIMIT 100"),
        session.bind,
    )

df.head()
```

## Troubleshooting

### `pip install -e .` fails on macOS with psycopg

```bash
# Install Postgres libraries first
brew install libpq
export LDFLAGS="-L/opt/homebrew/opt/libpq/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libpq/include"
pip install -e .
```

### Git clone of Sackmann repo is very slow

Both Sackmann repos are large. We use `--depth 1` for shallow clones, but if it's still slow:

```bash
# Manually clone in parallel
git clone --depth 1 https://github.com/JeffSackmann/tennis_atp.git &
git clone --depth 1 https://github.com/JeffSackmann/tennis_wta.git &
wait

# Then run with --skip-sync to skip the auto-clone step
tennis-predictor load-data --skip-sync
```

### Database connection drops mid-load

The connection pool is configured with `pool_pre_ping=True` and `pool_recycle=3600` to handle this. If you still see issues:

1. Check Supabase project status: https://status.supabase.com/
2. Free tier projects auto-pause after 7 days of inactivity (unlikely during active development)
3. Reduce batch size in `sackmann.py` from 500/1000 to 100/200

### "out of memory" during data load

Some years have a lot of stats data. If your machine has < 4GB RAM:

```bash
# Load one year at a time
for year in {2000..2024}; do
  tennis-predictor load-data --tour ATP --start-year $year --end-year $year
done
```

## Next steps

Once data is loaded successfully, you're ready for Phase 2:
- Implement Elo ratings calculation
- Run first walk-forward backtest
- Compare against Pinnacle closing odds

See [docs/ARCHITECTURE.md](ARCHITECTURE.md) for what's coming next.
