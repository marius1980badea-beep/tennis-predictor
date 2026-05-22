# First Steps — 2-Minute Setup

This is the absolute minimum to get the project on GitHub and running locally.

## Step 1: Create the GitHub repo (30 seconds)

Go to https://github.com/new

- **Repository name**: `tennis-predictor`
- **Description**: `Tennis match prediction system with point-level Monte Carlo simulation`
- **Public**
- ⚠️ **DO NOT** check "Initialize with README", "Add .gitignore", or "Add license" — we already have these

Click **Create repository**.

## Step 2: Push the code (30 seconds)

Extract the ZIP I gave you, then:

```bash
cd tennis-predictor
git init
git add .
git commit -m "Initial commit: project foundation with Supabase schema and Sackmann loader"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tennis-predictor.git
git push -u origin main
```

Done — repo is live.

## Step 3: Configure environment (60 seconds)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate   # Windows

# Install
pip install -e ".[dev]"

# Set up config
cp .env.example .env
```

Open `.env` and fill in two secrets (the rest is pre-filled):
- `SUPABASE_SERVICE_ROLE_KEY` from https://supabase.com/dashboard/project/bjjqnqxyfzgkgnkwlgsc/settings/api
- `SUPABASE_DB_PASSWORD` from https://supabase.com/dashboard/project/bjjqnqxyfzgkgnkwlgsc/settings/database

## Step 4: Verify (10 seconds)

```bash
tennis-predictor health-check
```

If all green ✓ — you're ready to load data.

## Step 5: Load the data (15-45 minutes — go grab coffee)

```bash
tennis-predictor load-data
```

This loads ATP + WTA, 2000-2024. Watch the progress in your terminal.

## Next

Once data is in, see [docs/SETUP.md](docs/SETUP.md) for detailed exploration and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what comes next (Elo + backtest).
