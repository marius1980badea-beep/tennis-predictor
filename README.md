# 🎾 Tennis Predictor

> An end-to-end tennis match prediction system for sports betting analysis. Combines surface-adjusted Elo ratings with point-level Monte Carlo simulation to identify value bets across multiple betting markets.

[![Tests](https://github.com/YOUR_USERNAME/tennis-predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/tennis-predictor/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚠️ Disclaimer

This software is for **educational and research purposes**. Gambling involves financial risk and may be illegal in your jurisdiction. Past performance does not guarantee future results. Bet only with money you can afford to lose. If gambling becomes a problem, seek professional help.

---

## 🎯 What this does

Tennis Predictor builds probabilistic forecasts for tennis matches using historical data, then identifies bets where the model's probability exceeds the implied probability in bookmaker odds (positive expected value bets).

**Key design decisions:**

- **Point-level simulation**: instead of predicting "who wins?" directly, we model each player's probability of winning a point on their serve, then run 10,000 Monte Carlo simulations. This unlocks ALL derivative markets (match winner, set betting, total games, etc.) from a single base model.
- **Niche market focus**: bookmaker markets like set betting, total games, and underdog wins-a-set are statistically less efficient than the headline match winner market. We target where the edge is biggest.
- **Walk-forward backtesting**: every metric we report is strictly out-of-sample. Models are trained only on data available before each prediction date.
- **Closing Line Value (CLV)**: the primary evaluation metric. If our predictions consistently beat the closing line at sharp books (Pinnacle), we have real edge. Profit can come and go from variance; CLV doesn't lie.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      n8n (Orchestration)                    │
│  Scheduled triggers, data sync, predictions, notifications  │
└────────────┬─────────────────────────┬──────────────────────┘
             │                         │
             ▼                         ▼
   ┌──────────────────┐      ┌──────────────────┐
   │  Anthropic API   │      │  Python Service  │
   │  (News & Quotes) │      │   (Railway)      │
   │  Phase 2         │      │  FastAPI         │
   └──────────────────┘      │  - Features      │
             │               │  - Elo Engine    │
             │               │  - Simulator     │
             │               │  - Backtest      │
             ▼               └──────────────────┘
   ┌──────────────────────────────────┐
   │       Supabase (PostgreSQL)      │
   │   12 tables, 3 views, pgvector   │
   └──────────────────────────────────┘
```

**Tech stack:**
- **Language**: Python 3.11+
- **Database**: Supabase (PostgreSQL 17 + pgvector)
- **Compute**: Railway (FastAPI microservice)
- **Orchestration**: n8n Cloud
- **AI**: Anthropic Claude (Haiku for extraction, Sonnet for analysis)
- **Data**: Jeff Sackmann's tennis datasets, tennis-data.co.uk odds archive

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full deep-dive.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Git
- A Supabase project (one is already set up: `tennis-predictor`)
- Anthropic API key (only needed for Phase 2 news layer)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/tennis-predictor.git
cd tennis-predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

You'll need to fill in:
- `SUPABASE_SERVICE_ROLE_KEY` — get from [Supabase Dashboard > Settings > API](https://supabase.com/dashboard/project/bjjqnqxyfzgkgnkwlgsc/settings/api)
- `SUPABASE_DB_PASSWORD` — get from [Supabase Dashboard > Settings > Database](https://supabase.com/dashboard/project/bjjqnqxyfzgkgnkwlgsc/settings/database)
- `ANTHROPIC_API_KEY` — get from [Anthropic Console](https://console.anthropic.com/) (only for Phase 2)

### 3. Verify setup

```bash
tennis-predictor health-check
```

Expected output:
```
✓ Configuration loaded
✓ Supabase REST API reachable
✓ Direct DB connection OK
✓ All 12 required tables present
All checks passed! System is ready.
```

### 4. Load initial data

This is a one-time operation that takes 15-45 minutes:

```bash
# Load both ATP and WTA, 2000-2024 (default range from .env)
tennis-predictor load-data

# Or load only ATP for a quicker test
tennis-predictor load-data --tour ATP --start-year 2020 --end-year 2024
```

### 5. Verify data loaded

```bash
tennis-predictor stats
```

You should see something like:
```
Players (ATP):           ~5,000
Players (WTA):           ~4,500
Matches (ATP):           ~75,000
Matches (WTA):           ~70,000
Match stats rows:        ~290,000
```

---

## 📋 CLI Commands

| Command | Description |
|---------|-------------|
| `tennis-predictor health-check` | Verify all services and tables |
| `tennis-predictor load-data [options]` | Load Sackmann tennis data |
| `tennis-predictor stats` | Show database statistics |
| `tennis-predictor ingestion-log [--last N]` | Show recent data ingestion log |

Full options: `tennis-predictor COMMAND --help`

---

## 🧪 Testing

```bash
# Run all unit tests (fast, no DB needed)
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov

# Run integration tests (requires DB connection)
pytest tests/integration -v -m integration

# Run only anti-leakage tests (critical for backtest validity)
pytest tests -v -m anti_leakage
```

---

## 📁 Project Structure

```
tennis-predictor/
├── src/tennis_predictor/
│   ├── config.py              # Pydantic Settings (env vars)
│   ├── logging_config.py      # Structured logging (structlog)
│   ├── cli.py                 # Click CLI entry point
│   ├── data/
│   │   ├── loaders/           # Data ingestion (Sackmann, odds, news)
│   │   ├── validators/        # Pandera schemas (future)
│   │   └── storage/           # Supabase client + SQLAlchemy
│   ├── features/              # Feature engineering (Phase 2)
│   ├── models/                # Elo, simulator, ML models (Phase 2)
│   ├── backtest/              # Walk-forward engine (Phase 2)
│   ├── decision/              # Pick selection, Kelly staking (Phase 3)
│   ├── pipelines/             # End-to-end workflows (Phase 3)
│   └── interface/             # CLI, dashboard (Streamlit) (Phase 4)
│
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── unit/                  # Fast tests, no external deps
│   └── integration/           # Tests requiring DB
│
├── scripts/                   # Standalone scripts
│   └── initial_data_load.py   # First-time data load
│
├── migrations/                # SQL migrations (source of truth)
├── sql/                       # Reference SQL queries
├── notebooks/                 # Jupyter notebooks for exploration
├── docs/                      # Architecture, setup, decisions
│
├── pyproject.toml             # Project metadata + dependencies
├── .env.example               # Template for environment variables
└── README.md                  # This file
```

---

## 🛣️ Roadmap

### ✅ Phase 1: Foundation (current)
- [x] Supabase schema (12 tables, 3 views, RLS, indexes)
- [x] Sackmann data loader (ATP + WTA, 2000-2024)
- [x] CLI tooling (health-check, load-data, stats)
- [x] Unit tests for transformations
- [x] CI/CD with GitHub Actions

### 🚧 Phase 2: First Predictions (next 2-3 weeks)
- [ ] Surface-adjusted Elo engine
- [ ] First walk-forward backtest (Elo baseline)
- [ ] Historical odds ingestion (tennis-data.co.uk)
- [ ] CLV evaluation vs. Pinnacle

### 📅 Phase 3: Point-Level Simulator
- [ ] Per-player serve win probability model (LightGBM)
- [ ] Monte Carlo match simulator (vectorized NumPy)
- [ ] Multi-market predictions (match winner, set betting, total games)
- [ ] Probability calibration

### 📅 Phase 4: News Intelligence
- [ ] News crawling (RSS feeds, multilingual)
- [ ] Claude-based fact extraction (injuries, coach changes, personal events)
- [ ] News features integration into model
- [ ] Pre-match risk filtering

### 📅 Phase 5: Production
- [ ] Railway deployment (FastAPI service)
- [ ] n8n daily prediction workflow
- [ ] Streamlit dashboard
- [ ] Telegram alerts for picks

---

## 📚 Key References

**Data sources:**
- [Jeff Sackmann's tennis_atp](https://github.com/JeffSackmann/tennis_atp) — ATP match data
- [Jeff Sackmann's tennis_wta](https://github.com/JeffSackmann/tennis_wta) — WTA match data
- [tennis-data.co.uk](http://www.tennis-data.co.uk/alldata.php) — Historical odds from multiple bookmakers

**Methodological inspiration:**
- Klaassen & Magnus (2003) — "Forecasting the winner of a tennis match"
- Barnett & Clarke (2005) — "Combining player statistics to predict outcomes of tennis matches"
- Kovalchik (2016) — "Searching for the GOAT of tennis win prediction"
- Dixon & Coles (1997) — "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
- Various academic work on Closing Line Value as betting performance metric

---

## 🤝 Contributing

This is currently a personal/research project. If you have suggestions or find bugs, please open an issue.

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details, including a gambling disclaimer.
