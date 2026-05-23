# Commit 3.1 — Loader cote din tennis-data.co.uk

**Goal**: Download historical odds (one Excel/year) from tennis-data.co.uk,
parse them, and stage them into a new DB table `historical_odds_raw`. No
fuzzy matching yet — that's Commit 3.2.

**Validated locally**: 47 unit tests + 1 end-to-end smoke test pass.

---

## Files in this ZIP

| Path in ZIP | Where it goes in repo |
|---|---|
| `migrations/010_create_historical_odds_raw.sql` | `migrations/010_create_historical_odds_raw.sql` |
| `src/tennis_predictor/data/loaders/tennis_data_uk.py` | same path in repo |
| `src/tennis_predictor/data/loaders/load_odds.py`     | same path in repo |
| `tests/unit/test_tennis_data_uk.py`                  | same path in repo |
| `data/odds/.gitignore`                                | same path in repo (gitignores `raw/`) |
| `data/odds/raw/.gitkeep`                              | same path in repo |

> Ignore the file `src/tennis_predictor/data/storage/db.py` in the ZIP — it's
> a sandbox stub I needed to run tests offline. **Do NOT copy it to your repo**;
> the real one already exists there.
> Same for `scratch_smoke_test.py`.

---

## Integration steps (în ordine)

### 1) Aplică migrarea pe Supabase

Migrarea creează tabela `historical_odds_raw` cu:
- Coloane raw pentru match identity (winner_name, loser_name, match_date, etc.)
- Coloane normalizate pentru cote (bookmaker_code, winner_odds, vig, etc.)
- `match_id` nullable + `match_confidence` (umplute în Commit 3.2)
- Unique constraint pe (source, year, tour, date, tournament, winner, loser, bookmaker) → re-runs idempotente
- 4 indecși (lookup pentru fuzzy match, partial pentru unmatched, forward pentru match_id, scan pe bookmaker)
- RLS enabled (backend-only via service_role)

```bash
# Verifică prima dată cu un dry-run / inspect
psql $DATABASE_URL -f migrations/010_create_historical_odds_raw.sql --dry-run

# Apply (sau via Supabase dashboard SQL editor)
psql $DATABASE_URL -f migrations/010_create_historical_odds_raw.sql
```

Sanity check după aplicare:
```sql
\d historical_odds_raw
SELECT COUNT(*) FROM historical_odds_raw;  -- should be 0
```

### 2) Copiază codul Python în repo

```cmd
cd "C:\Users\Marius Badea\Documents\GitHub\tennis-predictor"
# Copy these files from the ZIP (NOT the storage/db.py stub):
#   src/tennis_predictor/data/loaders/tennis_data_uk.py
#   src/tennis_predictor/data/loaders/load_odds.py
#   tests/unit/test_tennis_data_uk.py
# Plus the .gitignore + .gitkeep under data/odds/
```

### 3) Verifică dependențele

Dacă nu sunt deja în `pyproject.toml`, adaugă:
- `openpyxl` (pentru .xlsx, 2013+)
- `xlrd` (pentru .xls, 2001-2012). **Notă**: `xlrd >= 2.0` nu mai citește `.xls`.
   Folosește `xlrd==1.2.0` sau o alternativă (`pyxlsb`, `aspose-cells`). Cel mai
   simplu: `pip install "xlrd==1.2.0"` cu `--break-system-packages` dacă pe Windows.

```cmd
.venv\Scripts\activate.bat
pip install openpyxl "xlrd==1.2.0" requests
```

> Dacă rulezi `pip install` și `uv` se plânge, fă-o prin `uv pip install`.

### 4) Rulează testele

```cmd
python -m pytest tests/unit/test_tennis_data_uk.py -v
```

Așteptat: **47 passed**.

### 5) Dry-run pe un singur an (NU bagă în DB)

```cmd
python -m tennis_predictor.data.loaders.load_odds --year 2024 --tour ATP --dry-run
```

Aștept-așteptat:
- Downloads `2024.xlsx` în `data/odds/raw/atp_2024.xlsx` (~1-2 MB)
- Raportează ~2,700 raw matches, ~2,700 odds rows × bookmakers active
- ~2,500-2,700 PS rows (Pinnacle), similar pentru B365 și AVG
- Vig warnings ar trebui să fie sub 5% din rows (dacă e mai mult, e ceva ciudat)

### 6) Run real pe un singur an, apoi verifică DB

```cmd
python -m tennis_predictor.data.loaders.load_odds --year 2024 --tour ATP
```

Apoi în Supabase:
```sql
-- Cât avem total?
SELECT COUNT(*) FROM historical_odds_raw;

-- Distribuție per bookmaker
SELECT bookmaker_code, COUNT(*), AVG(vig)::numeric(5,4) AS avg_vig
FROM historical_odds_raw
WHERE source_year = 2024
GROUP BY bookmaker_code
ORDER BY COUNT(*) DESC;
-- Expected: PS, B365, AVG, MAX should dominate. PS avg_vig ~0.02-0.025.

-- Sample row inspection
SELECT match_date, tournament_name, winner_name, loser_name,
       bookmaker_code, winner_odds, loser_odds, vig
FROM historical_odds_raw
WHERE source_year = 2024 AND bookmaker_code = 'PS'
ORDER BY match_date DESC
LIMIT 10;
```

### 7) Full range (după ce single-year e OK)

```cmd
# Pinnacle ATP coverage: 2003-2024 (22 years, ~50k matches, ~400-600k odds rows)
python -m tennis_predictor.data.loaders.load_odds --years 2003-2024 --tour ATP

# WTA: 2007-2024
python -m tennis_predictor.data.loaders.load_odds --years 2007-2024 --tour WTA
```

Estimated time: 5-10 min for download (network bound), 1-2 min for inserts.

---

## Decizii din Commit 3.1 (justifications)

1. **Staging table separat (`historical_odds_raw`)** — Marius a ales asta la
   întrebarea inițială. Avantaj: separație curată între load și matching;
   permite re-rulare a fuzzy matching-ului în Commit 3.2 fără re-download.

2. **Long-format** (one row per match × bookmaker) — facilitează queries de
   tipul "toate cotele Pinnacle din 2018", și matchează schema viitoare a
   `historical_odds`.

3. **MAX și AVG ca pseudo-bookmakers** — Oddsportal aggregates utile pentru
   analiză în Commit 3.4 (compară edge vs Pinnacle alone vs Avg-of-soft-books).

4. **Vig warnings, nu errors** — cote ciudate sunt loggate la DEBUG, nu blochează
   insertion. Validation reală în Commit 3.2 (după ce avem match_id).

5. **ON CONFLICT DO NOTHING** — re-rulări sunt idempotente. Util când vrei să
   adaugi un an nou fără să afectezi ce-i deja încărcat.

6. **EARLIEST_YEAR constants** — ATP 2001 (B365 disponibil), WTA 2007 (cel mai
   vechi). Pinnacle ATP începe 2003, dar nu blocăm 2001-2002 fiindcă tot avem
   B365 și alți bookmakeri.

7. **Stub `db.py`** în ZIP = doar pentru testele mele în sandbox. Nu-l copia
   în repo (deja există acolo real).

---

## Known limitations / next-commit work

- **Match linkage**: 0 rows linked yet. Commit 3.2 va face fuzzy matching.
- **xls files (pre-2013)**: au nevoie de `xlrd==1.2.0`. Dacă instalarea eșuează
  pe Windows din vreun motiv, putem face fallback pe `aspose-cells` sau
  convertim manual cu LibreOffice. Spune-mi dacă întâmpini probleme.
- **Pre-2003 ATP dates**: notes.txt zice "prior to 2003 the date shown for all
  matches played in a single tournament is the start date". Asta înseamnă ±7-day
  tolerance când facem fuzzy match în Commit 3.2 pe pre-2003 ATP, dar pentru
  Commit 3.1 doar stocăm — e tradăbil în 3.2.
- **Romanian odds files (Iași, Cluj)**: pe site sunt liste de turnee românești.
  Acolo n-avem nevoie de nimic special.

---

## Testing summary

```
============================== 47 passed in 0.60s ==============================
```

- `TestBuildUrl`               — 7 tests (URL pattern validation)
- `TestDownloadYearFile`       — 4 tests (input validation, cache behavior)
- `TestComputeImplied`         — 8 tests (probability + vig math)
- `TestExtractMatchMetadata`   — 8 tests (raw row → metadata dict)
- `TestExtractOddsRows`        — 8 tests (pipeline orchestration)
- `TestParseYearSpec`          — 8 tests (CLI argument parsing)
- `TestBookmakerCodes`         — 4 tests (schema sanity)

Plus 1 end-to-end smoke test cu Excel sintetic — produs 8 OddsRow din 3 meciuri,
match-uite Pinnacle row exact pe implied prob și vig.

---

**ETA Commit 3.1 din kickoff: ~20 min.**
**Real**: probabil 30-45 min cu integration și verificări. (Estimate buffer +30% s-a confirmat.)

Următor: **Commit 3.2** — fuzzy player name matching cu pg_trgm.
