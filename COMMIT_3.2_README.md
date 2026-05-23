# Commit 3.2 — Fuzzy player name matching

**Goal**: Pentru fiecare row din `historical_odds_raw` (66k odds rows, ~17k
unique matches), găsim match-ul corespunzător în `matches` și populăm
`match_id` + `match_confidence` + `matched_at`.

**Validated locally**: 47 unit tests pass, plus 21 real-world ATP/WTA name
pairs match perfect (1.0).

---

## Fișiere în ZIP

| Path în ZIP | Unde se copiază în repo |
|---|---|
| `src/tennis_predictor/data/matching/__init__.py` | same |
| `src/tennis_predictor/data/matching/odds_match.py` | same |
| `src/tennis_predictor/data/matching/cli.py` | same |
| `tests/unit/test_odds_match.py` | same |
| `COMMIT_3.2_README.md` | (acest fișier, doar reference) |

**NU copia** `src/tennis_predictor/data/storage/db.py` din ZIP — e stub pentru
testele mele de sandbox. Real-ul există deja la tine.

---

## Dependențe noi

Două librării noi, ambele pure-Python + C extensions:

```cmd
.venv\Scripts\activate.bat
pip install rapidfuzz unidecode
```

- **rapidfuzz** (~MIT license, ~1MB) — Levenshtein și `token_set_ratio` accelerate cu C
- **unidecode** (GPL, ~200KB) — diacritics removal (Garín → Garin)

Adaugă la `pyproject.toml`:
```toml
dependencies = [
  # existing...
  "rapidfuzz>=3.0",
  "unidecode>=1.3",
]
```

---

## Pași integrare

### 1) Copiază fișierele Python

```cmd
cd "C:\Users\Marius Badea\Documents\GitHub\tennis-predictor"
# Copy from ZIP:
#   src/tennis_predictor/data/matching/__init__.py
#   src/tennis_predictor/data/matching/odds_match.py
#   src/tennis_predictor/data/matching/cli.py
#   tests/unit/test_odds_match.py
```

### 2) Verifică testele

```cmd
python -m pytest tests/unit/test_odds_match.py -v
```

Așteptat: **47 passed**.

### 3) Confirm schema assumption

Modulul `cli.py` are SQL care presupune următoarele coloane în baza ta:

| Tabel | Coloane folosite |
|---|---|
| `matches` | `match_id`, `tournament_id`, `winner_id`, `loser_id`, `match_date`, `tour`, `surface` |
| `players` | `player_id`, `name_full` |
| `tournaments` | `tournament_id`, `name` |

**Verifică rapid** că aceste coloane chiar există:

```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'tournaments' AND column_name IN ('tournament_id', 'name');
-- Should return both rows
```

Dacă coloana de nume în `tournaments` se numește diferit (ex. `tournament_name`),
modifică în `cli.py`:
```python
SQL_CANDIDATES = text("""...t.name AS tournament_name...""")
                              # ↑ schimbă aici
```

### 4) Dry run pe ATP cu limit mic

Înainte să rulăm pe toate cele 17k identități, încercăm pe primele 200 ca să
validăm că:
- SQL-ul funcționează
- Pipeline-ul produce match-uri rezonabile
- Confidence distribution arată sensibil

```cmd
python -m tennis_predictor.data.matching.cli --tour ATP --limit 200 --dry-run
```

Așteptat în output:
- Match rate ~98%+ (mai multe 1.00 confidence)
- Câțiva în 0.85-0.94 (nume cu diacritice complexe sau tournament naming differ)
- 0-5 below 0.70 (probabil meciuri Davis Cup/Olympics cu ambiguitate)

### 5) Dry run full ATP

```cmd
python -m tennis_predictor.data.matching.cli --tour ATP --dry-run
```

Asta load-ează ~150k matches din `matches` în memorie + ~17k identități. RAM
usage: ~100-200 MB. Timp: ~30-60 sec.

**Inspectează raportul** atent:
- **Match rate**: dorim >95% pentru ATP. Sub 85% e suspect — investigăm.
- **Below 0.70 bucket**: identifică patterns. Sunt toate dintr-un an specific?
  Un tournament specific? Putem adăuga alias-uri în `TOURNAMENT_ALIASES` și
  re-rulăm.
- **No candidates**: ar trebui aproape zero. Dacă nu, înseamnă că anumite
  odds au date care nu există în `matches` (probabil pre-2013 sau 2025+ pentru
  care Sackmann n-a fost încărcat încă).

### 6) Real run ATP

```cmd
python -m tennis_predictor.data.matching.cli --tour ATP
```

Odată confirmat dry run, rularea reală apply-ează UPDATE-urile pe Supabase.
Idempotent — re-rularea sare peste rows deja matched (`match_id IS NULL`
filter).

Sanity check post-run:
```sql
-- Cât % din rows sunt acum linked?
SELECT 
  tour,
  COUNT(*) AS total,
  COUNT(*) FILTER (WHERE match_id IS NOT NULL) AS matched,
  (100.0 * COUNT(*) FILTER (WHERE match_id IS NOT NULL) / COUNT(*))::numeric(5,2) AS pct
FROM historical_odds_raw
GROUP BY tour;
```

Așteptat: ATP ~95-99%, WTA TBD.

### 7) Repetă pentru WTA

```cmd
python -m tennis_predictor.data.matching.cli --tour WTA --dry-run
python -m tennis_predictor.data.matching.cli --tour WTA
```

### 8) Investighează unmatched

Pentru rows care n-au fost matched, vezi pattern-uri:
```sql
-- Per an, per tournament name, ce a rămas nematched?
SELECT 
  source_year,
  tournament_name,
  COUNT(DISTINCT (match_date, winner_name, loser_name)) AS unique_unmatched
FROM historical_odds_raw
WHERE match_id IS NULL AND tour = 'ATP'
GROUP BY 1, 2
HAVING COUNT(*) > 0
ORDER BY unique_unmatched DESC
LIMIT 20;
```

Dacă vezi un tournament care domină unmatched (ex. "Davis Cup Finals 2025"),
e clar că trebuie adăugat ca alias sau că meciurile nu există în `matches`
(Sackmann nu acoperă Davis Cup, de exemplu).

---

## Cum funcționează algorithm

**Normalisation**:
- `unidecode("Garín")` → "Garin" (strip diacritics)
- lowercase, strip dots/dashes/apostrophes
- Sackmann "Novak Djokovic" → compact "djokovic n"
- tennis-data "Djokovic N." → normalized "djokovic n"
- Exact string match → similarity = 1.0

**Scoring** (composite):
- 45% × winner_sim (rapidfuzz.token_set_ratio / 100)
- 45% × loser_sim
- 10% × tournament_sim (with alias support)
- Threshold default 0.70 (configurabil)

**Date filtering**:
- Doar candidates în `±1 zi` de odds row date (configurabil cu `--date-window`)
- Pentru pre-2003 (dacă vom încărca .xls files), ar putea trebui ±7 zile

**Tournament aliases**:
- "French Open" ↔ "Roland Garros" ↔ "Internationaux de France"
- "ATP Finals" ↔ "Nitto ATP Finals" ↔ "Barclays ATP World Tour Finals"
- Și altele în `TOURNAMENT_ALIASES` list în `odds_match.py`

**Performance**:
- Bulk pre-load matches → indexat pe match_date în Python
- ~17k identități × ~3-5 candidates fiecare = ~50-85k similarity calls
- @ ~1ms/call = ~1 min total
- Apoi batch UPDATE de 500 rows odată

---

## Sanity test: real names

Au fost testate 21 nume reale ATP+WTA, toate au scor 1.0 perfect:

| TD name | Sackmann full | Compact | Sim |
|---|---|---|---|
| Djokovic N. | Novak Djokovic | djokovic n | 1.0 |
| Garín C. | Cristian Garín | garin c | 1.0 |
| Bautista Agut R. | Roberto Bautista Agut | bautista agut r | 1.0 |
| Auger-Aliassime F. | Felix Auger-Aliassime | auger aliassime f | 1.0 |
| De Minaur A. | Alex de Minaur | de minaur a | 1.0 |
| Świątek I. | Iga Świątek | swiatek i | 1.0 |
| Čilić M. | Marin Čilić | cilic m | 1.0 |
| Davidovich Fokina A. | Alejandro Davidovich Fokina | davidovich fokina a | 1.0 |
| ... (toate 21) | ... | ... | 1.0 |

**Composite test** pe 2024 French Open Final (Alcaraz vs Zverev, French Open
↔ Roland Garros alias): scor 1.0 perfect.

---

## Known limitations

1. **2025-2026 odds**: orphan-uri (`match_id` rămâne NULL) pentru meciurile
   din 2025-2026 până când reload-ăm Sackmann. Așa cum am discutat: Sackmann
   refresh + Elo re-train sunt prerequisite pentru Faza 4. Nu blocking pentru
   3.3 (CLV pe 2013-2024 e suficient).

2. **Davis Cup / Hopman Cup / Olympics**: aceste meciuri probabil nu sunt în
   Sackmann data (focused on ATP/WTA tour). Vor apărea ca "no candidates" în
   raport. Așteptat.

3. **Tournament alias dictionary** e mic la v1. Vom adăuga pe parcurs ce
   descoperim în unmatched reports.

4. **Same-day, same-player edge case**: dacă doi jucători joacă două meciuri
   în aceeași zi (rar — Davis Cup/Laver Cup), tournament_sim e tiebreaker.
   Dar dacă turneul e identic, ne putem prinde la match-ul greșit. Foarte
   rar în practică.

---

## Testing summary

```
============================== 47 passed in 0.30s ==============================
```

- TestNormalize (8) — diacritics, dots, dashes, apostrophes, whitespace
- TestCompactSackmann (7) — single/multi-token, hyphenated, diacritics
- TestNameSimilarity (8) — perfect/partial/wrong-initial/missing
- TestTournamentSimilarity (6) — perfect/alias/fuzzy/empty
- TestScoreMatch (5) — composite scores in various scenarios
- TestResolveMatch (5) — candidate set resolution
- TestBucketLabel (5) — confidence binning
- TestTournamentAliases (2) — schema sanity of alias table

Plus offline real-world sanity check pe 21 nume ATP+WTA: toate 1.0 perfect.

---

**ETA Commit 3.2 din kickoff: ~25 min.**
**Real**: ~45 min cu sanity + diagnostic. Estimate buffer confirm +30-50%.

Următor: **Commit 3.3** — CLV calculator + value bet detection.
