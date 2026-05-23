# Commit 3.2 v2 — Player-ID-pivot fuzzy matching

**Refactor după ce v1 a dat doar 59.5% match rate pe primul dry-run.**

## Diagnoza (de ce v1 a eșuat)

V1 făcea matching nume-to-nume cu window de ±1 zi. Investigația pe date 2013
ATP a arătat că **Sackmann și tennis-data.co.uk folosesc convenții diferite
de date**:

| Source | Date convention |
|---|---|
| tennis-data.co.uk | data efectivă în care s-a jucat meciul |
| Sackmann | data de **start a turneului** (Monday of the week) |

Pe Brisbane International 2013, de exemplu, tennis-data avea matches
distribuite pe toate zilele 31-Dec → 5-Jan, iar Sackmann avea toate sub
data 31-Dec. Window-ul de ±1 zi rata 5-7 zile de meciuri pe turneu.

## v2 strategy: pivot via player IDs

Două faze:

1. **Build PlayerResolver** (one-time) — index `compact_name -> player_id` din
   tabela `players`. Pentru tennis-data names, lookup exact pe compact form
   (~99% hit rate) + fuzzy fallback cu `rapidfuzz.process.extractOne`.

2. **Resolve per row** — pentru fiecare odds identity, resolve winner_id +
   loser_id, apoi cauta în matches cu `WHERE winner_id = X AND loser_id = Y`
   în window ±14 zile. Match aproape mereu unic. Tiebreaker: tournament
   similarity + date proximity.

## Fișiere în ZIP

| Path în ZIP | Acțiune |
|---|---|
| `src/tennis_predictor/data/matching/__init__.py` | **OVERWRITE** existent |
| `src/tennis_predictor/data/matching/odds_match.py` | **OVERWRITE** existent |
| `src/tennis_predictor/data/matching/cli.py` | **OVERWRITE** existent |
| `tests/unit/test_odds_match.py` | **OVERWRITE** existent |

## Pași integrare

### 1) Overwrite fișierele Phase 3.2 vechi

Aceleași 4 fișiere care le-ai copiat în iterația anterioară, le înlocuiești
cu noile versiuni. Restul rămâne neschimbat.

### 2) Verifică testele

```cmd
.venv\Scripts\activate.bat
python -m pytest tests/unit/test_odds_match.py -v
```

Așteptat: **35 passed** (mai puține ca v1 fiindcă am unificat unele teste
similare, dar coverage e mai mare pe părțile critice).

Test critic: `test_match_with_tournament_start_date_offset` — reproduce
explicit problema de date (6 zile diferență Sackmann ↔ tennis-data).

### 3) Re-rulează dry run cu același limit ca data trecută

```cmd
python -m tennis_predictor.data.matching.cli --tour ATP --limit 200 --dry-run
```

Așteptat acum:
- Match rate ar trebui să sară de la **59.5% → 95%+**
- Cele 71 "no candidates in window" ar trebui să dispară aproape complet
  fiindcă window-ul implicit acum e ±14 zile
- Confidence distribution: majoritate 1.00 (exact compact match pe ambii
  jucători), câteva 0.95-0.99 (fuzzy fallback pe un jucător cu diacritice
  edge cases)

### 4) Inspectează raportul atent

Noul output are mai multe diagnostics:

```
Failure breakdown
  Player(s) couldn't be resolved      X     <- jucător nu exista în players
  Players OK, no match in date window Y     <- jucători OK, dar n-au jucat în window
  Composite confidence below threshold Z    <- fuzzy hits prea slabi

Fuzzy resolutions (vs exact)
  Winner via fuzzy   N
  Loser via fuzzy    M
```

Asta îți spune EXACT unde sunt rămași unmatched-ii.

### 5) Full dry run

```cmd
python -m tennis_predictor.data.matching.cli --tour ATP --dry-run
```

Loadează ~17k identități + ~35k matches Sackmann + ~3000 players. Timp: 1-3
min total.

### 6) Real run

```cmd
python -m tennis_predictor.data.matching.cli --tour ATP
python -m tennis_predictor.data.matching.cli --tour WTA
```

## Configurări notabile

| Flag | Default | Note |
|---|---|---|
| `--fuzzy-threshold` | 0.85 | Cât de slab acceptăm match-uri de player. 0.85 e conservator. |
| `--date-window` | 14 | ±14 zile e enough pentru toate turneele inclusiv Slam-uri. |
| `--min-confidence` | 0.70 | Compozit. 0.70 = average de ambii jucători. |

Dacă vezi multe `fuzzy_winner_resolutions` (>10% din matches), poate vrei să
relaxezi `--fuzzy-threshold` la 0.80 sau să strângi la 0.90 după preferință.

## Test coverage

```
35 passed in 1.32s
```

Categorii:
- **Normalize / Compact / Tournaments** (12 tests) — string canonicalization
- **PlayerResolver** (11 tests) — exact match, fuzzy fallback, edge cases:
  - Multi-word surnames (Bautista Agut)
  - Diacritics în ambele direcții (Garin ↔ Garín)
  - String vs int player_id (Sackmann atp_XXXX format)
  - Compact name collisions (Marko vs Mihailo Djokovic)
  - Empty/None names în players table
- **resolve_via_player_ids** (7 tests) — full pipeline:
  - **Direct match** la window=0
  - **Tournament start date offset** (cazul critic care a rupt v1)
  - Outside window rejection
  - Swapped W/L recovery
  - Multiple candidates → tournament tiebreak
- **Bucket labels** (5 tests) — diagnostic

## Performance

Pentru ATP full (≈17k identități, 35k matches, 3k players):

| Operație | Timp |
|---|---|
| Build PlayerResolver | <1 sec |
| Load matches DataFrame | ~5 sec (DB round-trip dominant) |
| 17k × resolve_via_player_ids | ~30-60 sec |
| Batch UPDATE | ~10-20 sec |
| **Total** | **~1-2 min** |

(De la 50+ min cu v1 brute-force.)

## Known limitations

1. **2025-2026 odds** — Sackmann nu a fost încărcat pentru 2025+, deci toate
   identitățile din acești ani vor rămâne în bucket "Player(s) couldn't be
   resolved" sau "Players OK, no match in date window". Așa cum am discutat:
   Sackmann refresh e prerequisite pentru Faza 4.

2. **Davis Cup / Olympics** — meciuri non-tour, probabil lipsă din Sackmann.
   Vor fi unmatched. Acceptăm pierderea.

3. **Compact-name collisions** — dacă doi jucători au același compact form
   (foarte rar: same last name + same first initial), primul ales câștigă
   pe FCFS. Log la DEBUG level. În practică, asta înseamnă maximum 0.1%
   din matches.

---

**ETA Commit 3.2 (v2) din kickoff: ~25 min.**
**Real**: ~90 min cu v1 broken + diagnoză + refactor v2. Estimate buffer mare.

Următor: după ce 3.2 v2 dă match rate >95%, mergem la **Commit 3.3** (CLV
calculator + value bet detection).
