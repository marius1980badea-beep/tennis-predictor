"""Loader for historical match odds from tennis-data.co.uk.

The site provides annual Excel files containing match results plus near-closing
odds from 13+ bookmakers, including Pinnacle (PSW/PSL). We use these as the
historical "true line" benchmark for computing Closing Line Value (CLV) in
Phase 3.3.

Schema reference:   http://www.tennis-data.co.uk/notes.txt
File index:         http://www.tennis-data.co.uk/alldata.php

Per the source's own notes: "Betting odds for matches generally represent the
most recent before play starts" -- i.e. near-closing lines, which is exactly
what we want for CLV measurement.

This module does NOT touch the database. It produces in-memory ``OddsRow``
records that the CLI module (``load_odds.py``) inserts into
``historical_odds_raw``. Keeping IO out of the parsing layer makes the
parsing logic easy to unit-test offline with synthetic DataFrames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "http://www.tennis-data.co.uk"

# URL path suffix: ATP files at /{year}/, WTA files at /{year}w/
TOUR_PATH_SUFFIX = {"ATP": "", "WTA": "w"}

# File extension switches from .xls to .xlsx in 2013 for both tours.
XLSX_FROM_YEAR = 2013

# Earliest year with usable odds. ATP 2000 has match results but no odds.
# Pinnacle specifically starts 2003 ATP / 2007 WTA; the loader still picks up
# Bet365 etc. for 2001-2002 ATP, which is useful for soft-book backtests.
EARLIEST_YEAR = {"ATP": 2001, "WTA": 2007}

# Bookmaker codes match `bookmakers.bookmaker_code` pre-inserted in Phase 1.
# Map: { code -> (winner_odds_column, loser_odds_column) }
BOOKMAKER_COLUMNS: dict[str, tuple[str, str]] = {
    "B365": ("B365W", "B365L"),
    "B&W":  ("B&WW",  "B&WL"),
    "CB":   ("CBW",   "CBL"),
    "EX":   ("EXW",   "EXL"),
    "LB":   ("LBW",   "LBL"),
    "GB":   ("GBW",   "GBL"),
    "IW":   ("IWW",   "IWL"),
    "PS":   ("PSW",   "PSL"),      # Pinnacle Sports -- our CLV benchmark
    "SB":   ("SBW",   "SBL"),
    "SJ":   ("SJW",   "SJL"),
    "UB":   ("UBW",   "UBL"),
    # Aggregated odds from Oddsportal -- pseudo-bookmakers, useful for analysis
    "MAX":  ("MaxW",  "MaxL"),
    "AVG":  ("AvgW",  "AvgL"),
}

# Vig sanity ranges per bookmaker. Values outside these trigger DEBUG warnings.
# Pinnacle is the sharpest book -> tightest margins.
VIG_RANGES: dict[str, tuple[float, float]] = {
    "PS":   (0.005, 0.040),   # Pinnacle: 0.5%-4%
    "B365": (0.020, 0.090),
    "MAX":  (-0.030, 0.030),  # MAX aggregates best odds -> can be ~zero or negative
    "AVG":  (0.020, 0.090),
}
# All other soft books default to a wider tolerance band:
DEFAULT_VIG_RANGE = (0.020, 0.130)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OddsRow:
    """One bookmaker's odds for one match, normalized and ready to insert."""

    source: str
    source_year: int
    tour: str
    match_date: date
    tournament_name: str
    series_or_tier: Optional[str]
    court: Optional[str]
    surface: Optional[str]
    round: Optional[str]
    best_of: Optional[int]
    winner_name: str
    loser_name: str
    winner_rank: Optional[int]
    loser_rank: Optional[int]
    comment: Optional[str]
    bookmaker_code: str
    winner_odds: float
    loser_odds: float
    winner_implied_prob: float
    loser_implied_prob: float
    vig: float


@dataclass
class YearLoadReport:
    """Per-year summary, displayed by the CLI as a Rich table."""

    tour: str
    year: int
    raw_matches: int          # rows present in the Excel file
    skipped_matches: int      # rows with no usable odds at all
    odds_rows: int            # total (match × bookmaker) rows produced
    pinnacle_rows: int        # rows with PS odds
    bet365_rows: int          # rows with B365 odds
    avg_rows: int             # rows with AVG odds
    vig_warnings: int         # rows where computed vig was out of expected range
    parse_errors: int         # rows where odds were malformed (e.g. <= 1.0)


# ---------------------------------------------------------------------------
# URL + download
# ---------------------------------------------------------------------------

def build_url(year: int, tour: str) -> str:
    """Construct the tennis-data.co.uk URL for ``year`` and ``tour``."""
    if tour not in TOUR_PATH_SUFFIX:
        raise ValueError(f"tour must be ATP or WTA, got {tour!r}")
    suffix = TOUR_PATH_SUFFIX[tour]
    ext = "xlsx" if year >= XLSX_FROM_YEAR else "xls"
    return f"{BASE_URL}/{year}{suffix}/{year}.{ext}"


def download_year_file(
    year: int,
    tour: str,
    cache_dir: Path,
    *,
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """Download a single annual file, caching to ``cache_dir``.

    Returns the local path to the (cached or freshly-downloaded) file. The
    cache filename is ``{tour}_{year}.{ext}`` for easy inspection.
    """
    if year < EARLIEST_YEAR[tour]:
        raise ValueError(
            f"{tour} odds available from {EARLIEST_YEAR[tour]} onward; "
            f"asked for {year}"
        )

    url = build_url(year, tour)
    ext = url.rsplit(".", 1)[-1]
    cache_dir = Path(cache_dir)
    local_path = cache_dir / f"{tour.lower()}_{year}.{ext}"

    if local_path.exists() and not force:
        logger.debug("Using cached file %s", local_path)
        return local_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    local_path.write_bytes(response.content)
    logger.info("Saved %s (%d bytes)", local_path, len(response.content))
    return local_path


# ---------------------------------------------------------------------------
# Excel reading
# ---------------------------------------------------------------------------

def read_year_excel(filepath: Path) -> pd.DataFrame:
    """Read the first sheet of an ``.xls`` or ``.xlsx`` file.

    Pandas auto-selects the engine (``openpyxl`` for ``.xlsx``, ``xlrd`` for
    ``.xls``). Both must be available; declared as dependencies in pyproject.
    """
    return pd.read_excel(filepath, sheet_name=0)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def _to_float(value: object) -> Optional[float]:
    """Coerce an Excel cell to ``float``; return ``None`` for missing/invalid."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", ".")
        if not cleaned:
            return None
        value = cleaned
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _to_int(value: object) -> Optional[int]:
    """Coerce to ``int`` via ``float`` (handles values like ``"NR"`` -> None)."""
    as_float = _to_float(value)
    if as_float is None:
        return None
    return int(as_float)


def _to_str(value: object) -> Optional[str]:
    """Strip and return non-empty strings; ``None`` for missing/empty/NaN."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s or None


# ---------------------------------------------------------------------------
# Implied probability + vig
# ---------------------------------------------------------------------------

def compute_implied(
    winner_odds: float,
    loser_odds: float,
) -> tuple[float, float, float]:
    """Return ``(winner_implied_prob, loser_implied_prob, vig)``.

    Implied probability = 1 / decimal_odds.
    Vig (overround) = winner_implied + loser_implied - 1.

    Raises ``ValueError`` if either odd is <= 1.0 (impossible market).
    """
    if winner_odds <= 1.0 or loser_odds <= 1.0:
        raise ValueError(f"odds must be > 1.0, got W={winner_odds} L={loser_odds}")
    w_imp = 1.0 / winner_odds
    l_imp = 1.0 / loser_odds
    vig = w_imp + l_imp - 1.0
    return w_imp, l_imp, vig


def _vig_out_of_range(vig: float, bookmaker: str) -> bool:
    low, high = VIG_RANGES.get(bookmaker, DEFAULT_VIG_RANGE)
    return not (low <= vig <= high)


# ---------------------------------------------------------------------------
# Row-level extraction
# ---------------------------------------------------------------------------

def extract_match_metadata(
    row: pd.Series,
    tour: str,
    year: int,
) -> Optional[dict]:
    """Extract non-odds metadata from one Excel row.

    Returns ``None`` if essentials are missing (no Winner / Loser / Date).
    The returned dict is suitable for ``OddsRow(**meta, ...)``.
    """
    winner = _to_str(row.get("Winner"))
    loser = _to_str(row.get("Loser"))
    if not winner or not loser:
        return None

    # Notes.txt labels the column "Data" (typo) but actual files use "Date".
    # Fall back to "Data" for robustness against any older file variants.
    raw_date = row.get("Date")
    if raw_date is None or pd.isna(raw_date):
        raw_date = row.get("Data")
    if raw_date is None or pd.isna(raw_date):
        return None

    if hasattr(raw_date, "date"):
        match_date = raw_date.date()
    else:
        try:
            match_date = pd.Timestamp(raw_date).date()
        except (ValueError, TypeError):
            return None

    # ATP files have "Series", WTA files have "Tier". Use whichever exists.
    series_or_tier = _to_str(row.get("Series")) or _to_str(row.get("Tier"))

    return {
        "source": "tennis-data.co.uk",
        "source_year": year,
        "tour": tour,
        "match_date": match_date,
        "tournament_name": _to_str(row.get("Tournament")) or "UNKNOWN",
        "series_or_tier": series_or_tier,
        "court": _to_str(row.get("Court")),
        "surface": _to_str(row.get("Surface")),
        "round": _to_str(row.get("Round")),
        "best_of": _to_int(row.get("Best of")),
        "winner_name": winner,
        "loser_name": loser,
        "winner_rank": _to_int(row.get("WRank")),
        "loser_rank": _to_int(row.get("LRank")),
        "comment": _to_str(row.get("Comment")),
    }


def extract_odds_rows(
    df: pd.DataFrame,
    *,
    tour: str,
    year: int,
) -> tuple[list[OddsRow], YearLoadReport]:
    """Iterate a DataFrame and emit one ``OddsRow`` per (match, bookmaker) cell.

    Skips matches with no usable odds in any column. Logs DEBUG warnings for
    rows whose computed vig is outside expected bookmaker ranges.
    """
    available_cols = set(df.columns)
    rows: list[OddsRow] = []
    skipped = 0
    vig_warnings = 0
    parse_errors = 0
    counts = {"PS": 0, "B365": 0, "AVG": 0}

    for _, row in df.iterrows():
        meta = extract_match_metadata(row, tour=tour, year=year)
        if meta is None:
            skipped += 1
            continue

        matched_any = False
        for code, (w_col, l_col) in BOOKMAKER_COLUMNS.items():
            if w_col not in available_cols or l_col not in available_cols:
                continue
            w = _to_float(row.get(w_col))
            l = _to_float(row.get(l_col))
            if w is None or l is None:
                continue
            try:
                w_imp, l_imp, vig = compute_implied(w, l)
            except ValueError:
                parse_errors += 1
                continue

            if _vig_out_of_range(vig, code):
                vig_warnings += 1
                logger.debug(
                    "vig out of range for %s %s/%s on %s: %.4f (odds %.2f/%.2f)",
                    code, meta["winner_name"], meta["loser_name"],
                    meta["match_date"], vig, w, l,
                )

            rows.append(OddsRow(
                **meta,
                bookmaker_code=code,
                winner_odds=w,
                loser_odds=l,
                winner_implied_prob=w_imp,
                loser_implied_prob=l_imp,
                vig=vig,
            ))
            matched_any = True
            if code in counts:
                counts[code] += 1

        if not matched_any:
            skipped += 1

    report = YearLoadReport(
        tour=tour,
        year=year,
        raw_matches=len(df),
        skipped_matches=skipped,
        odds_rows=len(rows),
        pinnacle_rows=counts["PS"],
        bet365_rows=counts["B365"],
        avg_rows=counts["AVG"],
        vig_warnings=vig_warnings,
        parse_errors=parse_errors,
    )
    return rows, report


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def load_year(
    year: int,
    tour: str,
    cache_dir: Path,
    *,
    force_download: bool = False,
) -> tuple[list[OddsRow], YearLoadReport]:
    """Download + parse + normalize one year."""
    filepath = download_year_file(
        year=year, tour=tour, cache_dir=cache_dir, force=force_download,
    )
    df = read_year_excel(filepath)
    return extract_odds_rows(df, tour=tour, year=year)


def load_years(
    years: Iterable[int],
    tour: str,
    cache_dir: Path,
    *,
    force_download: bool = False,
) -> tuple[list[OddsRow], list[YearLoadReport]]:
    """Load multiple years sequentially; returns concatenated rows + per-year reports."""
    all_rows: list[OddsRow] = []
    reports: list[YearLoadReport] = []
    for year in years:
        rows, report = load_year(
            year=year, tour=tour, cache_dir=cache_dir,
            force_download=force_download,
        )
        all_rows.extend(rows)
        reports.append(report)
    return all_rows, reports
