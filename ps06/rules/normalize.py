"""Text hygiene for rule matching (M-4, step 2).

The rules engine reads ``extracted_text`` and matches Spanish field labels and
values against it. Session-9's live Tier-2 test (see
``PS-06_Tier2_Results.md``/``PS-06_Tier1_Results.md``) found two live defects
in that text that would silently corrupt exact/substring matching if the
engine read it raw:

1. **Ligature characters** in text-layer extraction, e.g. ``Certiﬁcado`` using
   the single ``ﬁ`` codepoint rather than ``f`` + ``i`` — invisible to the eye,
   fatal to an exact match against ``"Certificado"``. Fixed by NFKC
   normalization, which decomposes typographic ligatures into their base
   letters.
2. **VLM hallucination artifacts**: the model wraps output in ```` ```markdown ````
   fences (Tier-1 finding 2) and, separately, sometimes invents a markdown
   image link that was never in the source document, e.g.
   ``![](https://example.com/image.png)`` (Tier-2 finding 2). Both need to be
   stripped before rule matching sees the text.

This module also carries the deterministic normalization primitives the
Lineage Engine's entity-resolution and flag detectors need to compare values
across documents: name folding (diacritics/case-insensitive), document-number
stripping (formatting-insensitive), and best-effort date parsing.

Pure text/string functions only — no I/O, no model calls.
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------------------
# VLM output hygiene
# ---------------------------------------------------------------------------

#: A fenced code block, e.g. ```` ```markdown\n...\n``` ```` or bare ```` ``` ````.
#: Only the fence markers are stripped, not the content inside — a real
#: transcription is usually wrapped in a fence, not garbage to discard.
_CODE_FENCE_RE = re.compile(r"^[ \t]*```[^\n]*\n?|\n?[ \t]*```[ \t]*$", re.MULTILINE)

#: A markdown image link, e.g. ``![](https://example.com/image.png)`` or
#: ``![alt text](url)``. The VLM has been observed inventing these outright
#: (Tier-2 finding 2) — no OCR'd document text is meaningfully an image link,
#: so this is safe to drop unconditionally rather than trying to distinguish
#: "real" from "hallucinated" ones.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def strip_vlm_artifacts(text: str) -> str:
    """Remove code fences and invented markdown image links from VLM output.

    Idempotent and safe to call on text-layer (non-VLM) extractions too — both
    patterns are absent there, so this is a no-op on that path.
    """
    text = _MARKDOWN_IMAGE_RE.sub("", text)
    text = _CODE_FENCE_RE.sub("", text)
    return text


def nfkc(text: str) -> str:
    """NFKC-normalize text, decomposing ligatures (e.g. ``ﬁ`` -> ``fi``).

    NFKC is compatibility decomposition + canonical composition: it collapses
    presentation-form ligatures and other compatibility variants into their
    plain-letter equivalents while still composing accented letters (e.g.
    ``á``) into a single codepoint, which is what every other normalization
    helper here expects.
    """
    return unicodedata.normalize("NFKC", text)


def normalize_extracted_text(text: str) -> str:
    """Full hygiene pass for rule matching: VLM-artifact stripping + NFKC.

    This is the one function the adapter/lineage/flags modules should call on
    any ``extracted_text`` before matching against it — it is intentionally the
    single choke point so a future hygiene fix only needs to land here.
    """
    return nfkc(strip_vlm_artifacts(text))


# ---------------------------------------------------------------------------
# Name folding (for fuzzy/exact person-name comparison)
# ---------------------------------------------------------------------------


def fold_name(name: Optional[str]) -> str:
    """Fold a name to a comparison-friendly form: NFKC, case-fold, strip accents.

    ``None``/empty input returns ``""`` rather than raising — callers compare
    folded values directly, and two absent names should compare unequal to any
    real name (``"" == ""`` would otherwise report a false match), which is left
    to the caller's own presence check, not this function's job.
    """
    if not name:
        return ""
    folded = nfkc(name).casefold()
    # Strip combining marks left after NFKC+casefold (e.g. "á" -> decompose via
    # NFD, drop U+0301, recompose is unnecessary since we only compare the
    # stripped form).
    decomposed = unicodedata.normalize("NFD", folded)
    without_marks = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    # Collapse internal whitespace so "Ana  Garcia" == "Ana Garcia".
    return re.sub(r"\s+", " ", without_marks).strip()


def names_match(a: Optional[str], b: Optional[str]) -> bool:
    """Exact match after :func:`fold_name`. Two absent names never match."""
    fa, fb = fold_name(a), fold_name(b)
    return bool(fa) and fa == fb


# ---------------------------------------------------------------------------
# Nationality normalization (code vs. adjective, e.g. "ESP" vs "Española")
# ---------------------------------------------------------------------------

#: ISO 3166-1 alpha-3/alpha-2 codes seen in source documents (e.g. a passport's
#: MRZ) mapped to the Spanish nationality adjective used elsewhere in the same
#: case's documents. Scoped to codes actually observed in test data — extend
#: as more folders are tested, not a general ISO 3166 lookup.
_NATIONALITY_CODE_TO_NAME: dict[str, str] = {
    "ESP": "Española",
    "ES": "Española",
}


def fold_nationality(value: Optional[str]) -> str:
    """Fold a nationality field for comparison: resolve an ISO code (e.g.
    ``ESP``) to its Spanish adjective (``Española``) before applying
    :func:`fold_name`, so a code and its spelled-out equivalent compare equal.
    """
    if not value:
        return ""
    resolved = _NATIONALITY_CODE_TO_NAME.get(nfkc(value).strip().upper(), value)
    return fold_name(resolved)


def names_fuzzy_match(a: Optional[str], b: Optional[str]) -> bool:
    """Loose match after :func:`fold_name`: exact, or one folded name contains
    the other as a whole-token subsequence (handles a missing middle name or
    a reordered apellido/nombre pair, both common in these documents).
    """
    fa, fb = fold_name(a), fold_name(b)
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    tokens_a, tokens_b = set(fa.split()), set(fb.split())
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    return bool(shorter) and shorter.issubset(longer)


# ---------------------------------------------------------------------------
# Document-number normalization (for exact person/ID comparison)
# ---------------------------------------------------------------------------

_DOC_NUMBER_STRIP_RE = re.compile(r"[\s\-.]+")


def fold_doc_number(num: Optional[str]) -> str:
    """Normalize a document number for comparison: NFKC, uppercase, strip
    whitespace/hyphens/periods (formatting-insensitive, per the source's
    "Document number (stripped of formatting)" entity-resolution criterion).
    """
    if not num:
        return ""
    return _DOC_NUMBER_STRIP_RE.sub("", nfkc(num)).upper()


def doc_numbers_match(a: Optional[str], b: Optional[str]) -> bool:
    """Exact match after :func:`fold_doc_number`. Two absent numbers never match."""
    fa, fb = fold_doc_number(a), fold_doc_number(b)
    return bool(fa) and fa == fb


# ---------------------------------------------------------------------------
# Date parsing (for the ±1 year DOB-match rule and generational gap math)
# ---------------------------------------------------------------------------

#: Formats observed/expected in the AEAD source's Spanish document fields.
#: Tried in order; the first that parses the full string wins.
_DATE_FORMATS: tuple[str, ...] = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%d %m %Y",
)

#: Spanish month names/abbreviations, e.g. ``30/Nov/2015`` or ``30 de
#: noviembre de 2015``. ``strptime``'s ``%b``/``%B`` are locale-dependent (and
#: default to English), so month names are resolved through this table
#: instead of relying on the platform locale.
_SPANISH_MONTHS: dict[str, int] = {
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "jun": 6,
    "julio": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "setiembre": 9, "sep": 9, "sept": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11,
    "diciembre": 12, "dic": 12,
}

#: Spanish number words needed to parse a spelled-out year, e.g. "MIL
#: NOVECIENTOS SESENTA" (1960). Scoped to the vocabulary actually seen in
#: source documents (whole hundreds/tens/units plus "mil"), not a general
#: Spanish numeral parser.
_SPANISH_UNITS: dict[str, int] = {
    "cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}
_SPANISH_TENS: dict[str, int] = {
    "diez": 10, "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
}
_SPANISH_HUNDREDS: dict[str, int] = {
    "cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300,
    "cuatrocientos": 400, "quinientos": 500, "seiscientos": 600,
    "setecientos": 700, "ochocientos": 800, "novecientos": 900,
}

_SPANISH_DE_DATE_RE = re.compile(
    r"^(\d{1,2})\s+de\s+([a-zñ]+)\s+de\s+(\d{3,4})$"
)
_SPANISH_SLASH_MONTH_RE = re.compile(r"^(\d{1,2})/([a-zñ]+)/(\d{3,4})$")
_SPANISH_SPACE_MONTH_RE = re.compile(r"^(\d{1,2})\s+([a-zñ]+)\s+(\d{3,4})$")
_SPANISH_WORDS_DATE_RE = re.compile(r"^(\d{1,2})\s+([a-zñ]+)\s+([a-zñ\s]+)$")


def _spanish_year_words_to_int(words: list[str]) -> Optional[int]:
    """Convert spelled-out Spanish year words (e.g. ``["mil", "novecientos",
    "sesenta"]`` -> 1960) to an int, or ``None`` if any token is unrecognized.
    """
    total = 0
    current = 0
    for word in words:
        if word == "y":
            continue
        if word == "mil":
            total += (current or 1) * 1000
            current = 0
        elif word in _SPANISH_HUNDREDS:
            current += _SPANISH_HUNDREDS[word]
        elif word in _SPANISH_TENS:
            current += _SPANISH_TENS[word]
        elif word in _SPANISH_UNITS:
            current += _SPANISH_UNITS[word]
        else:
            return None
    total += current
    return total or None


def _parse_spanish_named_date(candidate: str) -> Optional[date]:
    """Parse Spanish month-name dates: ``30/Nov/2015``, ``30 de noviembre de
    2015``, ``22 Octubre 2025`` (plain spaces, no "de"), or a fully
    spelled-out date including a worded year (e.g. ``27 SEPTIEMBRE MIL
    NOVECIENTOS SESENTA``). Returns ``None`` on no match.
    """
    lowered = candidate.lower()

    match = (
        _SPANISH_SLASH_MONTH_RE.match(lowered)
        or _SPANISH_DE_DATE_RE.match(lowered)
        or _SPANISH_SPACE_MONTH_RE.match(lowered)
    )
    if match:
        day, month_name, year = match.groups()
        month = _SPANISH_MONTHS.get(month_name)
        if month is None:
            return None
        try:
            return date(int(year), month, int(day))
        except ValueError:
            return None

    match = _SPANISH_WORDS_DATE_RE.match(lowered)
    if match:
        day, month_name, year_words = match.groups()
        month = _SPANISH_MONTHS.get(month_name)
        if month is None:
            return None
        year = _spanish_year_words_to_int(year_words.split())
        if year is None:
            return None
        try:
            return date(year, month, int(day))
        except ValueError:
            return None

    return None


def parse_date(value: Optional[str]) -> Optional[date]:
    """Best-effort parse of a Spanish-locale date field. Returns ``None``
    (never raises) if ``value`` is absent or matches no known format — an
    unparseable date is a missing-field/flag concern for the caller, not this
    function's job.
    """
    if not value:
        return None
    candidate = nfkc(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    return _parse_spanish_named_date(candidate)


def years_between(earlier: date, later: date) -> int:
    """Whole-year gap between two dates, ``later`` assumed >= ``earlier``.

    Uses calendar-aware whole years (not ``(later - earlier).days / 365.25``)
    so a birthday that hasn't occurred yet this year is not counted, matching
    how "age" and "generational gap in years" are meant colloquially in the
    source's 14-60/20-35 year bands.
    """
    years = later.year - earlier.year
    if (later.month, later.day) < (earlier.month, earlier.day):
        years -= 1
    return years


def dates_within_years(a: Optional[date], b: Optional[date], *, tolerance_years: int) -> bool:
    """Whether two dates are within ``tolerance_years`` of each other.

    Both ``None`` -> ``False`` (no evidence of a match). Used for the source's
    "Birth date (±1 year)" entity-resolution criterion.
    """
    if a is None or b is None:
        return False
    earlier, later = (a, b) if a <= b else (b, a)
    return years_between(earlier, later) <= tolerance_years


def add_months(d: date, months: int) -> date:
    """Add ``months`` calendar months to ``d``.

    Clamps the day to the resulting month's last day when the original day
    doesn't exist there (e.g. 31 Jan + 1 month -> 28/29 Feb), rather than
    raising or silently overflowing into the following month.
    """
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)
