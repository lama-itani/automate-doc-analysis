"""Deterministic Conflict Flag detectors (M-4, step 5).

The AEAD source's Data Aggregation agent lists nine named "Conflict Flags" in
its ``workflow_template.json`` ``goal`` (six defined there verbatim, plus the
three the session-10 scoping decision added concrete definitions for:
``DOCUMENTO_VENCIDO``, ``NOMBRE_INCONSISTENTE``, ``ID_INCONSISTENTE`` — see
:mod:`ps06.rules.rules_config`'s module docstring). This module is one pure,
stateless predicate function per flag.

Deliberately decoupled from :mod:`ps06.rules.schemas`/entity resolution: the
Lineage Engine (step 6, not yet built) is what assembles "these raw values
across documents all belong to the same resolved person" — this module only
answers "given values already grouped that way, do they conflict?". That
keeps every flag independently unit-testable against plain values (dates,
strings, counts) today, without depending on code that doesn't exist yet.

No I/O, no case/document identity, no config loading.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

from ps06.rules.normalize import (
    fold_doc_number,
    fold_nationality,
    names_fuzzy_match,
    years_between,
)
from ps06.rules.rules_config import GapBand


def document_incomplete(missing_fields: tuple[str, ...]) -> bool:
    """``DOCUMENT_INCOMPLETE``: the document is missing key fields.

    Directly consumes :attr:`ps06.rules.schemas.CanonicalDocument.missing_fields`
    (populated by the step-4 adapter) — non-empty means incomplete.
    """
    return bool(missing_fields)


def documento_vencido(fecha_vencimiento: Optional[date], reference_date: date) -> bool:
    """``DOCUMENTO_VENCIDO``: an identification's expiry date has passed.

    Definition per session-10 scoping decision #2: ``Fecha de vencimiento`` <
    ``reference_date``. An absent expiry date is not evidence of expiry
    (never fabricate a flag from missing data). ``reference_date`` is the
    caller's choice of what "has passed" means — the case's application date
    plus a validity buffer when available, falling back to the run's
    evaluation date otherwise (see ``semaphore.evaluate_document``) — this
    function only compares, it doesn't decide what the reference is.
    """
    return fecha_vencimiento is not None and fecha_vencimiento < reference_date


def nombre_ambiguo(distinct_candidate_count: int) -> bool:
    """``NOMBRE_AMBIGUO``: a name cannot be resolved to a single unique individual.

    Takes the number of distinct resolved-person candidates a name matched
    (computed by the Lineage Engine's entity resolution, step 6) — ambiguous
    when it matched more than one.
    """
    return distinct_candidate_count > 1


def nombre_inconsistente(names: Iterable[Optional[str]]) -> bool:
    """``NOMBRE_INCONSISTENTE``: the same person's name mismatches beyond fuzzy match.

    Definition per session-10 scoping decision #2: given every name recorded
    for what's presumed to be one resolved person, any pair that fails
    :func:`ps06.rules.normalize.names_fuzzy_match` is a conflict.
    """
    present = [n for n in names if n]
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            if not names_fuzzy_match(a, b):
                return True
    return False


def id_inconsistente(doc_numbers: Iterable[Optional[str]]) -> bool:
    """``ID_INCONSISTENTE``: the same person has differing document numbers.

    Definition per session-10 scoping decision #2: same resolved person,
    differing normalized (:func:`ps06.rules.normalize.fold_doc_number`)
    document numbers across documents.
    """
    folded = {fold_doc_number(d) for d in doc_numbers if d}
    return len(folded) > 1


def documento_cruzado(names: Iterable[Optional[str]]) -> bool:
    """``DOCUMENTO_CRUZADO``: one document number is assigned to two different names.

    Mirror of :func:`id_inconsistente` in the other direction: given every
    name recorded against what's presumed to be a single document number, any
    pair that fails :func:`ps06.rules.normalize.names_fuzzy_match` means the
    number is cross-assigned — same fuzzy-match tolerance as
    :func:`nombre_inconsistente`, since OCR/VLM noise (field-order swaps,
    missing middle names) is not evidence of a real cross-assignment.
    """
    present = [n for n in names if n]
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            if not names_fuzzy_match(a, b):
                return True
    return False


def fecha_conflicto(birth_dates: Iterable[Optional[date]]) -> bool:
    """``FECHA_CONFLICTO``: the same individual has two different birth dates.

    Any two distinct non-absent dates recorded for what's presumed to be one
    individual is a conflict — unlike entity-resolution matching, there is no
    tolerance window here: a real conflicting record is exactly the point.
    """
    present = {d for d in birth_dates if d is not None}
    return len(present) > 1


def nacionalidad_inconsistente(nationalities: Iterable[Optional[str]]) -> bool:
    """``NACIONALIDAD_INCONSISTENTE``: stated nationality differs across documents.

    Uses :func:`ps06.rules.normalize.fold_nationality`, which resolves an ISO
    code (e.g. ``ESP``) to its Spanish adjective (``Española``) before the
    usual case/accent/whitespace folding — a code and its spelled-out
    equivalent are the same nationality, not a conflict.
    """
    folded = {fold_nationality(n) for n in nationalities if n}
    return len(folded) > 1


def brecha_generacional(parent_birth: date, child_birth: date, hard_band: GapBand) -> bool:
    """``BRECHA_GENERACIONAL``: the parent/child age gap falls outside the hard band.

    Order-independent (mirrors :func:`ps06.rules.normalize.dates_within_years`'s
    earlier/later handling) so a chronologically reversed pair is still
    evaluated rather than producing a meaningless negative gap.
    """
    earlier, later = (
        (parent_birth, child_birth)
        if parent_birth <= child_birth
        else (child_birth, parent_birth)
    )
    gap = years_between(earlier, later)
    return not (hard_band.min_years <= gap <= hard_band.max_years)
