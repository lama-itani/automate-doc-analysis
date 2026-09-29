"""Case-level status derivation for the PS-06 status table (M-1).

A case's status is never stored — it is *derived* by aggregating the per-document
stages recorded in the status table. This is the mechanism the orchestrator uses to
know when a case is ready for the next stage (e.g. rules evaluation once every
document has reached ``OCR_DONE``) and to surface case-level failure.

M-1 documents only ever reach ``RECEIVED``, ``OCR_DONE``, or ``FAILED``.
``RULES_EVALUATED`` (M-4) is layered on top via the ``rules_evaluated`` flag
below, derived from the presence of a ``rule_evaluation`` row rather than any
per-document stage — ``RENDERED`` (M-5) is still not produced here.

The core function is pure (stages + flag in, status out) so it is trivially
unit-testable in isolation from the database.
"""

from __future__ import annotations

from typing import Iterable

from ps06.status.states import CaseStatus, DocumentStage


def derive_case_status(
    stages: Iterable[DocumentStage], *, rules_evaluated: bool = False
) -> CaseStatus:
    """Derive a case-level :class:`CaseStatus` from its documents' stages.

    Rules (evaluated in priority order):
      1. **any** document ``FAILED``                          -> :attr:`CaseStatus.FAILED`
      2. else **all** ``OCR_DONE`` and ``rules_evaluated``     -> :attr:`CaseStatus.RULES_EVALUATED`
      3. else **all** documents ``OCR_DONE``                   -> :attr:`CaseStatus.OCR_DONE`
      4. otherwise (still ingesting/processing)                -> :attr:`CaseStatus.RECEIVED`

    Failure dominates: a case with any failed document is ``FAILED`` regardless of
    the others or of ``rules_evaluated``, so the condition is visible rather than
    masked by in-progress or already-evaluated work. ``RULES_EVALUATED`` requires
    every document to have reached ``OCR_DONE`` first — it is a downstream stage,
    not a substitute for OCR completion.

    Raises :class:`ValueError` if ``stages`` is empty — a case with no documents has
    no derivable status; callers should treat "no documents" as a distinct condition
    (see :meth:`ps06.status.store.StatusStore.case_status`, which returns ``None``).
    """
    stage_list = list(stages)
    if not stage_list:
        raise ValueError("cannot derive case status from zero documents")

    if any(s is DocumentStage.FAILED for s in stage_list):
        return CaseStatus.FAILED
    if all(s is DocumentStage.OCR_DONE for s in stage_list):
        return CaseStatus.RULES_EVALUATED if rules_evaluated else CaseStatus.OCR_DONE
    return CaseStatus.RECEIVED
