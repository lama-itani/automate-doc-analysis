"""State vocabulary for the PS-06 status table (M-1).

Two distinct concepts live here:

* ``DocumentStage`` — the per-document state written by job invocations. One row
  per (case, document) in the status table carries exactly one of these. Under
  per-invocation token isolation, each document advances independently, so this
  is the level the table records directly.

* ``CaseStatus`` — the case-level status. This is **derived** by aggregating the
  stages of all a case's documents (see ``ps06.status.aggregate``); it is never
  written to a document row. ``RULES_EVALUATED`` and ``RENDERED`` are produced by
  later milestones (M-4 / M-5) and are included here for forward compatibility.

This module is pure: enums plus a validated transition map, no I/O.
"""

from __future__ import annotations

from enum import Enum


class DocumentStage(str, Enum):
    """Per-document stage recorded in the status table.

    ``str``-valued so the members serialize directly to the ``stage`` TEXT column
    and compare cleanly against raw DB values.
    """

    RECEIVED = "RECEIVED"
    OCR_DONE = "OCR_DONE"
    FAILED = "FAILED"


class CaseStatus(str, Enum):
    """Case-level status derived by aggregation over a case's document stages.

    Not stored per row. ``RULES_EVALUATED`` and ``RENDERED`` are placeholders for
    M-4 / M-5 and are not yet produced by M-1's aggregation.
    """

    RECEIVED = "RECEIVED"
    OCR_DONE = "OCR_DONE"
    RULES_EVALUATED = "RULES_EVALUATED"
    RENDERED = "RENDERED"
    FAILED = "FAILED"


# Allowed per-document transitions. A fresh registration starts at RECEIVED
# (handled by the store, not a transition). FAILED is reachable from any active
# stage; a FAILED document may be retried back into a processing stage as a fresh
# invocation, which is why FAILED -> RECEIVED / OCR_DONE are permitted.
_ALLOWED_TRANSITIONS: dict[DocumentStage, frozenset[DocumentStage]] = {
    DocumentStage.RECEIVED: frozenset({DocumentStage.OCR_DONE, DocumentStage.FAILED}),
    DocumentStage.OCR_DONE: frozenset({DocumentStage.FAILED}),
    DocumentStage.FAILED: frozenset({DocumentStage.RECEIVED, DocumentStage.OCR_DONE}),
}


def is_valid_transition(current: DocumentStage, target: DocumentStage) -> bool:
    """Return True if a document may move from ``current`` to ``target``.

    A no-op transition (``current == target``) is not allowed — callers that want
    to re-record the same stage (e.g. a retry of a still-processing document)
    should express that explicitly rather than relying on a self-transition.
    """

    return target in _ALLOWED_TRANSITIONS.get(current, frozenset())


def allowed_targets(current: DocumentStage) -> frozenset[DocumentStage]:
    """Return the set of stages reachable from ``current`` in one transition."""

    return _ALLOWED_TRANSITIONS.get(current, frozenset())
