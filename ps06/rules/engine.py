"""Case-level rules-engine entry point (M-4, step 10).

Wires the pure pieces built in steps 1-8 into one call per case: adapts a
case's persisted ``OcrResult``s into a :class:`~ps06.rules.schemas.CaseBundle`
(:mod:`ps06.rules.adapter`), resolves lineage (:mod:`ps06.rules.lineage`),
evaluates the VERDE/AMARILLO/ROJO semaphore (:mod:`ps06.rules.semaphore`), and
formats the result into a :class:`~ps06.rules.snapshot.ReportSnapshot`
(:mod:`ps06.rules.snapshot`).

Readiness (whether every document in the case has reached ``OCR_DONE``) is
the caller's responsibility, not this module's — :func:`build_case_bundle`
already raises :class:`~ps06.rules.adapter.MissingExtractionError` if a
registered document has no persisted extraction, which is the actual
failure mode worth guarding against here.

There is no case-level ``FAILED`` stage to write on error — unlike
``ps06.status.states.DocumentStage``, ``CaseStatus.RULES_EVALUATED`` is
*derived* (see :func:`ps06.status.aggregate.derive_case_status`) purely from
whether a ``rule_evaluation`` row exists for the case. So this module's "no
silent/partial RULES_EVALUATED" contract is satisfied differently than
``ocr/job.py``'s: rather than recording a ``FAILED`` transition, it simply
never writes the ``rule_evaluation`` row until the snapshot is fully built,
and re-raises any exception unwrapped. A half-built case is never persisted,
so the derived status never lies.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from ps06.rules.adapter import build_case_bundle
from ps06.rules.lineage import resolve_lineage
from ps06.rules.rules_config import RulesConfig
from ps06.rules.semaphore import evaluate_semaphore
from ps06.rules.snapshot import ReportSnapshot, build_snapshot
from ps06.status.store import StatusStore


def _utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision, ``Z`` suffix).

    Mirrors ``ps06.ocr.envelope._utcnow_iso`` rather than importing it, same
    reasoning: a 3-line pure function isn't worth a cross-package dependency.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def evaluate_case(
    case_id: str,
    store: StatusStore,
    config: RulesConfig,
    evaluation_date: date,
) -> ReportSnapshot:
    """Evaluate one case's rules and persist the resulting snapshot.

    Builds the case's :class:`~ps06.rules.schemas.CaseBundle` first, outside
    the ``try`` — :class:`~ps06.rules.adapter.MissingExtractionError` from a
    caller running this before every document has a persisted extraction
    propagates unwrapped, the same way ``ocr/job.py``'s ``DocumentNotFound``
    does for a caller-contract violation, not a processing failure.

    Returns the persisted :class:`~ps06.rules.snapshot.ReportSnapshot` on
    success. Raises (without persisting anything) on any failure.
    """
    bundle = build_case_bundle(store, case_id)

    lineage = resolve_lineage(bundle, config)
    decision = evaluate_semaphore(bundle, lineage, config, evaluation_date)
    snapshot = build_snapshot(decision, bundle)
    _persist_snapshot(store, snapshot)

    return snapshot


def get_snapshot(store: StatusStore, case_id: str) -> ReportSnapshot | None:
    """Read one case's persisted :class:`ReportSnapshot`, or ``None`` if not yet evaluated.

    Mirrors :func:`ps06.ocr.envelope.get_extraction`'s shape — the shared
    accessor for the ``rule_evaluation`` table's ``payload`` column, so
    callers (e.g. ``rules/cli.py show-case``) don't need ad hoc raw SQL.
    """
    row = store.connection.execute(
        "SELECT payload FROM rule_evaluation WHERE case_id = ?;",
        (case_id,),
    ).fetchone()
    if row is None:
        return None
    return ReportSnapshot.model_validate_json(row["payload"])


def _persist_snapshot(store: StatusStore, snapshot: ReportSnapshot) -> None:
    """Write ``snapshot`` into the ``rule_evaluation`` table.

    A re-run's row is overwritten (``INSERT OR REPLACE``), matching
    ``document_extraction``'s "latest attempt only" semantics. Runs on the
    same connection ``store`` wraps, mirroring ``ocr/job.py::_persist_extraction``.
    """
    conn = store.connection
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO rule_evaluation (case_id, evaluated_at, payload)
            VALUES (?, ?, ?);
            """,
            (
                snapshot.case_id,
                _utcnow_iso(),
                json.dumps(snapshot.model_dump(mode="json")),
            ),
        )
