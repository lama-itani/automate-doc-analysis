"""Accessors for the PS-06 status table (M-1).

``StatusStore`` wraps a migrated SQLite connection (see :mod:`ps06.status.db`)
and is the single choke point for reading and writing document status. All
stage changes go through :meth:`StatusStore.transition`, which enforces the
validated state machine in :mod:`ps06.status.states`, stamps ``last_updated``,
manages ``attempt_count``, and sets/clears ``error_detail``.

Case-level status is *not* computed here — it is derived by aggregation in
:mod:`ps06.status.aggregate` (Step 5), which consumes the rows this store returns.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict

from ps06.status.aggregate import derive_case_status
from ps06.status.states import CaseStatus, DocumentStage, is_valid_transition


# --- Errors -----------------------------------------------------------------


class StatusStoreError(Exception):
    """Base class for status-store errors."""


class DocumentNotFound(StatusStoreError):
    """Raised when a (case_id, document_id) row does not exist."""


class DocumentAlreadyExists(StatusStoreError):
    """Raised when registering a (case_id, document_id) that already exists."""


class InvalidTransition(StatusStoreError):
    """Raised when a stage change violates the state machine."""


class TransitionConflict(StatusStoreError):
    """Raised when a document's stage changed concurrently mid-transition.

    :meth:`StatusStore.transition` validates against the stage it read, then
    applies a compare-and-swap ``UPDATE`` guarded on that same stage. If another
    writer moved the row in between, the guarded update matches zero rows and
    this is raised instead of silently clobbering the other writer's change
    (HARDENING_NOTES #2). Under the orchestrator's one-owner-per-document
    invariant this should not occur in normal operation; when it does it is a
    real, visible conflict — never a silent lost update.
    """


# --- Row model --------------------------------------------------------------


class DocumentStatus(BaseModel):
    """One row of the ``document_status`` table."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    document_id: str
    stage: DocumentStage
    attempt_count: int
    last_updated: str  # ISO-8601 UTC
    error_detail: Optional[str] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DocumentStatus":
        """Build a model from a ``sqlite3.Row``."""
        return cls(
            case_id=row["case_id"],
            document_id=row["document_id"],
            stage=DocumentStage(row["stage"]),
            attempt_count=row["attempt_count"],
            last_updated=row["last_updated"],
            error_detail=row["error_detail"],
        )


def _utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision, ``Z`` suffix)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


class StatusStore:
    """Read/write accessor over a migrated status database.

    The caller owns the connection lifecycle; the store does not close it.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection, for callers writing to other tables in
        the same migrated database (e.g. M-2's ``document_extraction``) that
        need to share this store's transaction/connection lifecycle rather
        than opening a second one."""
        return self._conn

    # -- writes --------------------------------------------------------------

    def register_document(
        self, case_id: str, document_id: str
    ) -> DocumentStatus:
        """Register a new document at :attr:`DocumentStage.RECEIVED`, attempt 1.

        Raises :class:`DocumentAlreadyExists` if the row already exists — use
        :meth:`transition` to move an existing document.
        """
        now = _utcnow_iso()
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO document_status
                        (case_id, document_id, stage, attempt_count,
                         last_updated, error_detail)
                    VALUES (?, ?, ?, 1, ?, NULL);
                    """,
                    (case_id, document_id, DocumentStage.RECEIVED.value, now),
                )
        except sqlite3.IntegrityError as exc:
            raise DocumentAlreadyExists(
                f"document already registered: case={case_id!r} "
                f"document={document_id!r}"
            ) from exc
        return self.get_or_raise(case_id, document_id)

    def transition(
        self,
        case_id: str,
        document_id: str,
        target: DocumentStage,
        error_detail: Optional[str] = None,
    ) -> DocumentStatus:
        """Move a document to ``target``, enforcing the state machine.

        Behaviour:
          * The row must exist (else :class:`DocumentNotFound`).
          * The move must be allowed by
            :func:`ps06.status.states.is_valid_transition`
            (else :class:`InvalidTransition`).
          * ``attempt_count`` is incremented when leaving
            :attr:`DocumentStage.FAILED` (i.e. a retry starts a new attempt);
            other transitions keep the current count.
          * ``error_detail`` is stored when ``target`` is
            :attr:`DocumentStage.FAILED` and cleared otherwise.
          * ``last_updated`` is always stamped.

        Concurrency: the ``UPDATE`` is a compare-and-swap guarded on the stage
        this call validated against (``WHERE ... AND stage = <current>``). If a
        concurrent writer moved the row between the read and the write, the
        guarded update matches zero rows and :class:`TransitionConflict` is
        raised rather than overwriting the other writer's change — closing the
        read-modify-write race (HARDENING_NOTES #2). (An explicit
        ``BEGIN IMMEDIATE`` around read+write is the alternative fix; the
        compare-and-swap achieves the same guarantee without depending on the
        connection's isolation-level configuration.)
        """
        current = self.get_or_raise(case_id, document_id)
        if not is_valid_transition(current.stage, target):
            raise InvalidTransition(
                f"illegal transition {current.stage.value} -> {target.value} "
                f"for case={case_id!r} document={document_id!r}"
            )

        # Leaving FAILED means a fresh processing attempt.
        new_attempt = current.attempt_count + (
            1 if current.stage is DocumentStage.FAILED else 0
        )
        new_error = error_detail if target is DocumentStage.FAILED else None
        now = _utcnow_iso()

        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE document_status
                   SET stage = ?, attempt_count = ?, last_updated = ?,
                       error_detail = ?
                 WHERE case_id = ? AND document_id = ? AND stage = ?;
                """,
                (
                    target.value,
                    new_attempt,
                    now,
                    new_error,
                    case_id,
                    document_id,
                    current.stage.value,  # compare-and-swap guard
                ),
            )
            if cursor.rowcount != 1:
                raise TransitionConflict(
                    f"stage changed concurrently mid-transition (expected "
                    f"{current.stage.value}) for case={case_id!r} "
                    f"document={document_id!r}"
                )
        return self.get_or_raise(case_id, document_id)

    # -- reads ---------------------------------------------------------------

    def get(
        self, case_id: str, document_id: str
    ) -> Optional[DocumentStatus]:
        """Return the document's status, or ``None`` if it does not exist."""
        row = self._conn.execute(
            """
            SELECT * FROM document_status
             WHERE case_id = ? AND document_id = ?;
            """,
            (case_id, document_id),
        ).fetchone()
        return DocumentStatus.from_row(row) if row is not None else None

    def get_or_raise(self, case_id: str, document_id: str) -> DocumentStatus:
        """Like :meth:`get` but raises :class:`DocumentNotFound` if absent."""
        status = self.get(case_id, document_id)
        if status is None:
            raise DocumentNotFound(
                f"no such document: case={case_id!r} document={document_id!r}"
            )
        return status

    def list_for_case(self, case_id: str) -> list[DocumentStatus]:
        """Return all document statuses for a case, ordered by ``document_id``."""
        rows = self._conn.execute(
            """
            SELECT * FROM document_status
             WHERE case_id = ?
             ORDER BY document_id;
            """,
            (case_id,),
        ).fetchall()
        return [DocumentStatus.from_row(r) for r in rows]

    def case_status(self, case_id: str) -> Optional[CaseStatus]:
        """Return the derived case-level status, or ``None`` if the case has no
        registered documents.

        Aggregates the case's document stages via
        :func:`ps06.status.aggregate.derive_case_status`, along with whether a
        ``rule_evaluation`` row exists for the case (M-4's ``RULES_EVALUATED``).
        """
        docs = self.list_for_case(case_id)
        if not docs:
            return None
        rules_evaluated = (
            self._conn.execute(
                "SELECT 1 FROM rule_evaluation WHERE case_id = ?;", (case_id,)
            ).fetchone()
            is not None
        )
        return derive_case_status(
            (d.stage for d in docs), rules_evaluated=rules_evaluated
        )
