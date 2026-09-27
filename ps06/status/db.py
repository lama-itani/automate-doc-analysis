"""SQLite connection + schema migrations for the PS-06 status table (M-1).

Pilot-phase storage. Deliberately dependency-free (stdlib ``sqlite3`` only) and
small: a ``schema_version`` table drives an idempotent, ordered migration runner
so ``connect()`` can be called safely on a fresh or already-migrated database.

The SQLite -> hardened-store migration is a tracked open item (ENS-High audit
readiness), not part of M-1.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

#: In-memory database sentinel accepted by :func:`connect` (useful for tests).
MEMORY = ":memory:"


def connect(db_path: str | Path = MEMORY) -> sqlite3.Connection:
    """Open (creating if needed) the status database and run migrations.

    Returns a connection with:
      * ``row_factory`` = :class:`sqlite3.Row` for name-based column access,
      * foreign keys enabled, and
      * the schema migrated up to the latest version.

    Passing :data:`MEMORY` (the default) yields an ephemeral database — each call
    is a distinct in-memory DB, which suits unit tests.
    """

    if db_path != MEMORY:
        # Ensure the parent directory exists for file-backed databases.
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    migrate(conn)
    return conn


# --- Migrations -------------------------------------------------------------
#
# Each migration is a (version, apply_fn) pair. Versions are applied in ascending
# order and only when strictly greater than the DB's current recorded version.
# To evolve the schema in a later milestone, append a new pair — never edit or
# reorder an existing one.


def _migration_0001_initial(conn: sqlite3.Connection) -> None:
    """Create the ``document_status`` table (one row per case+document)."""

    conn.execute(
        """
        CREATE TABLE document_status (
            case_id       TEXT    NOT NULL,
            document_id   TEXT    NOT NULL,
            stage         TEXT    NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 1,
            last_updated  TEXT    NOT NULL,   -- ISO-8601 UTC
            error_detail  TEXT,               -- populated only on FAILED
            PRIMARY KEY (case_id, document_id)
        );
        """
    )
    # Case-level aggregation queries filter by case_id, so index it.
    conn.execute(
        "CREATE INDEX idx_document_status_case_id ON document_status (case_id);"
    )


def _migration_0002_document_extraction(conn: sqlite3.Connection) -> None:
    """Create the ``document_extraction`` table (M-2: per-document OCR output).

    One row per (case, document), holding the job's raw extraction result as a
    JSON blob (see ``ps06.ocr.envelope.OcrResult``). A retried document's row is
    overwritten, not accumulated — the table reflects the latest attempt only,
    consistent with ``document_status.attempt_count`` tracking retries rather
    than history.
    """

    conn.execute(
        """
        CREATE TABLE document_extraction (
            case_id            TEXT    NOT NULL,
            document_id        TEXT    NOT NULL,
            extracted_at       TEXT    NOT NULL,   -- ISO-8601 UTC
            processing_seconds REAL    NOT NULL,
            payload            TEXT    NOT NULL,   -- OcrResult, JSON-serialized
            PRIMARY KEY (case_id, document_id)
        );
        """
    )


#: Ordered migration ledger. Append-only.
MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _migration_0001_initial),
    (2, _migration_0002_document_extraction),
]

#: Latest schema version defined in this module.
SCHEMA_VERSION: int = max(version for version, _ in MIGRATIONS)


def _current_version(conn: sqlite3.Connection) -> int:
    """Return the DB's recorded schema version (0 if never migrated)."""

    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);"
    )
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version;").fetchone()
    return row["v"] if row and row["v"] is not None else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations in order. Idempotent.

    Each pending migration and its version stamp are committed together in one
    transaction, so an interrupted run leaves the DB at a clean version boundary.
    Returns the schema version after migrating.
    """

    current = _current_version(conn)
    for version, apply_fn in sorted(MIGRATIONS, key=lambda m: m[0]):
        if version <= current:
            continue
        with conn:  # BEGIN/COMMIT; rolls back on exception
            apply_fn(conn)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?);", (version,)
            )
        current = version
    return current
