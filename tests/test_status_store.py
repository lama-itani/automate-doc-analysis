"""Unit tests for the M-1 status table: states, db/migrations, store, aggregation."""

from __future__ import annotations

import sqlite3

import pytest

from ps06.status import db
from ps06.status.aggregate import derive_case_status
from ps06.status.states import (
    CaseStatus,
    DocumentStage,
    allowed_targets,
    is_valid_transition,
)
from ps06.status.store import (
    DocumentAlreadyExists,
    DocumentNotFound,
    DocumentStatus,
    InvalidTransition,
    StatusStore,
)


@pytest.fixture()
def store() -> StatusStore:
    """A StatusStore over a fresh in-memory, migrated database."""
    return StatusStore(db.connect())


# --- states -----------------------------------------------------------------


class TestStates:
    def test_str_valued_enums(self):
        assert DocumentStage.OCR_DONE == "OCR_DONE"
        assert CaseStatus.FAILED == "FAILED"

    @pytest.mark.parametrize(
        "current,target",
        [
            (DocumentStage.RECEIVED, DocumentStage.OCR_DONE),
            (DocumentStage.RECEIVED, DocumentStage.FAILED),
            (DocumentStage.OCR_DONE, DocumentStage.FAILED),
            (DocumentStage.FAILED, DocumentStage.RECEIVED),
            (DocumentStage.FAILED, DocumentStage.OCR_DONE),
        ],
    )
    def test_valid_transitions(self, current, target):
        assert is_valid_transition(current, target)

    @pytest.mark.parametrize(
        "current,target",
        [
            (DocumentStage.OCR_DONE, DocumentStage.RECEIVED),  # no backward progress
            (DocumentStage.RECEIVED, DocumentStage.RECEIVED),  # no self-transition
            (DocumentStage.OCR_DONE, DocumentStage.OCR_DONE),
            (DocumentStage.FAILED, DocumentStage.FAILED),
        ],
    )
    def test_invalid_transitions(self, current, target):
        assert not is_valid_transition(current, target)

    def test_allowed_targets(self):
        assert allowed_targets(DocumentStage.OCR_DONE) == frozenset(
            {DocumentStage.FAILED}
        )


# --- db / migrations --------------------------------------------------------


class TestDbMigrations:
    def test_connect_migrates_to_latest(self):
        conn = db.connect()
        assert db._current_version(conn) == db.SCHEMA_VERSION == 1

    def test_schema_shape(self):
        conn = db.connect()
        cols = {r["name"]: r for r in conn.execute("PRAGMA table_info(document_status)")}
        assert set(cols) == {
            "case_id",
            "document_id",
            "stage",
            "attempt_count",
            "last_updated",
            "error_detail",
        }
        # composite primary key on (case_id, document_id)
        assert cols["case_id"]["pk"] == 1
        assert cols["document_id"]["pk"] == 2
        # error_detail is the only nullable column
        assert cols["error_detail"]["notnull"] == 0
        indexes = {r["name"] for r in conn.execute("PRAGMA index_list('document_status')")}
        assert "idx_document_status_case_id" in indexes

    def test_migrate_is_idempotent(self):
        conn = db.connect()
        db.migrate(conn)
        db.migrate(conn)
        count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == 1

    def test_file_backed_persists(self, tmp_path):
        path = tmp_path / "nested" / "status.db"
        conn1 = db.connect(path)
        StatusStore(conn1).register_document("C", "d")
        conn1.close()
        assert path.exists()
        # reconnect: migration is a no-op and data survives
        conn2 = db.connect(path)
        assert db._current_version(conn2) == 1
        assert StatusStore(conn2).get("C", "d") is not None


# --- store: register --------------------------------------------------------


class TestRegister:
    def test_register_starts_at_received_attempt_1(self, store):
        s = store.register_document("C1", "docA")
        assert isinstance(s, DocumentStatus)
        assert s.stage is DocumentStage.RECEIVED
        assert s.attempt_count == 1
        assert s.error_detail is None
        assert s.last_updated.endswith("Z")

    def test_duplicate_register_raises(self, store):
        store.register_document("C1", "docA")
        with pytest.raises(DocumentAlreadyExists):
            store.register_document("C1", "docA")

    def test_same_document_id_across_cases_is_independent(self, store):
        store.register_document("C1", "shared")
        store.register_document("C2", "shared")  # different case, must not clash
        assert store.get("C1", "shared").case_id == "C1"
        assert store.get("C2", "shared").case_id == "C2"


# --- store: transitions -----------------------------------------------------


class TestTransitions:
    def test_happy_path_no_attempt_bump_clears_error(self, store):
        store.register_document("C1", "docA")
        s = store.transition("C1", "docA", DocumentStage.OCR_DONE)
        assert s.stage is DocumentStage.OCR_DONE
        assert s.attempt_count == 1  # normal progress, not a retry
        assert s.error_detail is None

    def test_illegal_transition_raises_and_leaves_row_unchanged(self, store):
        store.register_document("C1", "docA")
        store.transition("C1", "docA", DocumentStage.OCR_DONE)
        with pytest.raises(InvalidTransition):
            store.transition("C1", "docA", DocumentStage.RECEIVED)
        # unchanged
        assert store.get("C1", "docA").stage is DocumentStage.OCR_DONE

    def test_transition_missing_document_raises(self, store):
        with pytest.raises(DocumentNotFound):
            store.transition("C1", "ghost", DocumentStage.OCR_DONE)

    def test_failed_stores_error_detail(self, store):
        store.register_document("C1", "docA")
        s = store.transition("C1", "docA", DocumentStage.FAILED, error_detail="knox 401")
        assert s.stage is DocumentStage.FAILED
        assert s.error_detail == "knox 401"
        assert s.attempt_count == 1

    def test_retry_out_of_failed_bumps_attempt_and_clears_error(self, store):
        store.register_document("C1", "docA")
        store.transition("C1", "docA", DocumentStage.FAILED, error_detail="boom")
        s = store.transition("C1", "docA", DocumentStage.OCR_DONE)  # retry
        assert s.stage is DocumentStage.OCR_DONE
        assert s.attempt_count == 2
        assert s.error_detail is None

    def test_retry_to_received_then_ocr_counts_one_new_attempt(self, store):
        store.register_document("C1", "docA")
        store.transition("C1", "docA", DocumentStage.FAILED, error_detail="x")
        store.transition("C1", "docA", DocumentStage.RECEIVED)  # leaving FAILED -> attempt 2
        s = store.transition("C1", "docA", DocumentStage.OCR_DONE)  # progress, no bump
        assert s.attempt_count == 2


# --- store: reads -----------------------------------------------------------


class TestReads:
    def test_get_missing_returns_none(self, store):
        assert store.get("C1", "nope") is None

    def test_get_or_raise_missing(self, store):
        with pytest.raises(DocumentNotFound):
            store.get_or_raise("C1", "nope")

    def test_list_for_case_ordered_and_scoped(self, store):
        store.register_document("C1", "b")
        store.register_document("C1", "a")
        store.register_document("C2", "z")
        docs = store.list_for_case("C1")
        assert [d.document_id for d in docs] == ["a", "b"]  # ordered by document_id
        assert all(d.case_id == "C1" for d in docs)  # scoped to the case


# --- aggregation (pure) -----------------------------------------------------


class TestDeriveCaseStatus:
    def test_all_received(self):
        assert derive_case_status([DocumentStage.RECEIVED]) is CaseStatus.RECEIVED

    def test_all_ocr_done(self):
        assert (
            derive_case_status([DocumentStage.OCR_DONE, DocumentStage.OCR_DONE])
            is CaseStatus.OCR_DONE
        )

    def test_mixed_in_progress_is_received(self):
        assert (
            derive_case_status([DocumentStage.RECEIVED, DocumentStage.OCR_DONE])
            is CaseStatus.RECEIVED
        )

    @pytest.mark.parametrize(
        "stages",
        [
            [DocumentStage.OCR_DONE, DocumentStage.FAILED],
            [DocumentStage.RECEIVED, DocumentStage.FAILED],
            [DocumentStage.FAILED],
        ],
    )
    def test_failure_dominates(self, stages):
        assert derive_case_status(stages) is CaseStatus.FAILED

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            derive_case_status([])


# --- aggregation via store --------------------------------------------------


class TestCaseStatusViaStore:
    def test_no_documents_returns_none(self, store):
        assert store.case_status("unknown-case") is None

    def test_derived_progression(self, store):
        store.register_document("C1", "a")
        store.register_document("C1", "b")
        assert store.case_status("C1") is CaseStatus.RECEIVED

        store.transition("C1", "a", DocumentStage.OCR_DONE)
        assert store.case_status("C1") is CaseStatus.RECEIVED  # b still pending

        store.transition("C1", "b", DocumentStage.OCR_DONE)
        assert store.case_status("C1") is CaseStatus.OCR_DONE  # all done

        store.transition("C1", "b", DocumentStage.FAILED, error_detail="boom")
        assert store.case_status("C1") is CaseStatus.FAILED  # failure dominates
