"""Tests for ps06.rules.engine.evaluate_case — the case-level entry point (M-4, step 10)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from ps06.classification.classifier import ClassificationResult, DocumentType
from ps06.ocr.envelope import OcrResult
from ps06.ocr.extraction import ExtractionResult
from ps06.rules import engine
from ps06.rules.adapter import MissingExtractionError
from ps06.rules.rules_config import RulesConfig
from ps06.status import db
from ps06.status.store import StatusStore

CONFIG = RulesConfig()
EVAL_DATE = date(2026, 9, 29)


def _extraction(**overrides) -> ExtractionResult:
    defaults = dict(
        file_type="pdf",
        extracted_text="",
        page_count=1,
        pages_via_text_layer=1,
        pages_via_vlm=0,
        orientation_corrections={},
        acroform_fields={},
        xfa_fields={},
    )
    defaults.update(overrides)
    return ExtractionResult(**defaults)


def _result(*, case_id: str, document_id: str, classification: ClassificationResult) -> OcrResult:
    return OcrResult.build(
        case_id=case_id,
        document_id=document_id,
        file_path=f"{document_id}.pdf",
        extraction=_extraction(),
        processing_seconds=1.0,
        model_name="test-model",
        classification=classification,
    )


def _persist_extraction(store: StatusStore, result: OcrResult) -> None:
    conn = store.connection
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO document_extraction
                (case_id, document_id, extracted_at, processing_seconds, payload)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                result.case_id,
                result.document_id,
                result.extracted_at,
                result.processing_seconds,
                json.dumps(result.model_dump(mode="json")),
            ),
        )


@pytest.fixture()
def store() -> StatusStore:
    return StatusStore(db.connect())


def _seed_other_document(store: StatusStore, case_id: str, document_id: str) -> None:
    store.register_document(case_id, document_id)
    _persist_extraction(
        store,
        _result(
            case_id=case_id,
            document_id=document_id,
            classification=ClassificationResult(document_type=DocumentType.OTHER),
        ),
    )


class TestEvaluateCase:
    def test_happy_path_persists_row_and_returns_matching_snapshot(self, store):
        _seed_other_document(store, "case-1", "doc-1")

        snapshot = engine.evaluate_case("case-1", store, CONFIG, EVAL_DATE)

        assert snapshot.case_id == "case-1"
        assert snapshot.estado == "AMARILLO"  # no expected documents present

        row = store.connection.execute(
            "SELECT * FROM rule_evaluation WHERE case_id = ?;",
            ("case-1",),
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload"])
        assert payload["case_id"] == "case-1"
        assert payload["estado"] == snapshot.estado

    def test_missing_extraction_propagates_without_persisting(self, store):
        store.register_document("case-1", "doc-1")  # registered, never extracted

        with pytest.raises(MissingExtractionError):
            engine.evaluate_case("case-1", store, CONFIG, EVAL_DATE)

        row = store.connection.execute(
            "SELECT * FROM rule_evaluation WHERE case_id = ?;",
            ("case-1",),
        ).fetchone()
        assert row is None

    def test_internal_failure_reraises_without_persisting(self, store, monkeypatch):
        _seed_other_document(store, "case-1", "doc-1")

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated snapshot failure")

        monkeypatch.setattr(engine, "build_snapshot", _boom)

        with pytest.raises(RuntimeError):
            engine.evaluate_case("case-1", store, CONFIG, EVAL_DATE)

        row = store.connection.execute(
            "SELECT * FROM rule_evaluation WHERE case_id = ?;",
            ("case-1",),
        ).fetchone()
        assert row is None

    def test_rerun_overwrites_previous_row(self, store):
        _seed_other_document(store, "case-1", "doc-1")
        engine.evaluate_case("case-1", store, CONFIG, EVAL_DATE)
        engine.evaluate_case("case-1", store, CONFIG, EVAL_DATE)

        rows = store.connection.execute(
            "SELECT * FROM rule_evaluation WHERE case_id = ?;",
            ("case-1",),
        ).fetchall()
        assert len(rows) == 1
