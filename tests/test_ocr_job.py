"""Tests for ps06.ocr.job.run — the per-document OCR job orchestrator (M-2, step 7).

Exercises the full sequencing (status transitions + document_extraction
persistence) against an in-memory StatusStore and a fake OpenAI-shaped
client — no network, no live Cloudera/Workbench credentials.
"""

from __future__ import annotations

import json

import pytest

from ps06.ocr import job
from ps06.ocr.auth import FakeAuthProvider
from ps06.ocr.extraction import OcrJobConfig
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import DocumentNotFound, InvalidTransition, StatusStore


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMsg(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, ocr_text: str) -> None:
        self._ocr_text = ocr_text

    def create(self, **kwargs):
        return _FakeCompletion(self._ocr_text)


class _FakeChat:
    def __init__(self, ocr_text: str) -> None:
        self.completions = _FakeCompletions(ocr_text)


class _FakeOpenAIClient:
    """Stands in for openai.OpenAI: always returns a fixed OCR text, no
    orientation-detection branching needed since these tests exercise the
    text-layer fast path (no VLM call at all)."""

    def __init__(self, *, base_url: str, api_key: str, ocr_text: str = "unused") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _FakeChat(ocr_text)


def _client_factory(*, base_url: str, api_key: str) -> _FakeOpenAIClient:
    return _FakeOpenAIClient(base_url=base_url, api_key=api_key)


@pytest.fixture()
def store() -> StatusStore:
    return StatusStore(db.connect())


def _config(**overrides) -> OcrJobConfig:
    defaults = dict(
        endpoint_url="http://inference",
        model_name="model-x",
        min_text_layer_chars=1,
    )
    defaults.update(overrides)
    return OcrJobConfig(**defaults)


class TestJobRun:
    def test_happy_path_transitions_to_ocr_done_and_persists(self, store, pdf_with_text_layer):
        store.register_document("case-1", "doc-1")

        result = job.run(
            case_id="case-1",
            document_id="doc-1",
            file_path=str(pdf_with_text_layer),
            config=_config(),
            store=store,
            auth=FakeAuthProvider(["tok1"]),
            client_factory=_client_factory,
        )

        status = store.get_or_raise("case-1", "doc-1")
        assert status.stage == DocumentStage.OCR_DONE
        assert status.error_detail is None

        row = store.connection.execute(
            "SELECT * FROM document_extraction WHERE case_id = ? AND document_id = ?;",
            ("case-1", "doc-1"),
        ).fetchone()
        assert row is not None
        assert row["processing_seconds"] == result.processing_seconds
        payload = json.loads(row["payload"])
        assert payload["case_id"] == "case-1"
        assert payload["extraction"]["file_type"] == "pdf"

    def test_extraction_failure_transitions_to_failed_with_error_detail_and_reraises(
        self, store, tmp_path
    ):
        store.register_document("case-1", "doc-1")
        bad_file = tmp_path / "doc.unsupported"
        bad_file.write_text("not a real document")

        with pytest.raises(Exception):
            job.run(
                case_id="case-1",
                document_id="doc-1",
                file_path=str(bad_file),
                config=_config(),
                store=store,
                auth=FakeAuthProvider(["tok1"]),
                client_factory=_client_factory,
            )

        status = store.get_or_raise("case-1", "doc-1")
        assert status.stage == DocumentStage.FAILED
        assert status.error_detail
        assert "UnsupportedFileTypeError" in status.error_detail

        row = store.connection.execute(
            "SELECT * FROM document_extraction WHERE case_id = ? AND document_id = ?;",
            ("case-1", "doc-1"),
        ).fetchone()
        assert row is None

    def test_missing_document_raises_document_not_found_without_failed_transition(
        self, store, pdf_with_text_layer
    ):
        with pytest.raises(DocumentNotFound):
            job.run(
                case_id="case-1",
                document_id="doc-missing",
                file_path=str(pdf_with_text_layer),
                config=_config(),
                store=store,
                auth=FakeAuthProvider(["tok1"]),
                client_factory=_client_factory,
            )

        assert store.get("case-1", "doc-missing") is None

    def test_already_ocr_done_document_propagates_invalid_transition(
        self, store, pdf_with_text_layer
    ):
        store.register_document("case-1", "doc-1")
        store.transition("case-1", "doc-1", DocumentStage.OCR_DONE)

        with pytest.raises(InvalidTransition):
            job.run(
                case_id="case-1",
                document_id="doc-1",
                file_path=str(pdf_with_text_layer),
                config=_config(),
                store=store,
                auth=FakeAuthProvider(["tok1"]),
                client_factory=_client_factory,
            )

        # The document must not be silently flipped to FAILED for a
        # caller-contract bug (calling run() on an already-done document) —
        # InvalidTransition on the *second* transition (OCR_DONE) propagates
        # from inside the try block, so it IS recorded as FAILED with a
        # populated error_detail, per the "every exception path" contract.
        status = store.get_or_raise("case-1", "doc-1")
        assert status.stage == DocumentStage.FAILED
        assert "InvalidTransition" in status.error_detail
