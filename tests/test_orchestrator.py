"""Tests for the M-3 orchestrator core loop (ps06/orchestrator/orchestrator.py).

Uses a deterministic in-thread `ScriptedLauncher` (mirrors how test_ocr_job.py
fakes the OpenAI client) so concurrency, retry, crash-reconciliation, and
attempt-cap behaviour are exercised without real threads or network. The fake
writes status transitions on the *same* store the orchestrator reads, exactly as
a real per-document invocation would.
"""

from __future__ import annotations

import pytest

from ps06.ocr.auth import FakeAuthProvider
from ps06.ocr.extraction import OcrJobConfig
from ps06.orchestrator.launcher import LocalThreadJobLauncher
from ps06.orchestrator.orchestrator import (
    CRASH_ERROR_DETAIL,
    OrchestratorConfig,
    process_case,
)
from ps06.status import db
from ps06.status.states import CaseStatus, DocumentStage
from ps06.status.store import DocumentNotFound, StatusStore


@pytest.fixture
def store() -> StatusStore:
    return StatusStore(db.connect())


def _fast() -> OrchestratorConfig:
    # No idle sleeps in tests; behaviour is independent of the poll interval.
    return OrchestratorConfig(poll_interval_seconds=0.0)


class _Counter:
    def __init__(self) -> None:
        self.n = 0
        self.peak = 0

    def inc(self) -> None:
        self.n += 1
        self.peak = max(self.peak, self.n)

    def dec(self) -> None:
        self.n -= 1


class _Handle:
    """Done-immediately handle. Decrements the shared counter the first time the
    orchestrator observes it done — mirroring its removal from `in_flight`, so
    `counter.peak` tracks true peak concurrency."""

    def __init__(self, case_id, document_id, counter, exc=None) -> None:
        self.case_id = case_id
        self.document_id = document_id
        self._counter = counter
        self._exc = exc
        self._counted_done = False

    def done(self) -> bool:
        if not self._counted_done:
            self._counted_done = True
            self._counter.dec()
        return True

    def exception(self):
        return self._exc


class ScriptedLauncher:
    """Applies a scripted per-attempt outcome to the store, synchronously.

    `script[document_id]` is a queue of actions consumed one per launch:
    ``"ocr_done"`` (transition OCR_DONE), ``"fail"`` (transition FAILED),
    ``"crash"`` (leave RECEIVED and expose an exception on the handle — an
    invocation that died without recording a terminal status). Empty/absent
    queue defaults to ``"ocr_done"``.
    """

    def __init__(self, store: StatusStore, script: dict[str, list[str]] | None = None) -> None:
        self._store = store
        self._script = {k: list(v) for k, v in (script or {}).items()}
        self.launches: list[str] = []
        self.counter = _Counter()

    def launch(self, case_id: str, document_id: str, file_path: str) -> _Handle:
        self.launches.append(document_id)
        self.counter.inc()
        queue = self._script.get(document_id)
        action = queue.pop(0) if queue else "ocr_done"
        exc = None
        if action == "ocr_done":
            self._store.transition(case_id, document_id, DocumentStage.OCR_DONE)
        elif action == "fail":
            self._store.transition(
                case_id, document_id, DocumentStage.FAILED, error_detail="scripted failure"
            )
        elif action == "crash":
            exc = RuntimeError("worker died")  # status left at RECEIVED
        else:  # pragma: no cover - test author error
            raise ValueError(f"unknown scripted action: {action!r}")
        return _Handle(case_id, document_id, self.counter, exc)


class TestHappyPath:
    def test_all_documents_reach_ocr_done(self, store):
        docs = {"d1": "/f/1.pdf", "d2": "/f/2.pdf", "d3": "/f/3.pdf"}
        for d in docs:
            store.register_document("C1", d)
        launcher = ScriptedLauncher(store)

        summary = process_case("C1", docs, launcher, store, _fast())

        assert summary.case_status is CaseStatus.OCR_DONE
        assert summary.total == 3
        assert summary.succeeded == 3
        assert summary.failed == 0
        assert all(o.stage is DocumentStage.OCR_DONE for o in summary.documents)
        assert sorted(launcher.launches) == ["d1", "d2", "d3"]

    def test_already_done_document_is_not_relaunched(self, store):
        # Resumability: a document already OCR_DONE is left untouched.
        docs = {"d1": "/f/1.pdf", "d2": "/f/2.pdf"}
        for d in docs:
            store.register_document("C1", d)
        store.transition("C1", "d1", DocumentStage.OCR_DONE)  # pre-completed
        launcher = ScriptedLauncher(store)

        summary = process_case("C1", docs, launcher, store, _fast())

        assert launcher.launches == ["d2"]
        assert summary.case_status is CaseStatus.OCR_DONE


class TestConcurrencyBound:
    def test_never_exceeds_max_concurrency(self, store):
        docs = {f"d{i}": f"/f/{i}.pdf" for i in range(5)}
        for d in docs:
            store.register_document("C1", d)
        launcher = ScriptedLauncher(store)
        config = OrchestratorConfig(max_concurrency=2, poll_interval_seconds=0.0)

        summary = process_case("C1", docs, launcher, store, config)

        assert launcher.counter.peak == 2  # bound honoured, never 3+
        assert summary.succeeded == 5


class TestRetry:
    def test_failed_then_succeeds_counts_second_attempt(self, store):
        store.register_document("C1", "d1")
        launcher = ScriptedLauncher(store, {"d1": ["fail", "ocr_done"]})

        summary = process_case("C1", {"d1": "/f/1.pdf"}, launcher, store, _fast())

        outcome = summary.documents[0]
        assert outcome.stage is DocumentStage.OCR_DONE
        assert outcome.attempt_count == 2  # one retry as a fresh invocation
        assert outcome.error_detail is None
        assert launcher.launches == ["d1", "d1"]

    def test_attempts_exhausted_ends_failed(self, store):
        store.register_document("C1", "d1")
        launcher = ScriptedLauncher(store, {"d1": ["fail", "fail", "fail"]})
        config = OrchestratorConfig(max_attempts=3, poll_interval_seconds=0.0)

        summary = process_case("C1", {"d1": "/f/1.pdf"}, launcher, store, config)

        outcome = summary.documents[0]
        assert outcome.stage is DocumentStage.FAILED
        assert outcome.attempt_count == 3
        assert outcome.error_detail == "scripted failure"
        assert summary.case_status is CaseStatus.FAILED
        assert summary.failed == 1
        assert len(launcher.launches) == 3  # capped at max_attempts


class TestCrashReconciliation:
    def test_crash_without_status_is_recorded_failed(self, store):
        # Invocation exits without recording a terminal status; the orchestrator
        # records FAILED itself with a populated error_detail — never silent.
        store.register_document("C1", "d1")
        launcher = ScriptedLauncher(store, {"d1": ["crash"]})
        config = OrchestratorConfig(max_attempts=1, poll_interval_seconds=0.0)

        summary = process_case("C1", {"d1": "/f/1.pdf"}, launcher, store, config)

        outcome = summary.documents[0]
        assert outcome.stage is DocumentStage.FAILED
        assert outcome.attempt_count == 1
        assert CRASH_ERROR_DETAIL in outcome.error_detail
        assert "RuntimeError" in outcome.error_detail  # enriched from the handle
        assert summary.case_status is CaseStatus.FAILED

    def test_crash_then_retry_succeeds(self, store):
        store.register_document("C1", "d1")
        launcher = ScriptedLauncher(store, {"d1": ["crash", "ocr_done"]})
        config = OrchestratorConfig(max_attempts=3, poll_interval_seconds=0.0)

        summary = process_case("C1", {"d1": "/f/1.pdf"}, launcher, store, config)

        outcome = summary.documents[0]
        assert outcome.stage is DocumentStage.OCR_DONE
        assert outcome.attempt_count == 2  # crash -> FAILED -> retry -> OCR_DONE
        assert launcher.launches == ["d1", "d1"]


class TestIntegrationLocalThreadLauncher:
    def test_end_to_end_real_threads_reach_ocr_done(
        self, tmp_path, pdf_with_text_layer, fake_openai_client_factory
    ):
        # Real seams composed: orchestrator + LocalThreadJobLauncher (real threads,
        # real job.run, real status writes on a WAL file DB). Fast-path fixture ->
        # no VLM/network call. The orchestrator reads on its own connection while
        # workers write on theirs — validates the WAL + CAS concurrency story.
        db_path = tmp_path / "ps06.db"
        store = StatusStore(db.connect(db_path))
        docs = {"d1": str(pdf_with_text_layer), "d2": str(pdf_with_text_layer)}
        for d in docs:
            store.register_document("C1", d)

        cfg = OcrJobConfig(endpoint_url="http://localhost/v1", model_name="test-model")
        with LocalThreadJobLauncher(
            db_path,
            cfg,
            auth_factory=lambda: FakeAuthProvider(["tok"]),
            client_factory=fake_openai_client_factory,
            max_workers=2,
        ) as launcher:
            summary = process_case(
                "C1",
                docs,
                launcher,
                store,
                OrchestratorConfig(max_concurrency=2, poll_interval_seconds=0.01),
            )

        assert summary.case_status is CaseStatus.OCR_DONE
        assert summary.succeeded == 2
        assert all(o.stage is DocumentStage.OCR_DONE for o in summary.documents)


class TestCallerContract:
    def test_unregistered_document_raises(self, store):
        store.register_document("C1", "known")
        launcher = ScriptedLauncher(store)
        with pytest.raises(DocumentNotFound):
            process_case(
                "C1", {"known": "/f/1.pdf", "ghost": "/f/2.pdf"}, launcher, store, _fast()
            )
        # fail-fast: nothing was launched
        assert launcher.launches == []
