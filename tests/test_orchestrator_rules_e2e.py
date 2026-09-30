"""End-to-end test: ingest -> OCR/classification -> rules, in one call (M-4, step 12).

Exercises the real seams composed together — `LocalThreadJobLauncher` (real
threads, real `job.run`, real status writes) driven by `process_case`, which
then (opt-in via `evaluate_rules=True`) calls `ps06.rules.engine.evaluate_case`
once every document reaches `OCR_DONE`. This is the debugging entry point for
the full pipeline: run this test alone to exercise ingest through rules
without a live Workbench/Knox.
"""

from __future__ import annotations

from datetime import date

from ps06.ocr.auth import FakeAuthProvider
from ps06.ocr.extraction import OcrJobConfig
from ps06.orchestrator.launcher import LocalThreadJobLauncher
from ps06.orchestrator.orchestrator import OrchestratorConfig, process_case
from ps06.rules.engine import get_snapshot
from ps06.rules.rules_config import RulesConfig
from ps06.status import db
from ps06.status.states import CaseStatus, DocumentStage
from ps06.status.store import StatusStore


def test_all_docs_ocr_done_triggers_rules_evaluation(
    tmp_path, pdf_with_text_layer, fake_openai_client_factory
):
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
            evaluate_rules=True,
            rules_config=RulesConfig(),
            evaluation_date=date(2026, 9, 30),
        )

    assert summary.case_status is CaseStatus.RULES_EVALUATED
    assert summary.succeeded == 2

    row = store.connection.execute(
        "SELECT * FROM rule_evaluation WHERE case_id = ?;", ("C1",)
    ).fetchone()
    assert row is not None

    snapshot = get_snapshot(store, "C1")
    assert snapshot is not None
    assert snapshot.case_id == "C1"
    assert snapshot.estado in {"VERDE", "AMARILLO", "ROJO"}
    assert snapshot.s1_rows  # doc inventory populated


def test_failed_document_skips_rules_evaluation(
    tmp_path, pdf_with_text_layer, fake_openai_client_factory
):
    db_path = tmp_path / "ps06.db"
    store = StatusStore(db.connect(db_path))
    docs = {"d1": str(pdf_with_text_layer), "d2": "/nonexistent/missing.pdf"}
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
            OrchestratorConfig(max_concurrency=2, max_attempts=1, poll_interval_seconds=0.01),
            evaluate_rules=True,
        )

    assert summary.case_status is CaseStatus.FAILED
    assert any(o.stage is DocumentStage.FAILED for o in summary.documents)

    row = store.connection.execute(
        "SELECT * FROM rule_evaluation WHERE case_id = ?;", ("C1",)
    ).fetchone()
    assert row is None  # hook never fires on a partial/failed case
