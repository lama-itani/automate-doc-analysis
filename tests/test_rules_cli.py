"""Smoke tests for ps06.rules.cli (M-4, step 11)."""

from __future__ import annotations

import json

from ps06.classification.classifier import ClassificationResult, DocumentType
from ps06.ocr.envelope import OcrResult
from ps06.ocr.extraction import ExtractionResult
from ps06.rules import cli
from ps06.status import db
from ps06.status.store import StatusStore


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        file_type="pdf",
        extracted_text="",
        page_count=1,
        pages_via_text_layer=1,
        pages_via_vlm=0,
        orientation_corrections={},
        acroform_fields={},
        xfa_fields={},
    )


def _seed_other_document(db_path, case_id="case-1", document_id="doc-1") -> None:
    conn = db.connect(str(db_path))
    try:
        store = StatusStore(conn)
        store.register_document(case_id, document_id)
        result = OcrResult.build(
            case_id=case_id,
            document_id=document_id,
            file_path=f"{document_id}.pdf",
            extraction=_extraction(),
            processing_seconds=1.0,
            model_name="test-model",
            classification=ClassificationResult(document_type=DocumentType.OTHER),
        )
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
    finally:
        conn.close()


class TestRunCase:
    def test_happy_path_prints_summary_and_persists(self, tmp_path, capsys):
        db_path = tmp_path / "ps06.db"
        _seed_other_document(db_path)

        exit_code = cli.main(
            ["--db", str(db_path), "run-case", "--case", "case-1", "--evaluation-date", "2026-09-29"]
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "case 'case-1': AMARILLO" in out

        conn = db.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT * FROM rule_evaluation WHERE case_id = ?;", ("case-1",)
            ).fetchone()
        finally:
            conn.close()
        assert row is not None

    def test_json_flag_prints_full_snapshot(self, tmp_path, capsys):
        db_path = tmp_path / "ps06.db"
        _seed_other_document(db_path)

        cli.main(
            [
                "--db", str(db_path), "run-case", "--case", "case-1",
                "--evaluation-date", "2026-09-29", "--json",
            ]
        )

        out = capsys.readouterr().out
        json_start = out.index("{")
        payload = json.loads(out[json_start:])
        assert payload["case_id"] == "case-1"
        assert payload["estado"] == "AMARILLO"

    def test_missing_extraction_prints_error_and_returns_nonzero(self, tmp_path, capsys):
        db_path = tmp_path / "ps06.db"
        conn = db.connect(str(db_path))
        try:
            StatusStore(conn).register_document("case-1", "doc-1")
        finally:
            conn.close()

        exit_code = cli.main(
            ["--db", str(db_path), "run-case", "--case", "case-1", "--evaluation-date", "2026-09-29"]
        )

        assert exit_code == 1
        err = capsys.readouterr().err
        assert "error:" in err


class TestShowCase:
    def test_show_after_run_round_trips(self, tmp_path, capsys):
        db_path = tmp_path / "ps06.db"
        _seed_other_document(db_path)
        cli.main(
            ["--db", str(db_path), "run-case", "--case", "case-1", "--evaluation-date", "2026-09-29"]
        )
        capsys.readouterr()  # discard run-case output

        exit_code = cli.main(["--db", str(db_path), "show-case", "--case", "case-1"])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "case 'case-1': AMARILLO" in out

    def test_show_before_run_prints_error_and_returns_nonzero(self, tmp_path, capsys):
        db_path = tmp_path / "ps06.db"
        _seed_other_document(db_path)

        exit_code = cli.main(["--db", str(db_path), "show-case", "--case", "case-1"])

        assert exit_code == 1
        err = capsys.readouterr().err
        assert "no rule evaluation found" in err
