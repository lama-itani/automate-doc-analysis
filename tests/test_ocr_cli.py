"""Smoke tests for ps06.ocr.cli (M-2, step 8).

Uses the pdf_with_text_layer fixture (tests/conftest.py) so the text-layer
fast path is exercised and no VLM/network call happens — ReauthHTTPClient
still mints a token and constructs a real openai.OpenAI client (object
construction only, no network), so --auth fake with FakeAuthProvider is
sufficient to exercise the whole CLI offline.
"""

from __future__ import annotations

import json

import pytest

from ps06.ocr import cli
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore


def _run_argv(db_path, *extra):
    return [
        "--db", str(db_path),
        "run",
        "--case", "case-1",
        "--doc", "doc-1",
        "--endpoint-url", "http://fake",
        "--model-name", "test-model",
        "--auth", "fake",
        "--fake-token", "tok1",
        "--fake-token", "tok2",
        *extra,
    ]


def _register(db_path):
    conn = db.connect(str(db_path))
    try:
        StatusStore(conn).register_document("case-1", "doc-1")
    finally:
        conn.close()


class TestRun:
    def test_happy_path_prints_summary_and_persists(
        self, tmp_path, pdf_with_text_layer, capsys
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)

        exit_code = cli.main(_run_argv(db_path, "--file", str(pdf_with_text_layer)))

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "case-1/doc-1" in out
        assert "page_count=1" in out

        conn = db.connect(str(db_path))
        try:
            status = StatusStore(conn).get_or_raise("case-1", "doc-1")
        finally:
            conn.close()
        assert status.stage == DocumentStage.OCR_DONE

    def test_json_flag_prints_valid_ocr_result_json(
        self, tmp_path, pdf_with_text_layer, capsys
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)

        exit_code = cli.main(
            _run_argv(db_path, "--file", str(pdf_with_text_layer), "--json")
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        json_start = out.index("{")
        payload = json.loads(out[json_start:])
        assert payload["case_id"] == "case-1"
        assert payload["document_id"] == "doc-1"
        assert payload["extraction"]["file_type"] == "pdf"

    def test_auto_register_flag_registers_unregistered_document(
        self, tmp_path, pdf_with_text_layer
    ):
        db_path = tmp_path / "ps06.db"

        exit_code = cli.main(
            _run_argv(
                db_path, "--file", str(pdf_with_text_layer), "--auto-register"
            )
        )

        assert exit_code == 0
        conn = db.connect(str(db_path))
        try:
            status = StatusStore(conn).get_or_raise("case-1", "doc-1")
        finally:
            conn.close()
        assert status.stage == DocumentStage.OCR_DONE

    def test_fake_auth_without_tokens_exits_with_error(self, tmp_path, pdf_with_text_layer):
        db_path = tmp_path / "ps06.db"
        _register(db_path)

        argv = [
            "--db", str(db_path),
            "run",
            "--case", "case-1",
            "--doc", "doc-1",
            "--file", str(pdf_with_text_layer),
            "--endpoint-url", "http://fake",
            "--model-name", "test-model",
            "--auth", "fake",
        ]
        with pytest.raises(SystemExit):
            cli.main(argv)


class TestShow:
    def test_show_after_run_matches(self, tmp_path, pdf_with_text_layer, capsys):
        db_path = tmp_path / "ps06.db"
        _register(db_path)
        cli.main(_run_argv(db_path, "--file", str(pdf_with_text_layer)))
        capsys.readouterr()

        exit_code = cli.main(["--db", str(db_path), "show", "--case", "case-1", "--doc", "doc-1", "--json"])

        assert exit_code == 0
        out = capsys.readouterr().out
        json_start = out.index("{")
        payload = json.loads(out[json_start:])
        assert payload["case_id"] == "case-1"
        assert payload["document_id"] == "doc-1"

    def test_show_with_no_extraction_row_errors(self, tmp_path, capsys):
        db_path = tmp_path / "ps06.db"
        _register(db_path)

        exit_code = cli.main(
            ["--db", str(db_path), "show", "--case", "case-1", "--doc", "doc-1"]
        )

        assert exit_code == 1
        err = capsys.readouterr().err
        assert "no extraction found" in err
