"""Smoke tests for ps06.ocr.cli (M-2, step 8; classification added M-2.5).

Uses the pdf_with_text_layer fixture (tests/conftest.py) so the text-layer
fast path is exercised for extraction. Classification (M-2.5) always calls
the LLM once, though, so tests that actually run a job pass the
fake_openai_client_factory fixture in via cli.main's client_factory
parameter — otherwise a real openai.OpenAI client would attempt a real
network call against the fake --endpoint-url.
"""

from __future__ import annotations

import json

import pytest

from ps06.ocr import cli
from ps06.ocr.envelope import get_extraction
from ps06.ocr.result_file import read_result_file
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore
from tests.conftest import FakeOpenAIClient


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
        self, tmp_path, pdf_with_text_layer, capsys, fake_openai_client_factory
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)

        exit_code = cli.main(
            _run_argv(db_path, "--file", str(pdf_with_text_layer)),
            client_factory=fake_openai_client_factory,
        )

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
        self, tmp_path, pdf_with_text_layer, capsys, fake_openai_client_factory
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)

        exit_code = cli.main(
            _run_argv(db_path, "--file", str(pdf_with_text_layer), "--json"),
            client_factory=fake_openai_client_factory,
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        json_start = out.index("{")
        payload = json.loads(out[json_start:])
        assert payload["case_id"] == "case-1"
        assert payload["document_id"] == "doc-1"
        assert payload["extraction"]["file_type"] == "pdf"

    def test_auto_register_flag_registers_unregistered_document(
        self, tmp_path, pdf_with_text_layer, fake_openai_client_factory
    ):
        db_path = tmp_path / "ps06.db"

        exit_code = cli.main(
            _run_argv(
                db_path, "--file", str(pdf_with_text_layer), "--auto-register"
            ),
            client_factory=fake_openai_client_factory,
        )

        assert exit_code == 0
        conn = db.connect(str(db_path))
        try:
            status = StatusStore(conn).get_or_raise("case-1", "doc-1")
        finally:
            conn.close()
        assert status.stage == DocumentStage.OCR_DONE

    def test_prints_and_persists_document_type_and_generation(
        self, tmp_path, pdf_with_text_layer, capsys
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)

        def client_factory(*, base_url, api_key):
            return FakeOpenAIClient(
                base_url=base_url, api_key=api_key, classification_text="APPLICATION"
            )

        exit_code = cli.main(
            _run_argv(
                db_path,
                "--file", str(pdf_with_text_layer),
                "--json",
                "--classification-max-tokens", "16",
                "--classification-temperature", "0.5",
                "--classification-max-text-chars", "500",
                "--classification-min-chars", "5",
            ),
            client_factory=client_factory,
        )

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "document_type=APPLICATION generation=None" in out
        json_start = out.index("{")
        payload = json.loads(out[json_start:])
        assert payload["classification"]["document_type"] == "APPLICATION"

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
    def test_show_after_run_matches(
        self, tmp_path, pdf_with_text_layer, capsys, fake_openai_client_factory
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)
        cli.main(
            _run_argv(db_path, "--file", str(pdf_with_text_layer)),
            client_factory=fake_openai_client_factory,
        )
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


class TestResultOut:
    """``run --result-out``: the job's report to the API (ps06.ocr.result_file)."""

    def test_success_writes_ok_file_matching_db(
        self, tmp_path, pdf_with_text_layer, fake_openai_client_factory
    ):
        db_path = tmp_path / "job.db"
        out = tmp_path / "spool" / "case-1" / "doc-1.json"

        exit_code = cli.main(
            _run_argv(
                db_path, "--file", str(pdf_with_text_layer), "--auto-register",
                "--result-out", str(out),
            ),
            client_factory=fake_openai_client_factory,
        )

        assert exit_code == 0
        rf = read_result_file(out)
        assert rf.ok and rf.case_id == "case-1" and rf.document_id == "doc-1"
        conn = db.connect(str(db_path))
        try:
            assert get_extraction(StatusStore(conn), "case-1", "doc-1") == rf.result
        finally:
            conn.close()

    def test_failure_writes_error_file_with_detail(
        self, tmp_path, fake_openai_client_factory
    ):
        db_path = tmp_path / "job.db"
        out = tmp_path / "spool" / "case-1" / "doc-1.json"
        argv = _run_argv(
            db_path, "--file", str(tmp_path / "missing.pdf"), "--auto-register",
            "--result-out", str(out),
        )

        with pytest.raises(FileNotFoundError):
            cli.main(argv, client_factory=fake_openai_client_factory)

        rf = read_result_file(out)
        assert not rf.ok
        assert rf.error_detail.startswith("FileNotFoundError: ")

    def test_unwritable_result_path_keeps_original_error(
        self, tmp_path, capsys, fake_openai_client_factory
    ):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        out = blocker / "doc-1.json"  # parent is a file: cannot write here
        argv = _run_argv(
            tmp_path / "job.db", "--file", str(tmp_path / "missing.pdf"),
            "--auto-register", "--result-out", str(out),
        )

        with pytest.raises(FileNotFoundError):
            cli.main(argv, client_factory=fake_openai_client_factory)

        assert "could not write result file" in capsys.readouterr().err
        assert not out.exists()

    def test_no_result_out_writes_nothing(
        self, tmp_path, pdf_with_text_layer, fake_openai_client_factory
    ):
        db_path = tmp_path / "job.db"
        exit_code = cli.main(
            _run_argv(db_path, "--file", str(pdf_with_text_layer), "--auto-register"),
            client_factory=fake_openai_client_factory,
        )
        assert exit_code == 0
        assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".json") == []