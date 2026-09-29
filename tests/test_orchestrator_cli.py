"""Tests for the M-3 orchestrator CLI (ps06/orchestrator/cli.py).

Exercises run-case / status / retry-failed via the real code path (real
LocalThreadJobLauncher + job.run) using the fast-path fixture (no VLM/network)
and --auth fake, mirroring test_ocr_cli.py's style.
"""

from __future__ import annotations

import pytest

from ps06.orchestrator.cli import main
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore


def _endpoint_args() -> list[str]:
    return [
        "--endpoint-url",
        "http://localhost/v1",
        "--model-name",
        "test-model",
        "--auth",
        "fake",
        "--fake-token",
        "tok",
        "--poll-interval",
        "0.0",
    ]


class TestRunCase:
    def test_run_case_drives_all_to_ocr_done(self, tmp_path, pdf_with_text_layer):
        db_path = tmp_path / "ps06.db"
        pdf = str(pdf_with_text_layer)
        rc = main(
            [
                "--db", str(db_path), "run-case", "--case", "C1",
                "--doc", f"d1={pdf}", "--doc", f"d2={pdf}",
                "--auto-register", "--max-concurrency", "2",
                *_endpoint_args(),
            ]
        )
        assert rc == 0
        store = StatusStore(db.connect(db_path))
        assert store.get("C1", "d1").stage is DocumentStage.OCR_DONE
        assert store.get("C1", "d2").stage is DocumentStage.OCR_DONE

    def test_unregistered_without_auto_register_errors(self, tmp_path, pdf_with_text_layer, capsys):
        db_path = tmp_path / "ps06.db"
        pdf = str(pdf_with_text_layer)
        rc = main(
            [
                "--db", str(db_path), "run-case", "--case", "C1",
                "--doc", f"ghost={pdf}", *_endpoint_args(),
            ]
        )
        assert rc == 1
        assert "error:" in capsys.readouterr().err

    def test_malformed_doc_pair_exits(self, tmp_path, pdf_with_text_layer):
        db_path = tmp_path / "ps06.db"
        with pytest.raises(SystemExit):
            main(
                [
                    "--db", str(db_path), "run-case", "--case", "C1",
                    "--doc", "no-equals-sign", *_endpoint_args(),
                ]
            )

    def test_fake_auth_requires_token(self, tmp_path, pdf_with_text_layer):
        db_path = tmp_path / "ps06.db"
        pdf = str(pdf_with_text_layer)
        with pytest.raises(SystemExit, match="fake-token"):
            main(
                [
                    "--db", str(db_path), "run-case", "--case", "C1",
                    "--doc", f"d1={pdf}", "--auto-register",
                    "--endpoint-url", "http://localhost/v1", "--model-name", "m",
                    "--auth", "fake", "--poll-interval", "0.0",
                ]
            )


class TestStatus:
    def test_status_reports_case(self, tmp_path, pdf_with_text_layer, capsys):
        db_path = tmp_path / "ps06.db"
        pdf = str(pdf_with_text_layer)
        main(
            [
                "--db", str(db_path), "run-case", "--case", "C1",
                "--doc", f"d1={pdf}", "--auto-register", *_endpoint_args(),
            ]
        )
        capsys.readouterr()  # drop run-case output
        rc = main(["--db", str(db_path), "status", "--case", "C1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "C1" in out
        assert "OCR_DONE" in out

    def test_status_empty_case(self, tmp_path, capsys):
        db_path = tmp_path / "ps06.db"
        rc = main(["--db", str(db_path), "status", "--case", "nope"])
        assert rc == 0
        assert "no documents" in capsys.readouterr().out


class TestRetryFailed:
    def test_retry_failed_re_drives_with_corrected_path(self, tmp_path, pdf_with_text_layer):
        db_path = tmp_path / "ps06.db"
        bad = tmp_path / "notes.txt"
        bad.write_text("unsupported")
        good = str(pdf_with_text_layer)

        # First run with a bad file and max-attempts=1 -> ends FAILED (1 attempt).
        rc = main(
            [
                "--db", str(db_path), "run-case", "--case", "C1",
                "--doc", f"d1={bad}", "--auto-register", "--max-attempts", "1",
                *_endpoint_args(),
            ]
        )
        assert rc == 1
        store = StatusStore(db.connect(db_path))
        assert store.get("C1", "d1").stage is DocumentStage.FAILED

        # retry-failed with a corrected path + higher budget -> now OCR_DONE.
        rc = main(
            [
                "--db", str(db_path), "retry-failed", "--case", "C1",
                "--doc", f"d1={good}", "--max-attempts", "3",
                *_endpoint_args(),
            ]
        )
        assert rc == 0
        store = StatusStore(db.connect(db_path))
        final = store.get("C1", "d1")
        assert final.stage is DocumentStage.OCR_DONE
        assert final.attempt_count == 2  # one retry as a fresh invocation

    def test_retry_failed_noop_when_none_failed(self, tmp_path, pdf_with_text_layer, capsys):
        db_path = tmp_path / "ps06.db"
        pdf = str(pdf_with_text_layer)
        main(
            [
                "--db", str(db_path), "run-case", "--case", "C1",
                "--doc", f"d1={pdf}", "--auto-register", *_endpoint_args(),
            ]
        )
        capsys.readouterr()
        rc = main(
            [
                "--db", str(db_path), "retry-failed", "--case", "C1",
                "--doc", f"d1={pdf}", *_endpoint_args(),
            ]
        )
        assert rc == 0
        assert "no FAILED documents" in capsys.readouterr().out
