"""Tests for jobs/ocr_job.py, the Cloudera AI (PBJ) job entry point.

The entry file calls ``run_job()`` as its last statement, so loading it runs
the job. ``_load`` patches ``ps06.ocr.cli.main`` first and then executes the
file as a plain module.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

from ps06.ocr import cli
from ps06.ocr.auth import CDSWAuthProvider
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore

JOB_FILE = Path(__file__).resolve().parent.parent / "jobs" / "ocr_job.py"
KERNEL_ARGV = ["ipykernel_launcher.py", "-f", "x.json"]


def _load(monkeypatch, main):
    """Execute the entry file with ``cli.main`` replaced by ``main``."""
    monkeypatch.setattr(cli, "main", main)
    spec = importlib.util.spec_from_file_location("ocr_job_under_test", JOB_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def job(monkeypatch):
    """The entry module, loaded with a harmless ``main`` that returns 0."""
    monkeypatch.setenv("JOB_ARGUMENTS", "show --case c --doc d")
    return _load(monkeypatch, lambda argv: 0)


class TestResolveArgs:
    def test_job_arguments_env_var_is_used(self, job):
        env = {"JOB_ARGUMENTS": "--db a.db run --case C1"}
        assert job.resolve_args(env, KERNEL_ARGV) == ["--db", "a.db", "run", "--case", "C1"]

    def test_quoted_values_with_spaces_stay_whole(self, job):
        env = {"JOB_ARGUMENTS": "--db 'my dir/a.db' run --case \"case 1\""}
        assert job.resolve_args(env, KERNEL_ARGV) == [
            "--db", "my dir/a.db", "run", "--case", "case 1",
        ]

    def test_unset_falls_back_to_sys_argv(self, job):
        argv = ["ocr_job.py", "--db", "a.db", "show"]
        assert job.resolve_args({}, argv) == ["--db", "a.db", "show"]

    @pytest.mark.parametrize("blank", ["", "   ", "\n"])
    def test_blank_env_var_counts_as_unset(self, job, blank):
        argv = ["ocr_job.py", "show"]
        assert job.resolve_args({"JOB_ARGUMENTS": blank}, argv) == ["show"]

    def test_kernel_argv_is_ignored(self, job):
        assert job.resolve_args({}, KERNEL_ARGV) == []

    def test_env_var_beats_kernel_argv(self, job):
        assert job.resolve_args({"JOB_ARGUMENTS": "show"}, KERNEL_ARGV) == ["show"]

    def test_unbalanced_quote_raises_normal_exception(self, job):
        with pytest.raises(ValueError):
            job.resolve_args({"JOB_ARGUMENTS": "--db 'oops"}, KERNEL_ARGV)

    def test_defaults_read_real_environment(self, job, monkeypatch):
        monkeypatch.setenv("JOB_ARGUMENTS", "show --case X")
        monkeypatch.setattr(sys, "argv", KERNEL_ARGV)
        assert job.resolve_args() == ["show", "--case", "X"]


class TestRunJob:
    def test_success_returns_without_exiting(self, job):
        assert job.run_job(["anything"]) is None

    def test_args_are_passed_to_main(self, monkeypatch):
        seen = []

        def main(argv):
            seen.append(list(argv))
            return 0

        monkeypatch.setenv("JOB_ARGUMENTS", "--db 'a b.db' show")
        _load(monkeypatch, main)
        assert seen == [["--db", "a b.db", "show"]]

    def test_nonzero_rc_raises_runtime_error_with_rc_and_args(self, job, monkeypatch):
        monkeypatch.setattr(cli, "main", lambda argv: 1)
        with pytest.raises(RuntimeError) as err:
            job.run_job(["show", "--case", "C1"])
        assert "exit code 1" in str(err.value)
        assert "C1" in str(err.value)

    @pytest.mark.parametrize("code", [2, 0, None, "error: bad auth"])
    def test_system_exit_becomes_runtime_error(self, job, monkeypatch, code):
        def main(argv):
            raise SystemExit(code)

        monkeypatch.setattr(cli, "main", main)
        with pytest.raises(RuntimeError) as err:
            job.run_job(["x"])
        assert not isinstance(err.value, SystemExit)
        assert isinstance(err.value.__cause__, SystemExit)
        assert repr(code) in str(err.value)

    def test_other_exceptions_propagate_unchanged(self, job, monkeypatch):
        def main(argv):
            raise KeyError("boom")

        monkeypatch.setattr(cli, "main", main)
        with pytest.raises(KeyError):
            job.run_job(["x"])

    def test_error_text_never_contains_fake_token(self, job, monkeypatch):
        monkeypatch.setattr(cli, "main", lambda argv: 1)
        args = ["run", "--fake-token", "SECRET1", "--fake-token=SECRET2", "--case", "C1"]
        with pytest.raises(RuntimeError) as err:
            job.run_job(args)
        text = str(err.value)
        assert "SECRET1" not in text and "SECRET2" not in text
        assert "C1" in text


class TestRealAuthOnly:
    FAKE_ARGS = [
        ["run", "--auth", "fake", "--fake-token", "SECRET1"],
        ["run", "--auth=fake", "--fake-token=SECRET1"],
        ["run", "--fake-token", "SECRET1"],
        ["run", "--fake-token=SECRET1"],
        ["run", "--fake", "SECRET1"],
        ["run", "--fake=SECRET1"],
        ["run", "--fa", "SECRET1"],
        ["run", "--auth", "fake"],
        ["--db", "a.db", "run", "--auth=fake"],
    ]

    @pytest.mark.parametrize("args", FAKE_ARGS)
    def test_fake_auth_is_refused_before_main_runs(self, job, monkeypatch, args):
        calls = []
        monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 0)
        with pytest.raises(RuntimeError, match="refuses --auth fake") as err:
            job.run_job(args)
        assert calls == []
        assert "SECRET1" not in str(err.value)

    @pytest.mark.parametrize(
        "args",
        [
            ["run", "--auth", "cdsw", "--case", "C1"],
            ["run", "--auth=cdsw"],
            ["run", "--case", "C1", "--file", "x.pdf"],
            ["run", "--case", "fake", "--doc", "fake-token"],
            ["show", "--case", "C1"],
        ],
    )
    def test_real_auth_and_lookalike_values_pass(self, job, monkeypatch, args):
        calls = []
        monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv) or 0)
        job.run_job(args)
        assert calls == [args]

    def test_guard_fires_when_the_file_is_loaded(self, monkeypatch):
        monkeypatch.setenv("JOB_ARGUMENTS", "run --auth fake --fake-token SECRET1")
        monkeypatch.setattr(cli, "main", lambda argv: 0)
        with pytest.raises(RuntimeError, match="refuses --auth fake"):
            spec = importlib.util.spec_from_file_location("ocr_job_guard", JOB_FILE)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))


# --- Kernel simulation -------------------------------------------------------


def _run_like_pbj(source: str):
    """Run ``source`` the way a PBJ job does: one top-level statement per cell."""
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    try:
        results = []
        for node in ast.parse(source).body:
            chunk = ast.get_source_segment(source, node)
            results.append(shell.run_cell(chunk, store_history=False))
        return results
    finally:
        InteractiveShell.clear_instance()


def _failure(result):
    return result.error_before_exec or result.error_in_exec


@pytest.fixture
def kernel(monkeypatch, tmp_path):
    pytest.importorskip("IPython")
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.setattr(sys, "argv", list(KERNEL_ARGV))
    return _run_like_pbj


def _use_fake_client(monkeypatch, client_factory):
    real_main = cli.main
    monkeypatch.setattr(
        cli, "main", lambda argv: real_main(argv, client_factory=client_factory)
    )


@pytest.fixture
def jwt_file(monkeypatch, tmp_path):
    """Point the CLI's default auth at a temp token file (real provider code)."""
    path = tmp_path / "jwt"
    path.write_text(json.dumps({"access_token": "test-token"}), encoding="utf-8")
    monkeypatch.setattr(cli, "CDSWAuthProvider", lambda: CDSWAuthProvider(str(path)))
    return path


def _run_args(db_path, file_path):
    # No auth flags: the CLI default is --auth cdsw, as in a real job.
    return [
        "--db", str(db_path), "run",
        "--case", "case-1", "--doc", "doc-1", "--file", str(file_path),
        "--endpoint-url", "http://fake", "--model-name", "test-model",
    ]


def _register(db_path):
    conn = db.connect(str(db_path))
    try:
        StatusStore(conn).register_document("case-1", "doc-1")
    finally:
        conn.close()


def _status(db_path):
    conn = db.connect(str(db_path))
    try:
        return StatusStore(conn).get_or_raise("case-1", "doc-1")
    finally:
        conn.close()


class TestKernelSimulation:
    def test_negative_control_sys_exit_zero_fails_a_chunk(self, kernel):
        results = kernel("x = 1\nimport sys\nsys.exit(0)\n")
        assert [_failure(r) is None for r in results] == [True, True, False]

    def test_successful_job_every_chunk_succeeds(
        self, kernel, jwt_file, monkeypatch, tmp_path, pdf_with_text_layer,
        fake_openai_client_factory,
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)
        _use_fake_client(monkeypatch, fake_openai_client_factory)
        monkeypatch.setenv(
            "JOB_ARGUMENTS", shlex.join(_run_args(db_path, pdf_with_text_layer))
        )

        results = kernel(JOB_FILE.read_text())

        assert len(results) > 1
        assert all(_failure(r) is None for r in results)
        assert _status(db_path).stage == DocumentStage.OCR_DONE

    def test_ocr_failure_fails_the_job_and_marks_document_failed(
        self, kernel, jwt_file, monkeypatch, tmp_path, fake_openai_client_factory
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)
        _use_fake_client(monkeypatch, fake_openai_client_factory)
        missing = tmp_path / "does_not_exist.pdf"
        monkeypatch.setenv("JOB_ARGUMENTS", shlex.join(_run_args(db_path, missing)))

        results = kernel(JOB_FILE.read_text())

        *earlier, last = results
        assert all(_failure(r) is None for r in earlier)
        # ps06.ocr.job.run re-raises the real error (here FileNotFoundError);
        # the entry file lets it through. Exception excludes SystemExit.
        assert isinstance(_failure(last), Exception)
        status = _status(db_path)
        assert status.stage == DocumentStage.FAILED
        assert status.error_detail

    def test_bad_arguments_fail_the_job_with_a_normal_exception(self, kernel, monkeypatch):
        monkeypatch.setenv("JOB_ARGUMENTS", "--no-such-flag")

        results = kernel(JOB_FILE.read_text())

        *earlier, last = results
        assert all(_failure(r) is None for r in earlier)
        assert isinstance(_failure(last), RuntimeError)

    def test_no_arguments_at_all_fails_the_job(self, kernel, monkeypatch):
        monkeypatch.delenv("JOB_ARGUMENTS", raising=False)

        results = kernel(JOB_FILE.read_text())

        assert isinstance(_failure(results[-1]), RuntimeError)

    def test_missing_token_file_fails_job_and_marks_document_failed(
        self, kernel, jwt_file, monkeypatch, tmp_path, pdf_with_text_layer,
        fake_openai_client_factory,
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)
        _use_fake_client(monkeypatch, fake_openai_client_factory)
        jwt_file.unlink()
        monkeypatch.setenv(
            "JOB_ARGUMENTS", shlex.join(_run_args(db_path, pdf_with_text_layer))
        )

        results = kernel(JOB_FILE.read_text())

        assert isinstance(_failure(results[-1]), Exception)
        status = _status(db_path)
        assert status.stage == DocumentStage.FAILED
        assert "not found" in status.error_detail

    def test_fake_token_in_job_arguments_fails_job_without_running_ocr(
        self, kernel, monkeypatch, tmp_path, pdf_with_text_layer
    ):
        db_path = tmp_path / "ps06.db"
        _register(db_path)
        args = _run_args(db_path, pdf_with_text_layer) + [
            "--auth", "fake", "--fake-token", "SECRET1",
        ]
        monkeypatch.setenv("JOB_ARGUMENTS", shlex.join(args))

        results = kernel(JOB_FILE.read_text())

        *earlier, last = results
        assert all(_failure(r) is None for r in earlier)
        assert isinstance(_failure(last), RuntimeError)
        assert _status(db_path).stage == DocumentStage.RECEIVED