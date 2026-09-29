"""Tests for the M-3 launcher seam (ps06/orchestrator/launcher.py).

`LocalThreadJobLauncher` is exercised end-to-end against a real `job.run` via
the `pdf_with_text_layer` fixture (fast path -> no VLM/network call) and the
real `FakeAuthProvider`. `WorkbenchJobLauncher` is asserted to be an honest
stub. No live credentials or network.
"""

from __future__ import annotations

import time

import pytest

from ps06.ocr.auth import FakeAuthProvider
from ps06.ocr.extraction import OcrJobConfig
from ps06.orchestrator.launcher import (
    LocalThreadJobLauncher,
    WorkbenchJobLauncher,
)
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore


def _config() -> OcrJobConfig:
    # Fast-path defaults (use_text_layer_fast_path=True) mean a text-layer PDF
    # never calls the endpoint, so endpoint_url/model_name are inert here.
    return OcrJobConfig(endpoint_url="http://localhost/v1", model_name="test-model")


def _wait_done(handle, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not handle.done():
        if time.monotonic() > deadline:  # pragma: no cover - safety valve
            raise AssertionError("job handle did not finish in time")
        time.sleep(0.01)


class TestLocalThreadJobLauncher:
    def test_runs_job_to_ocr_done(self, tmp_path, pdf_with_text_layer):
        db_path = tmp_path / "ps06.db"
        conn = db.connect(db_path)
        StatusStore(conn).register_document("C1", "d1")
        conn.close()

        with LocalThreadJobLauncher(
            db_path, _config(), auth_factory=lambda: FakeAuthProvider(["tok"])
        ) as launcher:
            handle = launcher.launch("C1", "d1", str(pdf_with_text_layer))
            _wait_done(handle)

        assert handle.exception() is None
        # Reconcile via the status table (the source of truth), fresh connection.
        store = StatusStore(db.connect(db_path))
        assert store.get("C1", "d1").stage is DocumentStage.OCR_DONE

    def test_failure_is_captured_and_recorded_failed(self, tmp_path):
        # An unsupported file type makes extract() raise; job.run records FAILED
        # (with a populated error_detail) and re-raises -> the Future captures it.
        db_path = tmp_path / "ps06.db"
        conn = db.connect(db_path)
        StatusStore(conn).register_document("C1", "bad")
        conn.close()

        bad_file = tmp_path / "notes.txt"
        bad_file.write_text("not a supported document")

        with LocalThreadJobLauncher(
            db_path, _config(), auth_factory=lambda: FakeAuthProvider(["tok"])
        ) as launcher:
            handle = launcher.launch("C1", "bad", str(bad_file))
            _wait_done(handle)

        assert handle.exception() is not None
        status = StatusStore(db.connect(db_path)).get("C1", "bad")
        assert status.stage is DocumentStage.FAILED
        assert status.error_detail  # never None/empty on FAILED

    def test_rejects_memory_db(self):
        with pytest.raises(ValueError, match="file-backed"):
            LocalThreadJobLauncher(
                db.MEMORY, _config(), auth_factory=lambda: FakeAuthProvider(["t"])
            )

    def test_rejects_zero_workers(self, tmp_path):
        with pytest.raises(ValueError, match="max_workers"):
            LocalThreadJobLauncher(
                tmp_path / "x.db",
                _config(),
                auth_factory=lambda: FakeAuthProvider(["t"]),
                max_workers=0,
            )


class TestWorkbenchJobLauncher:
    def test_launch_raises_not_implemented_naming_open_item(self, monkeypatch):
        monkeypatch.delenv("CDSW_APIV2_KEY", raising=False)
        launcher = WorkbenchJobLauncher()
        with pytest.raises(NotImplementedError, match="open item #4"):
            launcher.launch("C1", "d1", "/tmp/whatever.pdf")

    def test_launch_still_stub_even_with_key_present(self, monkeypatch):
        monkeypatch.setenv("CDSW_APIV2_KEY", "present-but-unused")
        launcher = WorkbenchJobLauncher()
        with pytest.raises(NotImplementedError, match="Workbench API v2"):
            launcher.launch("C1", "d1", "/tmp/whatever.pdf")
