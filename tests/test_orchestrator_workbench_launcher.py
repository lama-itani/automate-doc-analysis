"""Tests for WorkbenchJobLauncher (B2). A fake cmlapi client and a fake clock:
no network, no workbench."""

from __future__ import annotations

import shlex
import time
from types import SimpleNamespace

import pytest

from ps06.ocr.envelope import OcrResult, get_extraction
from ps06.ocr.extraction import ExtractionResult
from ps06.ocr.result_file import result_path, write_failure, write_success
from ps06.orchestrator.launcher import WorkbenchJobLauncher
from ps06.orchestrator.orchestrator import OrchestratorConfig, process_case
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore


def _result(document_id: str = "a") -> OcrResult:
    return OcrResult.build(
        case_id="C1",
        document_id=document_id,
        file_path="/x/a.pdf",
        extraction=ExtractionResult(
            file_type="pdf",
            extracted_text="--- Page 1 ---\nHola",
            page_count=1,
            pages_via_text_layer=1,
            pages_via_vlm=0,
            orientation_corrections={},
            acroform_fields={},
            xfa_fields={},
        ),
        processing_seconds=1.0,
        model_name="m",
    )


class FakeApi:
    """Stands in for the cmlapi module."""

    @staticmethod
    def CreateJobRequest(**kw):
        return SimpleNamespace(**kw)

    @staticmethod
    def CreateJobRunRequest(**kw):
        return SimpleNamespace(**kw)


class FakeClient:
    """Stands in for ``cmlapi.default_client()``. ``statuses`` is consumed one
    per ``get_job_run`` call; the last one repeats."""

    def __init__(self, statuses=("ENGINE_SUCCEEDED",)):
        self.statuses = list(statuses)
        self.created_jobs = []
        self.deleted_jobs = []
        self.stopped = []
        self.polls = 0
        self.fail_create_job = None
        self.fail_create_run = None
        self.poll_error = None
        self.on_run_created = None

    def create_job(self, request, project_id):
        if self.fail_create_job:
            raise self.fail_create_job
        self.created_jobs.append(request)
        return SimpleNamespace(id=f"job{len(self.created_jobs)}")

    def create_job_run(self, request, project_id, job_id):
        if self.fail_create_run:
            raise self.fail_create_run
        if self.on_run_created:
            self.on_run_created()
        return SimpleNamespace(id="run1")

    def get_job_run(self, project_id, job_id, run_id):
        self.polls += 1
        if self.poll_error:
            raise self.poll_error
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return SimpleNamespace(status=status)

    def stop_job_run(self, project_id, job_id, run_id):
        self.stopped.append((job_id, run_id))

    def delete_job(self, project_id, job_id):
        self.deleted_jobs.append(job_id)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture()
def store():
    s = StatusStore(db.connect())
    s.register_document("C1", "a")
    return s


@pytest.fixture()
def clock():
    return Clock()


def _launcher(store, tmp_path, client, clock, **kw):
    params = dict(
        store=store,
        spool_dir=tmp_path / "spool",
        endpoint_url="http://endpoint/v1",
        model_name="model",
        project_id="proj",
        client=client,
        cmlapi_module=FakeApi,
        min_poll_interval_seconds=0,
        result_grace_seconds=30,
        run_timeout_seconds=100,
        time_fn=clock,
    )
    params.update(kw)
    return WorkbenchJobLauncher(**params)


def _path(tmp_path, doc="a"):
    return result_path(tmp_path / "spool", "C1", doc)


def test_launch_creates_one_job_and_run_with_safe_arguments(store, tmp_path, clock):
    client = FakeClient()
    launcher = _launcher(store, tmp_path, client, clock, extra_args=["--pdf-dpi", "150"])

    launcher.launch("C1", "a", "/home/cdsw/in/a b.pdf")

    (req,) = client.created_jobs
    assert req.script == "jobs/ocr_job.py"
    assert req.project_id == "proj" and req.cpu == 1 and req.memory == 4
    assert "pbj-jupyterlab-python3.10" in req.runtime_identifier
    args = shlex.split(req.arguments)
    assert args[:4] == ["--db", "/tmp/ps06_job.db", "run", "--auto-register"]
    assert args[args.index("--file") + 1] == "/home/cdsw/in/a b.pdf"
    assert args[args.index("--result-out") + 1] == str(_path(tmp_path))
    assert args[-2:] == ["--pdf-dpi", "150"]
    joined = req.arguments.lower()
    assert "token" not in joined and "fake" not in joined and "--auth" not in joined


def test_launch_removes_stale_result_file(store, tmp_path, clock):
    path = _path(tmp_path)
    write_failure(path, "C1", "a", "old attempt")
    _launcher(store, tmp_path, FakeClient(), clock).launch("C1", "a", "/f.pdf")
    assert not path.exists()


def test_success_is_ingested_and_cleaned_up(store, tmp_path, clock):
    client = FakeClient(["ENGINE_SCHEDULING", "ENGINE_RUNNING", "ENGINE_SUCCEEDED"])
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")

    assert handle.done() is False
    assert handle.done() is False
    write_success(_path(tmp_path), _result())
    assert handle.done() is True

    assert handle.exception() is None
    assert store.get_or_raise("C1", "a").stage is DocumentStage.OCR_DONE
    assert get_extraction(store, "C1", "a") is not None
    assert not _path(tmp_path).exists()
    assert client.deleted_jobs == ["job1"]


def test_poll_is_throttled(store, tmp_path, clock):
    client = FakeClient(["ENGINE_RUNNING"])
    handle = _launcher(
        store, tmp_path, client, clock, min_poll_interval_seconds=10
    ).launch("C1", "a", "/f.pdf")

    handle.done()
    handle.done()
    assert client.polls == 1
    clock.now += 11
    handle.done()
    assert client.polls == 2


def test_failure_file_records_failed_with_error_detail_and_keeps_job(
    store, tmp_path, clock
):
    client = FakeClient(["ENGINE_FAILED"])
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")
    write_failure(_path(tmp_path), "C1", "a", "FileNotFoundError: gone")

    assert handle.done() is True
    status = store.get_or_raise("C1", "a")
    assert status.stage is DocumentStage.FAILED
    assert status.error_detail == "FileNotFoundError: gone"
    assert handle.exception() is None
    assert client.deleted_jobs == []  # kept for debugging


def test_ended_run_without_file_waits_for_grace_then_fails(store, tmp_path, clock):
    client = FakeClient(["ENGINE_FAILED"])
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")

    assert handle.done() is False  # inside the NFS grace window
    clock.now += 31
    assert handle.done() is True
    msg = str(handle.exception())
    assert "job1" in msg and "run1" in msg and "ENGINE_FAILED" in msg
    assert "without a result file" in msg
    assert store.get_or_raise("C1", "a").stage is DocumentStage.RECEIVED


def test_file_appearing_late_inside_grace_is_ingested(store, tmp_path, clock):
    client = FakeClient(["ENGINE_SUCCEEDED"])
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")

    assert handle.done() is False
    clock.now += 5
    write_success(_path(tmp_path), _result())
    assert handle.done() is True
    assert store.get_or_raise("C1", "a").stage is DocumentStage.OCR_DONE


def test_corrupt_result_file_fails_the_handle(store, tmp_path, clock):
    client = FakeClient(["ENGINE_SUCCEEDED"])
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")
    path = _path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{broken")

    assert handle.done() is True
    assert "invalid result file" in str(handle.exception())
    assert store.get_or_raise("C1", "a").stage is DocumentStage.RECEIVED


def test_result_for_a_document_not_in_received_fails_the_handle(
    store, tmp_path, clock
):
    client = FakeClient(["ENGINE_SUCCEEDED"])
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")
    store.transition("C1", "a", DocumentStage.OCR_DONE)
    write_success(_path(tmp_path), _result())

    assert handle.done() is True
    assert "could not record result" in str(handle.exception())


def test_run_that_never_ends_times_out_and_is_stopped(store, tmp_path, clock):
    client = FakeClient(["ENGINE_RUNNING"])
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")

    assert handle.done() is False
    clock.now += 101
    assert handle.done() is True
    assert client.stopped == [("job1", "run1")]
    assert "after 100s" in str(handle.exception())


def test_repeated_poll_errors_fail_the_handle(store, tmp_path, clock):
    client = FakeClient()
    client.poll_error = ConnectionError("api down")
    handle = _launcher(
        store, tmp_path, client, clock, max_poll_errors=3
    ).launch("C1", "a", "/f.pdf")

    assert handle.done() is False
    assert handle.done() is False
    assert handle.done() is True
    assert "after 3 attempts" in str(handle.exception())
    assert "api down" in str(handle.exception())


def test_a_good_poll_resets_the_error_count(store, tmp_path, clock):
    client = FakeClient(["ENGINE_RUNNING"])
    handle = _launcher(
        store, tmp_path, client, clock, max_poll_errors=2
    ).launch("C1", "a", "/f.pdf")

    client.poll_error = ConnectionError("blip")
    assert handle.done() is False
    client.poll_error = None
    assert handle.done() is False
    client.poll_error = ConnectionError("blip")
    assert handle.done() is False  # 1 error again, not 2


def test_create_job_error_returns_finished_failed_handle(store, tmp_path, clock):
    client = FakeClient()
    client.fail_create_job = RuntimeError("quota exceeded")
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")

    assert handle.done() is True
    assert "launch failed" in str(handle.exception())
    assert "quota exceeded" in str(handle.exception())


def test_create_run_error_deletes_the_orphan_job(store, tmp_path, clock):
    client = FakeClient()
    client.fail_create_run = RuntimeError("no capacity")
    handle = _launcher(store, tmp_path, client, clock).launch("C1", "a", "/f.pdf")

    assert handle.done() is True
    assert client.deleted_jobs == ["job1"]
    assert "no capacity" in str(handle.exception())


def test_bad_document_id_does_not_raise_from_launch(store, tmp_path, clock):
    handle = _launcher(store, tmp_path, FakeClient(), clock).launch(
        "C1", "../evil", "/f.pdf"
    )
    assert handle.done() is True
    assert "invalid document_id" in str(handle.exception())


def test_constructor_requires_project_id(store, tmp_path, monkeypatch):
    monkeypatch.delenv("CDSW_PROJECT_ID", raising=False)
    with pytest.raises(ValueError, match="CDSW_PROJECT_ID"):
        WorkbenchJobLauncher(
            store=store, spool_dir=tmp_path, endpoint_url="u", model_name="m",
            client=FakeClient(), cmlapi_module=FakeApi,
        )


def test_constructor_uses_env_project_id(store, tmp_path, monkeypatch):
    monkeypatch.setenv("CDSW_PROJECT_ID", "from-env")
    client = FakeClient()
    WorkbenchJobLauncher(
        store=store, spool_dir=tmp_path, endpoint_url="u", model_name="m",
        client=client, cmlapi_module=FakeApi,
    ).launch("C1", "a", "/f.pdf")
    assert client.created_jobs[0].project_id == "from-env"


def test_process_case_end_to_end_with_mixed_outcomes(tmp_path):
    """One document succeeds, one reports a failure, one run dies with no file.
    Every document ends terminal and every failure has an error_detail."""
    store = StatusStore(db.connect())
    for doc in ("a", "b", "c"):
        store.register_document("C1", doc)
    client = FakeClient(["ENGINE_SUCCEEDED"])
    launcher = _launcher(
        store, tmp_path, client, None, result_grace_seconds=0, time_fn=time.monotonic,
    )
    spool = tmp_path / "spool"

    real_launch = launcher.launch

    def launch(case_id, document_id, file_path):
        handle = real_launch(case_id, document_id, file_path)
        path = result_path(spool, case_id, document_id)
        if document_id == "a":
            write_success(path, _result("a"))
        elif document_id == "b":
            write_failure(path, case_id, document_id, "ValueError: bad pdf")
        return handle  # "c": the run ends and writes nothing

    launcher.launch = launch
    summary = process_case(
        "C1",
        {"a": "/a.pdf", "b": "/b.pdf", "c": "/c.pdf"},
        launcher,
        store,
        OrchestratorConfig(max_concurrency=3, max_attempts=1, poll_interval_seconds=0),
    )

    stages = {d.document_id: d for d in summary.documents}
    assert stages["a"].stage is DocumentStage.OCR_DONE
    assert stages["b"].stage is DocumentStage.FAILED
    assert stages["b"].error_detail == "ValueError: bad pdf"
    assert stages["c"].stage is DocumentStage.FAILED
    assert "without a result file" in stages["c"].error_detail
    assert summary.succeeded == 1 and summary.failed == 2