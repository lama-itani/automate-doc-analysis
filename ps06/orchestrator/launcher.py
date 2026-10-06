"""Per-document job launcher seam (M-3, step 3).

The No-TTL architecture launches **one CML Job invocation per document** via the
Workbench API v2; each invocation mints its own fresh JWT, processes one
document, writes a status transition, and exits. :class:`JobLauncher` is the
seam that lets the orchestrator (:mod:`ps06.orchestrator.orchestrator`, step 4)
launch and track those invocations without depending on a live Workbench —
mirroring how :class:`ps06.ocr.auth.AuthProvider` separates the real Cloudera
token minting from a testable fake.

Two implementations:

* :class:`LocalThreadJobLauncher` — runs :func:`ps06.ocr.job.run` in an
  in-process thread pool, each worker opening its **own** ``StatusStore``
  connection. This mirrors per-invocation isolation (a distinct connection and a
  freshly minted token per document) and is fully testable via
  :class:`ps06.ocr.auth.FakeAuthProvider` plus the ``client_factory`` seam — no
  network, no live credentials. This is the runnable dev/pilot path today.
* :class:`WorkbenchJobLauncher` — the production path: one CML Job (and one
  job run) per document via Workbench API v2. ``/home/cdsw`` is NFS, where
  SQLite WAL is unsafe across pods, so the job never writes the shared status
  DB. It reports through a result file (:mod:`ps06.ocr.result_file`); the
  handle's :meth:`done` reads that file and records ``OCR_DONE``/``FAILED`` in
  the shared DB, from the orchestrator's thread.

The orchestrator treats the **status table** as the source of truth for a
document's outcome; a :class:`JobHandle` only reports whether the invocation
process has finished and, if it crashed, the exception — so the orchestrator can
distinguish "job recorded a terminal status" from "job exited without recording
one" (a crash) and never leave a document silently stuck.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import time
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import openai

from ps06.ocr import job
from ps06.ocr.auth import AuthProvider
from ps06.ocr.extraction import OcrJobConfig
from ps06.ocr.result_file import ResultFileError, ingest, read_result_file, result_path
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore, StatusStoreError

logger = logging.getLogger(__name__)


@runtime_checkable
class JobHandle(Protocol):
    """A handle to one in-flight per-document invocation.

    Reports only whether the invocation *process* has finished and, if it ended
    by raising, the exception. It intentionally does **not** report the OCR
    outcome — the status table is the source of truth for that (an invocation
    always records ``OCR_DONE``/``FAILED`` before exiting normally). The
    orchestrator reconciles the two: a finished handle whose document is still
    ``RECEIVED`` means the invocation crashed without recording a terminal
    status.
    """

    case_id: str
    document_id: str

    def done(self) -> bool:
        """True once the invocation process has finished (successfully or not)."""
        ...

    def exception(self) -> BaseException | None:
        """The exception the invocation raised, or ``None`` (also ``None`` while
        still running)."""
        ...


class JobLauncher(Protocol):
    """Launches one per-document invocation and returns a handle to track it."""

    def launch(self, case_id: str, document_id: str, file_path: str) -> JobHandle:
        """Start processing one document (non-blocking). Returns a
        :class:`JobHandle`."""
        ...


@dataclass
class _FutureJobHandle:
    """:class:`JobHandle` backed by a :class:`concurrent.futures.Future`."""

    case_id: str
    document_id: str
    _future: Future[Any] = field(repr=False)

    def done(self) -> bool:
        return self._future.done()

    def exception(self) -> BaseException | None:
        # Future.exception() blocks until done; guard so a poll of a still-running
        # handle returns None rather than blocking the orchestrator's sweep.
        if not self._future.done():
            return None
        try:
            return self._future.exception()
        except CancelledError as exc:  # cancelled future — treat as the failure
            return exc


class LocalThreadJobLauncher:
    """Runs :func:`ps06.ocr.job.run` per document in an in-process thread pool.

    Each launched invocation opens its **own** ``StatusStore`` connection to
    ``db_path`` and mints its own token via ``auth_factory()`` — the same
    per-invocation isolation the production per-document CML Job provides,
    realized as threads for a local/dev harness. Because each worker opens a
    separate connection to the *same* file, ``db_path`` must be file-backed;
    ``:memory:`` connections are distinct databases and would not share state.
    Concurrent writers are safe thanks to WAL + ``busy_timeout``
    (:func:`ps06.status.db.connect`) and the compare-and-swap in
    :meth:`ps06.status.store.StatusStore.transition`.

    ``client_factory`` is forwarded through to :class:`ReauthHTTPClient` exactly
    as :func:`ps06.ocr.job.run` documents — the seam that lets tests substitute
    a fake OpenAI-shaped client. Size ``max_workers`` to the orchestrator's
    concurrency bound.
    """

    def __init__(
        self,
        db_path: str | Path,
        config: OcrJobConfig,
        auth_factory: Callable[[], AuthProvider],
        client_factory: Callable[..., Any] = openai.OpenAI,
        max_workers: int = 4,
    ) -> None:
        if str(db_path) == db.MEMORY:
            raise ValueError(
                "LocalThreadJobLauncher requires a file-backed db_path: each "
                "worker opens its own connection, and separate ':memory:' "
                "connections are distinct databases that share no state."
            )
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._db_path = str(db_path)
        self._config = config
        self._auth_factory = auth_factory
        self._client_factory = client_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ocr-job"
        )

    def launch(self, case_id: str, document_id: str, file_path: str) -> JobHandle:
        future = self._executor.submit(
            self._run_one, case_id, document_id, file_path
        )
        return _FutureJobHandle(case_id, document_id, future)

    def _run_one(self, case_id: str, document_id: str, file_path: str) -> None:
        """Body of one invocation: own connection, own token, run the OCR job.

        ``job.run`` records ``OCR_DONE`` on success or ``FAILED`` (with a
        populated ``error_detail``) on any failure, then re-raises. The re-raised
        exception is captured on the ``Future``; the status table already
        reflects the terminal state, so the orchestrator reconciles from there.
        """
        conn = db.connect(self._db_path)
        try:
            store = StatusStore(conn)
            job.run(
                case_id,
                document_id,
                file_path,
                self._config,
                store,
                self._auth_factory(),
                client_factory=self._client_factory,
            )
        finally:
            conn.close()

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> LocalThreadJobLauncher:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.shutdown(wait=True)
        return False


#: Runtime validated on the workbench (PBJ JupyterLab, Python 3.10).
DEFAULT_RUNTIME = (
    "docker.repository.cloudera.com/cloudera/cdsw/"
    "ml-runtime-pbj-jupyterlab-python3.10-standard:2026.08.1-b5"
)
#: Run-status suffixes that end a job run (``ENGINE_SUCCEEDED`` etc.).
_TERMINAL_SUFFIXES = ("SUCCEEDED", "FAILED", "STOPPED", "TIMEDOUT")
#: Pod-local status DB inside a job. Never the shared project-volume DB.
_JOB_LOCAL_DB = "/tmp/ps06_job.db"


def _slug(value: str, limit: int = 20) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")[:limit] or "x"


class _ImmediateFailureHandle:
    """Handle for a launch that failed before a run existed. Always done."""

    def __init__(self, case_id: str, document_id: str, exc: BaseException) -> None:
        self.case_id = case_id
        self.document_id = document_id
        self._exc = exc

    def done(self) -> bool:
        return True

    def exception(self) -> BaseException | None:
        return self._exc


class _WorkbenchJobHandle:
    """Tracks one job run and records its outcome in the shared status DB.

    :meth:`done` polls the run, then reads the result file and ingests it. Call
    it only from the thread that owns the launcher's ``store`` (the
    orchestrator's thread). It never raises: any problem is kept as
    :meth:`exception` and the handle reports done, so the orchestrator marks a
    document that is still ``RECEIVED`` as ``FAILED`` with that detail.
    """

    def __init__(
        self,
        launcher: WorkbenchJobLauncher,
        case_id: str,
        document_id: str,
        job_id: str,
        run_id: str,
        path: Path,
    ) -> None:
        self.case_id = case_id
        self.document_id = document_id
        self._launcher = launcher
        self._job_id = job_id
        self._run_id = run_id
        self._path = path
        self._started_at = launcher._time()
        self._last_poll: float | None = None
        self._terminal_at: float | None = None
        self._run_status = ""
        self._poll_errors = 0
        self._finished = False
        self._exc: BaseException | None = None

    def exception(self) -> BaseException | None:
        return self._exc if self._finished else None

    def done(self) -> bool:
        if self._finished:
            return True
        lc = self._launcher
        now = lc._time()
        if self._terminal_at is None:
            if self._last_poll is not None and (
                now - self._last_poll < lc._min_poll_interval
            ):
                return False
            self._last_poll = now
            status = self._poll()
            if self._finished:
                return True
            if status is None:
                return False
            if status.upper().endswith(_TERMINAL_SUFFIXES):
                self._terminal_at = now
                self._run_status = status
            elif now - self._started_at > lc._run_timeout:
                return self._timeout()
            else:
                return False
        return self._try_ingest(now)

    def _where(self) -> str:
        return f"job {self._job_id} run {self._run_id}"

    def _poll(self) -> str | None:
        lc = self._launcher
        try:
            run = lc._client.get_job_run(lc._project_id, self._job_id, self._run_id)
        except Exception as exc:  # API/network error; counted, never swallowed
            self._poll_errors += 1
            logger.warning(
                "poll failed (%d/%d): case=%s document=%s %s: %s: %s",
                self._poll_errors, lc._max_poll_errors, self.case_id,
                self.document_id, self._where(), type(exc).__name__, exc,
            )
            if self._poll_errors >= lc._max_poll_errors:
                self._fail(
                    f"could not poll {self._where()} after "
                    f"{self._poll_errors} attempts: {type(exc).__name__}: {exc}"
                )
            return None
        self._poll_errors = 0
        return str(getattr(run, "status", "") or "")

    def _timeout(self) -> bool:
        lc = self._launcher
        try:
            lc._client.stop_job_run(lc._project_id, self._job_id, self._run_id)
        except Exception as exc:  # best effort; the timeout failure is recorded below
            logger.warning(
                "could not stop %s: %s: %s", self._where(), type(exc).__name__, exc
            )
        return self._fail(
            f"{self._where()} still {self._run_status or 'running'} after "
            f"{lc._run_timeout:.0f}s; stop requested"
        )

    def _try_ingest(self, now: float) -> bool:
        lc = self._launcher
        try:
            rf = read_result_file(self._path)
        except ResultFileError as exc:
            return self._fail(f"{self._where()} ({self._run_status}): {exc}")
        if rf is None:
            if now - (self._terminal_at or now) < lc._result_grace:
                return False  # NFS may show the file late; keep waiting
            return self._fail(
                f"{self._where()} ended {self._run_status} without a result "
                f"file at {self._path}"
            )
        try:
            stage = ingest(lc._store, self.case_id, self.document_id, rf)
        except (ResultFileError, StatusStoreError) as exc:
            return self._fail(
                f"could not record result of {self._where()}: "
                f"{type(exc).__name__}: {exc}"
            )
        if not self._run_status.upper().endswith("SUCCEEDED"):
            logger.warning(
                "%s ended %s but wrote a result file; recorded %s",
                self._where(), self._run_status, stage.value,
            )
        if stage is DocumentStage.OCR_DONE:
            self._cleanup()
        self._finished = True
        return True

    def _fail(self, detail: str) -> bool:
        logger.warning(
            "case=%s document=%s: %s", self.case_id, self.document_id, detail
        )
        self._exc = RuntimeError(detail)
        self._finished = True
        return True

    def _cleanup(self) -> None:
        """After a recorded success: drop the result file (applicant data) and
        the job. Failures keep both, for debugging."""
        lc = self._launcher
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("could not delete %s: %s", self._path, exc)
        try:
            lc._client.delete_job(lc._project_id, self._job_id)
        except Exception as exc:  # best effort; the document is already recorded
            logger.warning(
                "could not delete job %s: %s: %s", self._job_id,
                type(exc).__name__, exc,
            )


class WorkbenchJobLauncher:
    """Production launcher: one CML Job + one run per document (API v2).

    ``store`` is the **shared** status DB and must be used only from the thread
    that calls :meth:`launch` and the handles' ``done()`` (the orchestrator's).
    Jobs get a pod-local DB and write ``<spool_dir>/<case>/<doc>.json``; the
    handle ingests it. A run that ends with no valid result file leaves the
    document ``RECEIVED``, and the orchestrator then records ``FAILED``.

    Never puts tokens in job arguments: jobs use ``--auth cdsw`` (default).
    ``launch`` does not raise: an API error returns a finished handle carrying
    the error. ``client``/``cmlapi_module`` are test seams; by default they are
    the real ``cmlapi`` and ``cmlapi.default_client()``.
    """

    def __init__(
        self,
        *,
        store: StatusStore,
        spool_dir: str | Path,
        endpoint_url: str,
        model_name: str,
        project_id: str | None = None,
        client: Any = None,
        cmlapi_module: Any = None,
        script: str = "jobs/ocr_job.py",
        runtime_identifier: str = DEFAULT_RUNTIME,
        cpu: int = 1,
        memory: int = 4,
        extra_args: Sequence[str] = (),
        run_timeout_seconds: float = 1800.0,
        result_grace_seconds: float = 30.0,
        min_poll_interval_seconds: float = 10.0,
        max_poll_errors: int = 5,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if cmlapi_module is None:
            try:
                import cmlapi as cmlapi_module  # only on the Cloudera AI workbench
            except ImportError as exc:
                raise RuntimeError(
                    "WorkbenchJobLauncher needs the 'cmlapi' package, which is "
                    "available on the Cloudera AI workbench"
                ) from exc
        if project_id is None:
            project_id = os.environ.get("CDSW_PROJECT_ID")
        if not project_id:
            raise ValueError("project_id is not set and CDSW_PROJECT_ID is unset")
        self._api = cmlapi_module
        self._client = client if client is not None else cmlapi_module.default_client()
        self._project_id = project_id
        self._store = store
        self._spool_dir = Path(spool_dir)
        self._endpoint_url = endpoint_url
        self._model_name = model_name
        self._script = script
        self._runtime = runtime_identifier
        self._cpu = cpu
        self._memory = memory
        self._extra_args = tuple(extra_args)
        self._run_timeout = run_timeout_seconds
        self._result_grace = result_grace_seconds
        self._min_poll_interval = min_poll_interval_seconds
        self._max_poll_errors = max_poll_errors
        self._time = time_fn

    def launch(self, case_id: str, document_id: str, file_path: str) -> JobHandle:
        job_id: str | None = None
        try:
            path = result_path(self._spool_dir, case_id, document_id)
            path.unlink(missing_ok=True)  # never ingest a previous attempt's file
            args = shlex.join(
                [
                    "--db", _JOB_LOCAL_DB, "run", "--auto-register",
                    "--case", case_id, "--doc", document_id, "--file", file_path,
                    "--endpoint-url", self._endpoint_url,
                    "--model-name", self._model_name,
                    "--result-out", str(path),
                    *self._extra_args,
                ]
            )
            name = f"ps06-{_slug(case_id)}-{_slug(document_id)}-{int(time.time())}"
            job = self._client.create_job(
                self._api.CreateJobRequest(
                    project_id=self._project_id,
                    name=name,
                    script=self._script,
                    arguments=args,
                    cpu=self._cpu,
                    memory=self._memory,
                    runtime_identifier=self._runtime,
                ),
                self._project_id,
            )
            job_id = job.id
            run = self._client.create_job_run(
                self._api.CreateJobRunRequest(
                    project_id=self._project_id, job_id=job_id
                ),
                self._project_id,
                job_id,
            )
        except Exception as exc:  # launch must not crash the orchestrator loop
            logger.error(
                "launch failed: case=%s document=%s: %s: %s",
                case_id, document_id, type(exc).__name__, exc,
            )
            if job_id is not None:
                self._delete_job_quietly(job_id)
            return _ImmediateFailureHandle(
                case_id, document_id,
                RuntimeError(f"launch failed: {type(exc).__name__}: {exc}"),
            )
        logger.info(
            "launched: case=%s document=%s job=%s run=%s",
            case_id, document_id, job_id, run.id,
        )
        return _WorkbenchJobHandle(self, case_id, document_id, job_id, run.id, path)

    def _delete_job_quietly(self, job_id: str) -> None:
        try:
            self._client.delete_job(self._project_id, job_id)
        except Exception as exc:  # best effort cleanup after a failed launch
            logger.warning(
                "could not delete job %s: %s: %s", job_id, type(exc).__name__, exc
            )