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
* :class:`WorkbenchJobLauncher` — the production path: launch a per-document CML
  Job via Workbench API v2. Ships as an honest ``NotImplementedError`` stub,
  exactly like :class:`ps06.ocr.auth.CDSWAuthProvider` — blocked on Build
  Handoff open item #4 (live Cloudera/Workbench credentials).

The orchestrator treats the **status table** as the source of truth for a
document's outcome; a :class:`JobHandle` only reports whether the invocation
process has finished and, if it crashed, the exception — so the orchestrator can
distinguish "job recorded a terminal status" from "job exited without recording
one" (a crash) and never leave a document silently stuck.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import openai

from ps06.ocr import job
from ps06.ocr.auth import AuthProvider
from ps06.ocr.extraction import OcrJobConfig
from ps06.status import db
from ps06.status.store import StatusStore


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


class WorkbenchJobLauncher:
    """Production launcher: one per-document CML Job via Workbench API v2.

    Not yet implemented. Launching a per-document job against the live Workbench
    requires the Workbench API v2 key tracked as Build Handoff open item #4,
    which is not available yet. :meth:`launch` raises rather than faking a launch
    so a caller that forgets to inject :class:`LocalThreadJobLauncher` in a
    dev/test context fails loudly instead of silently "succeeding" against no
    real Workbench — the same honest-stub stance as
    :class:`ps06.ocr.auth.CDSWAuthProvider`.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        project_id: str | None = None,
        job_id: str | None = None,
        api_key_env_var: str = "CDSW_APIV2_KEY",
    ) -> None:
        self._host = host
        self._project_id = project_id
        self._job_id = job_id
        self._api_key_env_var = api_key_env_var

    def launch(self, case_id: str, document_id: str, file_path: str) -> JobHandle:
        if not os.environ.get(self._api_key_env_var):
            raise NotImplementedError(
                f"WorkbenchJobLauncher requires a live Workbench API v2 key "
                f"(env var {self._api_key_env_var!r} is unset) and a real "
                f"per-document CML Job launch — blocked on Build Handoff open "
                f"item #4. Use LocalThreadJobLauncher for dev/test."
            )
        raise NotImplementedError(
            "Launching a per-document CML Job via Workbench API v2 is not yet "
            "implemented — blocked on Build Handoff open item #4 (live "
            "Cloudera/Workbench credentials)."
        )
