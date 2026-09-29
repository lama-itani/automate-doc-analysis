"""App-layer orchestrator core loop (M-3, step 4).

:func:`process_case` drives one case's documents to a terminal state by launching
one per-document invocation each (via a
:class:`~ps06.orchestrator.launcher.JobLauncher`), polling the **status table**
for outcomes, and retrying ``FAILED`` documents as fresh invocations — all within
a bounded concurrency budget. This is the "Orchestrator: launch / poll / retry
per-doc invocations" node of the No-TTL architecture.

Design points that carry the production-readiness bar:

* **The status table is the source of truth**, not the job handle. A handle only
  says whether the invocation *process* finished. After it finishes, the
  orchestrator reads the recorded stage: ``OCR_DONE`` (success) or ``FAILED``
  (the job recorded its own failure). If the document is somehow still
  ``RECEIVED`` — the invocation exited without recording a terminal status (a
  crash / OOM / kill) — the orchestrator records ``FAILED`` itself with a
  populated ``error_detail`` rather than leaving the document silently stuck.
* **Retries are fresh invocations.** A retry-eligible ``FAILED`` document is
  transitioned ``FAILED -> RECEIVED`` (which bumps ``attempt_count`` and clears
  ``error_detail``) before being relaunched, matching the milestone's
  "retry ``FAILED`` as a fresh invocation" contract. ``max_attempts`` caps the
  total number of attempts per document.
* **Bounded concurrency.** At most ``max_concurrency`` invocations are in flight
  at once.

Input contract: every document in ``documents`` must already be registered in the
status table (at ``RECEIVED`` from ingestion) — this orchestrator never calls
``register_document``. A missing row is a caller-contract bug and propagates as
:class:`~ps06.status.store.DocumentNotFound`, exactly as :func:`ps06.ocr.job.run`
treats it. The ``documents`` mapping supplies the ``document_id -> file path``
handoff (the status table does not store paths).

Scope boundary: the orchestrator *reports* case readiness (all documents
``OCR_DONE`` -> ``CaseStatus.OCR_DONE``); it does not trigger rules/aggregation —
that is M-4.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from ps06.orchestrator.launcher import JobHandle, JobLauncher
from ps06.status.states import CaseStatus, DocumentStage
from ps06.status.store import StatusStore

logger = logging.getLogger(__name__)

#: error_detail recorded when an invocation process finishes but left the
#: document without a terminal status — a crash the job's own except never
#: reached. Kept non-empty so the "no silent failures" bar holds for crashes too.
CRASH_ERROR_DETAIL = "invocation exited without recording a terminal status"


class OrchestratorConfig(BaseModel):
    """Tunables for one :func:`process_case` run.

    Per-document is the chunking unit (No-TTL plan §1); ``max_concurrency`` sizes
    the in-flight budget against Inference endpoint capacity, ``max_attempts``
    caps retries per document, and ``poll_interval_seconds`` is the delay between
    reconciliation sweeps of in-flight invocations.
    """

    model_config = ConfigDict(frozen=True)

    max_concurrency: int = Field(default=3, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    poll_interval_seconds: float = Field(default=1.0, ge=0.0)


class DocumentOutcome(BaseModel):
    """Final recorded state of one processed document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    stage: DocumentStage
    attempt_count: int
    error_detail: str | None = None


class CaseSummary(BaseModel):
    """Result of a :func:`process_case` run.

    ``case_status`` is the derived status over **all** the case's registered
    documents (via :meth:`StatusStore.case_status`), so it reflects real case
    readiness even if this call processed only a subset. ``documents`` lists the
    outcomes of the documents this call processed, in input order.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    case_status: CaseStatus | None
    documents: list[DocumentOutcome]
    total: int
    succeeded: int
    failed: int

    @classmethod
    def build(
        cls,
        case_id: str,
        case_status: CaseStatus | None,
        documents: list[DocumentOutcome],
    ) -> CaseSummary:
        succeeded = sum(1 for d in documents if d.stage is DocumentStage.OCR_DONE)
        failed = sum(1 for d in documents if d.stage is DocumentStage.FAILED)
        return cls(
            case_id=case_id,
            case_status=case_status,
            documents=documents,
            total=len(documents),
            succeeded=succeeded,
            failed=failed,
        )


def process_case(
    case_id: str,
    documents: Mapping[str, str],
    launcher: JobLauncher,
    store: StatusStore,
    config: OrchestratorConfig | None = None,
) -> CaseSummary:
    """Drive ``case_id``'s ``documents`` to terminal states and return a summary.

    ``documents`` maps ``document_id -> file path``; every document must already
    be registered (``RECEIVED``). Runs until every document is ``OCR_DONE`` or
    ``FAILED`` with attempts exhausted, then returns a :class:`CaseSummary`.
    Idempotent/resumable: documents already ``OCR_DONE`` are left alone, and
    ``FAILED`` documents with attempts remaining are retried.
    """
    config = config or OrchestratorConfig()

    # Fail fast on the caller contract before launching anything.
    for document_id in documents:
        store.get_or_raise(case_id, document_id)

    in_flight: dict[str, JobHandle] = {}

    while True:
        # 1. Reconcile any finished invocations against the status table.
        for document_id, handle in list(in_flight.items()):
            if not handle.done():
                continue
            del in_flight[document_id]
            _reconcile_finished(store, case_id, document_id, handle)

        # 2. Launch launchable documents up to the concurrency budget.
        for document_id in documents:
            if len(in_flight) >= config.max_concurrency:
                break
            if document_id in in_flight:
                continue
            status = store.get_or_raise(case_id, document_id)
            if not _is_launchable(status.stage, status.attempt_count, config.max_attempts):
                continue
            if status.stage is DocumentStage.FAILED:
                # Retry as a fresh invocation: bumps attempt_count, clears error.
                store.transition(case_id, document_id, DocumentStage.RECEIVED)
            logger.info(
                "launching invocation: case=%s document=%s", case_id, document_id
            )
            in_flight[document_id] = launcher.launch(
                case_id, document_id, documents[document_id]
            )

        # 3. Terminate when nothing is in flight and nothing remains launchable.
        if not in_flight:
            if not _any_launchable(store, case_id, documents, config.max_attempts):
                break
            # A just-reconciled failure became retry-eligible; loop to launch it
            # without an idle sleep.
            continue

        # 4. Wait before the next reconciliation sweep.
        if config.poll_interval_seconds:
            time.sleep(config.poll_interval_seconds)

    return _build_summary(store, case_id, documents)


def _is_launchable(stage: DocumentStage, attempt_count: int, max_attempts: int) -> bool:
    """True if a document in ``stage`` should be (re)launched.

    ``RECEIVED`` -> launch. ``FAILED`` -> retry only if attempts remain.
    ``OCR_DONE`` -> terminal (never relaunched).
    """
    if stage is DocumentStage.RECEIVED:
        return True
    if stage is DocumentStage.FAILED:
        return attempt_count < max_attempts
    return False  # OCR_DONE


def _any_launchable(
    store: StatusStore,
    case_id: str,
    documents: Mapping[str, str],
    max_attempts: int,
) -> bool:
    for document_id in documents:
        status = store.get_or_raise(case_id, document_id)
        if _is_launchable(status.stage, status.attempt_count, max_attempts):
            return True
    return False


def _reconcile_finished(
    store: StatusStore,
    case_id: str,
    document_id: str,
    handle: JobHandle,
) -> None:
    """Reconcile a finished invocation with the status table.

    The invocation records its own ``OCR_DONE``/``FAILED``; if the document is
    still ``RECEIVED`` the process died without recording one, so we record
    ``FAILED`` here with a populated ``error_detail`` (enriched with the handle's
    exception when present) — never a silent stuck document.
    """
    status = store.get_or_raise(case_id, document_id)
    if status.stage is not DocumentStage.RECEIVED:
        logger.info(
            "invocation finished: case=%s document=%s stage=%s attempt=%d",
            case_id,
            document_id,
            status.stage.value,
            status.attempt_count,
        )
        return

    detail = CRASH_ERROR_DETAIL
    exc = handle.exception()
    if exc is not None:
        detail = f"{CRASH_ERROR_DETAIL}: {type(exc).__name__}: {exc}"
    logger.warning(
        "invocation crashed without recording status: case=%s document=%s (%s)",
        case_id,
        document_id,
        detail,
    )
    store.transition(case_id, document_id, DocumentStage.FAILED, error_detail=detail)


def _build_summary(
    store: StatusStore,
    case_id: str,
    documents: Mapping[str, str],
) -> CaseSummary:
    outcomes = [
        DocumentOutcome(
            document_id=status.document_id,
            stage=status.stage,
            attempt_count=status.attempt_count,
            error_detail=status.error_detail,
        )
        for status in (store.get_or_raise(case_id, d) for d in documents)
    ]
    return CaseSummary.build(case_id, store.case_status(case_id), outcomes)
