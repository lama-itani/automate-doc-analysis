"""Per-document OCR job orchestrator (M-2, step 7).

Ties together the pieces built in steps 1-6: mints a fresh token and builds a
:class:`~ps06.ocr.http_client.ReauthHTTPClient` (per-invocation token
isolation — the whole reason this build exists), runs
:func:`ps06.ocr.extraction.extract`, wraps the result in an
:class:`~ps06.ocr.envelope.OcrResult`, persists it to the
``document_extraction`` table, and drives the document's status via
:class:`~ps06.status.store.StatusStore`.

Every failure path — extraction error, auth exhaustion, a bug, a persistence
failure — is caught by one ``except Exception``, always transitions the
document to :attr:`~ps06.status.states.DocumentStage.FAILED` with a populated
``error_detail``, and always re-raises so the process exits non-zero and the
failure stays visible. No silent failures, per the production-readiness bar
for this build (see the Build Handoff's "M-2 session 2" Progress Log entry).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import openai

from ps06.ocr.auth import AuthProvider
from ps06.ocr.envelope import OcrResult
from ps06.ocr.extraction import OcrJobConfig, extract
from ps06.ocr.http_client import ReauthHTTPClient
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore


def run(
    case_id: str,
    document_id: str,
    file_path: str,
    config: OcrJobConfig,
    store: StatusStore,
    auth: AuthProvider,
    client_factory: Callable[..., Any] = openai.OpenAI,
) -> OcrResult:
    """Run one document through OCR/extraction and record the outcome.

    ``client_factory`` is forwarded to :class:`ReauthHTTPClient` unchanged —
    the same test seam that lets its own tests substitute a fake
    ``OpenAI``-shaped client without network or live credentials.

    Assumes the document is already registered at
    :attr:`DocumentStage.RECEIVED` by ingestion — this job never calls
    ``register_document``. ``store.get_or_raise`` confirms the row exists;
    :class:`~ps06.status.store.DocumentNotFound` propagates unwrapped, since
    a missing row is a caller-contract bug, not a processing failure worth a
    ``FAILED`` transition.

    Returns the persisted :class:`OcrResult` on success. Raises (after
    recording a ``FAILED`` transition) on any failure.
    """
    store.get_or_raise(case_id, document_id)

    start = time.monotonic()
    try:
        client = ReauthHTTPClient(
            auth, config.endpoint_url, config.model_name, client_factory=client_factory
        )
        extraction = extract(file_path, config, client)
        processing_seconds = time.monotonic() - start
        result = OcrResult.build(
            case_id=case_id,
            document_id=document_id,
            file_path=file_path,
            extraction=extraction,
            processing_seconds=processing_seconds,
            model_name=config.model_name,
        )
        _persist_extraction(store, result)
        store.transition(case_id, document_id, DocumentStage.OCR_DONE)
    except Exception as exc:
        store.transition(
            case_id,
            document_id,
            DocumentStage.FAILED,
            error_detail=f"{type(exc).__name__}: {exc}",
        )
        raise

    return result


def _persist_extraction(store: StatusStore, result: OcrResult) -> None:
    """Write ``result`` into the ``document_extraction`` table.

    A retried document's row is overwritten (``INSERT OR REPLACE``), matching
    the migration's stated semantics: the table reflects the latest attempt
    only, not history. Runs on the same connection ``store`` wraps, inside the
    caller's ``try`` block, so a write failure here becomes part of the
    ``FAILED`` transition rather than a false ``OCR_DONE`` success.
    """
    conn = store.connection  # same connection StatusStore writes through
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
