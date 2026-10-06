"""Per-document result file: how a job reports back without touching shared SQLite.

``/home/cdsw`` is NFS. SQLite WAL is unsafe there across pods, so jobs never
write the shared status DB. A job runs ``ps06.ocr.job.run`` against a
pod-local DB, then writes one JSON result file to a spool folder on the
project volume. The API (the only writer of the shared DB) reads that file
and records ``OCR_DONE`` or ``FAILED`` with :func:`ingest`.

File layout: ``<spool_dir>/<case_id>/<document_id>.json``. Writes are atomic
(temp file + ``os.replace``), so a reader never sees a partial file.

No silent failures: a failure file always carries a non-empty
``error_detail``; a corrupt or mismatched file raises :class:`ResultFileError`.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from ps06.ocr.envelope import OcrResult
from ps06.ocr.job import _persist_extraction
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore


class ResultFileError(Exception):
    """A result file is unreadable, invalid, or does not match its document."""


class ResultFile(BaseModel):
    """Content of one result file. Exactly one of ``result``/``error_detail``."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    case_id: str
    document_id: str
    result: OcrResult | None = None
    error_detail: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> ResultFile:
        if self.ok:
            if self.result is None or self.error_detail is not None:
                raise ValueError("ok=true needs result and no error_detail")
            if (self.result.case_id, self.result.document_id) != (
                self.case_id,
                self.document_id,
            ):
                raise ValueError("result ids do not match the file ids")
        else:
            if self.result is not None or not (self.error_detail or "").strip():
                raise ValueError("ok=false needs a non-empty error_detail and no result")
        return self


def _check_id(name: str, value: str) -> None:
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"invalid {name} for a result file path: {value!r}")


def result_path(spool_dir: str | Path, case_id: str, document_id: str) -> Path:
    """``<spool_dir>/<case_id>/<document_id>.json``. Rejects path-like ids."""
    _check_id("case_id", case_id)
    _check_id("document_id", document_id)
    return Path(spool_dir) / case_id / f"{document_id}.json"


def _write_atomic(path: Path, payload: ResultFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload.model_dump_json())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_success(path: str | Path, result: OcrResult) -> None:
    """Write a success file for ``result``."""
    _write_atomic(
        Path(path),
        ResultFile(
            ok=True,
            case_id=result.case_id,
            document_id=result.document_id,
            result=result,
        ),
    )


def write_failure(
    path: str | Path, case_id: str, document_id: str, error_detail: str
) -> None:
    """Write a failure file. ``error_detail`` must be non-empty."""
    _write_atomic(
        Path(path),
        ResultFile(
            ok=False,
            case_id=case_id,
            document_id=document_id,
            error_detail=error_detail,
        ),
    )


def read_result_file(path: str | Path) -> ResultFile | None:
    """Return the parsed file, or ``None`` if it does not exist yet.

    Raises :class:`ResultFileError` if the file exists but is invalid.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ResultFileError(f"cannot read result file {path}: {exc}") from exc
    try:
        return ResultFile.model_validate_json(raw)
    except ValidationError as exc:
        raise ResultFileError(f"invalid result file {path}: {exc}") from exc


def ingest(
    store: StatusStore, case_id: str, document_id: str, result_file: ResultFile
) -> DocumentStage:
    """Record ``result_file`` in the shared status DB. Returns the new stage.

    Success: persist the extraction, then ``OCR_DONE``. Failure: ``FAILED``
    with the file's ``error_detail``. The document must be ``RECEIVED``
    (``InvalidTransition`` / ``DocumentNotFound`` propagate otherwise).
    """
    if (result_file.case_id, result_file.document_id) != (case_id, document_id):
        raise ResultFileError(
            f"result file is for {result_file.case_id}/{result_file.document_id}, "
            f"expected {case_id}/{document_id}"
        )
    store.get_or_raise(case_id, document_id)
    if result_file.ok and result_file.result is not None:
        _persist_extraction(store, result_file.result)
        store.transition(case_id, document_id, DocumentStage.OCR_DONE)
        return DocumentStage.OCR_DONE
    store.transition(
        case_id,
        document_id,
        DocumentStage.FAILED,
        error_detail=result_file.error_detail,
    )
    return DocumentStage.FAILED