"""PS-06 API service: one Application serving ``/api/*`` and the built UI at ``/``.

Flow: the UI uploads a case's documents (``POST /api/cases``). The API saves
them to the project volume, registers each one (``RECEIVED``) and starts one
background thread per case. That thread runs :func:`process_case` with a
:class:`WorkbenchJobLauncher` (one CML job per document, ``max_attempts=1``)
and owns its own :class:`StatusStore` connection, because the launcher and its
handles must run on the thread that owns the store.

``/home/cdsw`` is NFS, so this process is the only writer of the shared status
DB. Jobs report through result files that the launcher handles ingest.

Endpoints:

* ``GET  /api/health``                 liveness, model, whether the UI is built.
* ``GET  /api/cases``                  every case in the DB (survives restarts).
* ``POST /api/cases``                  upload + register + launch (202).
* ``GET  /api/cases/{id}``             per-document stage, label, error_detail.
* ``GET  /api/cases/{id}/verdict``     ``pending`` | ``failed`` | ``verdict``.

Rules run only when every document is ``OCR_DONE``. If any document is
``FAILED``, the verdict endpoint returns those documents and their errors.

Restart: the runner threads die with the process. Documents still
``RECEIVED`` with no live runner are reported as ``interrupted``, never as
silently in progress.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import threading
import unicodedata
from contextlib import contextmanager
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ps06.ocr.envelope import get_extraction
from ps06.ocr.extraction import MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS
from ps06.orchestrator.launcher import JobLauncher, WorkbenchJobLauncher
from ps06.orchestrator.orchestrator import OrchestratorConfig, process_case
from ps06.rules.adapter import build_case_bundle
from ps06.rules.engine import evaluate_case, get_snapshot
from ps06.rules.rules_config import RulesConfig
from ps06.rules.schemas import CanonicalDocument
from ps06.rules.snapshot import ReportSnapshot
from ps06.status import db
from ps06.status.states import CaseStatus, DocumentStage
from ps06.status.store import DocumentStatus, StatusStore, StatusStoreError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHUNK = 1 << 20
_MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
_UNSAFE_CHARS = re.compile(r"[/\\\x00-\x1f\x7f]")
_NOT_DETERMINED = "No determinado"

#: A launcher bound to the runner thread's own store.
LauncherFactory = Callable[[StatusStore], JobLauncher]


# --- Settings ---------------------------------------------------------------


@dataclass(frozen=True)
class ApiSettings:
    """Paths and model settings. Uploads and spool must be on the project
    volume, so jobs can read the files and write their result files."""

    db_path: Path
    upload_dir: Path
    spool_dir: Path
    ui_dist: Path
    endpoint_url: str
    model_name: str = DEFAULT_MODEL
    max_concurrency: int = 3
    poll_interval_seconds: float = 2.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> ApiSettings:
        """Build settings from ``PS06_*`` variables. ``PS06_ENDPOINT_URL`` is
        required; everything else has a default."""
        endpoint = env.get("PS06_ENDPOINT_URL", "").strip()
        if not endpoint:
            raise ValueError("PS06_ENDPOINT_URL is not set")
        data_dir = Path(env.get("PS06_DATA_DIR", "/home/cdsw/ps06_data"))
        raw_concurrency = env.get("PS06_MAX_CONCURRENCY", "3")
        try:
            max_concurrency = int(raw_concurrency)
        except ValueError as exc:
            raise ValueError(
                f"PS06_MAX_CONCURRENCY must be an integer, got {raw_concurrency!r}"
            ) from exc
        return cls(
            db_path=Path(env.get("PS06_DB_PATH", str(data_dir / "ps06.db"))),
            upload_dir=data_dir / "uploads",
            spool_dir=data_dir / "spool",
            ui_dist=Path(env.get("PS06_UI_DIST", str(_REPO_ROOT / "ui" / "dist"))),
            endpoint_url=endpoint,
            model_name=env.get("PS06_MODEL_NAME", DEFAULT_MODEL),
            max_concurrency=max_concurrency,
        )


def workbench_launcher_factory(settings: ApiSettings) -> LauncherFactory:
    """Default factory: a :class:`WorkbenchJobLauncher` per case runner."""

    def factory(store: StatusStore) -> JobLauncher:
        return WorkbenchJobLauncher(
            store=store,
            spool_dir=settings.spool_dir,
            endpoint_url=settings.endpoint_url,
            model_name=settings.model_name,
        )

    return factory


# --- Response models --------------------------------------------------------


class Bilingual(BaseModel):
    es: str
    en: str


class DocumentOut(BaseModel):
    """One document's progress. camelCase fields match ``ui/src/data.js``."""

    name: str
    stage: DocumentStage
    stage_label: Bilingual
    tries: int
    last_updated: str
    error_detail: Optional[str] = None
    type: Optional[str] = None
    pagesText: Optional[int] = None
    pagesVlm: Optional[int] = None
    seconds: Optional[float] = None


class CaseOut(BaseModel):
    id: str
    case_status: CaseStatus
    case_label: Bilingual
    running: bool
    interrupted: bool
    message: Optional[Bilingual] = None
    total: int
    ocr_done: int
    failed: int
    docs: list[DocumentOut]


class CaseListItem(BaseModel):
    id: str
    case_status: CaseStatus
    case_label: Bilingual
    running: bool
    interrupted: bool
    total: int
    ocr_done: int
    failed: int
    last_updated: str


class FailedDocOut(BaseModel):
    name: str
    tries: int
    error_detail: str


class IssueOut(BaseModel):
    code: str
    severity: Literal["ROJO", "AMARILLO"]
    title: Bilingual
    cause: Optional[Bilingual] = None
    hint: Optional[Bilingual] = None
    docs: list[str]


class S1Out(BaseModel):
    es: str
    en: str
    present: bool


class LineageOut(BaseModel):
    gen: str
    role: Bilingual
    name: Optional[str] = None
    birth: Optional[str] = None
    place: Optional[str] = None


class VerdictDocOut(BaseModel):
    name: str
    stage: DocumentStage
    tries: int
    type: Optional[str] = None
    pagesText: int
    pagesVlm: int
    seconds: float
    fields: list[str]
    missing: list[str]
    flags: list[str]


class MetaOut(BaseModel):
    model: str
    totalSeconds: float


class VerdictOut(BaseModel):
    """``result`` says which part is filled: ``failed`` lists the failed
    documents; ``pending`` lists unfinished ones; ``verdict`` carries the
    UI-shaped verdict plus the raw S0-S3 ``snapshot``."""

    id: str
    result: Literal["pending", "failed", "verdict"]
    case_status: CaseStatus
    failed: list[FailedDocOut] = []
    pending: list[str] = []
    verdict: Optional[Literal["VERDE", "AMARILLO", "ROJO"]] = None
    justification: Optional[Bilingual] = None
    issues: list[IssueOut] = []
    s1: list[S1Out] = []
    lineage: list[LineageOut] = []
    docs: list[VerdictDocOut] = []
    meta: Optional[MetaOut] = None
    snapshot: Optional[ReportSnapshot] = None


# --- Labels -----------------------------------------------------------------

_PROCESSING = Bilingual(es="En proceso", en="Processing")
_INTERRUPTED = Bilingual(es="Interrumpido", en="Interrupted")
_INTERRUPTED_MESSAGE = Bilingual(
    es="La aplicación se reinició durante el proceso. "
    "Los documentos en estado Interrumpido no terminaron.",
    en="The application restarted during processing. "
    "Documents marked Interrupted did not finish.",
)
_STAGE_LABELS = {
    DocumentStage.OCR_DONE: Bilingual(es="Lectura completada", en="Read complete"),
    DocumentStage.FAILED: Bilingual(es="Error", en="Failed"),
}
_CASE_LABELS = {
    CaseStatus.OCR_DONE: Bilingual(es="Listo para evaluar", en="Ready for evaluation"),
    CaseStatus.RULES_EVALUATED: Bilingual(es="Evaluado", en="Evaluated"),
    CaseStatus.RENDERED: Bilingual(es="Informe generado", en="Report rendered"),
    CaseStatus.FAILED: Bilingual(es="Con errores", en="Has errors"),
}
_S1_EN = {
    "Solicitud Principal": "Main Application",
    "Identificación del Solicitante": "Applicant's ID",
    "Certificado de Nacimiento del Solicitante": "Applicant's Birth Certificate",
    "Certificado de Nacimiento del Progenitor": "Parent's Birth Certificate",
    "Certificado de Nacimiento Español de origen": "Spanish-origin Birth Certificate",
}
_ROLE_EN = {"Solicitante": "Applicant", "Progenitor": "Parent", "Abuelo/a": "Grandparent"}
_JUSTIFICACION_EN = {
    "Se detectaron conflictos que impiden validar el caso automáticamente.":
        "Conflicts were found that prevent automatic validation of the case.",
    "El caso requiere revisión manual: faltan documentos o hay observaciones menores.":
        "The case needs manual review: documents are missing or there are minor remarks.",
    "El caso cumple todos los controles automáticos sin observaciones.":
        "The case passes all automatic checks with no remarks.",
}
_CORROBORATION_PREFIX = "Corroboración insuficiente"


def _bi(es: str, table: Mapping[str, str]) -> Bilingual:
    """Spanish text verbatim; English from ``table``, else the Spanish text."""
    return Bilingual(es=es, en=table.get(es, es))


# --- Case runner ------------------------------------------------------------


def _fail_unfinished(
    store: StatusStore, case_id: str, document_ids: Iterable[str], detail: str
) -> None:
    """Record ``FAILED`` with ``detail`` on every listed document still
    ``RECEIVED``. Errors are logged per document; none is skipped silently."""
    for document_id in document_ids:
        try:
            status = store.get(case_id, document_id)
            if status is not None and status.stage is DocumentStage.RECEIVED:
                store.transition(
                    case_id, document_id, DocumentStage.FAILED, error_detail=detail
                )
        except (StatusStoreError, sqlite3.Error) as exc:
            logger.error(
                "could not mark FAILED: case=%s document=%s: %s: %s",
                case_id, document_id, type(exc).__name__, exc,
            )


class CaseRunner:
    """One background thread per case. Each thread opens its own DB
    connection and launcher, runs :func:`process_case`, then closes both."""

    def __init__(
        self,
        db_path: Path,
        launcher_factory: LauncherFactory,
        config: OrchestratorConfig,
    ) -> None:
        self._db_path = db_path
        self._launcher_factory = launcher_factory
        self._config = config
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(self, case_id: str, documents: Mapping[str, str]) -> None:
        with self._lock:
            current = self._threads.get(case_id)
            if current is not None and current.is_alive():
                raise RuntimeError(f"case {case_id!r} is already running")
            thread = threading.Thread(
                target=self._run,
                args=(case_id, dict(documents)),
                name=f"ps06-case-{case_id}",
                daemon=True,
            )
            self._threads[case_id] = thread
        thread.start()

    def is_running(self, case_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(case_id)
        return thread is not None and thread.is_alive()

    def join(self, case_id: str, timeout: float | None = None) -> None:
        """Wait for a case's thread (tests and shutdown)."""
        with self._lock:
            thread = self._threads.get(case_id)
        if thread is not None:
            thread.join(timeout)

    def _run(self, case_id: str, documents: dict[str, str]) -> None:
        try:
            conn = db.connect(self._db_path)
        except (sqlite3.Error, OSError):
            # Without a connection nothing can be recorded. The documents stay
            # RECEIVED and show as interrupted once this thread ends.
            logger.exception("case runner could not open the DB: case=%s", case_id)
            return
        launcher: JobLauncher | None = None
        try:
            store = StatusStore(conn)
            try:
                launcher = self._launcher_factory(store)
                summary = process_case(case_id, documents, launcher, store, self._config)
                logger.info(
                    "case finished: case=%s status=%s ok=%d failed=%d",
                    case_id, summary.case_status, summary.succeeded, summary.failed,
                )
            except Exception as exc:  # recorded on every unfinished document below
                logger.exception("case runner failed: case=%s", case_id)
                _fail_unfinished(
                    store, case_id, documents,
                    f"case runner failed: {type(exc).__name__}: {exc}",
                )
        finally:
            shutdown = getattr(launcher, "shutdown", None)
            if callable(shutdown):
                shutdown()
            conn.close()


# --- Upload helpers ---------------------------------------------------------


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _check_name(kind: str, value: str) -> None:
    if (
        not value
        or value != value.strip()
        or value.startswith(".")
        or len(value) > 200
        or _UNSAFE_CHARS.search(value)
    ):
        raise HTTPException(400, f"invalid {kind}: {value!r}")


def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream ``upload`` to ``dest`` (new file). Enforces the OCR size limit."""
    size = 0
    with open(dest, "xb") as out:
        while chunk := upload.file.read(_CHUNK):
            size += len(chunk)
            if size > _MAX_BYTES:
                raise HTTPException(
                    413, f"{dest.name} is larger than {MAX_FILE_SIZE_MB} MB"
                )
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, f"{dest.name} is empty")


def _remove_dir(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("could not remove %s: %s", path, exc)


# --- Response builders ------------------------------------------------------


def _case_label(status: CaseStatus, interrupted: bool) -> Bilingual:
    if status is CaseStatus.RECEIVED:
        return _INTERRUPTED if interrupted else _PROCESSING
    return _CASE_LABELS[status]


def _doc_out(store: StatusStore, row: DocumentStatus, running: bool) -> DocumentOut:
    if row.stage is DocumentStage.RECEIVED:
        label = _PROCESSING if running else _INTERRUPTED
    else:
        label = _STAGE_LABELS[row.stage]
    out = DocumentOut(
        name=row.document_id,
        stage=row.stage,
        stage_label=label,
        tries=row.attempt_count,
        last_updated=row.last_updated,
        error_detail=row.error_detail,
    )
    if row.stage is DocumentStage.OCR_DONE:
        result = get_extraction(store, row.case_id, row.document_id)
        if result is not None:
            out.type = (
                result.classification.document_type.value
                if result.classification is not None
                else None
            )
            out.pagesText = result.extraction.pages_via_text_layer
            out.pagesVlm = result.extraction.pages_via_vlm
            out.seconds = result.processing_seconds
    return out


def _counts(rows: list[DocumentStatus]) -> tuple[int, int]:
    done = sum(1 for r in rows if r.stage is DocumentStage.OCR_DONE)
    failed = sum(1 for r in rows if r.stage is DocumentStage.FAILED)
    return done, failed


def _case_out(store: StatusStore, case_id: str, runner: CaseRunner) -> CaseOut:
    rows = store.list_for_case(case_id)
    if not rows:
        raise HTTPException(404, f"unknown case: {case_id!r}")
    running = runner.is_running(case_id)
    interrupted = not running and any(r.stage is DocumentStage.RECEIVED for r in rows)
    status = store.case_status(case_id)
    done, failed = _counts(rows)
    return CaseOut(
        id=case_id,
        case_status=status,
        case_label=_case_label(status, interrupted),
        running=running,
        interrupted=interrupted,
        message=_INTERRUPTED_MESSAGE if interrupted else None,
        total=len(rows),
        ocr_done=done,
        failed=failed,
        docs=[_doc_out(store, r, running) for r in rows],
    )


def _populated_fields(doc: CanonicalDocument) -> list[str]:
    data = doc.solicitud or doc.identidad or doc.certificado
    if data is None:
        return []
    return [k for k, v in data.model_dump().items() if v is not None and v != ""]


def _issues(snapshot: ReportSnapshot, rules_config: RulesConfig) -> list[IssueOut]:
    """One issue per ``problemas`` entry. Flags stay raw uppercase constants
    (binding decision); missing documents keep their verbatim Spanish label."""
    missing = {r.tipo for r in snapshot.s1_rows if r.estado == "Faltante"}
    rojo = set(rules_config.severity.rojo)
    docs_by_flag: dict[str, list[str]] = {}
    for row in snapshot.s2_rows:
        for flag in row.conflictos:
            docs_by_flag.setdefault(flag, []).append(row.documento)

    issues: list[IssueOut] = []
    for problema in snapshot.problemas:
        if snapshot.estado == "VERDE":
            break  # problemas is ["Ninguno"]
        if problema in missing:
            issues.append(IssueOut(
                code=f"FALTANTE: {problema}",
                severity="AMARILLO",
                title=Bilingual(
                    es=f"Falta: {problema}",
                    en=f"Missing: {_S1_EN.get(problema, problema)}",
                ),
                docs=[],
            ))
        elif problema.startswith(_CORROBORATION_PREFIX):
            issues.append(IssueOut(
                code="CORROBORACION_INSUFICIENTE",
                severity="AMARILLO",
                title=Bilingual(es=problema, en=problema),
                docs=[],
            ))
        else:
            issues.append(IssueOut(
                code=problema,
                severity="ROJO" if problema in rojo else "AMARILLO",
                title=Bilingual(es=problema, en=problema),
                docs=docs_by_flag.get(problema, []),
            ))
    return issues


def _lineage(snapshot: ReportSnapshot) -> list[LineageOut]:
    out = []
    for row in snapshot.s3_rows:
        fecha, _, lugar = row.fecha_lugar_nacimiento.partition(" / ")
        out.append(LineageOut(
            gen=row.generacion,
            role=_bi(row.relacion, _ROLE_EN),
            name=None if row.nombre_completo == _NOT_DETERMINED else row.nombre_completo,
            birth=None if fecha in ("", _NOT_DETERMINED) else fecha,
            place=None if lugar in ("", _NOT_DETERMINED) else lugar,
        ))
    return out


def _verdict_out(
    store: StatusStore,
    case_id: str,
    rows: list[DocumentStatus],
    snapshot: ReportSnapshot,
    rules_config: RulesConfig,
    default_model: str,
) -> VerdictOut:
    bundle = build_case_bundle(store, case_id)
    canonical = {d.document_id: d for d in bundle.documents}
    flags = {r.documento: list(r.conflictos) for r in snapshot.s2_rows}

    docs: list[VerdictDocOut] = []
    models: list[str] = []
    for row in rows:
        result = get_extraction(store, case_id, row.document_id)
        doc = canonical.get(row.document_id)
        if result is None or doc is None:
            # build_case_bundle already raises for a missing extraction; this
            # guards a row deleted in between.
            raise HTTPException(
                500, f"no extraction for document {row.document_id!r}"
            )
        models.append(result.model_name)
        docs.append(VerdictDocOut(
            name=row.document_id,
            stage=row.stage,
            tries=row.attempt_count,
            type=doc.document_type.value,
            pagesText=result.extraction.pages_via_text_layer,
            pagesVlm=result.extraction.pages_via_vlm,
            seconds=result.processing_seconds,
            fields=_populated_fields(doc),
            missing=list(doc.missing_fields),
            flags=flags.get(row.document_id, []),
        ))

    return VerdictOut(
        id=case_id,
        result="verdict",
        case_status=store.case_status(case_id),
        verdict=snapshot.estado,
        justification=_bi(snapshot.justificacion, _JUSTIFICACION_EN),
        issues=_issues(snapshot, rules_config),
        s1=[
            S1Out(es=r.tipo, en=_S1_EN.get(r.tipo, r.tipo), present=r.estado == "Presente")
            for r in snapshot.s1_rows
        ],
        lineage=_lineage(snapshot),
        docs=docs,
        meta=MetaOut(
            model=models[0] if models else default_model,
            totalSeconds=round(sum(d.seconds for d in docs), 1),
        ),
        snapshot=snapshot,
    )


# --- Application ------------------------------------------------------------


def create_app(
    settings: ApiSettings,
    *,
    launcher_factory: LauncherFactory | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
    rules_config: RulesConfig | None = None,
    evaluation_date_fn: Callable[[], date] = date.today,
) -> FastAPI:
    """Build the Application. ``launcher_factory`` defaults to the Workbench
    launcher; tests inject their own. Retries are out of scope, so
    ``max_attempts`` must be 1."""
    config = orchestrator_config or OrchestratorConfig(
        max_concurrency=settings.max_concurrency,
        max_attempts=1,
        poll_interval_seconds=settings.poll_interval_seconds,
    )
    if config.max_attempts != 1:
        raise ValueError("the API runs cases with max_attempts=1 (retry is out of scope)")
    rules = rules_config or RulesConfig()

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.spool_dir.mkdir(parents=True, exist_ok=True)
    db.connect(settings.db_path).close()  # fail fast; runs migrations once

    runner = CaseRunner(
        settings.db_path,
        launcher_factory or workbench_launcher_factory(settings),
        config,
    )
    upload_lock = threading.Lock()
    verdict_lock = threading.Lock()
    ui_built = (settings.ui_dist / "index.html").is_file()

    app = FastAPI(
        title="PS-06 API",
        version="0.1.0",
        description="Upload an immigration case file, follow per-document "
        "progress, and read the VERDE/AMARILLO/ROJO verdict with S0-S3.",
    )
    app.state.runner = runner
    app.state.settings = settings

    @contextmanager
    def open_store() -> Iterator[StatusStore]:
        # Opened and closed inside the endpoint: sync endpoints run on one
        # worker thread, and sqlite3 connections must stay on their thread.
        conn = db.connect(settings.db_path)
        try:
            yield StatusStore(conn)
        finally:
            conn.close()

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "model": settings.model_name, "ui_built": ui_built}

    @app.get("/api/cases", response_model=list[CaseListItem])
    def list_cases() -> list[CaseListItem]:
        with open_store() as store:
            return _list_cases(store)

    def _list_cases(store: StatusStore) -> list[CaseListItem]:
        cases = store.connection.execute(
            "SELECT case_id, MAX(last_updated) AS last FROM document_status "
            "GROUP BY case_id ORDER BY last DESC, case_id;"
        ).fetchall()
        items = []
        for row in cases:
            case = _case_out(store, row["case_id"], runner)
            items.append(CaseListItem(
                id=case.id,
                case_status=case.case_status,
                case_label=case.case_label,
                running=case.running,
                interrupted=case.interrupted,
                total=case.total,
                ocr_done=case.ocr_done,
                failed=case.failed,
                last_updated=row["last"],
            ))
        return items

    @app.post("/api/cases", status_code=202, response_model=CaseOut)
    def upload_case(
        case_id: str = Form(...),
        files: list[UploadFile] = File(...),
    ) -> CaseOut:
        with open_store() as store:
            return _upload_case(store, case_id, files)

    def _upload_case(
        store: StatusStore, case_id: str, files: list[UploadFile]
    ) -> CaseOut:
        case_id = _nfc(case_id)
        _check_name("case id", case_id)
        if not files:
            raise HTTPException(400, "no files uploaded")

        planned: dict[str, tuple[UploadFile, str]] = {}
        for upload in files:
            filename = _nfc(PurePosixPath((upload.filename or "").replace("\\", "/")).name)
            _check_name("file name", filename)
            suffix = PurePosixPath(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    415,
                    f"{filename}: unsupported type (supported: "
                    f"{', '.join(sorted(SUPPORTED_EXTENSIONS))})",
                )
            document_id = PurePosixPath(filename).stem
            _check_name("document id", document_id)
            if document_id in planned:
                raise HTTPException(
                    400,
                    f"two files resolve to document {document_id!r}: "
                    f"{planned[document_id][1]!r} and {filename!r}",
                )
            planned[document_id] = (upload, filename)

        case_dir = settings.upload_dir / case_id
        with upload_lock:
            if store.list_for_case(case_id) or runner.is_running(case_id):
                raise HTTPException(409, f"case {case_id!r} already exists")
            try:
                case_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                raise HTTPException(
                    409, f"upload folder for case {case_id!r} already exists"
                ) from None

            paths: dict[str, str] = {}
            try:
                for document_id, (upload, filename) in planned.items():
                    dest = case_dir / filename
                    _save_upload(upload, dest)
                    paths[document_id] = str(dest)
            except HTTPException:
                _remove_dir(case_dir)
                raise
            except OSError as exc:
                _remove_dir(case_dir)
                logger.error("upload failed: case=%s: %s", case_id, exc)
                raise HTTPException(500, f"could not save upload: {exc}") from exc

            registered: list[str] = []
            try:
                for document_id in paths:
                    store.register_document(case_id, document_id)
                    registered.append(document_id)
                runner.start(case_id, paths)
            except (StatusStoreError, sqlite3.Error, RuntimeError) as exc:
                detail = f"case start failed: {type(exc).__name__}: {exc}"
                logger.error("case=%s: %s", case_id, detail)
                _fail_unfinished(store, case_id, registered, detail)
                raise HTTPException(500, detail) from exc

        logger.info("case accepted: case=%s documents=%d", case_id, len(paths))
        return _case_out(store, case_id, runner)

    @app.get("/api/cases/{case_id}", response_model=CaseOut)
    def case_status(case_id: str) -> CaseOut:
        with open_store() as store:
            return _case_out(store, case_id, runner)

    @app.get("/api/cases/{case_id}/verdict", response_model=VerdictOut)
    def case_verdict(case_id: str) -> VerdictOut:
        with open_store() as store:
            return _case_verdict(store, case_id)

    def _case_verdict(store: StatusStore, case_id: str) -> VerdictOut:
        rows = store.list_for_case(case_id)
        if not rows:
            raise HTTPException(404, f"unknown case: {case_id!r}")
        status = store.case_status(case_id)

        failed = [r for r in rows if r.stage is DocumentStage.FAILED]
        if failed:
            out = []
            for r in failed:
                if not r.error_detail:
                    logger.error(
                        "FAILED without error_detail: case=%s document=%s",
                        case_id, r.document_id,
                    )
                out.append(FailedDocOut(
                    name=r.document_id,
                    tries=r.attempt_count,
                    error_detail=r.error_detail or "FAILED without error_detail",
                ))
            return VerdictOut(id=case_id, result="failed", case_status=status, failed=out)

        if status not in (CaseStatus.OCR_DONE, CaseStatus.RULES_EVALUATED):
            return VerdictOut(
                id=case_id,
                result="pending",
                case_status=status,
                pending=[r.document_id for r in rows if r.stage is not DocumentStage.OCR_DONE],
            )

        with verdict_lock:
            snapshot = get_snapshot(store, case_id)
            if snapshot is None:
                try:
                    snapshot = evaluate_case(case_id, store, rules, evaluation_date_fn())
                except Exception as exc:  # surfaced as a 500 with its cause
                    logger.exception("rules evaluation failed: case=%s", case_id)
                    raise HTTPException(
                        500, f"rules evaluation failed: {type(exc).__name__}: {exc}"
                    ) from exc
        return _verdict_out(store, case_id, rows, snapshot, rules, settings.model_name)

    if ui_built:
        app.mount("/", StaticFiles(directory=settings.ui_dist, html=True), name="ui")
    else:
        logger.warning("UI not built: %s has no index.html; serving /api only", settings.ui_dist)

    return app
