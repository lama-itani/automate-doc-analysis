"""Tests for the PS-06 API service (``ps06/api/app.py``).

``GatedLauncher`` stands in for :class:`WorkbenchJobLauncher`: its handles stay
running until the test opens the gate, then ingest a result file on the runner
thread (the same way the real handle does). No network, no ``cmlapi``.
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ps06.api.app as api_app
from ps06.api.app import ApiSettings, create_app
from ps06.classification.classifier import ClassificationResult, DocumentType
from ps06.classification.field_extraction import FieldExtractionResult
from ps06.ocr.envelope import OcrResult
from ps06.ocr.extraction import ExtractionResult
from ps06.ocr.result_file import ResultFile, ingest
from ps06.orchestrator.orchestrator import OrchestratorConfig
from ps06.rules.snapshot import ReportSnapshot, S1Row, S2Row, S3Row
from ps06.rules.rules_config import RulesConfig
from ps06.status import db
from ps06.status.states import DocumentStage
from ps06.status.store import StatusStore

PDF = b"%PDF-1.4 test bytes"
WAIT = 10.0


# --- Fakes ------------------------------------------------------------------


def _ok_result(case_id: str, document_id: str) -> OcrResult:
    is_id = document_id.startswith("ID")
    return OcrResult.build(
        case_id=case_id,
        document_id=document_id,
        file_path=f"/x/{document_id}.pdf",
        extraction=ExtractionResult(
            file_type="pdf",
            extracted_text="texto suficiente para el documento",
            page_count=2,
            pages_via_text_layer=0,
            pages_via_vlm=2,
            orientation_corrections={},
            acroform_fields={},
            xfa_fields={},
        ),
        processing_seconds=12.5,
        model_name="test-model",
        classification=ClassificationResult(
            document_type=DocumentType.ID_DOCUMENT if is_id else DocumentType.OTHER
        ),
        field_extraction=FieldExtractionResult(
            fields={"nombres": "JOSE", "apellidos": "MONTALVA"} if is_id else {}
        ),
    )


class _Handle:
    def __init__(self, launcher: GatedLauncher, case_id: str, document_id: str) -> None:
        self.case_id = case_id
        self.document_id = document_id
        self._launcher = launcher
        self._done = False

    def done(self) -> bool:
        if self._done:
            return True
        if not self._launcher.gate.is_set():
            return False
        error = self._launcher.failures.get(self.document_id)
        if error is None:
            rf = ResultFile(
                ok=True, case_id=self.case_id, document_id=self.document_id,
                result=_ok_result(self.case_id, self.document_id),
            )
        else:
            rf = ResultFile(
                ok=False, case_id=self.case_id, document_id=self.document_id,
                error_detail=error,
            )
        ingest(self._launcher.store, self.case_id, self.document_id, rf)
        self._done = True
        return True

    def exception(self) -> BaseException | None:
        return None


class GatedLauncher:
    def __init__(self, store: StatusStore, gate: threading.Event, failures: dict) -> None:
        self.store = store
        self.gate = gate
        self.failures = failures
        self.launched: list[tuple[str, str, str]] = []

    def launch(self, case_id: str, document_id: str, file_path: str) -> _Handle:
        self.launched.append((case_id, document_id, file_path))
        return _Handle(self, case_id, document_id)


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.gate = threading.Event()
        self.failures: dict[str, str] = {}
        self.launchers: list[GatedLauncher] = []
        self.factory_error: Exception | None = None
        self.evaluations = 0
        self.settings = ApiSettings(
            db_path=tmp_path / "data" / "ps06.db",
            upload_dir=tmp_path / "data" / "uploads",
            spool_dir=tmp_path / "data" / "spool",
            ui_dist=tmp_path / "ui" / "dist",
            endpoint_url="http://localhost/v1",
        )

    def factory(self, store: StatusStore) -> GatedLauncher:
        if self.factory_error is not None:
            raise self.factory_error
        launcher = GatedLauncher(store, self.gate, self.failures)
        self.launchers.append(launcher)
        return launcher

    def eval_date(self) -> date:
        self.evaluations += 1
        return date(2026, 10, 6)

    def app(self):
        return create_app(
            self.settings,
            launcher_factory=self.factory,
            orchestrator_config=OrchestratorConfig(max_attempts=1, poll_interval_seconds=0.01),
            evaluation_date_fn=self.eval_date,
        )

    def store(self) -> StatusStore:
        return StatusStore(db.connect(self.settings.db_path))


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


@pytest.fixture
def client(h):
    app = h.app()
    with TestClient(app) as c:
        c.app_ref = app
        yield c
    h.gate.set()  # never leave a runner thread blocked


def _upload(client, case_id="C1", names=("ID_JOSE.pdf", "NACPEPE.pdf"), data=PDF):
    files = [("files", (n, data, "application/pdf")) for n in names]
    return client.post("/api/cases", data={"case_id": case_id}, files=files)


def _finish(client, h, case_id="C1"):
    h.gate.set()
    client.app_ref.state.runner.join(case_id, WAIT)
    assert not client.app_ref.state.runner.is_running(case_id)


# --- Upload -----------------------------------------------------------------


def test_upload_saves_registers_and_launches(client, h):
    r = _upload(client)
    assert r.status_code == 202
    body = r.json()
    assert body["id"] == "C1"
    assert body["running"] is True
    assert body["interrupted"] is False
    assert body["case_label"] == {"es": "En proceso", "en": "Processing"}
    assert [d["name"] for d in body["docs"]] == ["ID_JOSE", "NACPEPE"]
    assert {d["stage"] for d in body["docs"]} == {"RECEIVED"}

    case_dir = h.settings.upload_dir / "C1"
    assert (case_dir / "ID_JOSE.pdf").read_bytes() == PDF
    _finish(client, h)
    launched = {(c, d, p) for c, d, p in h.launchers[0].launched}
    assert launched == {
        ("C1", "ID_JOSE", str(case_dir / "ID_JOSE.pdf")),
        ("C1", "NACPEPE", str(case_dir / "NACPEPE.pdf")),
    }


def test_status_after_ocr_shows_extraction_details(client, h):
    _upload(client)
    _finish(client, h)
    body = client.get("/api/cases/C1").json()
    assert body["case_status"] == "OCR_DONE"
    assert body["running"] is False and body["interrupted"] is False
    doc = body["docs"][0]
    assert doc["stage"] == "OCR_DONE"
    assert doc["stage_label"] == {"es": "Lectura completada", "en": "Read complete"}
    assert doc["type"] == "ID_DOCUMENT"
    assert (doc["pagesText"], doc["pagesVlm"], doc["seconds"]) == (0, 2, 12.5)
    assert doc["error_detail"] is None


@pytest.mark.parametrize(
    ("names", "status"),
    [
        (("notes.txt",), 415),
        (("a.pdf", "a.png"), 400),  # same stem
        ((".hidden.pdf",), 400),
    ],
)
def test_upload_rejects_bad_files(client, h, names, status):
    r = _upload(client, names=names)
    assert r.status_code == status
    assert not (h.settings.upload_dir / "C1").exists()
    assert client.get("/api/cases/C1").status_code == 404


def test_upload_rejects_empty_file_and_cleans_folder(client, h):
    r = _upload(client, names=("a.pdf",), data=b"")
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]
    assert not (h.settings.upload_dir / "C1").exists()


def test_upload_rejects_oversize_file(client, h, monkeypatch):
    monkeypatch.setattr(api_app, "_MAX_BYTES", 4)
    r = _upload(client, names=("a.pdf",))
    assert r.status_code == 413
    assert not (h.settings.upload_dir / "C1").exists()


@pytest.mark.parametrize("case_id", ["../x", "a/b", ".", " C1", "a\\b"])
def test_upload_rejects_unsafe_case_id(client, case_id):
    assert _upload(client, case_id=case_id).status_code == 400


def test_upload_strips_client_path_from_file_name(client, h):
    r = _upload(client, names=("C:\\scans\\ID_JOSE.pdf",))
    assert r.status_code == 202
    assert (h.settings.upload_dir / "C1" / "ID_JOSE.pdf").exists()


def test_duplicate_case_is_409(client, h):
    assert _upload(client).status_code == 202
    assert _upload(client).status_code == 409
    _finish(client, h)
    assert _upload(client).status_code == 409


# --- Verdict ----------------------------------------------------------------


def test_verdict_pending_while_running(client, h):
    _upload(client)
    body = client.get("/api/cases/C1/verdict").json()
    assert body["result"] == "pending"
    assert body["pending"] == ["ID_JOSE", "NACPEPE"]
    assert body["verdict"] is None
    assert h.evaluations == 0


def test_verdict_returns_failed_documents_and_skips_rules(client, h):
    h.failures["NACPEPE"] = "FileNotFoundError: /x/NACPEPE.pdf"
    _upload(client)
    _finish(client, h)

    body = client.get("/api/cases/C1/verdict").json()
    assert body["result"] == "failed"
    assert body["case_status"] == "FAILED"
    assert body["failed"] == [
        {"name": "NACPEPE", "tries": 1, "error_detail": "FileNotFoundError: /x/NACPEPE.pdf"}
    ]
    assert body["snapshot"] is None
    assert h.evaluations == 0

    status = client.get("/api/cases/C1").json()
    failed = [d for d in status["docs"] if d["name"] == "NACPEPE"][0]
    assert failed["stage_label"] == {"es": "Error", "en": "Failed"}
    assert failed["error_detail"].startswith("FileNotFoundError")


def test_verdict_runs_rules_once_and_matches_ui_shape(client, h):
    _upload(client)
    _finish(client, h)

    body = client.get("/api/cases/C1/verdict").json()
    assert body["result"] == "verdict"
    assert body["case_status"] == "RULES_EVALUATED"
    assert body["verdict"] in {"VERDE", "AMARILLO", "ROJO"}
    assert body["verdict"] == body["snapshot"]["estado"]
    assert body["justification"]["es"] == body["snapshot"]["justificacion"]
    assert len(body["s1"]) == 5
    assert body["s1"][0] == {"es": "Solicitud Principal", "en": "Main Application", "present": False}
    assert [p["gen"] for p in body["lineage"]] == ["G1", "G2", "G3"]
    assert body["lineage"][0]["role"] == {"es": "Solicitante", "en": "Applicant"}
    assert body["meta"] == {"model": "test-model", "totalSeconds": 25.0}
    assert any(i["code"] == "FALTANTE: Solicitud Principal" for i in body["issues"])

    id_doc = [d for d in body["docs"] if d["name"] == "ID_JOSE"][0]
    assert id_doc["type"] == "ID_DOCUMENT"
    assert set(id_doc["fields"]) == {"nombres", "apellidos"}
    assert "numero_id" in id_doc["missing"]
    assert "DOCUMENT_INCOMPLETE" in id_doc["flags"]

    again = client.get("/api/cases/C1/verdict").json()
    assert again["snapshot"] == body["snapshot"]
    assert h.evaluations == 1  # stored snapshot reused


def test_rules_failure_is_500_with_cause(client, h, monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("bad rules")

    monkeypatch.setattr(api_app, "evaluate_case", boom)
    _upload(client)
    _finish(client, h)
    r = client.get("/api/cases/C1/verdict")
    assert r.status_code == 500
    assert r.json()["detail"] == "rules evaluation failed: ValueError: bad rules"


def test_unknown_case_is_404(client):
    assert client.get("/api/cases/NOPE").status_code == 404
    assert client.get("/api/cases/NOPE/verdict").status_code == 404


# --- Failures and restart ---------------------------------------------------


def test_runner_crash_marks_documents_failed(client, h):
    h.factory_error = RuntimeError("cmlapi missing")
    _upload(client)
    client.app_ref.state.runner.join("C1", WAIT)

    body = client.get("/api/cases/C1/verdict").json()
    assert body["result"] == "failed"
    details = {d["error_detail"] for d in body["failed"]}
    assert details == {"case runner failed: RuntimeError: cmlapi missing"}


def test_received_documents_after_restart_are_interrupted(h):
    store = h.store()
    store.register_document("C9", "ID_JOSE")
    store.connection.close()

    with TestClient(h.app()) as c:
        body = c.get("/api/cases/C9").json()
        assert body["interrupted"] is True
        assert body["running"] is False
        assert body["case_label"] == {"es": "Interrumpido", "en": "Interrupted"}
        assert body["message"]["en"].startswith("The application restarted")
        assert body["docs"][0]["stage_label"]["en"] == "Interrupted"

        cases = c.get("/api/cases").json()
        assert [x["id"] for x in cases] == ["C9"]
        assert cases[0]["interrupted"] is True
        assert c.get("/api/cases/C9/verdict").json()["result"] == "pending"


# --- Response builders (unit) -----------------------------------------------


def _snapshot(**kw) -> ReportSnapshot:
    base = dict(
        case_id="C1",
        estado="ROJO",
        justificacion="Se detectaron conflictos que impiden validar el caso automáticamente.",
        problemas=(
            "Certificado de Nacimiento del Progenitor",
            "ID_INCONSISTENTE",
            "DOCUMENT_INCOMPLETE",
            "Corroboración insuficiente (<2 documentos) para: G1",
        ),
        s1_rows=(S1Row(tipo="Certificado de Nacimiento del Progenitor",
                       descripcion="BIRTH_CERT (G2)", estado="Faltante"),),
        s2_rows=(S2Row(documento="NACPAPA", evaluacion_credencial="Con observaciones",
                       campos_faltantes=("sexo",), consistencia="Con conflictos",
                       conflictos=("DOCUMENT_INCOMPLETE",)),),
        s3_rows=(
            S3Row(generacion="G1", nombre_completo="JOSE MONTALVA",
                  fecha_lugar_nacimiento="1985-05-28 / No determinado", relacion="Solicitante"),
            S3Row(generacion="G2", nombre_completo="No determinado",
                  fecha_lugar_nacimiento="No determinado", relacion="Progenitor"),
        ),
    )
    base.update(kw)
    return ReportSnapshot(**base)


def test_issues_map_problemas_to_severity_and_documents():
    issues = api_app._issues(_snapshot(), RulesConfig())
    assert [(i.code, i.severity, i.docs) for i in issues] == [
        ("FALTANTE: Certificado de Nacimiento del Progenitor", "AMARILLO", []),
        ("ID_INCONSISTENTE", "ROJO", []),
        ("DOCUMENT_INCOMPLETE", "AMARILLO", ["NACPAPA"]),
        ("CORROBORACION_INSUFICIENTE", "AMARILLO", []),
    ]
    assert issues[0].title.en == "Missing: Parent's Birth Certificate"
    assert issues[1].title.es == "ID_INCONSISTENTE"  # raw constant


def test_verde_has_no_issues():
    snap = _snapshot(estado="VERDE", problemas=("Ninguno",))
    assert api_app._issues(snap, RulesConfig()) == []


def test_lineage_parses_name_birth_and_place():
    g1, g2 = api_app._lineage(_snapshot())
    assert (g1.name, g1.birth, g1.place) == ("JOSE MONTALVA", "1985-05-28", None)
    assert (g2.name, g2.birth, g2.place) == (None, None, None)
    assert g2.role.en == "Parent"


# --- Settings and static UI -------------------------------------------------


def test_max_attempts_must_be_one(h):
    with pytest.raises(ValueError, match="max_attempts=1"):
        create_app(h.settings, launcher_factory=h.factory,
                   orchestrator_config=OrchestratorConfig(max_attempts=2))


def test_from_env_requires_endpoint():
    with pytest.raises(ValueError, match="PS06_ENDPOINT_URL"):
        ApiSettings.from_env({})


def test_from_env_defaults_and_bad_concurrency():
    s = ApiSettings.from_env({"PS06_ENDPOINT_URL": "https://e/v1", "PS06_DATA_DIR": "/d"})
    assert s.db_path == Path("/d/ps06.db")
    assert s.upload_dir == Path("/d/uploads")
    assert s.spool_dir == Path("/d/spool")
    assert s.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    with pytest.raises(ValueError, match="PS06_MAX_CONCURRENCY"):
        ApiSettings.from_env({"PS06_ENDPOINT_URL": "x", "PS06_MAX_CONCURRENCY": "two"})


def test_health_without_ui(client):
    assert client.get("/api/health").json()["ui_built"] is False
    assert client.get("/").status_code == 404


def test_serves_built_ui(h):
    h.settings.ui_dist.mkdir(parents=True)
    (h.settings.ui_dist / "index.html").write_text("<html>PS-06</html>")
    with TestClient(h.app()) as c:
        assert c.get("/api/health").json()["ui_built"] is True
        r = c.get("/")
        assert r.status_code == 200 and "PS-06" in r.text
        assert c.get("/api/cases").status_code == 200
