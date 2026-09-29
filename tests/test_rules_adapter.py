"""Tests for the OcrResult -> CanonicalDocument/CaseBundle adapter (M-4, step 4)."""

from __future__ import annotations

import json

import pytest

from ps06.classification.classifier import ClassificationResult, DocumentType
from ps06.ocr.envelope import OcrResult, get_extraction
from ps06.ocr.extraction import ExtractionResult
from ps06.rules.adapter import MissingExtractionError, adapt_document, build_case_bundle
from ps06.rules.schemas import CertificadoData, IdData, SolicitudData, SOLICITUD_SOURCE_LABELS
from ps06.status import db
from ps06.status.store import StatusStore


def _extraction(
    *, extracted_text: str = "", acroform_fields: dict[str, str] | None = None,
    xfa_fields: dict[str, str] | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        file_type="pdf",
        extracted_text=extracted_text,
        page_count=1,
        pages_via_text_layer=1,
        pages_via_vlm=0,
        orientation_corrections={},
        acroform_fields=acroform_fields or {},
        xfa_fields=xfa_fields or {},
    )


def _result(
    *, case_id: str = "c1", document_id: str = "d1", extraction: ExtractionResult | None = None,
    classification: ClassificationResult | None = None,
) -> OcrResult:
    return OcrResult.build(
        case_id=case_id,
        document_id=document_id,
        file_path=f"{document_id}.pdf",
        extraction=extraction or _extraction(),
        processing_seconds=1.0,
        model_name="test-model",
        classification=classification,
    )


def _persist(store: StatusStore, result: OcrResult) -> None:
    conn = store.connection
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


# --- get_extraction ----------------------------------------------------------


def test_get_extraction_returns_none_when_absent():
    store = StatusStore(db.connect())
    assert get_extraction(store, "c1", "d1") is None


def test_get_extraction_round_trips_persisted_result():
    store = StatusStore(db.connect())
    result = _result(classification=ClassificationResult(document_type=DocumentType.OTHER))
    _persist(store, result)
    assert get_extraction(store, "c1", "d1") == result


# --- adapt_document: APPLICATION ---------------------------------------------


def test_adapt_document_application_matches_acroform_fields_by_label():
    result = _result(
        extraction=_extraction(
            acroform_fields={"Nombre del solicitante": "Ana García", "Nacionalidad": "Boliviana"}
        ),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION, generation="G1"),
    )
    doc = adapt_document(result)
    assert doc.solicitud.nombre_solicitante == "Ana García"
    assert doc.solicitud.nacionalidad == "Boliviana"
    remaining = set(SOLICITUD_SOURCE_LABELS) - {"nombre_solicitante", "nacionalidad"}
    assert remaining <= set(doc.missing_fields)


def test_adapt_document_application_case_insensitive_whitespace_match():
    result = _result(
        extraction=_extraction(acroform_fields={"  nombre DEL Solicitante  ": "Ana"}),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud.nombre_solicitante == "Ana"


def test_adapt_document_application_no_form_fields_leaves_solicitud_empty():
    result = _result(
        extraction=_extraction(),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud == SolicitudData()
    assert doc.missing_fields == tuple(SOLICITUD_SOURCE_LABELS.keys())


def test_adapt_document_application_firma_never_inferred():
    result = _result(
        extraction=_extraction(acroform_fields={"Firma": "X"}),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud.firma is False
    assert "firma" in doc.missing_fields


def test_adapt_document_application_unmatched_form_fields_ignored():
    result = _result(
        extraction=_extraction(acroform_fields={"txtField17": "whatever"}),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)  # should not raise
    assert doc.solicitud == SolicitudData()


# --- adapt_document: ID_DOCUMENT / BIRTH_CERT --------------------------------


def test_adapt_document_id_document_always_none_with_full_missing_fields():
    result = _result(
        classification=ClassificationResult(document_type=DocumentType.ID_DOCUMENT)
    )
    doc = adapt_document(result)
    assert doc.identidad is None
    assert doc.missing_fields == tuple(IdData.model_fields.keys())


def test_adapt_document_birth_cert_always_none_with_full_missing_fields():
    result = _result(
        classification=ClassificationResult(document_type=DocumentType.BIRTH_CERT, generation="G2")
    )
    doc = adapt_document(result)
    assert doc.certificado is None
    assert doc.missing_fields == tuple(CertificadoData.model_fields.keys())


# --- adapt_document: OTHER / EMPTY / classification=None --------------------


@pytest.mark.parametrize("doc_type", [DocumentType.OTHER, DocumentType.EMPTY])
def test_adapt_document_other_and_empty_have_no_missing_fields(doc_type):
    result = _result(classification=ClassificationResult(document_type=doc_type))
    doc = adapt_document(result)
    assert doc.solicitud is None
    assert doc.identidad is None
    assert doc.certificado is None
    assert doc.missing_fields == ()


def test_adapt_document_classification_none_flags_missing_type_and_generation():
    result = _result(classification=None)
    doc = adapt_document(result)
    assert doc.document_type is DocumentType.OTHER
    assert doc.generation is None
    assert doc.missing_fields == ("document_type", "generation")


def test_adapt_document_normalizes_text():
    result = _result(
        extraction=_extraction(extracted_text="```markdown\nCertiﬁcado\n```"),
        classification=ClassificationResult(document_type=DocumentType.OTHER),
    )
    doc = adapt_document(result)
    assert "```" not in doc.normalized_text
    assert "Certificado" in doc.normalized_text


# --- build_case_bundle --------------------------------------------------------


def test_build_case_bundle_assembles_all_documents():
    store = StatusStore(db.connect())
    store.register_document("c1", "app")
    store.register_document("c1", "id")
    store.register_document("c1", "cert")

    _persist(
        store,
        _result(
            document_id="app",
            classification=ClassificationResult(document_type=DocumentType.APPLICATION),
        ),
    )
    _persist(
        store,
        _result(
            document_id="id",
            classification=ClassificationResult(document_type=DocumentType.ID_DOCUMENT),
        ),
    )
    _persist(
        store,
        _result(
            document_id="cert",
            classification=ClassificationResult(document_type=DocumentType.BIRTH_CERT, generation="G1"),
        ),
    )

    bundle = build_case_bundle(store, "c1")
    assert len(bundle.documents) == 3
    assert bundle.application.document_id == "app"
    assert {d.document_id for d in bundle.identity_documents} == {"id"}
    assert {d.document_id for d in bundle.birth_certificates} == {"cert"}


def test_build_case_bundle_raises_when_extraction_missing():
    store = StatusStore(db.connect())
    store.register_document("c1", "app")
    with pytest.raises(MissingExtractionError):
        build_case_bundle(store, "c1")
