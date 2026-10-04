"""Tests for the OcrResult -> CanonicalDocument/CaseBundle adapter (M-4, step 4)."""

from __future__ import annotations

import json

import pytest

from ps06.classification.classifier import ClassificationResult, DocumentType
from ps06.classification.field_extraction import FieldExtractionResult
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
    field_extraction: FieldExtractionResult | None = None,
) -> OcrResult:
    return OcrResult.build(
        case_id=case_id,
        document_id=document_id,
        file_path=f"{document_id}.pdf",
        extraction=extraction or _extraction(),
        processing_seconds=1.0,
        model_name="test-model",
        classification=classification,
        field_extraction=field_extraction,
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


def test_adapt_document_application_firma_signed_when_field_non_empty():
    result = _result(
        extraction=_extraction(acroform_fields={"Firma": "X"}),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud.firma is True
    assert "firma" not in doc.missing_fields


def test_adapt_document_application_firma_missing_when_field_absent():
    result = _result(
        extraction=_extraction(acroform_fields={"Nombre del solicitante": "Ana"}),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud.firma is False
    assert "firma" in doc.missing_fields


def test_adapt_document_application_firma_missing_when_field_empty():
    result = _result(
        extraction=_extraction(acroform_fields={"Firma": "   "}),
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


# --- adapt_document: APPLICATION, real Anexo I/III/IV Textfield-N layouts ----


def test_adapt_document_anexo_iv_maps_textfield_n_positionally():
    # Field names/order taken from pymupdf inspection of the real Anexo IV
    # AcroForm template (and a filled real-world sample).
    result = _result(
        extraction=_extraction(
            extracted_text="A N E X O IV\nMODELO DE SOLICITUD...",
            acroform_fields={
                "Textfield": "Ricardo",
                "Textfield_p1": "Ricardo",
                "Textfield-0": "Minaya",
                "Textfield-1": "Sainz",
                "Textfield-2": "Española",
                "Textfield-3": "Soltero",
                "Textfield-4": "XDD882743",
                "Textfield-5": "105a St James Rd, Sutton",
                "Textfield-6": "Surrey",
                "Textfield-7": "Reino Unido",
                "Textfield-8": "7366835219",
                "Textfield-9": "r.minaya.sainz@gmail.com",
                "Textfield-10": "Ciudad de Mexico",
                "Textfield-11": "005030",
                "Textfield-12": "315",
                "Textfield-13": "30/Nov/2015",
            },
        ),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION, generation="G1"),
    )
    doc = adapt_document(result)
    s = doc.solicitud
    assert s.nombre_solicitante == "Ricardo"
    assert s.apellido_padre == "Minaya"
    assert s.apellido_madre == "Sainz"
    assert s.nacionalidad == "Española"
    assert s.estado_civil == "Soltero"
    assert s.numero_id == "XDD882743"
    assert s.domicilio == "105a St James Rd, Sutton"
    assert s.provincia == "Surrey"
    assert s.pais == "Reino Unido"
    assert s.telf_contacto == "7366835219"
    assert s.email == "r.minaya.sainz@gmail.com"
    assert s.registro_civil_inscripcion == "Ciudad de Mexico"
    assert s.tomo == "005030"
    assert s.folio == "315"
    assert s.fecha_ejercicio_opcion == "30/Nov/2015"
    # firma and checkbox-derived fields are never inferred -> stay missing.
    assert "firma" in doc.missing_fields


def test_adapt_document_anexo_iv_maps_signing_block_date_and_place():
    result = _result(
        extraction=_extraction(
            extracted_text="A N E X O IV\nMODELO DE SOLICITUD...",
            acroform_fields={
                "Textfield": "Ricardo",
                "Textfield-0": "Minaya",
                "Textfield-4": "XDD882743",
                "En": "Londres",
                "a": "22",
                "de": "Octubre",
                "de-0": "2025",
            },
        ),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION, generation="G1"),
    )
    doc = adapt_document(result)
    assert doc.solicitud.lugar_presentacion == "Londres"
    assert doc.solicitud.fecha_presentacion == "22/Octubre/2025"
    assert "lugar_presentacion" not in doc.missing_fields
    assert "fecha_presentacion" not in doc.missing_fields


def test_adapt_document_anexo_iv_fecha_presentacion_unset_when_day_missing():
    result = _result(
        extraction=_extraction(
            extracted_text="A N E X O IV\nMODELO DE SOLICITUD...",
            acroform_fields={
                "Textfield": "Ricardo",
                "de": "Octubre",
                "de-0": "2025",
            },
        ),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION, generation="G1"),
    )
    doc = adapt_document(result)
    assert doc.solicitud.fecha_presentacion is None
    assert "fecha_presentacion" in doc.missing_fields


def test_adapt_document_anexo_i_does_not_map_signing_block_fields():
    result = _result(
        extraction=_extraction(
            extracted_text="ANEXO I\nModelo de solicitud...",
            acroform_fields={
                "Textfieldad": "Registro Civil de Madrid",
                "Textfield": "Lucia",
                "En": "Londres",
                "a": "22",
                "de": "Octubre",
                "de-0": "2025",
            },
        ),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud.lugar_presentacion is None
    assert doc.solicitud.fecha_presentacion is None
    assert "lugar_presentacion" in doc.missing_fields
    assert "fecha_presentacion" in doc.missing_fields


def test_adapt_document_anexo_iii_shares_anexo_iv_field_map():
    result = _result(
        extraction=_extraction(
            extracted_text="ANEXO III\nModelo de solicitud...",
            acroform_fields={
                "Textfield": "Lucia",
                "Textfield-0": "Perez",
                "Textfield-4": "AB123456",
            },
        ),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud.nombre_solicitante == "Lucia"
    assert doc.solicitud.apellido_padre == "Perez"
    assert doc.solicitud.numero_id == "AB123456"


def test_adapt_document_anexo_i_uses_distinct_shifted_field_map():
    # Anexo I's layout differs from III/IV: "Textfieldad" is the registry
    # destination field, and "Textfield-3a" (not "-5") is domicilio, which
    # shifts provincia/pais/telf/email up by one index.
    result = _result(
        extraction=_extraction(
            extracted_text="ANEXO I\nModelo de solicitud...",
            acroform_fields={
                "Textfieldad": "Registro Civil de Madrid",
                "Textfield": "Marta",
                "Textfield-3a": "Calle Mayor 1",
                "Textfield-5": "Madrid",
                "Textfield-6": "España",
                "Textfield-9": "Italiana",
            },
        ),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud.registro_civil_inscripcion == "Registro Civil de Madrid"
    assert doc.solicitud.nombre_solicitante == "Marta"
    assert doc.solicitud.domicilio == "Calle Mayor 1"
    assert doc.solicitud.provincia == "Madrid"
    assert doc.solicitud.pais == "España"
    assert doc.solicitud.nacionalidad_origen_progenitor == "Italiana"


def test_adapt_document_non_anexo_source_still_uses_label_matching():
    # No recognizable Anexo header -> falls back to exact-label matching,
    # unaffected by the Textfield-N positional maps.
    result = _result(
        extraction=_extraction(
            extracted_text="Some other application form",
            acroform_fields={"Nombre del solicitante": "Ana"},
        ),
        classification=ClassificationResult(document_type=DocumentType.APPLICATION),
    )
    doc = adapt_document(result)
    assert doc.solicitud.nombre_solicitante == "Ana"


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


def test_adapt_document_id_document_full_fields_populates_identidad():
    fields = {
        "tipo_id": "Pasaporte",
        "numero_id": "XDD882743",
        "apellidos": "Minaya Sainz",
        "nombres": "Ricardo",
        "nacionalidad": "Española",
        "sexo": "M",
        "fecha_nacimiento": "01/02/1990",
        "fecha_emision": "01/01/2020",
        "fecha_vencimiento": "01/01/2030",
    }
    result = _result(
        classification=ClassificationResult(document_type=DocumentType.ID_DOCUMENT),
        field_extraction=FieldExtractionResult(fields=fields),
    )
    doc = adapt_document(result)
    assert doc.identidad == IdData(**fields)
    assert doc.missing_fields == ()


def test_adapt_document_id_document_partial_fields_reports_missing():
    result = _result(
        classification=ClassificationResult(document_type=DocumentType.ID_DOCUMENT),
        field_extraction=FieldExtractionResult(fields={"nombres": "Ricardo"}),
    )
    doc = adapt_document(result)
    assert doc.identidad.nombres == "Ricardo"
    assert doc.identidad.numero_id is None
    assert "numero_id" in doc.missing_fields
    assert "nombres" not in doc.missing_fields


def test_adapt_document_birth_cert_full_fields_populates_certificado():
    fields = {
        "grado_certificado": "G2",
        "nombre": "Maria Lopez",
        "fecha_nacimiento": "05/05/1965",
        "sexo": "F",
        "lugar_inscripcion": "Madrid",
        "fecha_inscripcion": "10/05/1965",
        "progenitor1_nombre": "Jose Lopez",
        "progenitor1_num_doc": "12345678",
        "progenitor1_nacionalidad": "Española",
        "progenitor2_nombre": "Carmen Ruiz",
        "progenitor2_num_doc": "87654321",
        "progenitor2_nacionalidad": "Española",
        "apostillado": "APOSTILLE / La Haya",
    }
    result = _result(
        classification=ClassificationResult(document_type=DocumentType.BIRTH_CERT),
        field_extraction=FieldExtractionResult(fields=fields),
    )
    doc = adapt_document(result)
    assert doc.certificado.grado_certificado == "G2"
    assert doc.certificado.nombre == "Maria Lopez"
    assert doc.certificado.apostillado is True
    assert doc.missing_fields == ()


def test_adapt_document_birth_cert_missing_apostillado_defaults_false_and_missing():
    result = _result(
        classification=ClassificationResult(document_type=DocumentType.BIRTH_CERT),
        field_extraction=FieldExtractionResult(fields={"nombre": "Maria Lopez"}),
    )
    doc = adapt_document(result)
    assert doc.certificado.apostillado is False
    assert "apostillado" in doc.missing_fields


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
