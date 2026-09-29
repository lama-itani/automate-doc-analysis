"""Tests for the canonical rules schemas (M-4, step 1)."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from ps06.classification.classifier import DocumentType
from ps06.rules.schemas import (
    CERTIFICADO_SOURCE_LABELS,
    GENERATION_BY_ROLE,
    ID_SOURCE_LABELS,
    ROLE_BY_GENERATION,
    SOLICITUD_SOURCE_LABELS,
    CanonicalDocument,
    CaseBundle,
    CertificadoData,
    IdData,
    Person,
    Role,
    SolicitudData,
)


# --- Document data models ---------------------------------------------------


def test_solicitud_defaults_all_none_except_firma():
    s = SolicitudData()
    assert s.nombre_solicitante is None
    assert s.firma is False


def test_solicitud_is_frozen():
    s = SolicitudData(nombre_solicitante="Ana")
    with pytest.raises(ValidationError):
        s.nombre_solicitante = "Otro"  # type: ignore[misc]


def test_id_and_certificado_construct_with_partial_fields():
    idd = IdData(numero_id="X1234567", apellidos="García", fecha_vencimiento="2020-01-01")
    assert idd.numero_id == "X1234567"
    assert idd.nombres is None
    cert = CertificadoData(nombre="Ana García", progenitor1_nombre="Luis García")
    assert cert.progenitor1_nombre == "Luis García"
    assert cert.apostillado is False
    assert cert.grado_certificado is None


def test_certificado_generation_literal_rejects_bad_value():
    CertificadoData(grado_certificado="G2")  # valid
    with pytest.raises(ValidationError):
        CertificadoData(grado_certificado="G9")  # type: ignore[arg-type]


# --- Spanish source-label maps (coherency) ----------------------------------


def test_source_label_maps_cover_every_field_verbatim():
    # Every canonical field (except the pydantic config) has a Spanish label,
    # and the labels preserve the source's accents/spacing verbatim.
    assert set(SOLICITUD_SOURCE_LABELS) == set(SolicitudData().model_dump())
    assert set(ID_SOURCE_LABELS) == set(IdData().model_dump())
    assert set(CERTIFICADO_SOURCE_LABELS) == set(CertificadoData().model_dump())
    assert SOLICITUD_SOURCE_LABELS["pais"] == "País"
    assert ID_SOURCE_LABELS["numero_id"] == "Número de ID"
    assert ID_SOURCE_LABELS["fecha_vencimiento"] == "Fecha de vencimiento"
    assert CERTIFICADO_SOURCE_LABELS["progenitor1_num_doc"] == "Nº Documento progenitor 1 G_X"


# --- Roles / generations ----------------------------------------------------


def test_role_generation_bijection_matches_source():
    assert ROLE_BY_GENERATION["G1"] is Role.SOLICITANTE
    assert ROLE_BY_GENERATION["G2"] is Role.PROGENITOR
    assert ROLE_BY_GENERATION["G3"] is Role.ABUELO
    # round-trips
    for gen, role in ROLE_BY_GENERATION.items():
        assert GENERATION_BY_ROLE[role] == gen
    assert Role.SOLICITANTE.value == "Solicitante"
    assert Role.ABUELO.value == "Abuelo/a"


# --- Person -----------------------------------------------------------------


def test_person_carries_parsed_date_and_audit_ids():
    p = Person(
        nombre="Ana García",
        fecha_nacimiento=date(1990, 5, 1),
        num_documento="X1234567",
        role=Role.SOLICITANTE,
        generation="G1",
        source_document_ids=("d1", "d2"),
    )
    assert p.fecha_nacimiento == date(1990, 5, 1)
    assert p.source_document_ids == ("d1", "d2")


# --- CanonicalDocument / CaseBundle -----------------------------------------


def _doc(doc_id: str, dtype: DocumentType, **kw) -> CanonicalDocument:
    return CanonicalDocument(
        case_id="c1", document_id=doc_id, source_file=f"{doc_id}.pdf",
        document_type=dtype, **kw,
    )


def test_canonical_document_missing_fields_seam_defaults_empty():
    doc = _doc("d1", DocumentType.BIRTH_CERT)
    assert doc.missing_fields == ()
    doc2 = _doc("d2", DocumentType.BIRTH_CERT, missing_fields=("nombre", "fecha_nacimiento"))
    assert "nombre" in doc2.missing_fields


def test_case_bundle_accessors_group_by_type():
    bundle = CaseBundle(
        case_id="c1",
        documents=(
            _doc("app", DocumentType.APPLICATION, solicitud=SolicitudData(nombre_solicitante="Ana")),
            _doc("bc1", DocumentType.BIRTH_CERT, certificado=CertificadoData(nombre="Ana")),
            _doc("bc2", DocumentType.BIRTH_CERT, certificado=CertificadoData(nombre="Luis")),
            _doc("id1", DocumentType.ID_DOCUMENT, identidad=IdData(numero_id="X1")),
            _doc("other", DocumentType.OTHER),
        ),
    )
    assert bundle.application.document_id == "app"
    assert {d.document_id for d in bundle.birth_certificates} == {"bc1", "bc2"}
    assert {d.document_id for d in bundle.identity_documents} == {"id1"}
    assert bundle.by_type(DocumentType.EMPTY) == ()


def test_case_bundle_application_none_when_absent():
    bundle = CaseBundle(case_id="c1", documents=(_doc("bc1", DocumentType.BIRTH_CERT),))
    assert bundle.application is None
