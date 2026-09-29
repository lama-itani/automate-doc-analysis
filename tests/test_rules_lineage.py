"""Tests for the Lineage Engine's entity resolution + role assignment (M-4, step 6)."""

from __future__ import annotations

from datetime import date

from ps06.classification.classifier import DocumentType
from ps06.rules.lineage import resolve_lineage
from ps06.rules.rules_config import RulesConfig
from ps06.rules.schemas import CanonicalDocument, CaseBundle, CertificadoData, IdData, SolicitudData

CONFIG = RulesConfig()


def _doc(
    document_id: str,
    document_type: DocumentType,
    *,
    generation=None,
    solicitud=None,
    identidad=None,
    certificado=None,
) -> CanonicalDocument:
    return CanonicalDocument(
        case_id="case-1",
        document_id=document_id,
        source_file=f"{document_id}.pdf",
        document_type=document_type,
        generation=generation,
        solicitud=solicitud,
        identidad=identidad,
        certificado=certificado,
    )


def _bundle(*documents: CanonicalDocument) -> CaseBundle:
    return CaseBundle(case_id="case-1", documents=tuple(documents))


# --- zero documents ------------------------------------------------------


def test_zero_documents_all_roles_unresolved():
    result = resolve_lineage(_bundle(), CONFIG)
    assert result.g1.person is None
    assert result.g2.person is None
    assert result.g3.person is None
    assert result.all_flags == frozenset()


# --- application-only -----------------------------------------------------


def test_application_only_resolves_g1_from_single_record():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(
            nombre_solicitante="Ana",
            apellido_padre="Garcia",
            apellido_madre="Lopez",
            numero_id="12345678",
        ),
    )
    result = resolve_lineage(_bundle(app), CONFIG)
    assert result.g1.person is not None
    assert result.g1.person.nombre == "Ana Garcia Lopez"
    assert result.g1.corroborating_document_ids == ("app",)
    assert result.g2.person is None
    assert result.g3.person is None


# --- G1 resolved from application + matching ID (2-of-3 -> corroborated) --


def test_g1_resolved_and_corroborated_from_application_and_id():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="12345678"),
    )
    id_doc = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="12345678", fecha_nacimiento="01/01/1990"),
    )
    result = resolve_lineage(_bundle(app, id_doc), CONFIG)
    assert result.g1.person is not None
    assert result.g1.person.fecha_nacimiento == date(1990, 1, 1)
    assert set(result.g1.corroborating_document_ids) == {"app", "id1"}
    assert result.g1.flags == ()


# --- ambiguous G2: two disjoint candidate groups ---------------------------


def test_ambiguous_g2_when_two_disjoint_groups():
    g1_app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="1"),
    )
    g1_id = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="1", fecha_nacimiento="01/01/1990"),
    )
    g1_cert = _doc(
        "cert_g1",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(
            nombre="Ana Garcia",
            fecha_nacimiento="01/01/1990",
            progenitor1_nombre="Pedro Ruiz",
            progenitor1_num_doc="P1",
            progenitor2_nombre="Maria Diaz",
            progenitor2_num_doc="P2",
        ),
    )
    # A G2-tagged cert whose subject matches neither progenitor above -> a
    # second, disjoint candidate group for G2.
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Jose Torres", fecha_nacimiento="01/01/1960", sexo="M"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g1_cert, g2_cert), CONFIG)
    assert result.g1.person is not None
    assert result.g2.person is None
    assert result.g2.ambiguous is True
    assert "NOMBRE_AMBIGUO" in result.g2.flags


# --- duplicate same-generation certs: agreeing -> stronger corroboration --


def test_duplicate_agreeing_certs_strengthen_corroboration():
    cert_a = _doc(
        "cert_a",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
    )
    cert_b = _doc(
        "cert_b",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
    )
    result = resolve_lineage(_bundle(cert_a, cert_b), CONFIG)
    assert result.g1.person is not None
    assert set(result.g1.corroborating_document_ids) == {"cert_a", "cert_b"}
    assert result.g1.flags == ()


# --- duplicate same-generation certs: disagreeing -> conflict flag, no crash --


def test_duplicate_disagreeing_certs_raise_conflict_not_crash():
    cert_a = _doc(
        "cert_a",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990", nacionalidad=None),
    )
    cert_b = _doc(
        "cert_b",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="05/05/1990", nacionalidad=None),
    )
    result = resolve_lineage(_bundle(cert_a, cert_b), CONFIG)
    assert result.g1.person is not None
    assert "FECHA_CONFLICTO" in result.g1.flags


# --- birth cert with no generation tag: inert ------------------------------


def test_untagged_birth_cert_contributes_nothing():
    untagged = _doc(
        "cert_untagged",
        DocumentType.BIRTH_CERT,
        generation=None,
        certificado=CertificadoData(nombre="Someone Else", fecha_nacimiento="01/01/1900"),
    )
    result = resolve_lineage(_bundle(untagged), CONFIG)
    assert result.g1.person is None
    assert result.g2.person is None
    assert result.g3.person is None


# --- G2 resolved from a single progenitor-named record, no matching doc ---


def test_g2_resolved_from_single_progenitor_record_insufficient_corroboration():
    g1_cert = _doc(
        "cert_g1",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(
            nombre="Ana Garcia",
            fecha_nacimiento="01/01/1990",
            progenitor1_nombre="Pedro Ruiz",
            progenitor1_num_doc="P1",
        ),
    )
    result = resolve_lineage(_bundle(g1_cert), CONFIG)
    assert result.g2.person is not None
    assert result.g2.person.nombre == "Pedro Ruiz"
    # Only one source document named this person -> not corroborated.
    assert len(result.g2.corroborating_document_ids) == 1


# --- brecha_generacional: None-DOB guard (no flag, no rejection) ----------


def test_generational_gap_check_skipped_when_child_dob_missing():
    g1_app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="1"),
    )
    g1_id = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="1", fecha_nacimiento="01/01/1990"),
    )
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Pedro Ruiz", sexo="M"),  # no DOB
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g2_cert), CONFIG)
    assert result.g1.person.fecha_nacimiento == date(1990, 1, 1)
    assert result.g2.person is not None
    assert result.g2.person.fecha_nacimiento is None
    assert result.g2.rejected is False
    assert "BRECHA_GENERACIONAL" not in result.g2.flags


# --- brecha_generacional: real rejection -----------------------------------


def test_generational_gap_check_rejects_outside_hard_band():
    g1_app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="1"),
    )
    g1_id = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="1", fecha_nacimiento="01/01/1990"),
    )
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1985", sexo="M"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g2_cert), CONFIG)
    assert result.g2.person is not None
    assert result.g2.rejected is True
    assert "BRECHA_GENERACIONAL" in result.g2.flags


# --- ambiguous G1 blocks progenitor-derived G2 pool -------------------------


def test_ambiguous_g1_means_g2_pool_only_from_own_tagged_cert():
    g1_cert_a = _doc(
        "cert_g1_a",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990", progenitor1_nombre="Pedro Ruiz"),
    )
    g1_cert_b = _doc(
        "cert_g1_b",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Jose Torres", fecha_nacimiento="01/01/1960", progenitor1_nombre="Carlos Diaz"),
    )
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Maria Lopez", fecha_nacimiento="01/01/1950", sexo="F"),
    )
    result = resolve_lineage(_bundle(g1_cert_a, g1_cert_b, g2_cert), CONFIG)
    assert result.g1.ambiguous is True
    # G2 must resolve only from its own tagged cert (Maria Lopez), not the
    # ambiguous G1's progenitor names (Pedro Ruiz / Carlos Diaz).
    assert result.g2.person is not None
    assert result.g2.person.nombre == "Maria Lopez"


# --- cross-role DOCUMENTO_CRUZADO ------------------------------------------


def test_documento_cruzado_when_doc_number_shared_by_two_names():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="SHARED123"),
    )
    g1_id = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(
            nombres="Ana", apellidos="Garcia", numero_id="SHARED123", fecha_nacimiento="01/01/1990"
        ),
    )
    g1_cert = _doc(
        "cert_g1",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(
            nombre="Ana Garcia",
            fecha_nacimiento="01/01/1990",
            progenitor1_nombre="Someone Different",
            progenitor1_num_doc="SHARED123",
        ),
    )
    result = resolve_lineage(_bundle(app, g1_id, g1_cert), CONFIG)
    assert result.g1.person is not None
    assert "DOCUMENTO_CRUZADO" in result.cross_case_flags
    assert "DOCUMENTO_CRUZADO" not in result.g1.flags
    assert "DOCUMENTO_CRUZADO" not in result.g2.flags


# --- full G1->G2->G3 chain resolves cleanly to VERDE-eligible shape --------


def test_full_three_generation_chain_resolves():
    # Each generation's own G-tagged cert is its sole source (no ID documents
    # exist yet to bridge a progenitor mention with a same-person tagged
    # cert on two criteria) -- this is today's realistic shape given the
    # adapter's deferred identidad/certificado extraction.
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="1"),
    )
    g1_id = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="1", fecha_nacimiento="01/01/1990"),
    )
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960"),
    )
    g3_cert = _doc(
        "cert_g3",
        DocumentType.BIRTH_CERT,
        generation="G3",
        certificado=CertificadoData(nombre="Carlos Ruiz", fecha_nacimiento="01/01/1930"),
    )
    result = resolve_lineage(_bundle(app, g1_id, g2_cert, g3_cert), CONFIG)
    assert result.g1.person.nombre == "Ana Garcia"
    assert result.g2.person.nombre == "Pedro Ruiz"
    assert result.g3.person.nombre == "Carlos Ruiz"
    assert result.g1.rejected is False
    assert result.g2.rejected is False
    assert result.g3.rejected is False
    assert result.all_flags == frozenset()
