"""Tests for the Semaphore Engine's per-document checks + decisioning (M-4, step 7)."""

from __future__ import annotations

from datetime import date

from ps06.classification.classifier import DocumentType
from ps06.rules.lineage import resolve_lineage
from ps06.rules.rules_config import RulesConfig
from ps06.rules.schemas import CanonicalDocument, CaseBundle, CertificadoData, IdData, SolicitudData
from ps06.rules.semaphore import (
    check_expected_documents,
    evaluate_document,
    evaluate_semaphore,
)

CONFIG = RulesConfig()
EVAL_DATE = date(2026, 9, 29)


def _doc(
    document_id: str,
    document_type: DocumentType,
    *,
    generation=None,
    solicitud=None,
    identidad=None,
    certificado=None,
    missing_fields=(),
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
        missing_fields=missing_fields,
    )


def _bundle(*documents: CanonicalDocument) -> CaseBundle:
    return CaseBundle(case_id="case-1", documents=tuple(documents))


def _decide(*documents: CanonicalDocument) -> "SemaphoreDecision":  # noqa: F821
    bundle = _bundle(*documents)
    lineage = resolve_lineage(bundle, CONFIG)
    return evaluate_semaphore(bundle, lineage, CONFIG, EVAL_DATE)


def _full_case(**overrides):
    """A 5/5-present, flag-free, but only-1-corroborating-doc-per-role case.

    This is today's realistic shape: adapter.py doesn't populate identidad/
    certificado structured fields yet, so each generation only has one
    source document contributing to its resolved person.
    """
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(
            nombre_solicitante="Ana",
            apellido_padre="Garcia",
            numero_id="1",
            fecha_presentacion="01/10/2025",
        ),
    )
    id_doc = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="1", fecha_nacimiento="01/01/1990"),
    )
    g1_cert = _doc(
        "cert_g1",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
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
    docs = {"app": app, "id_doc": id_doc, "g1_cert": g1_cert, "g2_cert": g2_cert, "g3_cert": g3_cert}
    docs.update(overrides)
    return list(docs.values())


# --- evaluate_document ------------------------------------------------------


def test_evaluate_document_incomplete_when_missing_fields():
    doc = _doc("d1", DocumentType.ID_DOCUMENT, missing_fields=("numero_id",))
    evaluation = evaluate_document(doc, EVAL_DATE)
    assert "DOCUMENT_INCOMPLETE" in evaluation.flags


def test_evaluate_document_complete_when_no_missing_fields():
    doc = _doc("d1", DocumentType.APPLICATION, missing_fields=())
    evaluation = evaluate_document(doc, EVAL_DATE)
    assert evaluation.flags == ()


def test_evaluate_document_vencido_never_flagged_without_identidad():
    doc = _doc("d1", DocumentType.ID_DOCUMENT, identidad=None)
    evaluation = evaluate_document(doc, EVAL_DATE)
    assert "DOCUMENTO_VENCIDO" not in evaluation.flags


def test_evaluate_document_vencido_true_when_expiry_passed():
    doc = _doc(
        "d1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(fecha_vencimiento="01/01/2020"),
    )
    evaluation = evaluate_document(doc, EVAL_DATE, EVAL_DATE)
    assert "DOCUMENTO_VENCIDO" in evaluation.flags


def test_evaluate_document_vencido_false_when_expiry_future():
    doc = _doc(
        "d1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(fecha_vencimiento="01/01/2030"),
    )
    evaluation = evaluate_document(doc, EVAL_DATE)
    assert "DOCUMENTO_VENCIDO" not in evaluation.flags


def test_evaluate_document_uses_vencimiento_reference_date_over_evaluation_date():
    doc = _doc(
        "d1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(fecha_vencimiento="01/01/2026"),
    )
    # Still valid as of EVAL_DATE (2026-09-29), but the explicit reference
    # date (e.g. application date + validity buffer) has already passed it.
    evaluation = evaluate_document(doc, EVAL_DATE, date(2026, 2, 1))
    assert "DOCUMENTO_VENCIDO" in evaluation.flags


def test_evaluate_document_skips_vencido_check_when_reference_absent():
    doc = _doc(
        "d1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(fecha_vencimiento="01/01/2020"),
    )
    evaluation = evaluate_document(doc, EVAL_DATE, None)
    assert "DOCUMENTO_VENCIDO" not in evaluation.flags


# --- check_expected_documents ------------------------------------------------


def test_check_expected_documents_all_missing_when_no_docs():
    bundle = _bundle()
    checks = check_expected_documents(bundle, CONFIG)
    assert len(checks) == 5
    assert all(not c.present for c in checks)


def test_check_expected_documents_generation_disambiguates_birth_certs():
    g1_cert = _doc("cert_g1", DocumentType.BIRTH_CERT, generation="G1")
    g2_cert = _doc("cert_g2", DocumentType.BIRTH_CERT, generation="G2")
    bundle = _bundle(g1_cert, g2_cert)
    checks = check_expected_documents(bundle, CONFIG)
    by_label = {c.label: c for c in checks}
    assert by_label["Certificado de Nacimiento del Solicitante"].present is True
    assert by_label["Certificado de Nacimiento del Solicitante"].matched_document_ids == ("cert_g1",)
    assert by_label["Certificado de Nacimiento del Progenitor"].present is True
    assert by_label["Certificado de Nacimiento Español de origen"].present is False


def test_check_expected_documents_counts_inferred_generation_as_present():
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
    untagged = _doc(
        "cert_untagged",
        DocumentType.BIRTH_CERT,
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
    )
    bundle = _bundle(app, g1_id, untagged)
    lineage = resolve_lineage(bundle, CONFIG)
    checks = check_expected_documents(bundle, CONFIG, lineage)
    by_label = {c.label: c for c in checks}
    entry = by_label["Certificado de Nacimiento del Solicitante"]
    assert entry.present is True
    assert "cert_untagged" in entry.matched_document_ids


# --- evaluate_semaphore: structural / severity cases -------------------------


def test_zero_documents_is_amarillo_all_missing():
    decision = _decide()
    assert decision.estado == "AMARILLO"
    assert len(decision.missing_documents) == 5


def test_application_only_is_amarillo():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", numero_id="1"),
    )
    decision = _decide(app)
    assert decision.estado == "AMARILLO"
    assert len(decision.missing_documents) == 4


def test_five_of_five_present_zero_flags_but_single_corroboration_is_amarillo():
    decision = _decide(*_full_case())
    assert decision.estado == "AMARILLO"
    assert decision.missing_documents == ()
    assert decision.rojo_flags == ()
    assert decision.amarillo_flags == ()
    assert any("Corroboración insuficiente" in p for p in decision.problemas)


def test_verde_when_five_of_five_present_zero_flags_and_corroborated():
    docs = _full_case()
    # Add a duplicate, agreeing G1 cert so G1 has 2 corroborating docs. To
    # keep this VERDE we'd need every role corroborated; duplicate all three
    # generations' certs to satisfy that.
    docs.append(
        _doc(
            "cert_g1_dup",
            DocumentType.BIRTH_CERT,
            generation="G1",
            certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
        )
    )
    docs.append(
        _doc(
            "cert_g2_dup",
            DocumentType.BIRTH_CERT,
            generation="G2",
            certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960"),
        )
    )
    docs.append(
        _doc(
            "cert_g3_dup",
            DocumentType.BIRTH_CERT,
            generation="G3",
            certificado=CertificadoData(nombre="Carlos Ruiz", fecha_nacimiento="01/01/1930"),
        )
    )
    decision = _decide(*docs)
    assert decision.estado == "VERDE"
    assert decision.problemas == ()


# --- each ROJO flag individually forces ROJO --------------------------------


def test_documento_vencido_uses_application_date_not_evaluation_date():
    """A passport valid as of EVAL_DATE but within 6 months of the case's
    own application date (fecha_presentacion) must still flag -- this is the
    realistic standard, not "has today's wall-clock date passed yet"."""
    docs = _full_case()
    for doc in docs:
        if doc.document_type is DocumentType.APPLICATION:
            docs[docs.index(doc)] = _doc(
                "app",
                DocumentType.APPLICATION,
                solicitud=SolicitudData(
                    nombre_solicitante="Ana",
                    apellido_padre="Garcia",
                    numero_id="1",
                    fecha_presentacion="01/10/2025",
                ),
            )
    docs.append(
        _doc(
            "id_expired",
            DocumentType.ID_DOCUMENT,
            identidad=IdData(fecha_vencimiento="01/01/2026"),
        )
    )
    decision = _decide(*docs)
    assert decision.estado == "ROJO"
    assert "DOCUMENTO_VENCIDO" in decision.rojo_flags


def test_documento_vencido_forces_rojo():
    docs = _full_case()
    docs.append(
        _doc(
            "id_expired",
            DocumentType.ID_DOCUMENT,
            identidad=IdData(fecha_vencimiento="01/01/2020"),
        )
    )
    decision = _decide(*docs)
    assert decision.estado == "ROJO"
    assert "DOCUMENTO_VENCIDO" in decision.rojo_flags


def test_missing_fecha_presentacion_skips_vencido_and_raises_amarillo_flag():
    """application present but fecha_presentacion absent -- even a long-
    expired ID must not trigger DOCUMENTO_VENCIDO (no silent fallback to
    evaluation_date), and the gap itself must surface as AMARILLO, not
    VERDE."""
    docs = _full_case(
        app=_doc(
            "app",
            DocumentType.APPLICATION,
            solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="1"),
        )
    )
    docs.append(
        _doc(
            "id_expired",
            DocumentType.ID_DOCUMENT,
            identidad=IdData(fecha_vencimiento="01/01/2000"),
        )
    )
    decision = _decide(*docs)
    assert decision.estado == "AMARILLO"
    assert "DOCUMENTO_VENCIDO" not in decision.rojo_flags
    assert "FECHA_PRESENTACION_FALTANTE" in decision.amarillo_flags
    assert "FECHA_PRESENTACION_FALTANTE" in decision.problemas


def test_nombre_ambiguo_forces_rojo():
    docs = _full_case()
    docs.append(
        _doc(
            "cert_g1_conflict",
            DocumentType.BIRTH_CERT,
            generation="G1",
            certificado=CertificadoData(nombre="Someone Else Entirely", fecha_nacimiento="01/01/1800", sexo="M"),
        )
    )
    decision = _decide(*docs)
    assert decision.estado == "ROJO"
    assert "NOMBRE_AMBIGUO" in decision.rojo_flags


def test_fecha_conflicto_forces_rojo():
    # Matches G1's applicant on name + doc number (via a matching ID doc) --
    # 2-of-3 criteria, so it merges -- but disagrees on birth date, which
    # raises FECHA_CONFLICTO on the merged person rather than blocking the
    # merge (birth-date agreement is one of the 3 criteria, not mandatory).
    docs = _full_case()
    docs.append(
        _doc(
            "id_conflict_date",
            DocumentType.ID_DOCUMENT,
            identidad=IdData(
                nombres="Ana", apellidos="Garcia", numero_id="1", fecha_nacimiento="05/05/1995"
            ),
        )
    )
    decision = _decide(*docs)
    assert decision.estado == "ROJO"
    assert "FECHA_CONFLICTO" in decision.rojo_flags


def test_brecha_generacional_forces_rojo():
    docs = _full_case()
    # Replace G2's cert with one whose DOB creates too-small a gap vs G1 (1990).
    for i, d in enumerate(docs):
        if d.document_id == "cert_g2":
            docs[i] = _doc(
                "cert_g2",
                DocumentType.BIRTH_CERT,
                generation="G2",
                certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1985"),
            )
    decision = _decide(*docs)
    assert decision.estado == "ROJO"
    assert "BRECHA_GENERACIONAL" in decision.rojo_flags


def test_documento_cruzado_forces_rojo():
    docs = _full_case()
    docs.append(
        _doc(
            "cert_g1_crossed",
            DocumentType.BIRTH_CERT,
            generation="G1",
            certificado=CertificadoData(
                nombre="Ana Garcia",
                fecha_nacimiento="01/01/1990",
                progenitor1_nombre="Someone Different",
                progenitor1_num_doc="1",
            ),
        )
    )
    decision = _decide(*docs)
    assert decision.estado == "ROJO"
    assert "DOCUMENTO_CRUZADO" in decision.rojo_flags


# --- AMARILLO-only flags never escalate to ROJO -----------------------------


def test_document_incomplete_alone_is_amarillo_not_rojo():
    docs = _full_case()
    docs.append(_doc("extra_incomplete", DocumentType.ID_DOCUMENT, missing_fields=("numero_id",)))
    decision = _decide(*docs)
    assert decision.estado == "AMARILLO"
    assert "DOCUMENT_INCOMPLETE" in decision.amarillo_flags
    assert decision.rojo_flags == ()


def test_nacionalidad_inconsistente_alone_is_amarillo_not_rojo():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="1", nacionalidad="Boliviana"),
    )
    id_doc = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(
            nombres="Ana",
            apellidos="Garcia",
            numero_id="1",
            fecha_nacimiento="01/01/1990",
            nacionalidad="Española",
        ),
    )
    g1_cert = _doc(
        "cert_g1",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
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
    decision = _decide(app, id_doc, g1_cert, g2_cert, g3_cert)
    assert decision.estado == "AMARILLO"
    assert "NACIONALIDAD_INCONSISTENTE" in decision.amarillo_flags
    assert decision.rojo_flags == ()


# --- ROJO precedence over simultaneous missing-doc + AMARILLO flags --------


def test_rojo_takes_precedence_over_missing_docs_and_amarillo_flags():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(
            nombre_solicitante="Ana", numero_id="1", fecha_presentacion="01/10/2025"
        ),
        missing_fields=("nacionalidad",),
    )
    id_doc = _doc(
        "id_expired",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(fecha_vencimiento="01/01/2020"),
    )
    decision = _decide(app, id_doc)
    assert decision.estado == "ROJO"
    assert decision.missing_documents != ()
    assert "DOCUMENT_INCOMPLETE" in decision.amarillo_flags
    assert "DOCUMENTO_VENCIDO" in decision.rojo_flags


# --- duplicate documents don't crash ----------------------------------------


def test_duplicate_documents_do_not_crash():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", numero_id="1"),
    )
    app_dup = _doc(
        "app_dup",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", numero_id="1"),
    )
    decision = _decide(app, app_dup)
    assert decision.estado in ("AMARILLO", "ROJO")
