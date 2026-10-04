"""Tests for the report-snapshot assembly layer (M-4, step 8)."""

from __future__ import annotations

from datetime import date

from ps06.classification.classifier import DocumentType
from ps06.rules.lineage import resolve_lineage
from ps06.rules.rules_config import RulesConfig
from ps06.rules.schemas import CanonicalDocument, CaseBundle, CertificadoData, IdData, SolicitudData
from ps06.rules.semaphore import evaluate_semaphore
from ps06.rules.snapshot import build_snapshot

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


def _snapshot(*documents: CanonicalDocument):
    bundle = _bundle(*documents)
    lineage = resolve_lineage(bundle, CONFIG)
    decision = evaluate_semaphore(bundle, lineage, CONFIG, EVAL_DATE)
    return build_snapshot(decision, bundle), decision


def _full_case_corroborated():
    """5/5 present, zero flags, every role corroborated by >=2 documents -> VERDE."""
    docs = [
        _doc(
            "app",
            DocumentType.APPLICATION,
            solicitud=SolicitudData(
                nombre_solicitante="Ana",
                apellido_padre="Garcia",
                numero_id="1",
                fecha_presentacion="01/10/2025",
            ),
        ),
        _doc(
            "id1",
            DocumentType.ID_DOCUMENT,
            identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="1", fecha_nacimiento="01/01/1990"),
        ),
        _doc(
            "cert_g1",
            DocumentType.BIRTH_CERT,
            generation="G1",
            certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
        ),
        _doc(
            "cert_g1_dup",
            DocumentType.BIRTH_CERT,
            generation="G1",
            certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
        ),
        _doc(
            "cert_g2",
            DocumentType.BIRTH_CERT,
            generation="G2",
            certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960"),
        ),
        _doc(
            "cert_g2_dup",
            DocumentType.BIRTH_CERT,
            generation="G2",
            certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960"),
        ),
        _doc(
            "cert_g3",
            DocumentType.BIRTH_CERT,
            generation="G3",
            certificado=CertificadoData(nombre="Carlos Ruiz", fecha_nacimiento="01/01/1930"),
        ),
        _doc(
            "cert_g3_dup",
            DocumentType.BIRTH_CERT,
            generation="G3",
            certificado=CertificadoData(nombre="Carlos Ruiz", fecha_nacimiento="01/01/1930"),
        ),
    ]
    return docs


# --- estado / justificacion / problemas passthrough -------------------------


def test_verde_snapshot_has_no_problemas_placeholder():
    snapshot, decision = _snapshot(*_full_case_corroborated())
    assert snapshot.estado == "VERDE" == decision.estado
    assert snapshot.justificacion == decision.justificacion
    assert snapshot.problemas == ("Ninguno",)


def test_amarillo_snapshot_passes_through_problemas():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", numero_id="1"),
    )
    snapshot, decision = _snapshot(app)
    assert snapshot.estado == "AMARILLO" == decision.estado
    assert snapshot.problemas == decision.problemas
    assert snapshot.problemas != ()


def test_rojo_snapshot_passes_through_problemas():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(
            nombre_solicitante="Ana", numero_id="1", fecha_presentacion="01/10/2025"
        ),
    )
    id_doc = _doc(
        "id_expired",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(fecha_vencimiento="01/01/2020"),
    )
    snapshot, decision = _snapshot(app, id_doc)
    assert snapshot.estado == "ROJO" == decision.estado
    assert snapshot.problemas == decision.problemas


# --- S1 rows -----------------------------------------------------------------


def test_s1_rows_cover_all_five_expected_documents():
    snapshot, _ = _snapshot()
    assert len(snapshot.s1_rows) == 5
    assert all(row.estado == "Faltante" for row in snapshot.s1_rows)


def test_s1_rows_mark_present_documents():
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", numero_id="1"),
    )
    snapshot, _ = _snapshot(app)
    by_tipo = {row.tipo: row for row in snapshot.s1_rows}
    assert by_tipo["Solicitud Principal"].estado == "Presente"
    assert by_tipo["Identificación del Solicitante"].estado == "Faltante"


# --- S2 rows -----------------------------------------------------------------


def test_s2_rows_carry_missing_fields_and_conflicts():
    doc = _doc("d1", DocumentType.ID_DOCUMENT, missing_fields=("numero_id",))
    snapshot, decision = _snapshot(doc)
    assert len(snapshot.s2_rows) == 1
    row = snapshot.s2_rows[0]
    assert row.documento == "d1"
    assert row.campos_faltantes == ("numero_id",)
    assert "DOCUMENT_INCOMPLETE" in row.conflictos
    assert row.evaluacion_credencial == "Con observaciones"
    assert row.consistencia == "Con conflictos"


def test_s2_rows_clean_document_has_no_observaciones():
    doc = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", numero_id="1"),
    )
    snapshot, _ = _snapshot(doc)
    row = next(r for r in snapshot.s2_rows if r.documento == "app")
    assert row.campos_faltantes == ()
    assert row.conflictos == ()
    assert row.evaluacion_credencial == "Sin observaciones"
    assert row.consistencia == "Consistente"


# --- S3 rows -----------------------------------------------------------------


def test_s3_rows_cover_all_three_generations_unresolved():
    snapshot, _ = _snapshot()
    assert len(snapshot.s3_rows) == 3
    by_gen = {row.generacion: row for row in snapshot.s3_rows}
    assert by_gen["G1"].nombre_completo == "No determinado"
    assert by_gen["G1"].relacion == "Solicitante"
    assert by_gen["G2"].relacion == "Progenitor"
    assert by_gen["G3"].relacion == "Abuelo/a"
    assert snapshot.s3_footnote is None


def test_s3_rows_resolved_person_has_name_and_dob():
    docs = _full_case_corroborated()
    snapshot, _ = _snapshot(*docs)
    by_gen = {row.generacion: row for row in snapshot.s3_rows}
    assert by_gen["G1"].nombre_completo == "Ana Garcia"
    assert "1990-01-01" in by_gen["G1"].fecha_lugar_nacimiento


def test_s3_footnote_set_when_role_ambiguous():
    docs = _full_case_corroborated()
    docs.append(
        _doc(
            "cert_g1_conflict",
            DocumentType.BIRTH_CERT,
            generation="G1",
            certificado=CertificadoData(nombre="Someone Else Entirely", fecha_nacimiento="01/01/1800", sexo="M"),
        )
    )
    snapshot, decision = _snapshot(*docs)
    assert decision.lineage.g1.ambiguous is True
    assert snapshot.s3_footnote is not None
    assert "G1" in snapshot.s3_footnote


def test_s3_footnote_set_when_role_rejected_for_generational_gap():
    docs = _full_case_corroborated()
    for i, d in enumerate(docs):
        if d.document_id in ("cert_g2", "cert_g2_dup"):
            docs[i] = _doc(
                d.document_id,
                DocumentType.BIRTH_CERT,
                generation="G2",
                certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1985"),
            )
    snapshot, decision = _snapshot(*docs)
    assert decision.lineage.g2.rejected is True
    assert snapshot.s3_footnote is not None
    assert "G2" in snapshot.s3_footnote
