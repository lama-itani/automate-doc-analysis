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


def test_g2_resolves_to_matching_progenitor_despite_unrelated_tagged_cert_name():
    """Two distinct, legitimate parents named on G1's cert must each resolve
    on their own merits, not get bucketed together into one false
    NOMBRE_AMBIGUO just because they don't match each other. An unrelated
    tagged G2 cert (matching neither parent) must not corrupt either slot
    either - it lands in its own disjoint candidate instead."""
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
    # third, disjoint candidate group for G2 (overflow), not a merge with
    # either real parent.
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Jose Torres", fecha_nacimiento="01/01/1960", sexo="M"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g1_cert, g2_cert), CONFIG)
    assert result.g1.person is not None
    # Pedro Ruiz (progenitor1) is the sole candidate in its slot -> resolves
    # cleanly and wins the collapse (only resolved, non-ambiguous candidate).
    assert result.g2.person is not None
    assert result.g2.person.nombre == "Pedro Ruiz"
    assert result.g2.ambiguous is False


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
    # No application/ID exists here, so no G1 person ever resolves to match
    # against -- not because untagged certs are categorically excluded.
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


# --- untagged birth cert resolved by subject-matching an applicant --------


def test_untagged_g1_cert_absorbed_via_subject_match():
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
        generation=None,
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
    )
    result = resolve_lineage(_bundle(app, g1_id, untagged), CONFIG)
    assert result.inferred_generations["cert_untagged"] == "G1"
    assert "cert_untagged" in result.g1.corroborating_document_ids


# --- untagged birth cert resolved by subject-matching a tagged G1's progenitor


def test_untagged_g2_cert_absorbed_via_progenitor_match():
    # Own-subject cert records never carry a doc number, so a resolved parent
    # person needs a DOB (from its own tagged cert here) for name+DOB to hit
    # min_criteria; a g1_cert's progenitor-only record has no DOB to merge on.
    g1_cert = _doc(
        "cert_g1", DocumentType.BIRTH_CERT, generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
    )
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960"),
    )
    untagged_g2 = _doc(
        "cert_untagged_g2",
        DocumentType.BIRTH_CERT,
        generation=None,
        certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960"),
    )
    result = resolve_lineage(_bundle(g1_cert, g2_cert, untagged_g2), CONFIG)
    assert result.inferred_generations["cert_untagged_g2"] == "G2"
    assert "cert_untagged_g2" in result.g2.corroborating_document_ids


# --- cascading inference: untagged G1 unlocks an untagged G2 next pass ----


def test_untagged_g1_cascades_into_g2_progenitor_pool():
    # Pass 1 infers cert_untagged_g1 as G1 via subject match; pass 2 then
    # rebuilds the G2 pool from its now-included progenitor1 fields.
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
    untagged_g1 = _doc(
        "cert_untagged_g1",
        DocumentType.BIRTH_CERT,
        generation=None,
        certificado=CertificadoData(
            nombre="Ana Garcia",
            fecha_nacimiento="01/01/1990",
            progenitor1_nombre="Pedro Ruiz",
            progenitor1_num_doc="P1",
        ),
    )
    result = resolve_lineage(_bundle(app, g1_id, untagged_g1), CONFIG)
    assert result.inferred_generations["cert_untagged_g1"] == "G1"
    assert result.g2.person is not None
    assert result.g2.person.nombre == "Pedro Ruiz"


# --- untagged cert not absorbed when parent generation is ambiguous -------


def test_untagged_cert_not_absorbed_when_parent_ambiguous():
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
        certificado=CertificadoData(nombre="Maria Lopez", fecha_nacimiento="01/01/1950"),
    )
    untagged = _doc(
        "cert_untagged",
        DocumentType.BIRTH_CERT,
        generation=None,
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
    )
    result = resolve_lineage(_bundle(cert_a, cert_b, untagged), CONFIG)
    assert result.g1.ambiguous is True
    assert "cert_untagged" not in result.inferred_generations


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


# --- G2 two-parent fix: per-slot matching -----------------------------------


def test_g2_two_distinct_progenitors_both_resolve_and_both_inferred():
    """Both parents named on G1's cert have their own untagged birth cert
    elsewhere in the bundle. Both must be recognized as G2 for inventory
    purposes (inferred_generations), even though only the better-corroborated
    one becomes the collapsed g2.person."""
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
    # Untagged cert matching progenitor1 by name only.
    pedro_cert = _doc(
        "cert_pedro",
        DocumentType.BIRTH_CERT,
        certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960", sexo="M"),
    )
    # Untagged cert matching progenitor2 by name only.
    maria_cert = _doc(
        "cert_maria",
        DocumentType.BIRTH_CERT,
        certificado=CertificadoData(nombre="Maria Diaz", fecha_nacimiento="01/01/1962", sexo="F"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g1_cert, pedro_cert, maria_cert), CONFIG)
    assert result.g1.person is not None
    assert result.g2.person is not None
    assert result.g2.person.nombre in ("Pedro Ruiz", "Maria Diaz")
    # Core regression assertion: both untagged certs are recognized as G2
    # present for inventory purposes, regardless of which one won the collapse.
    assert result.inferred_generations["cert_pedro"] == "G2"
    assert result.inferred_generations["cert_maria"] == "G2"


def test_g2_slot_match_ignores_missing_dob_and_doc_number():
    """Direct repro of the root-cause bug: a progenitor record (name+doc, no
    DOB) and an untagged cert's own-subject record (name+DOB, no doc number)
    can only ever share one signal (name) -- previously stuck below
    min_criteria=2 forever. The relaxed G2 gate must merge them."""
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
        ),
    )
    pedro_cert = _doc(
        "cert_pedro",
        DocumentType.BIRTH_CERT,
        certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960", sexo="M"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g1_cert, pedro_cert), CONFIG)
    assert result.g2.person is not None
    assert result.g2.person.nombre == "Pedro Ruiz"
    # Merged from both records: DOB came from the untagged cert, doc number
    # from the progenitor field -- proof the two actually merged, not that one
    # record alone happened to carry both.
    assert result.g2.person.fecha_nacimiento == date(1960, 1, 1)
    assert result.g2.person.num_documento == "P1"
    assert set(result.g2.corroborating_document_ids) == {"cert_g1", "cert_pedro"}


def test_g2_tiebreak_prefers_higher_name_similarity():
    """An own-tagged G2 cert whose subject name fuzzy-matches both progenitor
    slots (via whole-token-subset containment) is assigned to the slot with
    the higher graded name similarity, not the first slot by iteration order."""
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
            progenitor1_nombre="Maria Garcia",
            progenitor1_num_doc="P1",
            progenitor2_nombre="Maria Garcia Lopez",
            progenitor2_num_doc="P2",
        ),
    )
    # Subject name is a whole-token subset of both progenitor names, but a
    # closer sequence match to "Maria Garcia Lopez".
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Maria Garcia Lopez", fecha_nacimiento="01/01/1960", sexo="F"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g1_cert, g2_cert), CONFIG)
    assert result.g2.person is not None
    assert result.g2.person.num_documento == "P2"


def test_g2_tiebreak_exact_tie_goes_to_overflow():
    """When a name scores an exact tie against both progenitor slots, the
    record must not silently merge into either real parent's Person."""
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
            progenitor1_nombre="Maria Lopez",
            progenitor1_num_doc="P1",
            progenitor2_nombre="Maria Torres",
            progenitor2_num_doc="P2",
        ),
    )
    # "Maria" alone is a whole-token subset of, and scores identically
    # against, both progenitor names.
    g2_cert = _doc(
        "cert_g2",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Maria", fecha_nacimiento="01/01/1960", sexo="F"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g1_cert, g2_cert), CONFIG)
    # Both real parents remain their own distinct, uncorrupted single-record
    # candidates; the collapse picks one of them (not the tied "Maria" cert).
    assert result.g2.person is not None
    assert result.g2.person.nombre in ("Maria Lopez", "Maria Torres")
    assert result.g2.person.num_documento in ("P1", "P2")


def test_g3_hop_unaffected_by_g2_slot_split():
    """G3's child-pool hop still uses the unmodified _child_pool flat-pool
    behavior -- the G2 slot split must not leak into the G2->G3 hop."""
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
        ),
    )
    pedro_cert = _doc(
        "cert_pedro",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960"),
    )
    # G3's own tagged cert is its sole source here (no bridging progenitor
    # mention on G2's cert) -- this is the same shape as
    # test_full_three_generation_chain_resolves, which already proves G3
    # resolves cleanly in this shape; this test only confirms the G2 slot
    # split above it doesn't disturb that.
    g3_cert = _doc(
        "cert_g3",
        DocumentType.BIRTH_CERT,
        generation="G3",
        certificado=CertificadoData(nombre="Carlos Ruiz", fecha_nacimiento="01/01/1930"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g1_cert, pedro_cert, g3_cert), CONFIG)
    assert result.g2.person is not None
    assert result.g2.person.nombre == "Pedro Ruiz"
    assert result.g3.person is not None
    assert result.g3.person.nombre == "Carlos Ruiz"


# --- Step 4+5: name-only fallback for G1/G3 self-match + its new flag -----


def test_g1_self_match_dob_missing_fallback_resolves_with_flag():
    """App record has no DOB (solicitud never carries one) and a different
    doc number than the ID record -- only name agrees, so the strict 2-of-3
    gate fails (score=1). Since the app side's DOB is absent (not
    conflicting), the fallback must merge them anyway and flag the result."""
    app = _doc(
        "app",
        DocumentType.APPLICATION,
        solicitud=SolicitudData(nombre_solicitante="Ana", apellido_padre="Garcia", numero_id="1"),
    )
    id_doc = _doc(
        "id1",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="99", fecha_nacimiento="01/01/1990"),
    )
    result = resolve_lineage(_bundle(app, id_doc), CONFIG)
    assert result.g1.ambiguous is False
    assert result.g1.person is not None
    assert result.g1.person.fecha_nacimiento == date(1990, 1, 1)
    assert "GENERACION_INFERIDA_BAJA_CONFIANZA" in result.g1.flags


def test_g1_dob_conflict_both_sides_present_no_fallback():
    """Both ID records carry a DOB and they genuinely disagree (outside
    tolerance); doc numbers also differ, so only name agrees (score=1) --
    same shape as the fallback case, except DOB is present (not missing) on
    both sides. The fallback must NOT fire: this stays a real discrepancy,
    not a merge."""
    id_a = _doc(
        "id_a",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="1", fecha_nacimiento="01/01/1990"),
    )
    id_b = _doc(
        "id_b",
        DocumentType.ID_DOCUMENT,
        identidad=IdData(nombres="Ana", apellidos="Garcia", numero_id="2", fecha_nacimiento="01/01/1950"),
    )
    result = resolve_lineage(_bundle(id_a, id_b), CONFIG)
    assert result.g1.ambiguous is True
    assert result.g1.person is None
    assert "GENERACION_INFERIDA_BAJA_CONFIANZA" not in result.g1.flags


def test_g3_self_match_dob_missing_fallback_resolves_with_flag():
    """Two G3-tagged certs sharing a name -- a cert subject record never
    carries a doc number, so these can only ever agree on name; one side's
    DOB is missing, so the fallback must merge them and flag the result."""
    g3_cert_a = _doc(
        "cert_g3_a",
        DocumentType.BIRTH_CERT,
        generation="G3",
        certificado=CertificadoData(nombre="Carlos Ruiz", fecha_nacimiento="01/01/1930"),
    )
    g3_cert_b = _doc(
        "cert_g3_b",
        DocumentType.BIRTH_CERT,
        generation="G3",
        certificado=CertificadoData(nombre="Carlos Ruiz"),  # no DOB
    )
    result = resolve_lineage(_bundle(g3_cert_a, g3_cert_b), CONFIG)
    assert result.g3.ambiguous is False
    assert result.g3.person is not None
    assert result.g3.person.fecha_nacimiento == date(1930, 1, 1)
    assert "GENERACION_INFERIDA_BAJA_CONFIANZA" in result.g3.flags


def test_g2_ambiguous_parent_fallback_path_gets_name_only_fallback():
    """G1 is ambiguous (two disjoint tagged certs), so G2 takes the
    _child_pool + strict fallback branch (not the per-slot path). Two
    G2-tagged certs sharing a name, one missing DOB, must still merge via
    the same fallback and carry the new flag."""
    g1_cert_a = _doc(
        "cert_g1_a",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Ana Garcia", fecha_nacimiento="01/01/1990"),
    )
    g1_cert_b = _doc(
        "cert_g1_b",
        DocumentType.BIRTH_CERT,
        generation="G1",
        certificado=CertificadoData(nombre="Jose Torres", fecha_nacimiento="01/01/1960"),
    )
    g2_cert_a = _doc(
        "cert_g2_a",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Maria Lopez", fecha_nacimiento="01/01/1950"),
    )
    g2_cert_b = _doc(
        "cert_g2_b",
        DocumentType.BIRTH_CERT,
        generation="G2",
        certificado=CertificadoData(nombre="Maria Lopez"),  # no DOB
    )
    result = resolve_lineage(_bundle(g1_cert_a, g1_cert_b, g2_cert_a, g2_cert_b), CONFIG)
    assert result.g1.ambiguous is True
    assert result.g2.ambiguous is False
    assert result.g2.person is not None
    assert result.g2.person.nombre == "Maria Lopez"
    assert "GENERACION_INFERIDA_BAJA_CONFIANZA" in result.g2.flags


def test_g2_slot_pools_unaffected_by_g1g3_fallback():
    """Clean-G1 slot-based G2 resolution never passes fallback_match_fn into
    _resolve_role -- the new flag must never appear there, even though this
    scenario also merges name-only without a DOB on one side (via the
    pre-existing, separately-justified _g2_match_fn)."""
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
        ),
    )
    pedro_cert = _doc(
        "cert_pedro",
        DocumentType.BIRTH_CERT,
        certificado=CertificadoData(nombre="Pedro Ruiz", fecha_nacimiento="01/01/1960", sexo="M"),
    )
    result = resolve_lineage(_bundle(g1_app, g1_id, g1_cert, pedro_cert), CONFIG)
    assert result.g2.person is not None
    assert result.g2.person.nombre == "Pedro Ruiz"
    assert "GENERACION_INFERIDA_BAJA_CONFIANZA" not in result.g2.flags
    assert result.g3.rejected is False
