"""Adapt persisted ``OcrResult``s into the rules engine's canonical shapes (M-4, step 4).

This is the layer that does I/O: it reads a case's documents via
:func:`ps06.ocr.envelope.get_extraction` and composes :mod:`ps06.rules.schemas`
shapes + :mod:`ps06.rules.normalize` functions. It never fabricates data — per
the engine-first/fixture-driven scoping decision, the store today only holds
raw ``extracted_text`` plus whatever ``acroform_fields``/``xfa_fields`` a
fillable PDF form happened to expose, not full structured fields for ID
documents or birth certificates. Anything the engine will eventually need but
cannot be populated today is recorded in ``CanonicalDocument.missing_fields``
— never a silent VERDE on missing structured data (that degrade-to-AMARILLO
happens in a later step; this module's job is only to be honest about gaps).
"""

from __future__ import annotations

from typing import Literal, Optional

from ps06.classification.classifier import DocumentType
from ps06.ocr.envelope import OcrResult, get_extraction
from ps06.rules.normalize import normalize_extracted_text
from ps06.rules.schemas import (
    SOLICITUD_SOURCE_LABELS,
    CanonicalDocument,
    CaseBundle,
    CertificadoData,
    IdData,
    SolicitudData,
)
from ps06.status.store import StatusStore


class MissingExtractionError(Exception):
    """A case's registered document has no persisted ``OcrResult`` yet.

    Raised by :func:`build_case_bundle` rather than silently skipping the
    document: this function is only meant to run once every document in the
    case has reached ``OCR_DONE`` (at which point ``ocr/job.py`` guarantees a
    matching ``document_extraction`` row), so a missing row here is a real
    data-integrity bug, not a normal degrade-to-AMARILLO case.
    """


def _normalize_label(label: str) -> str:
    """Case/whitespace-insensitive key for matching form-field names to labels."""
    return " ".join(label.strip().lower().split())


#: Normalized SOLICITUD label -> canonical attribute name, built once at import.
_SOLICITUD_LABEL_INDEX: dict[str, str] = {
    _normalize_label(label): attr for attr, label in SOLICITUD_SOURCE_LABELS.items()
}

AnexoVariant = Literal["ANEXO_I", "ANEXO_III_IV"]

#: Anexo III and Anexo IV AcroForm templates name their fields identically
#: (``Textfield``/``Textfield-0``..``Textfield-13``) for the same 14 printed
#: slots, confirmed by inspecting both blank templates with pymupdf. Anexo IV's
#: template has a known defect: the "Registro Civil de" destination widget is
#: literally named ``Textfield`` too (same as ``nombre_solicitante``'s), so PDF
#: viewers/writers give them the same value — that field is unrecoverable from
#: the form and is intentionally left unmapped here (not guessed).
_ANEXO_III_IV_FIELD_MAP: dict[str, str] = {
    "Textfield": "nombre_solicitante",
    "Textfield-0": "apellido_padre",
    "Textfield-1": "apellido_madre",
    "Textfield-2": "nacionalidad",
    "Textfield-3": "estado_civil",
    "Textfield-4": "numero_id",
    "Textfield-5": "domicilio",
    "Textfield-6": "provincia",
    "Textfield-7": "pais",
    "Textfield-8": "telf_contacto",
    "Textfield-9": "email",
    "Textfield-10": "registro_civil_inscripcion",
    "Textfield-11": "tomo",
    "Textfield-12": "folio",
    "Textfield-13": "fecha_ejercicio_opcion",
}

#: Anexo I's AcroForm template uses a different, non-sequential field layout
#: (``Textfieldad`` for the registry destination, ``Textfield-3a`` for
#: domicilio shifts provincia/pais/telf/email up by one index relative to
#: Anexo III/IV) and has no tomo/folio/fecha_ejercicio_opcion fields, but adds
#: a free-text "nacionalidad de origen del progenitor/abuelo" box.
_ANEXO_I_FIELD_MAP: dict[str, str] = {
    "Textfieldad": "registro_civil_inscripcion",
    "Textfield": "nombre_solicitante",
    "Textfield-0": "apellido_padre",
    "Textfield-1": "apellido_madre",
    "Textfield-2": "nacionalidad",
    "Textfield-3": "estado_civil",
    "Textfield-4": "numero_id",
    "Textfield-3a": "domicilio",
    "Textfield-5": "provincia",
    "Textfield-6": "pais",
    "Textfield-7": "telf_contacto",
    "Textfield-8": "email",
    "Textfield-9": "nacionalidad_origen_progenitor",
}

_ANEXO_VARIANT_FIELD_MAPS: dict[AnexoVariant, dict[str, str]] = {
    "ANEXO_I": _ANEXO_I_FIELD_MAP,
    "ANEXO_III_IV": _ANEXO_III_IV_FIELD_MAP,
}


def _detect_anexo_variant(normalized_text: str) -> Optional[AnexoVariant]:
    """Identify which Anexo template a document's header text names.

    Whitespace is stripped before matching because the real Anexo IV header
    is letter-spaced in the source PDF ("A N E X O  IV"). Checked
    longest-prefix-first ("ANEXOIV"/"ANEXOIII" before bare "ANEXOI") so e.g.
    "ANEXO III" is never mistaken for "ANEXO I".
    """
    collapsed = "".join(normalized_text.upper().split())
    if "ANEXOIV" in collapsed or "ANEXOIII" in collapsed:
        return "ANEXO_III_IV"
    if "ANEXOI" in collapsed:
        return "ANEXO_I"
    return None


def _adapt_solicitud(
    acroform_fields: dict[str, str],
    xfa_fields: dict[str, str],
    anexo_variant: Optional[AnexoVariant] = None,
) -> tuple[SolicitudData, tuple[str, ...]]:
    """Map form fields to :class:`SolicitudData`, honestly recording gaps.

    When ``anexo_variant`` is recognized, real Anexo I/III/IV AcroForm fields
    use generic ``Textfield-N`` names rather than printed labels, so they are
    matched positionally via the per-template maps above. Otherwise (or for
    any field a positional map doesn't cover), fields fall back to exact
    (normalized) label matching — no fuzzy/substring matching, since no
    fixture evidence yet justifies anything more. Checkbox-only fields (civil
    vecindad choice, Anexo IV's 5-way fundamento, document checklists) are
    never inferred from a form field's string value in this step (no fixture
    shows what a reliable checkbox "selected" signal looks like); they are
    always left at their defaults and always reported missing. ``firma`` is
    signed iff its matched field has a non-empty (post-strip) value — no
    AcroForm signature-widget or text-layer heuristic yet, per user decision;
    revisit if the Tier-3 re-run shows this is insufficient.
    """
    fields = {**acroform_fields, **xfa_fields}
    positional_map = _ANEXO_VARIANT_FIELD_MAPS.get(anexo_variant, {})
    matched: dict[str, str] = {}
    for field_name, value in fields.items():
        attr = positional_map.get(field_name)
        if attr is None:
            attr = _SOLICITUD_LABEL_INDEX.get(_normalize_label(field_name))
        if attr is not None and attr != "firma":
            matched[attr] = value
        elif attr == "firma":
            matched["firma"] = bool(value and value.strip())

    missing = tuple(
        attr
        for attr in SOLICITUD_SOURCE_LABELS
        if (attr == "firma" and not matched.get("firma")) or (attr != "firma" and attr not in matched)
    )
    return SolicitudData(**matched), missing


def _adapt_id(fields: dict[str, str]) -> tuple[IdData, tuple[str, ...]]:
    """Build IdData from already canonical-attr-keyed field_extraction fields."""
    missing = tuple(attr for attr in IdData.model_fields if attr not in fields)
    return IdData(**fields), missing


def _adapt_certificado(fields: dict[str, str]) -> tuple[CertificadoData, tuple[str, ...]]:
    """Build CertificadoData; apostillado follows firma's non-empty-is-true rule (item 3)."""
    kwargs = {k: v for k, v in fields.items() if k != "apostillado"}
    kwargs["apostillado"] = bool(fields.get("apostillado"))
    missing = tuple(
        attr
        for attr in CertificadoData.model_fields
        if (attr == "apostillado" and not kwargs["apostillado"])
        or (attr != "apostillado" and attr not in fields)
    )
    return CertificadoData(**kwargs), missing


def adapt_document(result: OcrResult) -> CanonicalDocument:
    """Adapt one persisted ``OcrResult`` into a :class:`CanonicalDocument`."""
    normalized_text = normalize_extracted_text(result.extraction.extracted_text)

    if result.classification is None:
        # Pre-M-2.5 row: classification itself is unknown. Never guess.
        return CanonicalDocument(
            case_id=result.case_id,
            document_id=result.document_id,
            source_file=result.source_file,
            document_type=DocumentType.OTHER,
            generation=None,
            missing_fields=("document_type", "generation"),
            normalized_text=normalized_text,
        )

    document_type = result.classification.document_type
    generation = result.classification.generation

    solicitud = identidad = certificado = None
    missing_fields: tuple[str, ...] = ()

    if document_type is DocumentType.APPLICATION:
        anexo_variant = _detect_anexo_variant(normalized_text)
        solicitud, missing_fields = _adapt_solicitud(
            result.extraction.acroform_fields,
            result.extraction.xfa_fields,
            anexo_variant,
        )
    elif document_type is DocumentType.ID_DOCUMENT:
        if result.field_extraction is not None:
            identidad, missing_fields = _adapt_id(result.field_extraction.fields)
        else:
            missing_fields = tuple(IdData.model_fields.keys())
    elif document_type is DocumentType.BIRTH_CERT:
        if result.field_extraction is not None:
            certificado, missing_fields = _adapt_certificado(result.field_extraction.fields)
        else:
            missing_fields = tuple(CertificadoData.model_fields.keys())
    # OTHER / EMPTY: nothing structured is expected, missing_fields stays ().

    return CanonicalDocument(
        case_id=result.case_id,
        document_id=result.document_id,
        source_file=result.source_file,
        document_type=document_type,
        generation=generation,
        solicitud=solicitud,
        identidad=identidad,
        certificado=certificado,
        missing_fields=missing_fields,
        normalized_text=normalized_text,
    )


def build_case_bundle(store: StatusStore, case_id: str) -> CaseBundle:
    """Adapt every registered document of a case into a :class:`CaseBundle`.

    Raises :class:`MissingExtractionError` if any registered document has no
    persisted extraction yet — see that class's docstring for why this is a
    raise, not a skip.
    """
    documents = []
    for status in store.list_for_case(case_id):
        result = get_extraction(store, case_id, status.document_id)
        if result is None:
            raise MissingExtractionError(
                f"no persisted extraction for case={case_id!r} "
                f"document={status.document_id!r} (stage={status.stage.value})"
            )
        documents.append(adapt_document(result))

    return CaseBundle(case_id=case_id, documents=tuple(documents))
