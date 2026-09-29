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


def _adapt_solicitud(
    acroform_fields: dict[str, str], xfa_fields: dict[str, str]
) -> tuple[SolicitudData, tuple[str, ...]]:
    """Map form fields to :class:`SolicitudData`, honestly recording gaps.

    Only exact (normalized) label matches are used — no fuzzy/substring
    matching, since no fixture evidence yet justifies anything more. ``firma``
    is never inferred from a form field's string value in this step (no
    fixture shows what a checkbox value looks like); it is always left at its
    default ``False`` and always reported missing.
    """
    fields = {**acroform_fields, **xfa_fields}
    matched: dict[str, str] = {}
    for field_name, value in fields.items():
        attr = _SOLICITUD_LABEL_INDEX.get(_normalize_label(field_name))
        if attr is not None and attr != "firma":
            matched[attr] = value

    missing = tuple(
        attr for attr in SOLICITUD_SOURCE_LABELS if attr not in matched
    )
    return SolicitudData(**matched), missing


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
        solicitud, missing_fields = _adapt_solicitud(
            result.extraction.acroform_fields, result.extraction.xfa_fields
        )
    elif document_type is DocumentType.ID_DOCUMENT:
        missing_fields = tuple(IdData.model_fields.keys())
    elif document_type is DocumentType.BIRTH_CERT:
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
