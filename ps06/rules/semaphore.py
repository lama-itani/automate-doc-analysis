"""Per-document checks + VERDE/AMARILLO/ROJO decisioning (M-4, step 7).

The AEAD source's "Semaphore Engine" combines the Lineage Engine's resolved
roles/flags (:mod:`ps06.rules.lineage`) with per-document-only checks (is a
document missing key fields? has an ID expired?) and the 5-expected-document
structural checklist into one traffic-light decision per case.

This module owns: ``DOCUMENT_INCOMPLETE``/``DOCUMENTO_VENCIDO`` (the two
flags that only need a single document, never grouping), the expected-
document structural check, all ROJO/AMARILLO/VERDE severity classification
(via :class:`~ps06.rules.rules_config.SeverityMapping`), the corroboration
gate, and assembling a snapshot-ready decision object. It never detects a
flag that requires grouping raw values by resolved person — that is
:mod:`ps06.rules.lineage`'s job; this module only classifies severity from
flags lineage.py already raised.

Design decision confirmed with the user, not dictated by the extracted
source: a resolved role needs >=2 distinct source documents to count as
*corroborated*. Even with 5/5 expected documents present and zero conflict
flags, a role backed by only 1 document is downgraded to AMARILLO rather
than VERDE -- directly relevant today since the adapter (step 4) doesn't yet
populate ``identidad``/``certificado`` structured fields, so a case can look
flag-free simply because there was nothing to compare.

No I/O, no case/document identity beyond what's in ``CaseBundle`` already.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from ps06.classification.classifier import DocumentType
from ps06.rules import flags
from ps06.rules.lineage import LineageResult
from ps06.rules.normalize import parse_date
from ps06.rules.rules_config import RulesConfig
from ps06.rules.schemas import CanonicalDocument, CaseBundle, Generation

_MIN_CORROBORATING_DOCUMENTS = 2

Estado = Literal["VERDE", "AMARILLO", "ROJO"]


class DocumentEvaluation(BaseModel):
    """Per-document-only flags for one :class:`CanonicalDocument`."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_type: DocumentType
    generation: Optional[Generation] = None
    flags: tuple[str, ...] = ()


class ExpectedDocumentCheck(BaseModel):
    """Whether one of the 5 checklist entries is present in the case bundle."""

    model_config = ConfigDict(frozen=True)

    label: str
    document_type: DocumentType
    generation: Optional[Generation] = None
    present: bool
    matched_document_ids: tuple[str, ...] = ()


class SemaphoreDecision(BaseModel):
    """The final traffic-light decision for a case, with full supporting detail.

    Fields are kept separate rather than collapsed into one opaque value so
    the future ``snapshot.py`` (step 8) can build its S1-S3 rows directly
    from this object without re-deriving any of this module's logic.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    estado: Estado
    justificacion: str
    problemas: tuple[str, ...] = ()
    expected_document_checks: tuple[ExpectedDocumentCheck, ...] = ()
    document_evaluations: tuple[DocumentEvaluation, ...] = ()
    lineage: LineageResult
    rojo_flags: tuple[str, ...] = ()
    amarillo_flags: tuple[str, ...] = ()
    missing_documents: tuple[str, ...] = ()


def evaluate_document(doc: CanonicalDocument, evaluation_date: date) -> DocumentEvaluation:
    """Per-document-only flags: ``DOCUMENT_INCOMPLETE`` and ``DOCUMENTO_VENCIDO``.

    ``DOCUMENTO_VENCIDO`` is always ``False`` today because ``identidad`` is
    not yet populated by the adapter (step 4) -- this never crashes on that
    absence, it simply has no expiry date to compare.
    """
    hit_flags: list[str] = []
    if flags.document_incomplete(doc.missing_fields):
        hit_flags.append("DOCUMENT_INCOMPLETE")

    fecha_vencimiento = parse_date(doc.identidad.fecha_vencimiento) if doc.identidad else None
    if flags.documento_vencido(fecha_vencimiento, evaluation_date):
        hit_flags.append("DOCUMENTO_VENCIDO")

    return DocumentEvaluation(
        document_id=doc.document_id,
        document_type=doc.document_type,
        generation=doc.generation,
        flags=tuple(hit_flags),
    )


def check_expected_documents(
    bundle: CaseBundle, config: RulesConfig, lineage: Optional[LineageResult] = None
) -> tuple[ExpectedDocumentCheck, ...]:
    """Match each of ``config.expected_documents`` against the case bundle.

    ``lineage``, when given, also counts a birth cert inferred (not tagged)
    as the expected generation.
    """
    checks: list[ExpectedDocumentCheck] = []
    for expected in config.expected_documents:
        candidates = bundle.by_type(expected.document_type)
        if expected.generation is not None:
            candidates = tuple(
                d
                for d in candidates
                if d.generation == expected.generation
                or (lineage is not None and lineage.inferred_generations.get(d.document_id) == expected.generation)
            )
        checks.append(
            ExpectedDocumentCheck(
                label=expected.label,
                document_type=expected.document_type,
                generation=expected.generation,
                present=len(candidates) > 0,
                matched_document_ids=tuple(d.document_id for d in candidates),
            )
        )
    return tuple(checks)


def _corroboration_failures(lineage: LineageResult) -> tuple[str, ...]:
    """Roles that resolved a person but lack >=2 corroborating documents."""
    failures = []
    for role in lineage.roles:
        if role.person is not None and len(role.corroborating_document_ids) < _MIN_CORROBORATING_DOCUMENTS:
            failures.append(role.generation)
    return tuple(failures)


def _justificacion(estado: Estado) -> str:
    if estado == "ROJO":
        return "Se detectaron conflictos que impiden validar el caso automáticamente."
    if estado == "AMARILLO":
        return "El caso requiere revisión manual: faltan documentos o hay observaciones menores."
    return "El caso cumple todos los controles automáticos sin observaciones."


def evaluate_semaphore(
    bundle: CaseBundle,
    lineage: LineageResult,
    config: RulesConfig,
    evaluation_date: date,
) -> SemaphoreDecision:
    """Combine per-document checks + lineage flags into a final decision."""
    expected_document_checks = check_expected_documents(bundle, config, lineage)
    missing_documents = tuple(c.label for c in expected_document_checks if not c.present)

    document_evaluations = tuple(evaluate_document(doc, evaluation_date) for doc in bundle.documents)
    per_doc_flags: set[str] = set()
    for evaluation in document_evaluations:
        per_doc_flags.update(evaluation.flags)

    all_flags = per_doc_flags | set(lineage.all_flags)
    rojo_hits = tuple(sorted(all_flags & set(config.severity.rojo)))
    amarillo_hits = tuple(sorted(all_flags & set(config.severity.amarillo)))

    problemas: list[str] = list(missing_documents) + list(rojo_hits) + list(amarillo_hits)

    if rojo_hits:
        estado: Estado = "ROJO"
    elif missing_documents or amarillo_hits:
        estado = "AMARILLO"
    else:
        corroboration_failures = _corroboration_failures(lineage)
        if corroboration_failures:
            estado = "AMARILLO"
            problemas.append(
                "Corroboración insuficiente (<2 documentos) para: "
                + ", ".join(corroboration_failures)
            )
        else:
            estado = "VERDE"

    return SemaphoreDecision(
        case_id=bundle.case_id,
        estado=estado,
        justificacion=_justificacion(estado),
        problemas=tuple(problemas),
        expected_document_checks=expected_document_checks,
        document_evaluations=document_evaluations,
        lineage=lineage,
        rojo_flags=rojo_hits,
        amarillo_flags=amarillo_hits,
        missing_documents=missing_documents,
    )
